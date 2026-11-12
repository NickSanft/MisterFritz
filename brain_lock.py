"""One brain per memory store: the lock every entry point to the brain takes.

main_discord.py, main_local.py and main_telegram.py are three ways to run the
same brain, and two of them must never run on the same data at once. The memory store is an
embedded Chroma, which is not safe to open from two processes; the two would
also share the conversation database, the web and metrics ports, and, with two
Discord logins, the bot itself.

Each entry point calls hold() before it starts anything that listens or logs
in. It takes an exclusive lock on brain.lock inside CHROMA_DB_PATH and keeps it
for the life of the process. The operating system drops the lock when the
process ends, however it ends, so a crash or a kill leaves nothing to clean up:
the file stays, the lock does not. A PID file would be stale after every crash
and would need exactly that cleanup. A forked child (document_engine's
ingestion pool forks on Linux) closes its copy of the descriptor at once,
because on Linux a flock lasts until the last copy is closed, and a worker
orphaned by a killed brain would otherwise hold it for the next one.

The holder writes a note into the file - its entry point, process ID and start
time - so that the brain refused can say which one is in the way.

Where the lock cannot be taken at all (the file cannot be created, or the
filesystem does not support locks), hold() logs a warning and lets the brain
start without it. The lock guards against a mistake; refusing to start on such
a filesystem would be a worse failure than the one it prevents.

Not covered: a brain in a container and one on the host sharing a bind-mounted
store, since Docker Desktop's file sharing may not carry locks across.
"""
from __future__ import annotations

import errno
import logging
import os
import time

logger = logging.getLogger(__name__)

LOCK_FILE_NAME = "brain.lock"

# The byte that is locked: one far past the note. On Windows a lock is
# mandatory, so locking the note's own bytes would stop the refused process
# from reading it. Both systems allow locking past the end of a file.
_LOCK_OFFSET = 1 << 20

# The note is read from a file and goes into a log line, so it is shown only
# as printable characters, and only this many of them.
_NOTE_MAX_CHARS = 200

# Lock path -> the descriptor holding it, open for the life of the process.
# os.open's descriptors close on exec (PEP 446), so a subprocess never has one;
# a fork copies them all, which _forget_in_child undoes.
_HELD: dict[str, int] = {}


class BrainBusy(RuntimeError):
    """Another process holds the brain's lock on this memory store."""

    def __init__(self, store: str, holder: str) -> None:
        self.store = store
        self.holder = holder
        super().__init__(
            f"Another copy of the brain is already running on {store} ({holder}). "
            "main_discord.py, main_local.py and main_telegram.py all share the "
            "conversation database and the memory store, so run one at a time: "
            "stop that one, then start this again."
        )


class _Busy(Exception):
    """The lock call itself said another holder has it."""


def lock_path(store: str) -> str:
    return os.path.abspath(os.path.join(store, LOCK_FILE_NAME))


def hold(entry_point: str, store: str) -> None:
    """Take the brain's lock on `store` for the rest of this process's life.

    Raises BrainBusy, naming the holder, if another process has it. Calling it
    again in the process that holds it does nothing.
    """
    path = lock_path(store)
    if path in _HELD:
        return
    try:
        os.makedirs(store, exist_ok=True)
        # Never through a symlink: the note is written over whatever it opens.
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o644)
    except OSError as e:
        _start_unguarded(path, e)
        return
    try:
        _lock(fd)
    except _Busy:
        holder = _read_note(fd)
        os.close(fd)
        raise BrainBusy(os.path.abspath(store), holder) from None
    except OSError as e:
        os.close(fd)
        _start_unguarded(path, e)
        return
    _HELD[path] = fd
    _write_note(fd, path, entry_point)


def release_for_tests(store: str) -> None:
    """Drop the lock this process holds on `store`. Only tests call this: a
    brain keeps its lock until it exits."""
    fd = _HELD.pop(lock_path(store), None)
    if fd is not None:
        os.close(fd)


def _lock(fd: int) -> None:
    """Lock one byte of `fd` without waiting. _Busy if another holder has it;
    any other OSError means locks cannot be taken here at all."""
    if os.name == "nt":
        import msvcrt
        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as e:
            # Measured on Windows 11: a byte locked by another handle, in this
            # process or another, refuses with EACCES. EDEADLOCK is what the
            # C runtime reports for the same when it has retried.
            if e.errno in (errno.EACCES, errno.EDEADLOCK):
                raise _Busy from e
            raise
    else:
        import fcntl
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            # EWOULDBLOCK on a local filesystem; EACCES from Linux's CIFS
            # client (an SMB share) for the same refusal. A security module
            # that forbids locking also says EACCES, but then SQLite could not
            # lock its database in this same store either: the brain could
            # not run there anyway.
            if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                raise _Busy from e
            raise


def _forget_in_child() -> None:
    """In a forked child: close its copies of the held descriptors.

    Only close them. flock(LOCK_UN) here would release the parent's lock too,
    since parent and child share the one open file description.
    """
    for fd in _HELD.values():
        try:
            os.close(fd)
        except OSError:
            pass
    _HELD.clear()


if hasattr(os, "register_at_fork"):        # not on Windows, which never forks
    os.register_at_fork(after_in_child=_forget_in_child)


def _write_note(fd: int, path: str, entry_point: str) -> None:
    note = (f"{entry_point}, process {os.getpid()}, since "
            f"{time.strftime('%Y-%m-%d %H:%M:%S %z')}\n")
    try:
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, note.encode("utf-8"))
    except OSError as e:
        # The lock is held either way; only the note for the next one is lost.
        logger.warning("Holding %s, but could not write who holds it: %s", path, e)


def _read_note(fd: int) -> str:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, 4 * _NOTE_MAX_CHARS)
    except OSError as e:
        # Linux's CIFS client sends a flock to the server as a lock on the
        # whole file, note included, so there the holder's note is unreadable.
        return f"its note could not be read: {e.strerror or e}"
    text = "".join(c for c in raw.decode("utf-8", "replace") if c.isprintable())
    return text.strip()[:_NOTE_MAX_CHARS] or "it left no note saying which"


def _start_unguarded(path: str, error: OSError) -> None:
    logger.warning(
        "Could not lock %s (%s), so nothing stops a second brain starting on "
        "this memory store. Run main_discord.py, main_local.py and "
        "main_telegram.py one at a time.",
        path, error,
    )
