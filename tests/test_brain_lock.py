"""brain_lock - one brain per memory store, for main_discord.py and main_local.py.

Before this, nothing stopped two brains running on one store. main_local
refused only when something already held the panel's port, which a brain with
the panel switched off never does, and main_discord did not check at all. The
two would then share an embedded Chroma that one process may open at a time.

What matters, and is tested through real processes where it can be: a second
brain is refused and told which one is running; the lock goes when its holder
dies, however it dies; a lock file left behind stops nobody; and where locks
cannot be taken at all, the brain still starts and says so.
"""
import ast
import errno
import os
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import brain_lock

REPO = Path(__file__).resolve().parents[1]

# A second process that takes the lock and keeps it until the test ends it.
# It prints its own process ID: on Windows a venv's python.exe is a launcher,
# so Popen.pid is not the interpreter that holds the lock.
_HOLDER = textwrap.dedent("""
    import os, sys
    import brain_lock
    brain_lock.hold(sys.argv[1], sys.argv[2])
    print("held", os.getpid(), flush=True)
    sys.stdin.read()
""")

# The same, after forking one child that outlives it, as document_engine's
# ingestion workers can when a brain is killed on Linux. The child never forks
# again, and is gone within 30 s whatever happens.
_FORKING_HOLDER = textwrap.dedent("""
    import os, sys, time
    import brain_lock
    brain_lock.hold(sys.argv[1], sys.argv[2])
    child = os.fork()
    if child == 0:
        time.sleep(30)
        os._exit(0)
    print("held", os.getpid(), child, flush=True)
    sys.stdin.read()
""")


def _readline_within(stream, timeout: float) -> str:
    box = {"line": ""}
    t = threading.Thread(target=lambda: box.update(line=stream.readline()), daemon=True)
    t.start()
    t.join(timeout)
    return box["line"]


def _kill_quietly(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _StoreCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.store = os.path.join(tmp.name, "chroma_store")
        self.path = brain_lock.lock_path(self.store)
        # Cleanups run last-added first: the lock is dropped before the
        # directory goes, which Windows needs.
        self.addCleanup(brain_lock.release_for_tests, self.store)

    def another_process_holding(self, entry_point: str, script: str = _HOLDER) -> int:
        """Start a process holding the lock; return the PID that holds it."""
        proc = subprocess.Popen(
            [sys.executable, "-c", script, entry_point, self.store], cwd=REPO,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True)
        self.addCleanup(self._end, proc)
        line = _readline_within(proc.stdout, 30).split()
        if line[:1] != ["held"]:
            # Ended first, so its stderr reaches EOF; read before the cleanup
            # above closes the stream.
            if proc.poll() is None:
                proc.kill()
            proc.wait(10)
            self.fail(f"the holder never took the lock: {line} {proc.stderr.read()}")
        self.holder = proc
        self.holder_line = line
        return int(line[1])

    @staticmethod
    def _end(proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            proc.kill()
        proc.wait(10)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            stream.close()

    def held_in_this_process_by_hand(self, note: bytes) -> int:
        """A descriptor of our own holding the lock with `note` in the file, so
        that hold() meets a holder without a second process."""
        os.makedirs(self.store, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT)
        self.addCleanup(os.close, fd)
        brain_lock._lock(fd)
        os.lseek(fd, 0, os.SEEK_SET)    # _lock leaves Windows at the locked byte
        os.write(fd, note)
        return fd

    def take_within(self, seconds: float) -> None:
        """hold(), retried while a holder that was just killed lets go."""
        deadline = time.monotonic() + seconds
        while True:
            try:
                brain_lock.hold("main_local.py", self.store)
                return
            except brain_lock.BrainBusy as e:
                if time.monotonic() > deadline:
                    self.fail(f"the lock outlived the process that held it: {e}")
                time.sleep(0.05)

    def assert_locked(self):
        """Proved by trying: a second descriptor cannot take it."""
        fd = os.open(self.path, os.O_RDWR)
        try:
            with self.assertRaises(brain_lock._Busy):
                brain_lock._lock(fd)
        finally:
            os.close(fd)


class TestOneBrainPerStore(_StoreCase):
    def test_a_second_brain_is_refused_and_told_which_one_is_running(self):
        """Through a real second process. On Windows this also proves the note
        is readable while the lock is held, since Windows locks are mandatory:
        the locked byte is past the note for exactly that reason."""
        pid = self.another_process_holding("main_discord.py")
        with self.assertRaises(brain_lock.BrainBusy) as caught:
            brain_lock.hold("main_local.py", self.store)
        message = str(caught.exception)
        self.assertIn(f"main_discord.py, process {pid}, since ", message)
        self.assertIn(os.path.abspath(self.store), message)
        self.assertIn("one at a time", message)
        self.assertNotIn(self.path, brain_lock._HELD)

    def test_the_lock_goes_when_its_holder_is_killed(self):
        """The reason this is an OS lock and not a PID file: a brain that
        crashed, or was killed, cannot leave the next one locked out."""
        self.another_process_holding("main_discord.py")
        self.holder.kill()
        self.holder.wait(10)
        self.take_within(5)
        self.assert_locked()

    @unittest.skipIf(os.name == "nt", "Windows does not fork")
    def test_a_forked_child_that_outlives_its_holder_does_not_keep_it(self):
        """On Linux a flock lasts until the last copy of its descriptor is
        closed, and a fork copies them all. document_engine's ingestion pool
        forks there, so a worker orphaned by a killed brain would have kept the
        next one out until someone found and killed it by hand."""
        self.another_process_holding("main_discord.py", _FORKING_HOLDER)
        orphan = int(self.holder_line[2])
        self.addCleanup(_kill_quietly, orphan)
        self.holder.kill()
        self.holder.wait(10)
        os.kill(orphan, 0)          # alive, so it is not what let the lock go
        self.take_within(5)
        self.assert_locked()

    @unittest.skipIf(os.name == "nt", "flock is POSIX")
    def test_a_refusal_from_an_smb_share_counts_as_busy(self):
        """Linux's CIFS client refuses a held flock with EACCES, not
        EWOULDBLOCK. Taken for "cannot lock here", the second brain would
        start beside the first."""
        import fcntl
        refused = PermissionError(errno.EACCES, "Permission denied")
        with patch.object(fcntl, "flock", side_effect=refused):
            with self.assertRaises(brain_lock.BrainBusy):
                brain_lock.hold("main_local.py", self.store)

    def test_a_lock_file_left_behind_stops_nobody_and_gets_our_note(self):
        os.makedirs(self.store)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("main_discord.py, process 99999, since last week\n" + "x" * 500)
        brain_lock.hold("main_local.py", self.store)
        with open(self.path, encoding="utf-8") as f:
            note = f.read()
        self.assertTrue(note.startswith(f"main_local.py, process {os.getpid()}, since "), note)
        # Replaced, not written over the front of the old one.
        self.assertNotIn("99999", note)
        self.assertNotIn("xxx", note)
        self.assertEqual(note.count("\n"), 1, note)

    def test_it_holds_the_lock_once_taken(self):
        brain_lock.hold("main_local.py", self.store)
        self.assert_locked()

    def test_taking_it_again_in_the_process_that_holds_it_does_nothing(self):
        brain_lock.hold("main_local.py", self.store)
        brain_lock.hold("main_local.py", self.store)
        self.assertEqual([p for p in brain_lock._HELD if p == self.path], [self.path])
        self.assert_locked()

    def test_the_lock_lives_in_the_memory_store_and_makes_it_if_needed(self):
        self.assertFalse(os.path.exists(self.store))
        brain_lock.hold("main_local.py", self.store)
        self.assertTrue(os.path.isdir(self.store))
        self.assertEqual(self.path, os.path.join(os.path.abspath(self.store), "brain.lock"))
        self.assertTrue(os.path.isfile(self.path))

    def test_released_it_can_be_taken_again(self):
        brain_lock.hold("main_local.py", self.store)
        brain_lock.release_for_tests(self.store)
        fd = os.open(self.path, os.O_RDWR)
        try:
            brain_lock._lock(fd)        # does not raise: nobody holds it
        finally:
            os.close(fd)


class TestWhatTheRefusedBrainIsTold(_StoreCase):
    def test_the_holders_note_is_shown_as_printable_text_and_not_too_much(self):
        """The note comes from a file and goes into a log line."""
        self.held_in_this_process_by_hand(
            b"\x1b[31mmain_discord.py\r\n2026-11-12 ERROR forged line\x00" + b"x" * 5000)
        with self.assertRaises(brain_lock.BrainBusy) as caught:
            brain_lock.hold("main_local.py", self.store)
        holder = caught.exception.holder
        self.assertIn("main_discord.py", holder)
        for ch in "\x1b\r\n\x00":
            self.assertNotIn(ch, holder)
        self.assertLessEqual(len(holder), brain_lock._NOTE_MAX_CHARS)
        self.assertNotIn("\n", str(caught.exception))

    def test_a_holder_that_left_no_note_is_still_refused(self):
        self.held_in_this_process_by_hand(b"")
        with self.assertRaises(brain_lock.BrainBusy) as caught:
            brain_lock.hold("main_local.py", self.store)
        self.assertEqual(caught.exception.holder, "it left no note saying which")

    def test_a_note_that_cannot_be_read_is_not_called_missing(self):
        """On a Linux SMB mount the holder's lock covers its note too. It did
        leave one; saying it left none would send someone looking for a bug."""
        self.held_in_this_process_by_hand(b"main_discord.py, process 7, since 08:00\n")
        real_read = os.read

        def read(fd, n):
            # Only the note's read: os is patched for the whole process.
            if n == 4 * brain_lock._NOTE_MAX_CHARS:
                raise PermissionError(errno.EACCES, "Permission denied")
            return real_read(fd, n)

        with patch.object(brain_lock.os, "read", side_effect=read), \
             self.assertRaises(brain_lock.BrainBusy) as caught:
            brain_lock.hold("main_local.py", self.store)
        self.assertEqual(caught.exception.holder,
                         "its note could not be read: Permission denied")


class TestWhereLocksCannotBeTaken(_StoreCase):
    """The lock guards against a mistake. Where it cannot be taken at all, the
    brain starts as it did before the lock existed, and says so."""

    def assert_starts_unguarded(self):
        with self.assertLogs("brain_lock", level="WARNING") as logs:
            self.assertIsNone(brain_lock.hold("main_local.py", self.store))
        self.assertNotIn(self.path, brain_lock._HELD)
        self.assertIn("one at a time", " ".join(logs.output))
        return " ".join(logs.output)

    def test_a_filesystem_that_does_not_lock(self):
        """The lock call's own refusal, other than "someone has it"."""
        if os.name == "nt":
            import msvcrt
            target, name = msvcrt, "locking"
        else:
            import fcntl
            target, name = fcntl, "flock"
        with patch.object(target, name, side_effect=OSError(errno.ENOLCK, "No locks available")):
            self.assertIn("No locks available", self.assert_starts_unguarded())

    def test_a_store_path_that_is_a_file(self):
        with open(self.store, "w", encoding="utf-8"):
            pass
        self.assert_starts_unguarded()

    def test_a_lock_file_that_cannot_be_opened_is_not_mistaken_for_busy(self):
        """On Windows, opening a directory fails with EACCES: the same errno a
        held lock gives. Only the lock call's EACCES means busy."""
        os.makedirs(self.path)
        self.assert_starts_unguarded()

    @unittest.skipIf(os.name == "nt", "a symlink needs privileges on Windows")
    def test_a_symlinked_lock_file_is_not_followed(self):
        """The note is written over whatever the lock file is."""
        os.makedirs(self.store)
        target = os.path.join(self.tmp, "precious.txt")
        with open(target, "w", encoding="utf-8") as f:
            f.write("keep me")
        os.symlink(target, self.path)
        self.assert_starts_unguarded()
        with open(target, encoding="utf-8") as f:
            self.assertEqual(f.read(), "keep me")

    def test_a_note_that_cannot_be_written_still_leaves_the_lock_held(self):
        with patch.object(brain_lock.os, "ftruncate", side_effect=OSError(errno.EIO, "I/O error")), \
             self.assertLogs("brain_lock", level="WARNING") as logs:
            brain_lock.hold("main_local.py", self.store)
        self.assertIn("could not write who holds it", " ".join(logs.output))
        self.assert_locked()


def _dotted(node: ast.AST) -> str:
    return ast.unparse(node) if isinstance(node, (ast.Name, ast.Attribute)) else ""


class TestEveryEntryPointTakesIt(unittest.TestCase):
    ENTRY_POINTS = {"main_discord", "main_local", "main_telegram"}

    def test_every_module_that_starts_a_brain_locks_first_and_stops_when_refused(self):
        """Found by what they do, not from a list: only a running brain starts
        the metrics server. preflight is the one exception, and is skipped: it
        starts the server only to see that it answers, and never runs the
        brain. This is all main_telegram gets, since CI does not install the
        telegram extra it needs to be run."""
        found = set()
        for path in sorted(REPO.glob("*.py")):
            if path.stem in ("observability", "preflight"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            parents = {child: node for node in ast.walk(tree)
                       for child in ast.iter_child_nodes(node)}

            def scope(node):
                while not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                    node = parents[node]
                return node

            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
            starts = [c for c in calls if _dotted(c.func).endswith("start_metrics_server")]
            if not starts:
                continue
            found.add(path.stem)
            holds = [c for c in calls if _dotted(c.func) == "brain_lock.hold"]
            for start in starts:
                with self.subTest(module=path.name, line=start.lineno):
                    mine = [h for h in holds if scope(h) is scope(start)
                            and h.lineno < start.lineno]
                    self.assertTrue(mine, "starts the metrics server without taking "
                                          "brain_lock.hold first, in the same scope")
                    hold = mine[-1]
                    self.assertEqual(ast.literal_eval(hold.args[0]), path.name,
                                     "the note should name the script that holds it")
                    self.assertIn(_dotted(hold.args[1]), ("CHROMA_DB_PATH",
                                                          "fritz_utils.CHROMA_DB_PATH"))
                    # Inside a try whose BrainBusy handler stops the process.
                    node = parents[hold]
                    while not isinstance(node, ast.Try):
                        node = parents[node]
                    [handler] = [h for h in node.handlers
                                 if _dotted(h.type) == "brain_lock.BrainBusy"]
                    body = ast.unparse(handler)
                    self.assertTrue("sys.exit(1)" in body or "return 1" in body, body)
        self.assertEqual(found, self.ENTRY_POINTS)


class TestBothEntryPointsTakeIt(_StoreCase):
    """The real scripts, refused by a lock this test holds. Were either not to
    take it, it would start serving (or log in) and the run would time out."""

    TIMEOUT = 90     # main_discord imports the whole graph before it gets here

    def run_entry_point(self, script: str, **env) -> subprocess.CompletedProcess:
        brain_lock.hold("the test", self.store)
        try:
            return subprocess.run(
                [sys.executable, script], cwd=REPO, capture_output=True, text=True,
                timeout=self.TIMEOUT,
                env={**os.environ, "CHROMA_DB_PATH": self.store,
                     "DB_NAME": os.path.join(self.tmp, "fritz.db"),
                     "ROOT_USER": "discord-000000000000000000",
                     "AUTO_DOWNLOAD_NLTK": "false",
                     "METRICS_PORT": str(_free_port()),
                     "ADMIN_PANEL_PORT": str(_free_port()), **env})
        except subprocess.TimeoutExpired:
            self.fail(f"{script} was still running after {self.TIMEOUT}s: it "
                      "started beside a brain holding the lock")

    def assert_refused(self, result: subprocess.CompletedProcess):
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 1, output)
        self.assertIn("Another copy of the brain is already running on", output)
        self.assertIn(f"the test, process {os.getpid()}", output)

    def test_main_local(self):
        self.assert_refused(self.run_entry_point(
            "main_local.py", ADMIN_PANEL_PASSWORD="pw"))

    def test_main_discord(self):
        """A placeholder token, so a main_discord that went on past the lock
        fails at Discord's login instead: also exit status 1, and the refusal
        is already in the log by then. What tells the two apart is that the
        login's failure is an exception out of client.run."""
        result = self.run_entry_point("main_discord.py", DISCORD_BOT_TOKEN="test-placeholder")
        self.assert_refused(result)
        self.assertNotIn("Traceback", result.stdout + result.stderr,
                         "it went on past the refusal and tried to log in")


if __name__ == "__main__":
    unittest.main()
