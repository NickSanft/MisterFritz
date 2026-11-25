#!/usr/bin/env python3
"""His voice's files, fetched into his folder (plan 14, PR 10).

    python desktop/scripts/fetch_voice.py              # into %APPDATA%\\com.divora.fritz\\tts
    python desktop/scripts/fetch_voice.py --dest DIR   # anywhere else
    python desktop/scripts/fetch_voice.py --check      # are they all there, and right? (no network)

The desktop app's `kokoro_tts.rs` speaks with what this puts there, and
says "not installed" until it has. About 122 MB is fetched, once:

- Kokoro (the model, its config and DivoraVoice's pack of seven voices) and
  espeak-ng 1.52.0 (its program, its library and its data), from the
  DivoraVoiceMod release `voice-assets-v2` on github.com, as DivoraVoice
  ships them;
- ONNX Runtime's CPU build, `onnxruntime.dll`, out of PyPI's onnxruntime
  1.26.0 wheel (the version DivoraVoice's Kokoro was checked against on the
  CPU). Only the DLL is kept; nothing of the wheel is installed or run.

Every file is checked against the size and SHA-256 written here before it is
kept (written beside it as `<name>.part`, then renamed), and one already
there and right is not fetched again. Nothing fetched is run: espeak-ng's
data is unpacked from its zip (every path in it checked to stay inside
`espeak-ng-data`), and the zip and the wheel are deleted after.

Licences: Kokoro-82M, its config and voices are Apache-2.0 (hexgrad);
espeak-ng and its data are GPL-3.0, and Fritz only ever runs espeak-ng as a
program of its own, never linking it (source: github.com/espeak-ng/espeak-ng);
ONNX Runtime is MIT (Microsoft). Standard library only, so it runs in any
Python 3.10 or later.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable

RELEASE = "https://github.com/NickSanft/DivoraVoiceMod/releases/download/voice-assets-v2/"

# (name, bytes, sha256), as GitHub's release lists them.
FILES = (
    ("kokoro-v1.0.int8.onnx", 92_361_271, "6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb"),
    ("voices-divora.bin", 3_655_785, "7af093e89dfa847f05608115983366989ade8e4a24d3b3c6c2edb8f532341578"),
    ("kokoro-config.json", 2_351, "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f"),
    ("espeak-ng.exe", 27_136, "3080ec3822c1b266ef557c710bc79a97d20a7ab133a34bac308b81ab0afc733e"),
    ("libespeak-ng.dll", 477_184, "e737572df0a35a32b7bd444537c661c1c916b13b0b91351030c7f1d531307beb"),
)

# Unpacked to espeak-ng-data\, which must then hold phontab.
ESPEAK_DATA = ("espeak-ng-data.zip", 12_208_741, "fcdcbc1972ac225d8201db77124f32a2d3a32807ac9c9203d7ea16a82f901b04")
ESPEAK_DIR = "espeak-ng-data"
ESPEAK_NEEDS = "phontab"
# Which zip the data came from, beside it, so --check and a second run know.
ESPEAK_FROM = ".from-zip"

WHEEL = (
    "https://files.pythonhosted.org/packages/44/fc/026d0a7162b9c2153dac292baea9e027c42304dc1d9dc6f8ff5b4cfbaedd/"
    "onnxruntime-1.26.0-cp312-cp312-win_amd64.whl",
    13_026_427,
    "a26374dc7fbcaae593601086b242120e13f2310558df0991da6dd8b8fac00414",
)
WHEEL_DLL = "onnxruntime/capi/onnxruntime.dll"
RUNTIME = "onnxruntime.dll"
# The DLL as the wheel holds it (16,783,672 bytes). It needs Microsoft's
# Visual C++ runtime (MSVCP140, VCRUNTIME140), which Windows apps install.
RUNTIME_SHA256 = "efa14dc0f3e8d6e4fc7ce0b9b5aa70d6e27deec908af05c1fa9a37270f57460b"

CHUNK = 1 << 20
USER_AGENT = "fritz-fetch-voice/1"

# url -> a readable response; urllib's by default, a test's own otherwise.
Opener = Callable[[str], BinaryIO]


class FetchError(Exception):
    """A file couldn't be fetched, or wasn't what it should be."""


def default_dest() -> Path:
    """His folder: tts in the app's data folder (tauri.conf.json's identifier)."""
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise FetchError("APPDATA isn't set: say where with --dest")
    return Path(appdata) / "com.divora.fritz" / "tts"


def urlopen(url: str) -> BinaryIO:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=60)  # noqa: S310 (https only, pinned above)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def is_right(path: Path, size: int, sha: str) -> bool:
    return path.is_file() and path.stat().st_size == size and sha256_of(path) == sha


def download(opener: Opener, url: str, to: Path, size: int, sha: str, say: Callable[[str], None]) -> None:
    """`url` into `to`, by way of `to.part`: kept only if it is `size` bytes
    and hashes to `sha`, else nothing is left behind."""
    part = to.with_name(to.name + ".part")
    h = hashlib.sha256()
    got = 0
    try:
        with opener(url) as response, open(part, "wb") as out:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                got += len(block)
                if got > size:
                    raise FetchError(f"{to.name}: more than the {size:,} bytes it should be")
                h.update(block)
                out.write(block)
        if got != size:
            raise FetchError(f"{to.name}: {got:,} bytes, not {size:,}")
        if h.hexdigest() != sha:
            raise FetchError(f"{to.name}: its SHA-256 isn't the one it should have")
        os.replace(part, to)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    say(f"{to.name}: {human(size)}, checked")


def human(size: int) -> str:
    """`size` bytes as a person reads it: 2.4 KB, 92.4 MB."""
    return f"{size / 1e6:.1f} MB" if size >= 100_000 else f"{size / 1e3:.1f} KB"


def safe_members(zf: zipfile.ZipFile, top: str) -> list[zipfile.ZipInfo]:
    """The zip's entries, each checked to stay inside `top`."""
    members = zf.infolist()
    for m in members:
        p = PurePosixPath(m.filename.replace("\\", "/"))
        if p.is_absolute() or ".." in p.parts or ":" in m.filename or not p.parts or p.parts[0] != top:
            raise FetchError(f"the zip has an entry outside {top}: {m.filename!r}")
    return members


def unpack_espeak_data(zip_path: Path, dest: Path, sha: str) -> None:
    """espeak-ng-data from its zip, replacing any there before, and only once
    the whole of it is unpacked and has phontab."""
    fresh = dest / (ESPEAK_DIR + ".new")
    old = dest / (ESPEAK_DIR + ".old")
    for d in (fresh, old):
        shutil.rmtree(d, ignore_errors=True)
    fresh.mkdir()
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for m in safe_members(zf, ESPEAK_DIR):
                inner = PurePosixPath(m.filename.replace("\\", "/")).parts[1:]
                target = fresh.joinpath(*inner) if inner else fresh
                if m.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out, CHUNK)
        if not (fresh / ESPEAK_NEEDS).is_file():
            raise FetchError(f"{zip_path.name} has no {ESPEAK_DIR}/{ESPEAK_NEEDS}")
        (fresh / ESPEAK_FROM).write_text(sha, encoding="ascii")
        current = dest / ESPEAK_DIR
        moved = current.exists()
        if moved:
            current.rename(old)
        try:
            fresh.rename(current)
        except BaseException:
            # The old data back where it was, rather than left as .old for
            # the next run to delete.
            if moved and not current.exists():
                old.rename(current)
            raise
        shutil.rmtree(old, ignore_errors=True)
    except BaseException:
        shutil.rmtree(fresh, ignore_errors=True)
        raise


def espeak_data_is_right(dest: Path) -> bool:
    d = dest / ESPEAK_DIR
    marker = d / ESPEAK_FROM
    return (d / ESPEAK_NEEDS).is_file() and marker.is_file() and marker.read_text(encoding="ascii").strip() == ESPEAK_DATA[2]


def extract_runtime(wheel: Path, dest: Path) -> None:
    """onnxruntime.dll out of the wheel, checked, by way of a .part."""
    to = dest / RUNTIME
    part = to.with_name(to.name + ".part")
    try:
        with zipfile.ZipFile(wheel) as zf, zf.open(WHEEL_DLL) as src, open(part, "wb") as out:
            shutil.copyfileobj(src, out, CHUNK)
        if sha256_of(part) != RUNTIME_SHA256:
            raise FetchError(f"{RUNTIME} in the wheel isn't the one it should be")
        os.replace(part, to)
    except KeyError as e:
        part.unlink(missing_ok=True)
        raise FetchError(f"the wheel has no {WHEEL_DLL}") from e
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def runtime_is_right(dest: Path) -> bool:
    p = dest / RUNTIME
    return p.is_file() and sha256_of(p) == RUNTIME_SHA256


def fetch(dest: Path, opener: Opener | None = None, say: Callable[[str], None] = print) -> list[str]:
    """Everything his voice needs into `dest`; what was fetched, by name."""
    opener = opener or urlopen
    dest.mkdir(parents=True, exist_ok=True)
    fetched: list[str] = []
    for name, size, sha in FILES:
        if is_right(dest / name, size, sha):
            say(f"{name}: already there")
            continue
        download(opener, RELEASE + name, dest / name, size, sha, say)
        fetched.append(name)
    if espeak_data_is_right(dest):
        say(f"{ESPEAK_DIR}: already there")
    else:
        name, size, sha = ESPEAK_DATA
        zip_path = dest / name
        download(opener, RELEASE + name, zip_path, size, sha, say)
        try:
            unpack_espeak_data(zip_path, dest, sha)
        finally:
            zip_path.unlink(missing_ok=True)
        say(f"{ESPEAK_DIR}: unpacked")
        fetched.append(name)
    if runtime_is_right(dest):
        say(f"{RUNTIME}: already there")
    else:
        url, size, sha = WHEEL
        wheel = dest / "onnxruntime-1.26.0.whl"
        download(opener, url, wheel, size, sha, say)
        try:
            extract_runtime(wheel, dest)
        finally:
            wheel.unlink(missing_ok=True)
        say(f"{RUNTIME}: taken out of the wheel, checked")
        fetched.append(RUNTIME)
    return fetched


def check(dest: Path) -> list[str]:
    """What is missing or wrong in `dest`, by name; nothing if all is right."""
    wrong = [name for name, size, sha in FILES if not is_right(dest / name, size, sha)]
    if not espeak_data_is_right(dest):
        wrong.append(ESPEAK_DIR)
    if not runtime_is_right(dest):
        wrong.append(RUNTIME)
    return wrong


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch his voice's files (Kokoro, espeak-ng, ONNX Runtime).")
    parser.add_argument("--dest", type=Path, help="the folder (default: %%APPDATA%%\\com.divora.fritz\\tts)")
    parser.add_argument("--check", action="store_true", help="only say whether they are all there and right")
    args = parser.parse_args(argv)
    try:
        dest = args.dest or default_dest()
        if args.check:
            wrong = check(dest)
            print(f"{dest}: " + ("all there" if not wrong else "missing or wrong: " + ", ".join(wrong)))
            return 1 if wrong else 0
        fetch(dest)
    except (FetchError, OSError, zipfile.BadZipFile) as e:
        print(f"fetch_voice: {e}", file=sys.stderr)
        return 1
    print(f"{dest}: his voice is there")
    return 0


if __name__ == "__main__":
    sys.exit(main())
