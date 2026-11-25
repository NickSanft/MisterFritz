"""desktop/scripts/fetch_voice.py: his voice's files fetched into his folder,
each checked before it is kept, and nothing left half done. A fake opener
stands in for the network, and the pinned files for fakes of their own."""
import hashlib
import importlib.util
import io
import re
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "desktop" / "scripts" / "fetch_voice.py"
SHELL = ROOT / "desktop" / "src-tauri" / "src" / "kokoro_tts.rs"

spec = importlib.util.spec_from_file_location("fetch_voice", SCRIPT)
fv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fv)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def zipped(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


class Fakes:
    """Fake files, pinned in place of the real ones, served by `opener`."""

    def __init__(self):
        self.files = {name: f"fake {name}".encode() * 3 for name, _, _ in fv.FILES}
        self.espeak = zipped({"espeak-ng-data/phontab": b"phon", "espeak-ng-data/voices/gb": b"gb", "espeak-ng-data/": b""})
        self.dll = b"MZ a fake runtime"
        self.wheel = zipped({fv.WHEEL_DLL: self.dll, "onnxruntime/__init__.py": b"x"})
        self.served = {fv.RELEASE + n: b for n, b in self.files.items()}
        self.served[fv.RELEASE + fv.ESPEAK_DATA[0]] = self.espeak
        self.served["https://example.test/onnxruntime.whl"] = self.wheel
        self.asked: list[str] = []

    def patches(self):
        return [
            patch.object(fv, "FILES", tuple((n, len(b), sha(b)) for n, b in self.files.items())),
            patch.object(fv, "ESPEAK_DATA", (fv.ESPEAK_DATA[0], len(self.espeak), sha(self.espeak))),
            patch.object(fv, "WHEEL", ("https://example.test/onnxruntime.whl", len(self.wheel), sha(self.wheel))),
            patch.object(fv, "RUNTIME_SHA256", sha(self.dll)),
        ]

    def opener(self, url):
        self.asked.append(url)
        return io.BytesIO(self.served[url])


class TestFetchVoice(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.dest = Path(self.tmp.name) / "tts"
        self.fake = Fakes()
        for p in self.fake.patches():
            p.start()
            self.addCleanup(p.stop)
        self.said: list[str] = []

    def tearDown(self):
        self.tmp.cleanup()

    def fetch(self):
        return fv.fetch(self.dest, opener=self.fake.opener, say=self.said.append)

    def test_everything_is_fetched_checked_and_nothing_else_is_left(self):
        fetched = self.fetch()
        self.assertEqual(fetched, [n for n, _, _ in fv.FILES] + ["espeak-ng-data.zip", "onnxruntime.dll"])
        for name, data in self.fake.files.items():
            self.assertEqual((self.dest / name).read_bytes(), data)
        self.assertEqual((self.dest / "espeak-ng-data" / "phontab").read_bytes(), b"phon")
        self.assertEqual((self.dest / "espeak-ng-data" / "voices" / "gb").read_bytes(), b"gb")
        self.assertEqual((self.dest / "onnxruntime.dll").read_bytes(), self.fake.dll)
        left = sorted(p.name for p in self.dest.iterdir())
        self.assertEqual(left, sorted([*self.fake.files, "espeak-ng-data", "onnxruntime.dll"]),
                         "no zip, wheel, .part, .new or .old left behind")
        self.assertEqual(fv.check(self.dest), [])

    def test_a_second_run_fetches_nothing(self):
        self.fetch()
        self.fake.asked.clear()
        self.assertEqual(self.fetch(), [])
        self.assertEqual(self.fake.asked, [], "nothing asked of the network")
        # One file changed is the only one fetched again, even at its size.
        (self.dest / "voices-divora.bin").write_bytes(b"changed")
        self.assertEqual(self.fetch(), ["voices-divora.bin"])
        size = (self.dest / "kokoro-config.json").stat().st_size
        (self.dest / "kokoro-config.json").write_bytes(b"x" * size)
        self.assertEqual(self.fetch(), ["kokoro-config.json"], "the same size isn't the same file")

    def test_a_wrong_file_is_never_kept(self):
        name = "kokoro-v1.0.int8.onnx"
        good = self.fake.served[fv.RELEASE + name]
        for bad, why in ((b"x" * len(good), "SHA-256"), (good[:-1], "bytes, not"), (good + b"!", "more than")):
            with self.subTest(why=why):
                self.fake.served[fv.RELEASE + name] = bad
                with self.assertRaisesRegex(fv.FetchError, why):
                    self.fetch()
                self.assertFalse((self.dest / name).exists())
                self.assertEqual(list(self.dest.glob("*.part")), [])

    def test_a_download_cut_off_leaves_nothing(self):
        class Broken(io.BytesIO):
            def read(self, n=-1):
                raise ConnectionResetError("the line dropped")

        with patch.object(self.fake, "opener", lambda url: Broken(b"")):
            with self.assertRaises(ConnectionResetError):
                self.fetch()
        self.assertEqual(list(self.dest.iterdir()), [])

    def test_espeak_data_stays_inside_its_folder(self):
        for entry in ("../evil", "espeak-ng-data/../../evil", "/abs/evil", "C:/evil", "other/phontab", "espeak-ng-data\\..\\..\\evil",
                      "espeak-ng-data/phontab:evil", "espeak-ng-data/C:evil"):
            with self.subTest(entry=entry):
                bad = zipped({"espeak-ng-data/phontab": b"p", entry: b"evil"})
                self.fake.served[fv.RELEASE + fv.ESPEAK_DATA[0]] = bad
                with patch.object(fv, "ESPEAK_DATA", (fv.ESPEAK_DATA[0], len(bad), sha(bad))):
                    with self.assertRaisesRegex(fv.FetchError, "outside"):
                        self.fetch()
                self.assertFalse((Path(self.tmp.name) / "evil").exists())
                self.assertFalse((self.dest / "espeak-ng-data").exists())
                self.assertFalse((self.dest / "espeak-ng-data.zip").exists())

    def test_espeak_data_without_phontab_is_not_kept_and_the_old_stays(self):
        self.fetch()
        bad = zipped({"espeak-ng-data/voices/gb": b"gb"})
        self.fake.served[fv.RELEASE + fv.ESPEAK_DATA[0]] = bad
        (self.dest / "espeak-ng-data" / fv.ESPEAK_FROM).write_text("another zip", encoding="ascii")
        with patch.object(fv, "ESPEAK_DATA", (fv.ESPEAK_DATA[0], len(bad), sha(bad))):
            with self.assertRaisesRegex(fv.FetchError, "phontab"):
                self.fetch()
        self.assertTrue((self.dest / "espeak-ng-data" / "phontab").is_file(), "the old data is still there")
        self.assertFalse((self.dest / "espeak-ng-data.new").exists())

    def test_the_old_data_stays_if_the_new_cant_take_its_place(self):
        self.fetch()
        (self.dest / "espeak-ng-data" / fv.ESPEAK_FROM).write_text("another zip", encoding="ascii")
        real_rename = Path.rename

        def rename(path, target):
            if path.name == "espeak-ng-data.new":
                raise PermissionError("in use")
            return real_rename(path, target)

        with patch.object(Path, "rename", rename), self.assertRaises(PermissionError):
            self.fetch()
        self.assertEqual((self.dest / "espeak-ng-data" / "phontab").read_bytes(), b"phon", "the old data is back")
        self.assertFalse((self.dest / "espeak-ng-data.old").exists())
        self.assertFalse((self.dest / "espeak-ng-data.new").exists())
        self.assertFalse((self.dest / "espeak-ng-data.zip").exists())

    def test_the_runtime_must_be_the_one_pinned(self):
        with patch.object(fv, "RUNTIME_SHA256", "0" * 64):
            with self.assertRaisesRegex(fv.FetchError, "onnxruntime.dll"):
                self.fetch()
        self.assertFalse((self.dest / "onnxruntime.dll").exists())
        self.assertEqual([p.name for p in self.dest.glob("*.whl")], [], "the wheel is deleted")
        empty = zipped({"onnxruntime/__init__.py": b"x"})
        self.fake.served["https://example.test/onnxruntime.whl"] = empty
        with patch.object(fv, "WHEEL", ("https://example.test/onnxruntime.whl", len(empty), sha(empty))):
            with self.assertRaisesRegex(fv.FetchError, "no onnxruntime/capi"):
                self.fetch()

    def test_check_names_what_is_missing_or_wrong(self):
        self.assertEqual(fv.check(self.dest), [n for n, _, _ in fv.FILES] + ["espeak-ng-data", "onnxruntime.dll"])
        self.fetch()
        (self.dest / "onnxruntime.dll").write_bytes(b"tampered")
        (self.dest / "espeak-ng-data" / "phontab").unlink()
        self.assertEqual(fv.check(self.dest), ["espeak-ng-data", "onnxruntime.dll"])
        self.assertEqual(fv.main(["--dest", str(self.dest), "--check"]), 1)

    def test_main_fetches_into_the_data_folder_by_default(self):
        with patch.dict("os.environ", {"APPDATA": self.tmp.name}), patch.object(fv, "urlopen", self.fake.opener), \
             patch("builtins.print"):
            self.assertEqual(fv.main([]), 0)
            self.assertEqual(fv.main(["--check"]), 0)
        self.assertEqual(fv.check(Path(self.tmp.name) / "com.divora.fritz" / "tts"), [])
        with patch.dict("os.environ", {"APPDATA": ""}), patch("builtins.print"):
            self.assertEqual(fv.main([]), 1, "no APPDATA, no --dest: refused")

    def test_sizes_read_as_a_person_reads_them(self):
        self.assertEqual(fv.human(2_351), "2.4 KB")
        self.assertEqual(fv.human(92_361_271), "92.4 MB")


class TestThePinnedFiles(unittest.TestCase):
    """The real list: what the owner approved, and what the shell needs."""

    def test_the_owners_files_from_the_owners_release_and_pypi(self):
        self.assertEqual(fv.RELEASE, "https://github.com/NickSanft/DivoraVoiceMod/releases/download/voice-assets-v2/")
        names = [n for n, _, _ in fv.FILES] + [fv.ESPEAK_DATA[0]]
        self.assertEqual(names, ["kokoro-v1.0.int8.onnx", "voices-divora.bin", "kokoro-config.json",
                                 "espeak-ng.exe", "libespeak-ng.dll", "espeak-ng-data.zip"])
        self.assertTrue(fv.WHEEL[0].startswith("https://files.pythonhosted.org/"))
        self.assertTrue(fv.WHEEL[0].endswith("/onnxruntime-1.26.0-cp312-cp312-win_amd64.whl"))
        total = sum(s for _, s, _ in fv.FILES) + fv.ESPEAK_DATA[1] + fv.WHEEL[1]
        self.assertEqual(round(total / 1e6, 1), 121.8)
        for _, size, digest in [*fv.FILES, fv.ESPEAK_DATA, fv.WHEEL]:
            self.assertGreater(size, 0)
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertRegex(fv.RUNTIME_SHA256, r"^[0-9a-f]{64}$")
        self.assertTrue(all(u.startswith("https://") for u in (fv.RELEASE, fv.WHEEL[0])))

    def test_every_file_the_shell_needs_is_fetched(self):
        shell = SHELL.read_text(encoding="utf-8")
        consts = dict(re.findall(r'const (\w+_FILE): &str = "([^"]+)";', shell))
        self.assertEqual(set(consts.values()) - {n for n, _, _ in fv.FILES} - {fv.RUNTIME}, set())
        espeak = re.search(r"const ESPEAK_FILES: \[&str; 3\] = \[([^\]]+)\]", shell).group(1)
        for name in re.findall(r'"([^"]+)"', espeak):
            if "/" in name:
                self.assertEqual(name, f"{fv.ESPEAK_DIR}/{fv.ESPEAK_NEEDS}")
            else:
                self.assertIn(name, [n for n, _, _ in fv.FILES])


if __name__ == "__main__":
    unittest.main()
