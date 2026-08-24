"""Guards on the dependency declaration itself.

These are cheap and they protect changes that are otherwise invisible until
something breaks in production: a security-critical package silently dropped
from the lock, or the `fitz` pin coming back.
"""
import importlib.metadata as md
import pathlib
import re
import tomllib
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent


def _requirements_names() -> set[str]:
    """Normalised distribution names pinned in requirements.txt."""
    names = set()
    for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip the version spec, environment marker and extras.
        name = line.split(";")[0].split("[")[0]
        for sep in ("===", "==", ">=", "<=", "~=", "!=", ">", "<"):
            name = name.split(sep)[0]
        name = name.strip()
        if name:
            names.add(name.lower().replace("_", "-").replace(".", "-"))
    return names


def _pyproject() -> dict:
    with open(REPO / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


class TestFitzHazard(unittest.TestCase):
    """`import fitz` must come from PyMuPDF and nothing else.

    The package published on PyPI as `fitz` is unrelated 2016 neuroimaging
    software (`Fitz: Workflow Management for neuroimaging data`, Python 2.7)
    that installs into the SAME fitz/ directory PyMuPDF uses. Both RECORDs
    claim fitz/__init__.py, so whichever wheel lands last wins.

    If the neuroimaging one wins, `import fitz` SUCCEEDS — so
    document_engine's `except ImportError` guard never fires and
    PYMUPDF_AVAILABLE stays True — and `fitz.open()` then raises AttributeError
    deep inside PDF ingestion. Silent, guard-defeating breakage.
    """

    def test_no_fitz_pin_in_requirements(self):
        self.assertNotIn(
            "fitz", _requirements_names(),
            "requirements.txt pins `fitz`. That is the neuroimaging package, not "
            "PyMuPDF. Remove the pin; PyMuPDF provides `import fitz`.",
        )

    def test_no_fitz_in_any_pyproject_dependency_group(self):
        proj = _pyproject()["project"]
        groups = {"core": proj["dependencies"]}
        groups.update(proj["optional-dependencies"])
        for group, reqs in groups.items():
            for raw in reqs:
                name = raw.split(";")[0].split("[")[0]
                for sep in ("===", "==", ">=", "<=", "~=", "!=", ">", "<"):
                    name = name.split(sep)[0]
                self.assertNotEqual(
                    name.strip().lower(), "fitz",
                    f"pyproject group [{group}] declares `fitz`",
                )

    def test_pymupdf_is_declared_in_the_ocr_extra(self):
        ocr = _pyproject()["project"]["optional-dependencies"]["ocr"]
        self.assertTrue(
            any(r.lower().startswith("pymupdf") for r in ocr),
            "PyMuPDF must stay in the [ocr] extra — it is what provides `import fitz`",
        )

    def test_no_fitz_distribution_is_installed(self):
        """The acceptance signal. Fails on the pre-change environment.

        Note this asserts on the *distribution*, not on whether `import fitz`
        works: PyMuPDF installs the fitz/ package without registering a
        distribution called `fitz`.
        """
        try:
            version = md.version("fitz")
        except md.PackageNotFoundError:
            return
        self.fail(
            f"A distribution named `fitz` ({version}) is installed. It is the "
            "neuroimaging package and it fights PyMuPDF for the fitz/ directory. "
            "Uninstalling it is itself a trap: its RECORD lists fitz/__init__.py, "
            "so `pip uninstall fitz` deletes PyMuPDF's shim. Run "
            "`pip uninstall -y fitz && pip install --force-reinstall PyMuPDF`."
        )


class TestSecurityCriticalDependencies(unittest.TestCase):
    """Two packages whose absence silently un-fixes a security control.

    Neither is obvious from a call graph: nh3 is used in one helper, and
    Pygments is never imported by this codebase at all — markdown's codehilite
    imports it. A regeneration of the lock that "cleans up unused packages"
    would take both.
    """

    def test_nh3_is_pinned_and_declared(self):
        self.assertIn("nh3", _requirements_names())
        self.assertTrue(
            any(r.lower().startswith("nh3")
                for r in _pyproject()["project"]["dependencies"]),
            "nh3 sanitises rendered chat markdown; python-markdown passes raw "
            "HTML through and the template renders it with |safe. Dropping it "
            "reopens stored XSS.",
        )

    def test_pygments_is_pinned_and_declared(self):
        self.assertIn("pygments", _requirements_names())
        self.assertTrue(
            any(r.lower().startswith("pygments")
                for r in _pyproject()["project"]["dependencies"]),
            "Pygments backs markdown's codehilite; without it the extension "
            "raises at render time and every chat reply 500s.",
        )

    def test_nh3_actually_strips_a_script_tag(self):
        # Belt and braces: the pin existing is not the same as it working.
        import nh3
        self.assertNotIn("<script", nh3.clean("<script>alert(1)</script><p>ok</p>"))


class TestCoreIsTorchFree(unittest.TestCase):
    """Core must not pull the multi-GB GPU stack.

    agent_tools, bot_commands and main_discord all defer their
    image_generator / tts imports precisely so this stays true.
    """

    HEAVY = ("torch", "diffusers", "xformers", "coqui-tts", "easyocr",
             "nvidia-", "triton", "faster-whisper", "transformers")

    def test_no_heavy_package_in_core_dependencies(self):
        for raw in _pyproject()["project"]["dependencies"]:
            name = raw.split(";")[0].split("[")[0].strip().lower()
            for heavy in self.HEAVY:
                self.assertFalse(
                    name.startswith(heavy),
                    f"core dependency {raw!r} pulls the GPU stack; move it to an extra",
                )

    def test_heavy_modules_are_not_imported_at_module_level(self):
        """bot_commands and main_discord are on the bot's boot path."""
        for module in ("bot_commands.py", "main_discord.py", "agent_tools.py"):
            src = (REPO / module).read_text(encoding="utf-8")
            for line in src.splitlines():
                stripped = line.strip()
                if line.startswith(("import ", "from ")):     # column 0 == module level
                    self.assertNotIn("image_generator", stripped, f"{module}: {stripped}")
                    self.assertFalse(
                        stripped.startswith(("import tts", "from tts ")),
                        f"{module}: {stripped}",
                    )


class TestDependencyDeclaration(unittest.TestCase):
    def test_expected_extras_exist(self):
        extras = _pyproject()["project"]["optional-dependencies"]
        for name in ("voice", "image", "ocr", "telegram", "dev", "all"):
            self.assertIn(name, extras)

    def test_no_browser_extra(self):
        # DECISIONS #7: browser_tools.py is deleted rather than wired up, so
        # playwright never enters the dependency set.
        self.assertNotIn("browser",
                         _pyproject()["project"]["optional-dependencies"])
        self.assertNotIn("playwright", _requirements_names())

    def test_browser_tools_module_is_gone(self):
        self.assertFalse((REPO / "browser_tools.py").exists())

    def test_py_modules_lists_every_top_level_module(self):
        declared = set(_pyproject()["tool"]["setuptools"]["py-modules"])
        on_disk = {p.stem for p in REPO.glob("*.py")}
        self.assertEqual(
            on_disk - declared, set(),
            "a top-level module is missing from [tool.setuptools] py-modules",
        )
        self.assertEqual(
            declared - on_disk, set(),
            "py-modules lists a module that no longer exists",
        )

    def test_requirements_has_no_duplicate_pins(self):
        seen, dupes = set(), []
        for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name = line.split(";")[0].split("[")[0]
            for sep in ("===", "==", ">=", "<=", "~=", "!=", ">", "<"):
                name = name.split(sep)[0]
            name = name.strip().lower().replace("_", "-").replace(".", "-")
            if name in seen:
                dupes.append(name)
            seen.add(name)
        self.assertEqual(dupes, [], f"duplicate pins in requirements.txt: {dupes}")

    def test_no_neuroimaging_leftovers(self):
        # The fitz -> nipype chain, purged. Listed explicitly so a future
        # `pip freeze > requirements.txt` cannot quietly restore them.
        purged = {"nipype", "nibabel", "pyxnat", "traits", "prov", "rdflib",
                  "simplejson", "acres", "ci-info", "etelemetry", "looseversion",
                  "puremagic", "pydot", "configobj", "configparser", "httplib2",
                  "pathlib", "pygame", "pyttsx3", "pdf2image", "pytesseract",
                  "langchain-google-community"}
        present = purged & _requirements_names()
        self.assertEqual(
            present, set(),
            f"purged packages are back in requirements.txt: {sorted(present)}. "
            "That is what `pip freeze > requirements.txt` does — regenerate by "
            "hand from pyproject.toml instead (DECISIONS #9).",
        )


class TestDeclaredCoreDepsAreInstalled(unittest.TestCase):
    """Catches a core dependency silently dropped from the lock.

    Not hypothetical: prometheus-client is declared core but was missing from
    the working venv, and observability.py guards its import with try/except —
    so metrics degraded silently instead of failing loudly. A declared core dep
    that is not installed means requirements.txt and pyproject.toml have
    drifted apart.
    """

    @staticmethod
    def _normalise(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).strip().lower()

    def test_every_declared_core_dep_is_importable_as_a_distribution(self):
        declared = set()
        for raw in _pyproject()["project"]["dependencies"]:
            spec = raw.split(";")[0]           # drop environment markers
            name = re.split(r"[<>=!\[~]", spec)[0]
            declared.add(self._normalise(name))
        installed = {self._normalise(d.metadata["Name"])
                     for d in md.distributions() if d.metadata["Name"]}
        missing = sorted(declared - installed)
        self.assertEqual(
            missing, [],
            f"declared as core in pyproject.toml but not installed: {missing}. "
            "Either install them (pip install -e '.[dev]') or stop declaring "
            "them core — a guarded import means this degrades silently.",
        )


class TestHeavyImportsRunOffTheEventLoop(unittest.TestCase):
    """A deferred import is only half the fix.

    Moving `import image_generator` / `import tts` off module scope keeps the
    extras optional, but executing the statement inside an `async def` still
    runs the module body — torch, diffusers, TTS.api — on the event loop.
    Measured: ~10s for image_generator, ~17s for tts, the latter past
    discord.py's "heartbeat blocked for more than 10 seconds" threshold. Both
    imports therefore have to sit inside the callable handed to the worker
    pool, not beside it.

    This is a source check because the failure is a latency regression with no
    functional symptom — nothing raises, the bot just freezes for everyone.
    """

    def _source(self, name):
        return (REPO / name).read_text(encoding="utf-8")

    def test_gen_command_imports_inside_the_offloaded_callable(self):
        src = self._source("bot_commands.py")
        # The helper exists and carries the import...
        self.assertIn("def _render_image(", src)
        helper = src.split("def _render_image(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("from image_generator import generate_image", helper)
        # ...and gen_slash offloads it rather than importing inline.
        gen = src.split("async def gen_slash(", 1)[1].split("\n    @", 1)[0]
        self.assertIn("run_blocking(_render_image", gen)
        self.assertNotIn("from image_generator import", gen)

    def test_tts_load_imports_inside_the_offloaded_callable(self):
        src = self._source("main_discord.py")
        loader = src.split("def _load_tts(", 1)[1].split("\n        logger", 1)[0]
        self.assertIn("from tts import TTSEngine", loader)
        # The on_ready body must not import tts directly.
        on_ready = src.split("async def on_ready(", 1)[1].split("\n@", 1)[0]
        stripped = on_ready.replace(loader, "")
        self.assertNotIn("from tts import", stripped)
        self.assertIn("run_blocking(_load_tts)", on_ready)




class TestSdxlPipelineIsGuarded(unittest.TestCase):
    """The ~7 GB SDXL pipeline must load exactly once.

    Two concurrent /gen calls that both found _pipeline is None would each
    build one, and the second would OOM the GPU or silently double VRAM.
    conftest.py installs a MagicMock for image_generator before any test module
    imports, so this is the one place that has to reach the REAL module — and
    it is why the guard had no coverage at all.
    """

    def _real_source(self):
        return (REPO / "image_generator.py").read_text(encoding="utf-8")

    def test_get_pipeline_holds_the_lock(self):
        """Source-level: importing the real module needs torch + diffusers,
        which core installs deliberately do not have."""
        src = self._real_source()
        self.assertIn("_PIPELINE_LOCK = threading.Lock()", src)
        body = src.split("def get_pipeline(", 1)[1].split(chr(10) + "def ", 1)[0]
        self.assertIn("with _PIPELINE_LOCK:", body)
        # The None-check must be INSIDE the lock, or two callers can both pass
        # it before either assigns.
        lock_at = body.index("with _PIPELINE_LOCK:")
        check_at = body.index("if _pipeline is None:")
        self.assertLess(lock_at, check_at,
                        "the _pipeline is None check sits OUTSIDE the lock, so "
                        "two concurrent callers can both enter and build one")

    def test_generation_also_serialises_on_the_lock(self):
        """The render itself is single-GPU work; two at once thrash VRAM."""
        src = self._real_source()
        self.assertGreaterEqual(src.count("with _PIPELINE_LOCK:"), 2)

    def test_the_real_module_is_importable_when_the_extra_is_present(self):
        """Belt and braces: if [image] IS installed, prove the lock object is
        real rather than trusting the source read. Skipped on a core install,
        which is the normal case for CI."""
        import importlib
        import sys
        diffusers = importlib.util.find_spec("diffusers")
        if diffusers is None:
            self.skipTest("[image] extra not installed — source check covers it")
        stub = sys.modules.pop("image_generator", None)
        try:
            real = importlib.import_module("image_generator")
            import threading
            self.assertIsInstance(real._PIPELINE_LOCK, type(threading.Lock()))
        finally:
            if stub is not None:
                sys.modules["image_generator"] = stub

if __name__ == "__main__":
    unittest.main()


class TestTheLockActuallyLocks(unittest.TestCase):
    """The file's stated job is "the pinned closure of those roots".

    A root whose own dependencies are missing is not a closure — pip resolves
    them at build time, so two builds of the same commit can differ. This
    caught faster-whisper's `av` and `ctranslate2`, which nothing else pulls in.
    """

    def test_faster_whisper_transitives_are_pinned(self):
        pinned = _requirements_names()
        self.assertIn("faster-whisper", pinned)
        for dep in ("av", "ctranslate2"):
            self.assertIn(
                dep, pinned,
                f"faster-whisper requires {dep} and nothing else pulls it in, "
                "so leaving it out means the lock does not lock.",
            )

    def test_no_unbounded_floors(self):
        """An UNBOUNDED floor lets two builds of the same commit resolve
        differently. A bounded range (nh3>=0.3,<0.4) is a deliberate choice —
        patch updates for a security package, no major bump — so it passes.
        Only `>=` with nothing above it is the oversight worth catching."""
        unbounded = []
        for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if not line or "==" in line:
                continue
            if any(op in line for op in (">=", ">", "~=")) and "<" not in line:
                unbounded.append(line)
        self.assertEqual(
            sorted(unbounded), ["python-telegram-bot>=20.0"],
            "a new unbounded floor appeared in the lock; pin it to the resolved "
            "version, give it an upper bound, or document why it cannot be pinned",
        )
    def test_zstandard_is_present_for_its_real_reason(self):
        """It is langsmith's, not faster-whisper's — an earlier header claimed
        the latter, which sent someone looking in the wrong place."""
        self.assertIn("zstandard", _requirements_names())
