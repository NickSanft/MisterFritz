"""Guards on the dependency declaration itself.

These are cheap and they protect changes that are otherwise invisible until
something breaks in production: a security-critical package silently dropped
from the lock, or the `fitz` pin coming back.
"""
import importlib.metadata as md
import pathlib
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


if __name__ == "__main__":
    unittest.main()
