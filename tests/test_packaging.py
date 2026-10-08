"""Guards on the dependency declaration itself.

These are cheap and they protect changes that are otherwise invisible until
something breaks in production: a security-critical package silently dropped
from the lock, or the `fitz` pin coming back.
"""
import importlib.metadata as md
import ast
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



class TestEveryLockLineIsInstallable(unittest.TestCase):
    """requirements.txt is hand-edited on purpose (DECISIONS #9), so a stray
    line is a real possibility — and pip rejects the WHOLE file for one bad
    line, which is the kind of failure that only shows up on a fresh deploy.
    Found the hard way: a shell command substitution once pasted three lines
    of `distro` output into the header.
    """

    def test_no_line_is_anything_but_a_comment_or_a_requirement(self):
        from packaging.requirements import InvalidRequirement, Requirement
        for number, raw in enumerate(
                (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith(("#", "-")):
                continue
            with self.subTest(line=number):
                try:
                    Requirement(line.split("#")[0].strip())
                except InvalidRequirement as e:
                    self.fail(f"requirements.txt:{number} is not installable: {raw!r} ({e})")


class TestChromaCannotPhoneHome(unittest.TestCase):
    """Up to chromadb 1.5.2, Chroma's product telemetry imported posthog and
    POSTed to a hardcoded project key on collection operations, on by default.
    It carried counts, flags and collection UUIDs — never anything anyone
    wrote — but it was outbound traffic from an app whose premise is that
    everything runs locally. 1.5.3 reduced that client to a `pass` and dropped
    the dependency; the floor is 1.5.4 because 1.5.3 is yanked on PyPI and its
    settings raise ValidationError on import against this repo's own .env.

    Version numbers alone cannot carry that property, so the last test here
    looks at the client itself. If a future chromadb revives telemetry under a
    new transport, that is the test that fails.
    """

    LAST_SENDER = (1, 5, 2)
    FIRST_INSTALLABLE_INERT = (1, 5, 4)

    @staticmethod
    def _parts(spec: str) -> tuple:
        return tuple(int(p) for p in re.findall(r"\d+", spec)[:3])

    @staticmethod
    def _pinned(name: str) -> str:
        for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{name}=="):
                return line.split("==", 1)[1].split("#")[0].split(";")[0].strip()
        raise AssertionError(f"{name} is not pinned in requirements.txt")

    @staticmethod
    def _declared() -> str:
        [declared] = [d for d in _pyproject()["project"]["dependencies"]
                      if d.split(">")[0].strip().lower() == "chromadb"]
        return declared

    def test_the_lock_pins_a_chromadb_that_cannot_send(self):
        pinned = self._pinned("chromadb")
        self.assertGreaterEqual(self._parts(pinned), self.FIRST_INSTALLABLE_INERT, pinned)

    def test_the_floor_excludes_every_release_that_could_send(self):
        floor = self._parts(self._declared().split(">=", 1)[1])
        self.assertGreaterEqual(floor, self.FIRST_INSTALLABLE_INERT, floor)
        self.assertGreater(floor, self.LAST_SENDER, floor)

    def test_and_the_range_is_closed(self):
        """A bare `>=` asserts nothing about a version that does not exist yet:
        a 1.6 that revived telemetry would satisfy it. The claim is checked
        against the 1.5 line, so the dependency says so."""
        self.assertIn("<", self._declared().split(">=", 1)[1])

    def test_posthog_is_not_pinned_anywhere(self):
        """It only ever arrived as a chromadb dependency, and chromadb dropped
        it. A regenerated freeze must not quietly bring it back."""
        self.assertNotIn("posthog", _requirements_names())

    def test_the_installed_chromadb_does_not_require_posthog(self):
        """The pin states the intent; this is the environment matching it."""
        import importlib.metadata as md
        requires = md.requires("chromadb") or []
        self.assertEqual([r for r in requires if r.lower().startswith("posthog")], [])

    def test_the_installed_telemetry_client_sends_nothing(self):
        """The one assertion here about behaviour rather than version strings.

        chromadb still names this class as its product-telemetry client
        (config.py: chroma_product_telemetry_impl), so it is what would carry
        any revival. Its body being `pass` is the property every version
        assertion above is only a proxy for.
        """
        import ast
        import inspect
        import textwrap

        from chromadb.config import Settings
        from chromadb.telemetry.product.posthog import Posthog

        self.assertEqual(Settings().chroma_product_telemetry_impl,
                         "chromadb.telemetry.product.posthog.Posthog")
        [function] = ast.parse(textwrap.dedent(inspect.getsource(Posthog.capture))).body
        statements = [node for node in function.body
                      if not (isinstance(node, ast.Expr)
                              and isinstance(node.value, ast.Constant)
                              and isinstance(node.value.value, str))]   # the docstring
        self.assertEqual([type(node).__name__ for node in statements], ["Pass"],
                         "Chroma's telemetry client does something again; read what "
                         "it does and decide whether this app still wants it")


if __name__ == "__main__":
    unittest.main()


class TestNoEmbedderCanLoadAModelWhenAStoreIsOpened(unittest.TestCase):
    """Chroma builds an embedding function named in a collection's persisted
    schema while it deserializes that schema — which happens as the collection
    is opened, before any check this app makes gets a turn.

    fritz_utils.refuse_stored_embedders then refuses the store, so nothing is
    ever called with anyone's text. What keeps the build itself harmless is
    narrower: of the embedders chromadb can name, the only ones constructible
    in this environment talk to a remote API over httpx and send nothing from
    __init__. The ones that would load a model at that moment need a package
    that is not installed, so they warn and deserialize to None.

    That is a dependency property, so it is asserted here. Adding any of these
    — directly or as somebody's extra — turns a line in a store on disk into
    model loading at open time, which is the one chromadb CVE class that would
    then have an embedded path.
    """

    FORBIDDEN = ("fastembed", "sentence-transformers", "instructorembedding",
                 "open-clip-torch", "text2vec")

    IMPORTS = ("fastembed", "sentence_transformers", "InstructorEmbedding",
               "open_clip", "text2vec")

    def test_none_are_pinned_in_the_lock(self):
        pinned = _requirements_names()
        for name in self.FORBIDDEN:
            with self.subTest(package=name):
                self.assertNotIn(name, pinned, self.__doc__)

    def test_none_are_declared_anywhere_in_pyproject(self):
        """Extras included: an extra nobody installs today is still a line that
        says this is allowed, and the GPU extra is installed on the host that
        actually runs the bot."""
        project = _pyproject()["project"]
        declared = list(project.get("dependencies", []))
        for extra, deps in (project.get("optional-dependencies") or {}).items():
            declared.extend(f"{dep}   (extra: {extra})" for dep in deps)
        for raw in declared:
            name = raw.split(";")[0].split("[")[0].strip().lower()
            for forbidden in self.FORBIDDEN:
                with self.subTest(dependency=raw):
                    self.assertFalse(name.startswith(forbidden),
                                     f"{raw!r} lets a stored schema load a model")

    def test_none_can_be_imported_here(self):
        """The declaration is the intent; this is the environment matching it.

        A transitive dependency, or a stray `pip install`, counts just as much
        as a line in pyproject — chromadb only asks whether the import works.
        """
        import importlib.util
        for module in self.IMPORTS:
            with self.subTest(module=module):
                try:
                    spec = importlib.util.find_spec(module)
                except Exception:                      # pragma: no cover
                    spec = None
                self.assertIsNone(
                    spec,
                    f"{module} is importable, so a collection schema naming its "
                    "embedder would load a model while the store is being "
                    "opened — see requirements.txt at the chromadb pin")


class TestTheParsersAreNotTheVulnerableReleases(unittest.TestCase):
    """Three pins carry advisory content rather than currency.

    The counts below were derived by walking OSV's introduced/fixed events
    against the pinned version and deduplicating by CVE, because OSV's own
    version filter returns every advisory for a package whatever version you
    ask about, and it lists the same CVE under both GHSA and PYSEC ids. One
    consequence is recorded at the nltk pin: CVE-2026-33236 reads as unfixed in
    its GHSA record and fixed=3.9.4 in the PYSEC record for the same CVE.
    """

    # package -> (lowest release clearing every CVE that applied to the old
    #             pin, how many unique CVEs that was)
    CLEARED = {
        "pypdf": ((6, 19, 0), 45),
        "nltk": ((3, 10, 3), 43),
        "PyNaCl": ((1, 6, 2), 1),
    }

    @staticmethod
    def _parts(version: str) -> tuple:
        return tuple(int(p) for p in re.findall(r"\d+", version)[:3])

    def test_the_lock_is_at_or_past_every_fix(self):
        for package, (floor, count) in self.CLEARED.items():
            with self.subTest(package=package, cves=count):
                [line] = [ln for ln in (REPO / "requirements.txt")
                          .read_text(encoding="utf-8").splitlines()
                          if ln.lower().startswith(package.lower() + "==")]
                pinned = self._parts(line.split("==", 1)[1])
                self.assertGreaterEqual(
                    pinned, floor,
                    f"{package} is pinned below the release that clears "
                    f"{count} CVEs")

    def test_the_installed_versions_match_the_lock(self):
        """The pin is the intent; this is the environment matching it."""
        import importlib.metadata as md
        for package, (floor, _count) in self.CLEARED.items():
            with self.subTest(package=package):
                self.assertGreaterEqual(self._parts(md.version(package)), floor)

    def test_the_declared_floors_exclude_the_vulnerable_releases(self):
        """A lock pin protects this checkout; the floor protects anyone
        installing the package."""
        declared = {d.split(">")[0].strip().lower(): d
                    for d in _pyproject()["project"]["dependencies"]}
        for package in ("pypdf", "PyNaCl"):
            with self.subTest(package=package):
                spec = declared[package.lower()]
                self.assertIn(">=", spec)
                self.assertGreaterEqual(self._parts(spec.split(">=", 1)[1]),
                                        self.CLEARED[package][0][:2])

    def test_nltk_arrives_with_what_its_fix_needs(self):
        """defusedxml is new with nltk 3.10.3 and is part of how the Downloader
        traversal was closed, so the lock has to carry it."""
        self.assertIn("defusedxml", _requirements_names())

    def test_nltk_is_nobody_in_this_repo_importing_it(self):
        """It arrives under `unstructured`. If a module here starts importing
        nltk directly, the corpora question (AUTO_DOWNLOAD_NLTK) becomes this
        repo's own problem rather than a dependency's."""
        offenders = []
        for path in sorted(REPO.glob("*.py")):
            source = path.read_text(encoding="utf-8")
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith(("import nltk", "from nltk")):
                    offenders.append(f"{path.name}: {stripped}")
        self.assertEqual(offenders, [])


class TestThePyNaClCeilingIsRecordedWhileItIsCrossed(unittest.TestCase):
    """discord.py's voice extra declares `PyNaCl<1.6`, and this repo pins above
    it on purpose: CVE-2025-69277 is introduced=0, so no release under the
    ceiling avoids it.

    pip never evaluates that constraint, because the lock asks for PyNaCl
    directly rather than for discord.py[voice] — which is exactly why it needs
    saying somewhere a reader will find it.
    """

    @staticmethod
    def _declared_voice_ceiling():
        import importlib.metadata as md
        for requirement in md.requires("discord.py") or []:
            if requirement.lower().startswith("pynacl"):
                return requirement
        return None

    def test_the_pin_is_still_above_what_discord_py_declares(self):
        from packaging.requirements import Requirement
        declared = self._declared_voice_ceiling()
        self.assertIsNotNone(declared, "discord.py no longer declares PyNaCl")
        spec = Requirement(declared.split(";")[0]).specifier
        import importlib.metadata as md
        pinned = md.version("PyNaCl")
        if pinned in spec:
            self.fail(
                f"discord.py now admits PyNaCl {pinned} ({declared}). The "
                "explanation above the pin in requirements.txt is stale — "
                "remove it, and this test with it.")

    def test_the_crossing_is_explained_where_the_pin_lives(self):
        note = (REPO / "requirements.txt").read_text(encoding="utf-8")
        head = note[:note.index("PyNaCl==")]
        self.assertIn("discord.py", head.rsplit("# ---", 1)[-1][-1600:])
        self.assertIn("CVE-2025-69277", head[-1600:])

    def test_the_three_symbols_discord_py_actually_uses_exist(self):
        """The ceiling is only safe to cross while this holds. These are every
        nacl name discord.py's voice code touches."""
        import nacl.secret
        import nacl.utils
        self.assertTrue(hasattr(nacl.secret, "Aead"))
        self.assertTrue(hasattr(nacl.secret, "SecretBox"))
        self.assertTrue(hasattr(nacl.utils, "random"))


class TestTheImageDoesNotFetchCorporaAtImport(unittest.TestCase):
    """`unstructured` downloads two nltk corpora at IMPORT time unless told
    otherwise, so ingesting one .docx reaches the network from an application
    whose premise is that the models run locally — and it does it through the
    nltk Downloader, which is where CVE-2026-33236's path traversal was.

    Every assertion here is tied to the installed `unstructured`, not to a
    string this repo invented: if upstream renames the switch or changes which
    corpora it wants, these fail rather than leaving an image that silently
    starts downloading again.
    """

    @staticmethod
    def _tokenize_source() -> str:
        import unstructured.nlp.tokenize as tokenize
        return pathlib.Path(tokenize.__file__).read_text(encoding="utf-8")

    @staticmethod
    def _dockerfile() -> str:
        return (REPO / "Dockerfile").read_text(encoding="utf-8")

    def test_the_switch_is_the_one_unstructured_reads(self):
        source = self._tokenize_source()
        self.assertIn('os.getenv("AUTO_DOWNLOAD_NLTK"', source,
                      "unstructured no longer reads AUTO_DOWNLOAD_NLTK; the "
                      "Dockerfile is setting a variable nothing consults")
        self.assertIn("download_nltk_packages()", source)

    def test_the_download_still_happens_at_module_scope(self):
        """The reason this cannot be fixed by a guard at a call site: it runs on
        import, inside a module-level `if` on the environment variable, before
        any code of ours gets a turn."""
        import unstructured.nlp.tokenize as tokenize
        tree = ast.parse(pathlib.Path(tokenize.__file__).read_text(encoding="utf-8"))
        guarded = []
        for node in tree.body:                      # module scope only
            if not isinstance(node, ast.If):
                continue
            calls = [c for c in ast.walk(node) if isinstance(c, ast.Call)
                     and getattr(c.func, "id", None) == "download_nltk_packages"]
            names = [n.value for n in ast.walk(node)
                     if isinstance(n, ast.Constant) and isinstance(n.value, str)]
            if calls and "AUTO_DOWNLOAD_NLTK" in names:
                guarded.append(node)
        self.assertTrue(
            guarded,
            "unstructured no longer downloads at import under an "
            "AUTO_DOWNLOAD_NLTK check; re-read whether the Dockerfile's "
            "bake-and-disable is still the right shape")

    def test_the_image_switches_it_off(self):
        self.assertIn("ENV AUTO_DOWNLOAD_NLTK=false", self._dockerfile())

    def test_the_image_bakes_exactly_what_unstructured_asks_for(self):
        """Whatever corpora unstructured downloads must be the ones baked, or
        the image has the switch off and the files missing — which is a
        LookupError from inside a loader on the first .docx."""
        source = self._tokenize_source()
        wanted = set(re.findall(r'nltk\.download\(\s*"([^"]+)"', source))
        self.assertTrue(wanted, "could not read which corpora unstructured wants")
        dockerfile = self._dockerfile()
        for corpus in sorted(wanted):
            with self.subTest(corpus=corpus):
                self.assertIn(corpus, dockerfile)

    def test_the_bake_is_not_allowed_to_fail_quietly(self):
        """Unlike the Whisper model above it, which degrades to no
        transcription. A missing corpus with the switch off is a crash in a
        loader, so the build has to fail where somebody is watching."""
        dockerfile = self._dockerfile()
        bake = dockerfile[dockerfile.index("ENV NLTK_DATA"):
                          dockerfile.index("ENV AUTO_DOWNLOAD_NLTK")]
        self.assertIn("nltk.download", bake)
        self.assertNotIn("|| true", bake)

    def test_the_baked_location_is_one_nltk_will_look_in(self):
        """unstructured's own check appends "nltk_data" to any path that does
        not already end in it, so the download_dir and NLTK_DATA have to agree
        with that."""
        dockerfile = self._dockerfile()
        [declared] = re.findall(r"ENV NLTK_DATA=(\S+)", dockerfile)
        self.assertTrue(declared.endswith("nltk_data"), declared)
        self.assertIn(f"download_dir='{declared}'", dockerfile)

    def test_the_knob_is_documented_where_the_others_are(self):
        """CONTRIBUTING asks for a README row per knob, and this one needs its
        caveat recorded: setting it false without the corpora breaks ingestion."""
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        row = [line for line in readme.splitlines()
               if line.startswith("| `AUTO_DOWNLOAD_NLTK`")]
        self.assertTrue(row, "AUTO_DOWNLOAD_NLTK has no README row")
        self.assertIn("LookupError", row[0])


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
