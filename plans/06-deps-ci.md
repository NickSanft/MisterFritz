# 6. Purge the dependency freeze and fix CI

[← back to index](README.md)

**Effort:** L (1-3 days)  
**Depends on:** nothing

## Goal
Today requirements.txt is a raw 292-line pip freeze that carries a whole 2016-era neuroimaging stack into every install and every CI run, rooted in a single bogus pin (fitz==0.0.1.dev2) that literally fights PyMuPDF for ownership of the fitz/ package directory. When this is done, the repo declares its dependencies intentionally in a [project] table in pyproject.toml with optional-dependency groups (voice, image, ocr, telegram, browser, dev, all); requirements.txt is a regenerated, honest production lock with ~48 unused pins gone; CI installs a torch-free core+dev set with pip caching instead of pulling CUDA wheels on every push; the fitz/PyMuPDF shadowing hazard is eliminated and guarded by a regression test; and a per-module import smoke check proves the slimmed set actually still imports.

## Definition of done

- [ ] `requirements.txt` no longer contains a standalone `fitz==` pin; `PyMuPDF==1.26.7` (currently `requirements.txt:193`) remains.
- [ ] `pip install -r requirements.txt` into a clean venv, followed by `python -c "import fitz; assert fitz.open"`, succeeds — and `importlib.metadata` shows no distribution named `fitz`.
- [ ] `site-packages/fitz/` contains only PyMuPDF files (no `frontend.py`, no `tools/`).
- [ ] `pyproject.toml` has a `[project]` table with `dependencies` and `[project.optional-dependencies]` groups `voice`, `image`, `ocr`, `telegram`, `browser`, `dev`, `all`, plus a `[build-system]` and an explicit `[tool.setuptools] py-modules` list (flat layout — auto-discovery would error without it).
- [ ] `pip install -e ".[dev]"` into a clean venv installs no `torch`, no `nvidia-*`, no `diffusers`, no `coqui-tts`, no `easyocr`; the installed package count is materially smaller than today's 237-package closure.
- [ ] `pytest tests/` passes in BOTH the core-only venv and the full-extras venv, with `--cov-fail-under=60` still satisfied.
- [ ] `python scripts/check_imports.py` reports all 16 core modules importing cleanly in a core-only venv.
- [ ] `tests/test_packaging.py::test_no_neuroimaging_fitz_distribution` passes (it fails on `master` today — that is the acceptance signal).
- [ ] `.github/workflows/ci.yml` sets `cache: pip` on `actions/setup-python@v5`, installs `-e ".[dev]"`, and runs the import smoke step; a separate `full-deps` job installs `.[all]` against `https://download.pytorch.org/whl/cpu` on schedule/dispatch.
- [ ] CI wall time is measurably lower than the 3m59s-4m43s baseline captured in Step 0, with the actual number recorded in the PR body.
- [ ] `playwright` is declared in the `[browser]` extra — it was imported at `browser_tools.py:44` and declared nowhere.
- [ ] `Dockerfile` no longer installs `tesseract-ocr`; the image builds and passes the `release.yml` smoke check (`/health` + `/metrics | grep misterfritz`).
- [ ] A real `.xlsx`, `.docx`, text-PDF, and scanned-PDF each ingest successfully through `document_engine` on the full-extras venv.
- [ ] `ruff check .` and `python -m compileall -q .` are clean.
- [ ] `README.md:132`, `README.md:431`, `CONTRIBUTING.md:18-19`, and `scripts/setup.py:339,377` describe the core/extras install; a Phase 15 entry exists under `## [Unreleased]` in `CHANGELOG.md`.
- [ ] `.env.example` is unchanged — this item introduces no new env vars (stated explicitly so the reviewer doesn't go looking).

## Current state (verified against the working tree)
VERIFIED THIS SESSION (corrections to the audit are called out inline).

**1. The `fitz` pin is real and worse than described.** `requirements.txt:59` pins `fitz==0.0.1.dev2`. `.venv/Lib/site-packages/fitz-0.0.1.dev2.dist-info/METADATA` reads `Summary: Fitz: Workflow Mangement for neuroimaging data.` / `Author: Erik Kastman` / `Classifier: Development Status :: 2 - Pre-Alpha` / `Programming Language :: Python :: 2.7`, with `Requires-Dist:` configobj, configparser, httplib2, nibabel, nipype, numpy, pandas, pyxnat, scipy. PyMuPDF is pinned separately at `requirements.txt:193` (`PyMuPDF==1.26.7`) and is what `document_engine.py:78` (`import fitz  # PyMuPDF`) actually needs — `fitz.open()` at `document_engine.py:120`, `fitz.Matrix(2, 2)` at `document_engine.py:126`.

**NEW FINDING the audit missed — the two packages currently share one directory on disk.** `ls .venv/Lib/site-packages/fitz/` returns `__init__.py, frontend.py, table.py, tools/, utils.py`. `frontend.py` and `tools/` belong to the neuroimaging package; `table.py` and `utils.py` belong to PyMuPDF. `.venv/Lib/site-packages/fitz/__init__.py` is currently PyMuPDF's shim (`from pymupdf import *`), so `import fitz; fitz.open` works today — PyMuPDF happened to write last. Both RECORDs claim `fitz/__init__.py`:
  - `fitz-0.0.1.dev2.dist-info/RECORD` -> `fitz/__init__.py,sha256=pQk1zJ-zWDD7d0uPU87DkN2wWrwX7SEHgJJXyLsi-JA,331`
  - `pymupdf-1.26.7.dist-info/RECORD` -> `fitz/__init__.py,,`
  This is install-order-dependent roulette. If the neuroimaging wheel lands last, `import fitz` at `document_engine.py:78` **succeeds** (so `PYMUPDF_AVAILABLE = True` at line 80 and the `except ImportError` guard at 81-83 never fires) and then `fitz.open(file_path)` at line 120 raises `AttributeError` inside PDF ingestion. Silent, guard-defeating breakage.

**NEW OPERATIONAL GOTCHA:** because `fitz`'s RECORD lists `fitz/__init__.py`, a plain `pip uninstall fitz` **deletes PyMuPDF's shim** and breaks `import fitz`. PyMuPDF must be force-reinstalled afterwards. Any engineer who runs the obvious uninstall without this step will think the change broke PDF OCR.

**2. `pathlib==1.0.1` at `requirements.txt:169` — audit missed this.** `.venv/Lib/site-packages/pathlib.py` exists (41KB, the Python 2 backport). Nothing in the installed metadata requires it; it is a pure freeze artifact. It does not shadow today (`import pathlib` resolves to the CPython 3.13 stdlib package, verified), because stdlib packages win over site-packages modules — but it is py2-era code sitting on `sys.path` and pip has historically failed to uninstall it cleanly.

**3. Removal candidates — verified by grep across all tracked `.py`, not trusted.** Zero import sites anywhere in the repo (including `tests/`, `local-sim/`, `scripts/`, `skills/`) for: `pygame`, `pyttsx3`, `pdf2image`, `pytesseract`, `ffmpeg` / `ffmpeg-python`, `langchain_google_community`, `googleapiclient`, `google.oauth*`, `nibabel`, `nipype`, `pyxnat`, `configobj`, `configparser`, `rdflib`, `traits`, `prov`, `puremagic`, `pydot`, `simplejson`, `looseversion`.

**4. Dependency-closure analysis (script run against the live venv metadata).** Rooting the graph at the intentional direct-dependency set yields **237 kept / 101 orphaned**. Intersecting the orphan set with `requirements.txt` gives **50 pins that no intentional root needs**, including all the `fitz`->`nipype` leaves (`acres`, `ci-info`, `etelemetry`, `looseversion`, `puremagic`, `pydot`, `prov`, `rdflib`, `simplejson`, `traits`, `configobj`, `configparser`, `nibabel`, `nipype`, `pyxnat`, `httplib2`), the 6-package `google-*` chain rooted solely at `langchain-google-community` (`requirements.txt:115`), the httpx extras (`h2:83`, `hpack:85`, `hyperframe:94`, `brotli:19`, `socksio:234`), and the uvicorn[standard] extras (`httptools:89`, `watchfiles:270`, `websockets:273`).

**5. Two AUDIT CORRECTIONS.**
  - **`pandas` must STAY.** It is a `fitz`->`nipype` dep, but it is *independently* required by `unstructured[xlsx]`, and `document_engine.py:30` uses `UnstructuredExcelLoader`. Verified: `unstructured` declares `pandas; extra == "xlsx"`. Dropping pandas would break `.xlsx` ingestion at load time, silently, since the loader imports it lazily.
  - **`gruut_lang_de/es/fr` (`requirements.txt:78,80,81`) must STAY under the voice extra.** My closure script skipped extras-gated requirements and wrongly flagged them; `coqui-tts` declares `gruut[de,es,fr]>=2.4.0`. Likewise `matplotlib` is required by `coqui-tts`, not orphaned.

**6. httpx/uvicorn extras confirmed unneeded.** `agent_tools.py:38-39` constructs a plain `httpx.Client(timeout=httpx.Timeout(...))` — no `http2=True`, no socks proxy. `admin_panel.py:840-846` constructs `uvicorn.Config(app, host="127.0.0.1", port=ADMIN_PANEL_PORT, log_level="warning", access_log=False)` — no `reload`, no WebSocket routes (web chat is SSE via `StreamingResponse`).

**7. `pyproject.toml` (44 lines) has no `[project]` and no `[build-system]`** — only `[tool.ruff]:1`, `[tool.ruff.lint]:23`, `[tool.ruff.lint.per-file-ignores]:34`, `[tool.pytest.ini_options]:39` (`pythonpath = ["."]`, `testpaths = ["tests"]`).

**8. CI (`.github/workflows/ci.yml`, 48 lines).** `actions/setup-python@v5` at 16-18 with **no `cache:` key**; `pip install -r requirements.txt` at 20-21; `pip install pytest pytest-cov pytest-asyncio ruff` at 23-24; `compileall` 26-27; `ruff check .` 29-30; pytest with `--cov-fail-under=60` at 32-39. No `actions/cache`. Recent runs are green at ~4m (`gh run list`: 3m59s-4m43s across the last 8), so this is a *time and fragility* fix, not a broken-build fix — be honest about that.

**9. `Dockerfile:32` installs `tesseract-ocr`** in the runtime stage (comment at line 28 says "OCR fallback for scanned PDFs") but OCR is `easyocr` (`document_engine.py:70`, `get_ocr_reader()` at 86-91). `pytesseract` is never imported. Dead apt layer.

**10. `playwright` is imported at `browser_tools.py:44` (lazy, inside `_ensure_browser()`) and is NOT in `requirements.txt`.** Confirmed. The module's docstring at lines 12-14 already documents `pip install playwright && playwright install chromium` as a manual step, and the `except ImportError` at 45-49 returns a helpful message. Also note `browser_tools.py` is **not wired in anywhere** — nothing imports it (`get_browser_tools_description` at line 149 has no callers). It only needs a declared extra, not a core pin.

**11. `requirements.txt` is a stale/partial freeze, not a lock.** Lines 1-279 are `pip freeze` output; lines 280-292 are hand-appended (`watchdog`, `pydub`, `faster-whisper>=1.0.0`, `python-telegram-bot>=20.0`, `prometheus-client>=0.20.0`, `APScheduler>=3.10.0`, `starlette>=0.46`, `python-multipart>=0.0.22`, `Markdown>=3.5`). Proof it's stale: `faster-whisper`'s own transitive deps (`av`, `ctranslate2`, `zstandard`) are installed in the venv but appear **nowhere** in `requirements.txt`. So the "lock" doesn't even lock.

**12. BLOCKER for a torch-free core — `agent_tools.py:20` does a module-level `import image_generator`,** and `image_generator.py:4-6` imports `diffusers`, `torch`, and `xformers` at module level. `mister_fritz.py:17` imports `agent_tools`; `bot_commands`/`main_discord` import `mister_fritz`. So `import agent_tools` today drags in the entire multi-GB GPU stack. The only use is a single call at `agent_tools.py:306`. Tests only survive by stubbing `image_generator` in `sys.modules` (`tests/test_agent_tools.py:30`, `tests/test_mister_fritz.py:24`, `tests/test_bot_commands.py:25`, `tests/test_discord_commands.py:31`). Same pattern for `tts`: `bot_commands.py:26` and `main_discord.py:28` do `from tts import TTSEngine`, and `tts.py:5,8` import `torch` and `TTS.api` at module level.

**13. What the test suite actually needs at import time (drives the CI set).** `tests/test_document_engine.py:60` imports the **real** `document_engine` (stubbing only `easyocr`:43, `fitz`:45, `spacy`/`thinc`/`transformers`:49-52) — so `msoffcrypto`, `openpyxl`, `watchdog`, `langchain_*`, `chromadb`, `pydantic` are mandatory. `tests/test_stt.py:63` imports the **real** `stt`, whose `pydub` import is module-level (`stt.py:7-8`) while `faster_whisper` is deferred to inside `_get_model()` (`stt.py:38`) — so `pydub` is mandatory, `faster-whisper` is not. `tests/test_agent_tools.py` does **not** stub `ddgs`/`bs4` — so `ddgs` and `beautifulsoup4` are mandatory. `tests/test_admin_panel.py:28` uses `starlette.testclient.TestClient` (needs `httpx`). `tests/test_scheduler.py:7-8` needs `apscheduler`. **Nothing in the suite needs `torch`, `diffusers`, `xformers`, `coqui-tts`, `easyocr`, `PyMuPDF`, `faster-whisper`, `transformers`, or `python-telegram-bot`.** Verified separately: `unstructured`, `pypdf`, and `pandas` are *not* needed at import time either — I blocked them via an `__import__` guard and `from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader, UnstructuredExcelLoader, CSVLoader, TextLoader` still succeeded.

**14. Docs referencing the install flow:** `README.md:132` (`pip install -r requirements.txt`), `README.md:110,113` (an existing manual CUDA-vs-CPU torch section), `README.md:242`, `README.md:431` (`Install optional deps: pip install easyocr PyMuPDF pillow` — already an informal "ocr extra"), `CONTRIBUTING.md:18-19`, `scripts/setup.py:339,377`. `scripts/setup.py` is loaded via `exec_module` by `tests/test_setup_wizard.py:20` and must stay stdlib-only; that test asserts nothing about the pip strings, so rewording them is safe.

## Change sites

### `pyproject.toml:1 (insert above the existing [tool.ruff] block, which currently starts at line 1)`

Add [build-system], an intentional [project] table with optional-dependency groups, and explicit setuptools flat-layout config. This becomes the single source of truth for what we depend on and why; requirements.txt stays the pinned lock.

# NEW — prepended above the existing `[tool.ruff]` (currently line 1)
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "misterfritz"
version = "0.0.0"   # real version lives in fritz_utils.__version__; see openQuestions
description = "Sardonic English-butler AI assistant for Discord/Telegram/web"
requires-python = ">=3.12"

# Core = everything needed to import and run the bot's non-optional surface:
# Discord + LangGraph agent + Chroma memory + document RAG + admin/chat panel.
dependencies = [
    # chat surfaces
    "discord.py>=2.6",
    "PyNaCl>=1.5",                 # discord voice
    # agent / LLM stack
    "langchain>=1.1",
    "langchain-core>=1.2",
    "langchain-community>=0.4",
    "langchain-chroma>=1.1",
    "langchain-ollama>=1.0",
    "langchain-text-splitters>=1.1",
    "langgraph>=1.0",
    "langgraph-checkpoint-sqlite>=3.0",
    "chromadb>=1.3",
    "ollama>=0.6",
    "pydantic>=2.12",
    "typing_extensions>=4.15",
    # tools
    "httpx>=0.28",
    "beautifulsoup4>=4.14",
    "ddgs>=9.10",
    "pytz>=2025.2",
    "APScheduler>=3.10",
    # document ingestion (document_engine.py:13-14 imports these at module level)
    "watchdog>=6.0",
    "msoffcrypto-tool>=5.4",
    "openpyxl>=3.1",
    "unstructured>=0.18",
    "python-docx>=1.2",
    "pypdf>=6.4",
    "pandas>=2.0",                 # required by unstructured[xlsx] — NOT the nipype chain
    # audio preprocessing (stt.py:7-8 is module-level; the model itself is [voice])
    "pydub>=0.25",
    # admin panel + web chat
    "starlette>=0.46",
    "uvicorn>=0.38",
    "python-multipart>=0.0.22",
    "Jinja2>=3.1",
    "Markdown>=3.5",
    # config + observability
    "python-dotenv>=1.2",
    "prometheus-client>=0.20",
]

[project.optional-dependencies]
# Coqui TTS (tts.py) + Whisper STT model (stt.py:38). Pulls torch.
voice = ["coqui-tts>=0.27", "faster-whisper>=1.0"]
# SDXL image generation (image_generator.py:4-6).
image = ["diffusers>=0.36", "torch>=2.9", "xformers>=0.0.33", "accelerate>=1.12", "safetensors>=0.7", "transformers>=4.57", "pillow>=12.0"]
# Scanned-PDF OCR fallback (document_engine.py:67-91). PyMuPDF provides `import fitz`.
ocr = ["easyocr>=1.7", "PyMuPDF>=1.26", "pillow>=12.0"]
telegram = ["python-telegram-bot>=20.0"]
# browser_tools.py:44 — currently unwired, lazy-imported, previously undeclared.
browser = ["playwright>=1.40"]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "pytest-asyncio>=0.23", "ruff>=0.6"]
all = ["misterfritz[voice,image,ocr,telegram,browser]"]

# Flat module layout — setuptools auto-discovery would error with
# "Multiple top-level modules discovered", so enumerate explicitly.
[tool.setuptools]
py-modules = [
    "admin_panel", "agent_tools", "bot_adapters", "bot_commands", "browser_tools",
    "cards", "chat_auth", "document_engine", "file_tools", "fritz_utils",
    "image_generator", "main_discord", "main_telegram", "migrate_db", "mister_fritz",
    "observability", "prewarm", "privacy", "scheduler", "storage", "stt", "tts",
    "workspace_store",
]
packages = ["skills"]

### `agent_tools.py:20 and 305-306`

Move the module-level `import image_generator` into the one function that uses it. This is the single change that makes a torch-free install able to `import agent_tools` / `import mister_fritz` — without it the whole [project] core group is a fiction. Two lines.

# agent_tools.py:19-20 — BEFORE
import document_engine
import image_generator

# AFTER (line 20 deleted; document_engine stays — it imports fine on core deps)
import document_engine

# agent_tools.py:294-306 — BEFORE
@tool(parse_docstring=True)
def generate_image(prompt: str):
    """..."""
    with time_tool("generate_image"):
        return image_generator.generate_image(prompt)

# AFTER
@tool(parse_docstring=True)
def generate_image(prompt: str):
    """..."""
    with time_tool("generate_image"):
        # Deferred: image_generator imports torch/diffusers/xformers at module
        # level. Keeping it out of the import graph lets the bot start (and CI
        # run) without the multi-GB GPU stack. Install with `pip install .[image]`.
        import image_generator
        return image_generator.generate_image(prompt)

### `requirements.txt:1-292 (whole file regenerated; 291 newline-terminated lines + an unterminated final line)`

Regenerate from a clean venv installed via `pip install ".[voice,image,ocr,telegram]"`. Expect ~48 pins to disappear and the missing faster-whisper transitives (av, ctranslate2, zstandard) to appear. Add a header comment declaring it a generated lock and pointing at pyproject as the source of truth.

# HEADER to prepend to the regenerated file:
# ---------------------------------------------------------------------------
# GENERATED LOCK — do not hand-edit.
# Source of truth for *intent* is [project] / [project.optional-dependencies]
# in pyproject.toml. This file is the pinned production set consumed by the
# Dockerfile and by `pip install -r requirements.txt`.
#
# Regenerate:
#   python -m venv .venv-freeze && .venv-freeze/Scripts/activate   # (bin/activate on POSIX)
#   pip install ".[voice,image,ocr,telegram]"
#   pip freeze --exclude-editable > requirements.txt
# ---------------------------------------------------------------------------

# EXPECTED REMOVALS (review the diff against this list; each verified unused):
#   the fitz->nipype chain rooted at :59
#     fitz==0.0.1.dev2       (:59)   nipype==1.10.0      (:147)
#     nibabel==5.3.3         (:145)  pyxnat==1.6.4       (:213)
#     traits==7.1.0          (:254)  prov==2.1.1         (:179)
#     rdflib==7.5.0          (:216)  pydot==4.0.1        (:190)
#     puremagic==1.30        (:181)  simplejson==3.20.2  (:232)
#     looseversion==1.3.0    (:128)  configobj==5.0.9    (:31)
#     configparser==7.2.0    (:32)   acres==0.5.0        (:3)
#     ci-info==0.3.0         (:26)   etelemetry==0.3.1   (:53)
#     httplib2==0.31.0       (:88)
#   the google-* chain rooted at :115
#     langchain-google-community==3.0.2  (:115)
#     google-api-core==2.28.1            (:65)
#     google-api-python-client==2.187.0  (:66)
#     google-auth-httplib2==0.3.0        (:68)
#     google-auth-oauthlib==1.2.3        (:69)
#     google-cloud-core==2.5.0           (:70)
#     google-cloud-modelarmor==0.3.0     (:71)
#     grpcio-status==1.76.0              (:75)
#     proto-plus==1.27.0                 (:177)
#     uritemplate==4.2.0                 (:266)
#   never imported anywhere in the repo
#     pygame==2.6.1 (:191)   pyttsx3==2.99 (:211)   comtypes (:30)
#     pypiwin32==223 (:198)  pdf2image==1.17.0 (:170)
#     pytesseract==0.3.13 (:202)
#     ffmpeg==1.4 (:55)  ffmpeg-python==0.2.0 (:56)  future==1.0.0 (:64)
#   py2-era freeze artifact, nothing requires it
#     pathlib==1.0.1 (:169)
#   unrequested httpx extras (agent_tools.py:38 uses no http2/socks)
#     h2==4.3.0 (:83)  hpack==4.1.0 (:85)  hyperframe==6.1.0 (:94)
#     brotli==1.2.0 (:19)  socksio==1.0.0 (:234)
#   unrequested uvicorn[standard] extras (admin_panel.py:840-846 is plain)
#     httptools==0.7.1 (:89)  watchfiles==1.1.1 (:270)  websockets==15.0.1 (:273)
#   stray
#     typer-slim==0.20.0 (:258)   # typer itself is a chromadb dep and stays
#
# EXPECTED TO REMAIN despite looking like the nipype chain:
#   pandas   — required by unstructured[xlsx]; document_engine.py:30
#   scipy    — required by easyocr
#   numpy    — required by ~everything
#   gruut_lang_de/es/fr (:78,:80,:81) — coqui-tts declares gruut[de,es,fr]
#   matplotlib — coqui-tts dep
#   google-auth (:67), googleapis-common-protos (:72) — kubernetes/otel deps

### `.github/workflows/ci.yml:16-24 (add cache + swap install), plus a new job appended after line 48`

Enable pip caching on setup-python; install core+dev from pyproject instead of the full pinned freeze (no torch); add an import smoke step; add a separate opt-in full-deps job that installs .[all] against the CPU torch index so heavy-path resolution drift is still caught.

# ci.yml:16-24 — BEFORE
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Install test/lint tools
        run: pip install pytest pytest-cov pytest-asyncio ruff

# AFTER
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: pyproject.toml

      # Core + dev only. The voice/image/ocr extras pull torch (multi-GB CUDA
      # wheels) and nothing in tests/ imports them — every test that touches
      # image_generator / tts / easyocr / fitz stubs them in sys.modules.
      # The `full-deps` job below covers the heavy path.
      - name: Install dependencies
        run: pip install -e ".[dev]"

# ...compileall / ruff / pytest steps at 26-39 unchanged...
# NEW step, inserted after "Lint with ruff" (line 30):
      - name: Import smoke test (core deps only)
        run: python scripts/check_imports.py
        env:
          DISCORD_BOT_TOKEN: "ci-placeholder-token"
          ROOT_USER: "ci-root"
          OLLAMA_HOST: "http://127.0.0.1:11434"

# NEW job appended after line 48:
  full-deps:
    name: Resolve full dependency set (CPU torch)
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: pyproject.toml
      - name: Install everything against the CPU torch index
        run: |
          pip install --extra-index-url https://download.pytorch.org/whl/cpu \
            -e ".[all,dev]"
      - name: Assert no stray 'fitz' distribution
        run: |
          python - <<'PY'
          from importlib.metadata import distributions
          names = {d.metadata["Name"].lower() for d in distributions() if d.metadata["Name"]}
          assert "fitz" not in names, "the neuroimaging 'fitz' package is back — it shadows PyMuPDF"
          import fitz
          assert hasattr(fitz, "open"), f"'import fitz' resolved to {fitz.__file__}, not PyMuPDF"
          PY

# Also add to the `on:` block at ci.yml:3-8:
#   schedule:
#     - cron: "0 6 * * 1"   # Mondays 06:00 UTC
#   workflow_dispatch:

### `scripts/check_imports.py:new file`

Stdlib-only smoke checker that imports each first-party module in its own subprocess (so module-level side effects and sys.modules pollution can't cascade). Proves the slimmed core actually satisfies the real import graph.

"""Import-smoke check: every module here must import on core deps alone.

Run after any dependency change:  python scripts/check_imports.py
Stdlib-only, one subprocess per module so side effects stay isolated.
"""
import subprocess
import sys

# Importable with `pip install .` (no extras).
CORE_MODULES = [
    "fritz_utils", "observability", "storage", "workspace_store", "privacy",
    "chat_auth", "cards", "file_tools", "scheduler", "prewarm", "browser_tools",
    "stt", "document_engine", "agent_tools", "mister_fritz", "admin_panel",
]
# Require an extra; reported as skip unless the extra is installed.
EXTRA_MODULES = {
    "image_generator": "image",
    "tts": "voice",
    "bot_commands": "voice",     # bot_commands.py:26 -> from tts import TTSEngine
    "main_discord": "voice",     # main_discord.py:28 -> same
    "main_telegram": "telegram",
}


def _import_ok(mod):
    proc = subprocess.run(
        [sys.executable, "-c", "import " + mod],
        capture_output=True, text=True,
    )
    err = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
    return proc.returncode == 0, err


def main():
    failed = []
    for mod in CORE_MODULES:
        ok, err = _import_ok(mod)
        print("  %s  %s%s" % ("ok  " if ok else "FAIL", mod, "" if ok else "  -> " + err))
        if not ok:
            failed.append(mod)
    for mod, extra in EXTRA_MODULES.items():
        ok, _ = _import_ok(mod)
        print("  %s  %s  (extra: %s)" % ("ok  " if ok else "skip", mod, extra))
    if failed:
        print("\n%d core module(s) failed to import: %s" % (len(failed), ", ".join(failed)))
        return 1
    print("\nAll core modules import cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

### `tests/test_packaging.py:new file`

Regression guard that runs in the normal (core-only) CI job. Asserts no distribution literally named fitz is installed, and that every [project] dependencies entry is actually installed. Catches the exact class of bug this item fixes.

"""Guards against dependency-manifest regressions.

The repo previously pinned `fitz==0.0.1.dev2` — a 2016 neuroimaging package
that fights PyMuPDF for the `fitz/` directory on disk. `import fitz` in
document_engine.py:78 must resolve to PyMuPDF.
"""
import re
import tomllib
import unittest
from importlib.metadata import distributions
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _installed():
    return {
        re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower()
        for d in distributions()
        if d.metadata["Name"]
    }


class TestDependencyManifest(unittest.TestCase):
    def test_no_neuroimaging_fitz_distribution(self):
        self.assertNotIn(
            "fitz", _installed(),
            "The 'fitz' PyPI distribution (neuroimaging, py2.7) is installed. It "
            "overwrites PyMuPDF's fitz/__init__.py, so document_engine.py:78 can "
            "import successfully and still blow up at fitz.open() on line 120. "
            "Use PyMuPDF only.",
        )

    def test_declared_core_deps_are_installed(self):
        data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        declared = {
            re.sub(r"[-_.]+", "-", re.split(r"[\s\[<>=!;]", spec)[0]).lower()
            for spec in data["project"]["dependencies"]
        }
        self.assertEqual(
            set(), declared - _installed(),
            "pyproject [project].dependencies lists packages that aren't installed",
        )


if __name__ == "__main__":
    unittest.main()

### `Dockerfile:14, 25-33`

Copy pyproject.toml alongside requirements.txt (so the lock's provenance travels with the image), and drop the dead tesseract-ocr apt package.

# Dockerfile:14 — BEFORE
COPY requirements.txt .
# AFTER
COPY requirements.txt pyproject.toml ./

# Dockerfile:25-33 — BEFORE
# Runtime system deps:
#   ffmpeg      — audio processing (replaces bundled ffmpeg.exe)
#   libsndfile1 — required by soundfile / Coqui TTS
#   tesseract   — OCR fallback for scanned PDFs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# AFTER — OCR is easyocr (document_engine.py:70), which is pure-Python +
# torch. pytesseract was never imported; the tesseract binary was dead weight.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

### `tests/test_document_engine.py:40-45`

No functional change — the existing stubs already do the right thing. Update the stale comment so the next reader understands these stubs are now load-bearing (easyocr/fitz/transformers are genuinely absent in the CI env, not merely 'optional in this codebase').

# tests/test_document_engine.py:40-45 — BEFORE
# msoffcrypto + the OCR stack are optional in this codebase; the source code
# guards them with try/except at import time, so stubs only need to exist if
# something downstream imports them eagerly.
_ensure_mock("easyocr")
# PyMuPDF (fitz) — guarded by try/except in the source.
_ensure_mock("fitz")

# AFTER
# easyocr + PyMuPDF live in the optional `ocr` extra and are NOT installed in
# CI (see .github/workflows/ci.yml — core+dev only). document_engine.py:67-83
# guards both with try/except, but these stubs keep OCR_AVAILABLE /
# PYMUPDF_AVAILABLE True so the OCR branch stays reachable in tests.
# NOTE: stubbing `fitz` with a MagicMock means PYMUPDF_AVAILABLE is True even
# though fitz.open is a mock — intentional, and the reason
# tests/test_packaging.py guards the real distribution separately.
_ensure_mock("easyocr")
_ensure_mock("fitz")

### `CONTRIBUTING.md:18-19`

Point the dev setup at the extras instead of the full freeze, so contributors get a fast install that matches CI.

# BEFORE (lines 18-19)
pip install -r requirements.txt
pip install pytest pytest-cov pytest-asyncio ruff

# AFTER
# Core + test tooling — matches CI, no torch, installs in seconds.
pip install -e ".[dev]"

# Add the optional stacks you actually need:
#   pip install -e ".[voice]"     # Coqui TTS + faster-whisper
#   pip install -e ".[image]"     # SDXL image generation
#   pip install -e ".[ocr]"       # scanned-PDF OCR (easyocr + PyMuPDF)
#   pip install -e ".[telegram]"  # main_telegram.py
#   pip install -e ".[browser]"   # browser_tools.py (then: playwright install chromium)
#   pip install -e ".[all,dev]"   # everything
#
# CPU-only machines should prefix the torch-bearing extras with:
#   --extra-index-url https://download.pytorch.org/whl/cpu

### `README.md:105-115, 132, 431`

Replace the single `pip install -r requirements.txt` with the core/extras story; fold the existing manual CUDA-vs-CPU torch block (105-115) and the ad-hoc OCR line (431) into the extras vocabulary so there is one install idiom.

# README.md:132 — BEFORE
pip install -r requirements.txt

# AFTER
# Everything (matches the Docker image):
pip install -r requirements.txt

# Or install only what you need:
pip install -e "."              # core: Discord bot, agent, memory, doc RAG, web chat
pip install -e ".[voice]"       # + Coqui TTS and Whisper STT
pip install -e ".[image]"       # + SDXL image generation
pip install -e ".[ocr]"         # + scanned-PDF OCR
pip install -e ".[all]"         # + Telegram and Playwright browser tools

# README.md:431 — BEFORE
Install optional deps: `pip install easyocr PyMuPDF pillow`.
# AFTER
Install optional deps: `pip install -e ".[ocr]"`.

### `scripts/setup.py:339, 377`

Update the two user-facing install hints to the new idiom. Must stay stdlib-only — tests/test_setup_wizard.py:20 exec_module()s this file. No test asserts on these strings (verified), so this is a safe string edit.

# scripts/setup.py:339 — BEFORE
        warn("PyTorch is not installed yet (run `pip install -r requirements.txt`).")
# AFTER
        warn("PyTorch is not installed (it ships with the `image` / `voice` extras).")
        info("Install with: pip install -e \".[image,voice]\"")

# scripts/setup.py:377 — BEFORE
    print("    1. pip install -r requirements.txt   (if you haven't already)")
# AFTER
    print("    1. pip install -r requirements.txt   (everything), or")
    print("       pip install -e \".[voice,image]\"   (core + the bits you want)")

### `CHANGELOG.md:8 (under the existing `## [Unreleased]` at line 8, alongside the existing ### Added / ### Changed groupings)`

Phase-style entry matching the repo's prose convention.

### Removed
- **Phase 15 — dependency purge.** `requirements.txt` was a raw `pip freeze` carrying a
  2016 pre-alpha neuroimaging package called `fitz` ("Fitz: Workflow Mangement for
  neuroimaging data", Python 2.7) that had nothing to do with PyMuPDF beyond the name
  collision — and which fought PyMuPDF for ownership of `site-packages/fitz/`. Whichever
  wheel unpacked last won, so `import fitz` in `document_engine.py` was a coin flip;
  when the wrong one won, the `except ImportError` guard didn't fire and the failure
  surfaced as an `AttributeError` deep inside PDF OCR. It also dragged in nipype, nibabel,
  pyxnat, prov, rdflib, traits and friends. Gone, along with `langchain-google-community`
  (sole root of a six-package `google-*` chain), `pygame`, `pyttsx3`, `pdf2image`,
  `pytesseract`, `ffmpeg==1.4` (a junk placeholder, not a real ffmpeg binding),
  `ffmpeg-python`, and `pathlib==1.0.1` (the Python 2 backport, sitting on `sys.path`
  for no reason). ~48 pins removed. The `tesseract-ocr` apt package also left the
  Dockerfile — OCR has always been easyocr.

### Changed
- **Phase 15 — intentional dependency manifest.** `pyproject.toml` gained a real
  `[project]` table. Core is what the bot needs to boot: Discord, LangGraph, Chroma,
  document RAG, the admin panel and web chat. Everything heavy is an extra —
  `voice` (Coqui TTS + Whisper), `image` (SDXL), `ocr` (easyocr + PyMuPDF),
  `telegram`, `browser` (Playwright, which was imported by `browser_tools.py` but
  had never been declared anywhere), and `dev`. `requirements.txt` survives as the
  generated production lock the Dockerfile consumes.
- **Phase 15 — CI stops downloading CUDA.** `agent_tools` no longer imports
  `image_generator` at module level — the one call site now imports it lazily — which
  takes torch, diffusers and xformers out of the bot's import graph entirely. CI
  installs `-e ".[dev]"` with pip caching instead of the full freeze. A separate
  weekly `full-deps` job installs `.[all]` against the CPU torch index so drift in
  the heavy stack still gets caught, and asserts that `import fitz` really is PyMuPDF.
- New `scripts/check_imports.py` imports every first-party module in its own
  subprocess, so "we removed a dependency something quietly needed" fails loudly
  instead of at 2am in production.

## Steps

1. **Step 0 — capture the baseline.** Record current CI wall time (`gh run list --workflow=ci.yml --limit 5`; latest runs are 3m59s-4m43s) and the current `pytest --cov` total so you can prove the coverage gate at `.github/workflows/ci.yml:37` (`--cov-fail-under=60`) still passes afterwards. Commit nothing.
2. **Step 1 — lazy-import `image_generator` (commit on its own).** Delete `import image_generator` at `agent_tools.py:20`; add it inside `generate_image` just above `agent_tools.py:306`. Run the full suite locally. Nothing should change: `tests/test_agent_tools.py:30`, `tests/test_mister_fritz.py:24`, `tests/test_bot_commands.py:25` and `tests/test_discord_commands.py:31` all stub `image_generator` in `sys.modules` and the stub is now simply never consulted at import time. This commit alone is the load-bearing one — verify it standalone before anything else moves.
3. **Step 2 — add the `[project]` table to `pyproject.toml` (commit on its own).** Prepend `[build-system]`, `[project]`, `[project.optional-dependencies]`, and `[tool.setuptools]` above the existing `[tool.ruff]` at line 1. Do NOT touch `requirements.txt` yet. Verify the metadata is well-formed and flat-layout discovery is satisfied: `pip install -e . --no-deps --dry-run` (or `python -m build --sdist`). If setuptools complains about multiple top-level modules, the `py-modules` list is incomplete — note the empty untracked `main_discord/` and `testing/` directories in the repo root; they have no `__init__.py` so they should be ignored, but confirm.
4. **Step 3 — add `scripts/check_imports.py` and `tests/test_packaging.py` (commit on its own).** Run `python scripts/check_imports.py` in your CURRENT fat venv first — it must pass everything including the extras modules. This establishes that the checker itself is correct before you start removing packages. Run `pytest tests/test_packaging.py` — `test_no_neuroimaging_fitz_distribution` **will fail right now**, which is exactly the point. Land it together with Step 4, or mark it `@unittest.expectedFailure` if you need the commits split.
5. **Step 4 — build a clean reference venv and regenerate the lock.** In a scratch directory, not your working venv: `python -m venv .venv-freeze`; `.venv-freeze/Scripts/python -m pip install -U pip`; `.venv-freeze/Scripts/pip install ".[voice,image,ocr,telegram]"`; `.venv-freeze/Scripts/pip freeze --exclude-editable > requirements.txt.new`. Diff `requirements.txt.new` against the current `requirements.txt` and walk the diff against the EXPECTED REMOVALS / EXPECTED TO REMAIN lists in the requirements.txt codeSketch. Any removal not on that list, or any of the four 'must remain' entries (`pandas`, `scipy`, `gruut_lang_*`, `matplotlib`) missing, means stop and re-derive. Prepend the generated-lock header, then replace the file.
6. **Step 5 — repair your working venv (do not skip).** `pip uninstall fitz` removes `site-packages/fitz/__init__.py` because that path is listed in `fitz-0.0.1.dev2.dist-info/RECORD` — and that file is currently PyMuPDF's shim. After uninstalling, run `pip install --force-reinstall --no-deps PyMuPDF==1.26.7` and confirm with `python -c "import fitz; print(fitz.__file__); print(fitz.open)"`. Also delete the leftover `site-packages/fitz/frontend.py` and `site-packages/fitz/tools/` — neuroimaging files that neither RECORD will clean up. Cleanest alternative: throw the venv away and rebuild from the new `requirements.txt`.
7. **Step 6 — verify the slimmed set actually runs.** In the clean `.venv-freeze` (full extras): `python scripts/check_imports.py` -> all green. Then build a SECOND venv with `pip install -e ".[dev]"` (core only, no torch) and run `python scripts/check_imports.py` again — the 16 `CORE_MODULES` must all import; the 5 `EXTRA_MODULES` will report `skip`. Then run `pytest tests/ -v` in that core-only venv. Any failure here is the real signal that a package I classified as an extra is actually core.
8. **Step 7 — exercise the runtime paths the import check can't reach.** With the full-extras venv, ingest one real `.xlsx`, one real `.docx`, one text-bearing `.pdf`, and one scanned/image-only `.pdf` through `document_engine`. The scanned PDF is the one that proves `fitz.open()` -> `fitz.Matrix(2, 2)` -> `easyocr` still works end-to-end (`document_engine.py:120-130`). The `.xlsx` proves `pandas` survived. This is the step that catches a wrong `unstructured` extras call.
9. **Step 8 — rework CI (`.github/workflows/ci.yml`).** Add `cache: pip` + `cache-dependency-path: pyproject.toml` to the `actions/setup-python@v5` block at lines 16-18. Replace lines 20-24 with the single `pip install -e ".[dev]"`. Insert the `python scripts/check_imports.py` step after the ruff step (line 30). Add `schedule:` + `workflow_dispatch:` to the `on:` block (lines 3-8) and append the `full-deps` job. Push to a branch and watch the run — compare wall time to the Step 0 baseline and confirm `--cov-fail-under=60` still passes.
10. **Step 9 — Dockerfile.** Change line 14 to `COPY requirements.txt pyproject.toml ./` and drop `tesseract-ocr` from the apt list at lines 29-33 (updating the comment at line 28). Build the image locally and run the same health check `.github/workflows/release.yml:65-74` does: `docker run ... python local-sim/mock_fritz.py`, then `curl -f http://localhost:8000/health` and `curl -f http://localhost:8000/metrics | grep misterfritz`. Also confirm the `faster_whisper` model pre-bake at `Dockerfile:53` still succeeds (it is `|| true`, so read the build log rather than trusting the exit code).
11. **Step 10 — docs and changelog.** Update `CONTRIBUTING.md:18-19`, `README.md:132`, `README.md:431`, fold `README.md:105-115` into the extras vocabulary, and reword `scripts/setup.py:339` and `:377`. Add the Phase 15 CHANGELOG entry under `## [Unreleased]`. Re-run `pytest tests/test_setup_wizard.py` — it `exec_module()`s `scripts/setup.py` at line 20, so a syntax slip there fails loudly.
12. **Step 11 — final gate.** `ruff check .` clean, `python -m compileall -q .` clean, full suite green in both the core-only and full-extras venvs, `scripts/check_imports.py` green in both, and `pytest tests/test_packaging.py` green (no `fitz` distribution).

## Config and env changes

- NO new environment variables are introduced by this item, so `.env.example` needs no new knob. Stated explicitly because the repo convention is that every new knob is documented there — this change adds none.
- Two new CI-only knobs live in `.github/workflows/ci.yml`, not in env config: `cache: pip` / `cache-dependency-path: pyproject.toml` on `actions/setup-python@v5`, and `--extra-index-url https://download.pytorch.org/whl/cpu` in the new `full-deps` job.
- `Dockerfile` env vars are unchanged (`FFMPEG_PATH=ffmpeg` at :46, `FFPROBE_PATH=ffprobe` at :47, `WHISPER_*` at :50-52 all stay). Only the apt package list at :29-33 and the COPY at :14 change.
- New install idioms to publicise (docs only, no config): `pip install -e ".[dev]"` (matches CI), `.[voice]`, `.[image]`, `.[ocr]`, `.[telegram]`, `.[browser]`, `.[all]`.
- `browser_tools.py` gains a declared home (the `[browser]` extra) for the first time — `playwright` was imported at `browser_tools.py:44` but appeared in no requirements file. Its post-install step (`playwright install chromium`, already documented in the module docstring at lines 13-14) must be surfaced in CONTRIBUTING.md.

## Tests
### New

- `tests/test_packaging.py::TestDependencyManifest::test_no_neuroimaging_fitz_distribution` — asserts no distribution named `fitz` is installed. Fails against the repo as it stands today; passes after Step 4/5. This is the direct regression guard for the shadowing bug.
- `tests/test_packaging.py::TestDependencyManifest::test_declared_core_deps_are_installed` — parses `[project].dependencies` out of `pyproject.toml` with `tomllib` and asserts every entry resolves to an installed distribution. Catches typos in the new manifest and catches a core dep silently dropped from the lock.
- `scripts/check_imports.py` (a CI step, deliberately NOT pytest — importing `mister_fritz` / `document_engine` for real inside the pytest process would pollute `sys.modules` and wreck the carefully-ordered stubs in `tests/test_document_engine.py:43-58` and `tests/test_bot_commands.py:24-27`). One subprocess per module; 16 core modules must import, 5 extra-gated modules report skip.
- Optionally add a case to `tests/test_packaging.py` asserting `pyproject.toml`'s `[tool.setuptools].py-modules` covers every top-level `.py` at the repo root — a cheap guard against a new module being added and silently left out of packaging.

### Existing tests affected

- `tests/test_document_engine.py` — MOST AT RISK. Line 60 imports the REAL `document_engine`, so every module-level import in that file must be satisfiable by core deps: `msoffcrypto` (:13), `openpyxl` (:14), `watchdog` (:20-21), `langchain_ollama` (:23), `pydantic` (:24), `langchain_community.document_loaders` (:27-33), `langchain_text_splitters` (:35), `langchain_chroma` (:36), `langchain_core` (:37-39), `langgraph` (:40). Its stubs at :43 (`easyocr`), :45 (`fitz`), :49-52 (`spacy`/`thinc`/`transformers`) become genuinely load-bearing under a torch-free CI. No code change needed, but the comment block at :40-45 must be reworded (see changeSites). If this file goes red, the extras split is wrong.
- `tests/test_stt.py` — line 63 (and 67, 82, 104, 117, 135, 151, 164, 177, 188) does `import stt`, and `stt.py:7-8` imports `pydub` at module level. This is the sole reason `pydub` must be a CORE dependency and not sit in `[voice]`. `faster_whisper` is deferred to `stt.py:38` inside `_get_model()`, so it stays in `[voice]` safely.
- `tests/test_agent_tools.py` — line 30 stubs `image_generator`; after Step 1 that stub is inert at import time. Lines 36-44 import the real `agent_tools` and `mister_fritz`, and this file does NOT stub `ddgs` or `bs4`, which is why `ddgs` (`agent_tools.py:15`) and `beautifulsoup4` (`agent_tools.py:14`) must stay core. Expect green with no edits; an ImportError here means a core dep was misclassified.
- `tests/test_mister_fritz.py` — lines 23-25 stub `ddgs`/`image_generator`/`document_engine`; line 27 imports `langchain_core.messages`; lines 29-30 import the real `mister_fritz`, which pulls `langchain`, `langgraph`, `langgraph.checkpoint.sqlite`, `langchain_ollama`. Keeps those core.
- `tests/test_admin_panel.py` — line 28 `from starlette.testclient import TestClient` needs `httpx` installed; lines 47-54 reload `fritz_utils`/`workspace_store`/`privacy`/`admin_panel`, and `admin_panel.py:26-36` needs `uvicorn`, `starlette`, `Jinja2` (via `Jinja2Templates` at :33) and `markdown` (:35). All core.
- `tests/test_bot_commands.py` (stubs `tts` at :27) and `tests/test_discord_commands.py` (stubs `tts` at :27, `image_generator` at :31) — these are the ONLY reason `bot_commands`/`main_discord` are testable without the voice stack, since `bot_commands.py:26` and `main_discord.py:28` still do a module-level `from tts import TTSEngine`. No edits needed; just be aware the stubs are now mandatory rather than defensive.
- `tests/test_setup_wizard.py:20` — `_spec.loader.exec_module(fritz_setup)` executes `scripts/setup.py` at collection time. The Step 10 string edits at `scripts/setup.py:339,377` must not introduce a non-stdlib import. Verified: no test in this file asserts on the pip-install strings, so the wording change is safe.
- `tests/test_scheduler.py:7-8` — imports `apscheduler.triggers.cron` / `.interval` directly, pinning `APScheduler` to core.
- Coverage gate: `--cov=.` at `.github/workflows/ci.yml:35-37`. `image_generator.py`, `tts.py`, `main_telegram.py` and `browser_tools.py` are already never imported by the suite, so their coverage is already ~0 and the split shouldn't move the number — but re-check the total against the Step 0 baseline before merging, because `--cov-fail-under=60` has no headroom to spare.

### Manual verification

- `python -c "import fitz; print(fitz.__file__); print(fitz.open)"` in a freshly-built venv -> must print the PyMuPDF shim path and a builtin, not AttributeError. Re-run this after any `pip uninstall fitz`.
- `ls site-packages/fitz/` -> must contain only PyMuPDF's files. `frontend.py` and `tools/` are neuroimaging leftovers and must be gone.
- Ingest a scanned (image-only) PDF end-to-end through `document_engine` and confirm text comes out — the only path that exercises `fitz.open()` at `document_engine.py:120` and `fitz.Matrix(2, 2)` at :126 together with `easyocr`.
- Ingest a real `.xlsx` — proves `unstructured[xlsx]` + `pandas` survived the purge (the loader imports pandas lazily, so an import smoke test will NOT catch this).
- Ingest a real `.docx` and a text-bearing `.pdf` — proves `unstructured`/`python-docx` and `pypdf` survived.
- Start the bot with only `pip install -e "."` (no extras) and observe the failure mode. It will currently fail at `main_discord.py:28` -> `from tts import TTSEngine`; decide whether that's acceptable for this item (see openQuestions) — the fix is deferred, but the failure should at least be legible.
- Boot the admin panel and hit `/health` — confirms `starlette`/`uvicorn`/`Jinja2`/`Markdown` are intact and that dropping the uvicorn[standard] extras (`httptools`, `watchfiles`, `websockets`) didn't break anything. Then open `/chat` and send a message to confirm SSE streaming still works without `websockets` installed.
- Time the CI run and compare against the 3m59s-4m43s baseline. If the install step doesn't drop dramatically, check whether `chromadb`->`onnxruntime` or `unstructured` is the real cost centre rather than torch — that would change the story and should be reported honestly rather than papered over.
- `docker build .` and run the release smoke check (`.github/workflows/release.yml:65-74`) against the new image; confirm image size dropped and the `faster_whisper` pre-bake at `Dockerfile:53` still succeeds (read the log — it's `|| true`).

## Risks

- **Uninstalling `fitz` breaks PyMuPDF.** `fitz-0.0.1.dev2.dist-info/RECORD` claims `fitz/__init__.py`, and that file on disk is currently PyMuPDF's shim. `pip uninstall fitz` deletes it. Detection: `python -c "import fitz"` raises ModuleNotFoundError immediately after. Mitigation: Step 5's force-reinstall, or rebuild the venv from scratch. Anyone who skips this will conclude the change broke PDF OCR.
- **A dependency I classified as an extra is actually core.** The riskiest cases are lazy imports inside `langchain_community` loaders and `chromadb`'s optional backends, which no static analysis catches. Detection: Step 6's core-only venv running the full suite, plus Step 7's four real-document ingests. `unstructured`/`pypdf`/`pandas` are the specific trap — I proved they aren't needed at *import* time, which makes them easy to drop by mistake and only fail at first ingest.
- **Floating versions in `[project]`.** CI installs `-e ".[dev]"` with floors, not pins, so a bad upstream release can turn CI red on an unrelated PR. Detection: a CI failure with no corresponding code change. Mitigations in order of cost: (a) accept it and pin the offender reactively, (b) add a `constraints-ci.txt` generated from `requirements.txt`, (c) adopt pip-tools. Start with (a); escalate only if it bites twice.
- **CI and production now install different sets.** A `[voice]`/`[image]`-only breakage sails past the core CI job. Mitigation: the weekly `full-deps` job. Residual risk: up to a week of latency before drift surfaces. If that's unacceptable, run `full-deps` on every PR that touches `pyproject.toml` or `requirements.txt` via a `paths:` filter — cheap and targeted.
- **`pip install -e .` adds a second import-path mechanism.** Today imports work purely via `pythonpath = ["."]` at `pyproject.toml:43`. If setuptools' editable finder and the pytest pythonpath disagree (stale `__editable__` finder after a rename, say), you get confusing double-registration. Detection: `python -c "import fritz_utils; print(fritz_utils.__file__)"` must print the repo path, not a site-packages copy. Mitigation: keep `pythonpath = ["."]` as-is; harmless alongside an editable install of the same directory.
- **Losing `--cov-fail-under=60` headroom.** Any module that stops being imported drops the total. I checked and expect no change (the affected modules already sit at ~0), but the gate has no slack. Detection: Step 0 baseline vs. Step 8 CI run.
- **Dropping `tesseract-ocr` from the Dockerfile assumes nothing shells out to the binary.** I grepped for `tesseract` across all tracked files and found zero hits, so this is safe — but if someone later adds a `pytesseract` fallback they'll get a confusing runtime error. The CHANGELOG entry records why it left.
- **`pathlib==1.0.1` removal is safe but pip has historically choked on it.** If `pip uninstall pathlib` errors, delete `site-packages/pathlib.py` and `pathlib-1.0.1.dist-info/` by hand. Verify afterwards that `python -c "import pathlib; print(pathlib.__file__)"` still points at the stdlib.
- **Windows-vs-Linux freeze skew.** A `requirements.txt` regenerated on Windows will carry `pypiwin32`/`comtypes`-shaped entries and miss Linux-only wheels (`nvidia-*` under a CUDA torch, `triton`). The current file already has this problem (`comtypes` at :30, `pypiwin32` at :198, both with `sys_platform == "win32"` markers). Best regenerated inside the Docker builder stage, or at minimum reviewed for platform markers before commit.

## Rollback
No feature flag is warranted — this is packaging and CI, not runtime behaviour, and a flag would add more risk than it removes. Land it as the ordered sequence of small commits in `steps` so any single one reverts cleanly. (1) agent_tools.py lazy import (2 lines) — reverting restores the module-level `import image_generator` at line 20 with zero other consequences. (2) pyproject.toml [project] table — purely additive; reverting leaves [tool.ruff] and [tool.pytest.ini_options] exactly as they are today, and nothing at runtime reads it. (3) scripts/check_imports.py + tests/test_packaging.py — new files; delete to revert. (4) requirements.txt regeneration — `git checkout <base> -- requirements.txt` restores the old freeze verbatim. IMPORTANT: reverting the file does NOT repair a venv you already purged; after reverting you must `pip install -r requirements.txt` again, which will reinstall fitz==0.0.1.dev2 and re-open the shadowing race. Note this in the PR body. (5) .github/workflows/ci.yml — reverting restores `pip install -r requirements.txt`; CI goes back to ~4m and full deps. (6) Dockerfile — reverting restores tesseract-ocr and the single-file COPY, independent of everything else. If a production incident is traced to a missing package, the fastest safe fix is `pip install <package>` in the running environment plus a one-line addition to [project].dependencies — not a wholesale revert to the polluted freeze.

## Open questions for you to decide

- **Do `bot_commands.py:26` / `main_discord.py:28` get the same lazy-import treatment as `agent_tools.py:20`?** Both do `from tts import TTSEngine`, and `tts.py:5,8` import `torch` and `TTS.api` at module level — so with `[voice]` optional, `python main_discord.py` cannot even start without it. Making it lazy needs `TYPE_CHECKING` + a string annotation at `bot_commands.py:105` (`sayer: "TTSEngine"`) and a function-local import at `main_discord.py:112` (`sayer = await loop.run_in_executor(None, TTSEngine)`). I DEFERRED this: four edits touching annotation evaluation inside a discord.py Cog, which deserves its own commit and test pass. Practical consequence: `[voice]` is effectively mandatory for the Discord bot today, so `requirements.txt` (the production lock) must keep it. Owner decides whether to fold it in now or file it separately.
- **Should `requirements.txt` become a true generated lock (pip-tools / `uv pip compile`) or stay a hand-reviewed `pip freeze`?** I planned the freeze approach because it adds no new tooling and the Dockerfile already consumes the file unchanged. `pip-compile` would give reproducible resolution, hashes, and a real pyproject->lock pipeline. It's the right long-term answer and explicitly deferred — decide whether the extra dev dependency is worth it before this pattern calcifies.
- **What `version` goes in `[project]`?** `fritz_utils.__version__` is already documented in the CHANGELOG as "the single source of truth for the bot version" and is consumed at `admin_panel.py:51`. Options: (a) hardcode `version = "0.0.0"` and treat pyproject purely as a dependency manifest, (b) `dynamic = ["version"]` with `[tool.setuptools.dynamic] version = {attr = "fritz_utils.__version__"}` — which makes the build import `fritz_utils`, and `fritz_utils` reads env vars at import. I sketched (a) as the safe default; (b) is nicer but needs a check that `fritz_utils` imports cleanly with no env set.
- **Is `browser_tools.py` alive?** Nothing imports it — `get_browser_tools_description` at line 149 has no callers anywhere in the repo. I gave it a `[browser]` extra so `playwright` is at least declared, but the honest alternative is deleting the module. Owner's call; out of scope here either way.
- **Should `pandas` be a direct core dependency or expressed as `unstructured[xlsx,docx]`?** The extras form is more truthful about *why* it's there, but couples us to `unstructured`'s extras naming, which has churned across releases. I chose the direct pin plus an explanatory comment. Low stakes, but worth a decision so the next person doesn't 'clean it up'.
- **Is torch actually the CI bottleneck?** I could NOT verify this statically — I read the workflow and the run timings (3m59s-4m43s), not the per-step logs. `chromadb`->`onnxruntime` and `unstructured` are also chunky. The experiment that settles it: `gh run view <id> --log` on a recent CI run and read the per-step duration of "Install dependencies", or push a branch with `pip install -e ".[dev]"` and diff wall time. Do this before writing a number into the CHANGELOG.
- **Does removing the uvicorn[standard] extras (`httptools`, `watchfiles`, `websockets`) affect SSE throughput?** `admin_panel.py:840-846` uses plain `uvicorn.Config`, so uvicorn falls back to `h11` instead of `httptools`. Functionally identical, marginally slower parsing. Cannot be settled statically — if web-chat streaming latency matters (see the `latency-tax` item), benchmark `/chat/stream` before and after rather than guessing.
