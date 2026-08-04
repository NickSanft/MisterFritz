"""Shared test-environment setup.

pytest imports this file before collecting any test module, which is the only
moment early enough to matter here: the test modules import mister_fritz /
document_engine / fritz_utils at module scope (i.e. during collection, long
before any fixture body runs), and those imports have CWD-relative side
effects — fritz_utils reads its env-var config and writes .chat_cookie_secret,
mister_fritz opens SQLite databases and spawns a thread that writes
mister_fritz_diagram.png, document_engine writes its own diagram PNG and roots
a Chroma store at CHROMA_DB_PATH.

Hence everything below is a TOP-LEVEL STATEMENT, not a fixture. A fixture
would run far too late and the artifacts would land in the working tree.
"""
import os
import sys
import tempfile
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# 1. Point every CWD-relative artifact at a per-session temp directory.
#    setdefault (not assignment) so an outer environment / CI can still
#    override deliberately; individual test modules that need their own paths
#    (e.g. test_admin_panel.py) keep patching on top of these as before.
# ---------------------------------------------------------------------------

_SANDBOX = tempfile.mkdtemp(prefix="fritz-tests-")

os.environ.setdefault("DB_NAME", os.path.join(_SANDBOX, "fritz.db"))
os.environ.setdefault("CHAT_DB_NAME", os.path.join(_SANDBOX, "chat_history.db"))
os.environ.setdefault("SCHEDULE_DB", os.path.join(_SANDBOX, "schedule.db"))
os.environ.setdefault("CHROMA_DB_PATH", os.path.join(_SANDBOX, "chroma_store"))
os.environ.setdefault("WORKSPACES_ROOT", os.path.join(_SANDBOX, "workspaces"))
os.environ.setdefault("DOC_FOLDER", os.path.join(_SANDBOX, "input"))
# Stops fritz_utils writing .chat_cookie_secret into the CWD at import time.
os.environ.setdefault("CHAT_COOKIE_SECRET", "conftest-test-cookie-secret")
# mister_fritz and document_engine write graph-diagram PNGs to the CWD at
# import time; both honour this switch. Without it every test run dirties the
# two tracked *_diagram.png files at the repo root.
os.environ.setdefault("FRITZ_WRITE_DIAGRAMS", "0")

# ---------------------------------------------------------------------------
# 2. Heavy-module stubs, hoisted from the per-file preambles that used to be
#    duplicated across six test modules. They must exist before any test
#    module imports mister_fritz / agent_tools / bot_commands / main_discord.
#
#    ddgs performs network-capable init; image_generator imports the
#    torch/diffusers stack; tts imports torch + Coqui; document_engine drags
#    in the whole langchain document pipeline. A test that needs the real
#    module (tests/test_document_engine.py) evicts the MagicMock stub before
#    importing — see the isinstance check there.
# ---------------------------------------------------------------------------


def _ensure_mock(name: str) -> MagicMock:
    """Place a MagicMock in sys.modules unless the module is already present."""
    if name not in sys.modules:
        sys.modules[name] = MagicMock()
    return sys.modules[name]


_ensure_mock("ddgs")
_ensure_mock("image_generator")
_ensure_mock("document_engine")
_ensure_mock("tts")
