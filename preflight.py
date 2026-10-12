"""The boot the entrypoint performs, minus Discord.

Reached only through `python main_discord.py --preflight`, which is the image's
real ENTRYPOINT with one argument. That indirection is the point: what CI has
to prove is that THIS ENTRYPOINT works in THIS image, and a second entrypoint
would prove something about a program nobody ships.

Until now nothing built the image outside a tag push, and the tag push had
never once succeeded — four releases, four failures, no image ever pushed. The
only thing that ever ran against the image was a smoke test that asked for
`python local-sim/mock_fritz.py`, which an exec-form ENTRYPOINT appends rather
than replaces, so it would have run `python main_discord.py python
local-sim/mock_fritz.py` and main_discord reads no arguments at all. The
"smoke test" shared nothing with the application except metric names.

By the time any line below runs, the entire import graph of a real boot has
already loaded: main_discord imports it at module scope, which reaches 22 of
this repo's modules including the agent, the cogs, the document engine and the
admin panel. So there is no "did the imports work" check to write here —
REACHING THIS MODULE IS THAT CHECK. What is left is the handful of things the
import graph loads but never exercises, and the two modules it never loads.

Nothing here touches Discord, Ollama, Chroma or any network. `import
main_discord` was measured to open zero outbound sockets and preflight must not
be what changes that.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import traceback
import urllib.error
import urllib.request

# scripts/check_imports.py's CORE list, minus everything `import main_discord`
# already pulls in. These two are the whole remainder — and the entrypoint
# never runs them either, so a migration broken by a dependency bump would ship
# and stay silent until somebody needed it.
NEVER_REACHED_BY_A_BOOT = ("migrate_db", "migrate_identity")

# A sentence that needs both baked corpora: punkt_tab to split it,
# averaged_perceptron_tagger_eng to tag it.
CORPUS_PROBE = "The butler answered. He did not elaborate."


def _connectable(host: str) -> str:
    """The address to CONNECT to, given the one the server BOUND to.

    The image sets METRICS_HOST=0.0.0.0, and has to: Prometheus scrapes across
    the compose network and the kubelet probes /health on the pod IP, and
    neither can see a listener on the container's loopback. But 0.0.0.0 is not
    a connectable address — on Windows it is WinError 10049 — and relying on
    Linux treating it as localhost would make this check platform-dependent.
    """
    if host in ("", "0.0.0.0"):
        return "127.0.0.1"
    if host in ("::", "::0"):
        return "[::1]"
    return host


def _get(host: str, port: int, path: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}{path}", timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:        # a 503 from /metrics lands here
        return error.code, error.read()


def check_the_modules_a_boot_never_loads() -> str:
    for name in NEVER_REACHED_BY_A_BOOT:
        importlib.import_module(name)
    return f"imported {', '.join(NEVER_REACHED_BY_A_BOOT)}"


def check_the_configuration_is_usable() -> str:
    """The first thing the real entrypoint does.

    .dockerignore excludes config.json, so inside the image the environment is
    the only source: this failing means the container was started without
    DISCORD_BOT_TOKEN or ROOT_USER, which is also how it would fail at 3am.
    """
    from fritz_utils import __version__, validate_config
    validate_config()
    return f"validate_config passed, version {__version__}"


def check_the_command_surface_builds() -> str:
    """Every slash command, group and context menu, constructed and put on a
    tree — which is what `on_ready` does and nothing before READY does.

    This is where a discord.py bump lands. The unit tests build this cog too,
    but against the discord.py the floors resolved; here it is built against
    the one the lock pinned and the image installed.
    """
    import main_discord
    from bot_commands import FritzCommands

    async def _build():
        client = main_discord.client
        # Idempotent, because add_cog raises ClientException on a duplicate
        # name and this must be runnable twice in one process — the test suite
        # does exactly that. In a container it is a no-op: nothing has
        # registered anything yet.
        if client.get_cog("FritzCommands") is not None:
            await client.remove_cog("FritzCommands")
        # sayer=None is the documented degraded state (/voice reports itself
        # unavailable); schedule_manager=None keeps APScheduler stopped, since
        # starting a scheduler in a preflight would leave jobs behind.
        await client.add_cog(FritzCommands(client, None, None))
        return sorted(command.name for command in client.tree.get_commands())

    names = asyncio.run(_build())
    if len(names) < 8:
        raise AssertionError(f"only {len(names)} commands on the tree: {names}")
    return f"{len(names)} commands: {', '.join(names)}"


def check_the_document_corpora_are_present() -> str:
    """The one thing only an in-image run can prove.

    `unstructured` fetches two nltk corpora at IMPORT time unless
    AUTO_DOWNLOAD_NLTK is false, so the Dockerfile bakes them and sets the
    switch — deliberately without `|| true`, because with the switch off and
    the files missing the first .docx raises LookupError from inside a loader.
    Those are two independent build steps and nothing checked they agree.
    Tokenising one sentence does: it needs punkt_tab and
    averaged_perceptron_tagger_eng, and raises LookupError without them.
    """
    from unstructured.nlp.tokenize import pos_tag, sent_tokenize
    sentences = sent_tokenize(CORPUS_PROBE)
    tagged = pos_tag(CORPUS_PROBE)
    if len(sentences) != 2 or not tagged:
        raise AssertionError(f"tokenizer returned {sentences!r} / {tagged[:3]!r}")
    return f"punkt_tab split {len(sentences)}, tagger tagged {len(tagged)}"


def check_the_metrics_server_answers() -> str:
    """Started by the real entrypoint, scraped by Prometheus, and probed for
    liveness AND readiness by the kubelet — so an image where this does not
    come up is an image Kubernetes will restart forever.
    """
    import observability
    if not observability._PROMETHEUS_AVAILABLE:
        raise AssertionError(
            "prometheus_client did not import, so /metrics serves 503. It is "
            "pinned in requirements.txt and declared core in pyproject, so "
            "this means the install is broken, not that an extra is missing.")

    port = observability.start_metrics_server()
    # Read the same way start_metrics_server reads it, because it takes the
    # value from the environment rather than exposing it.
    host = _connectable(os.getenv("METRICS_HOST", "127.0.0.1"))

    status, body = _get(host, port, "/health")
    if status != 200:
        raise AssertionError(f"/health answered {status}")
    health = json.loads(body)
    if health.get("status") != "ok" or "uptime_sec" not in health:
        raise AssertionError(f"/health said {health!r}")

    status, body = _get(host, port, "/metrics")
    if status != 200:
        raise AssertionError(f"/metrics answered {status}")
    # Not just "some text came back": these are the names the Grafana
    # dashboards and the canary analysis template query by.
    for metric in (b"misterfritz_discord_messages_total", b"misterfritz_errors_total"):
        if metric not in body:
            raise AssertionError(f"/metrics does not expose {metric.decode()}")

    status, _ = _get(host, port, "/nope")
    if status != 404:
        raise AssertionError(f"an unknown path answered {status}, not 404")

    return f"{host}:{port} /health ok, /metrics {len(body)} bytes, 404 works"


# Ordered: cheapest and most structural first, so a failure early in the list
# is the one a reader sees rather than the first timeout.
CHECKS = (
    ("modules a boot never loads", check_the_modules_a_boot_never_loads),
    ("configuration", check_the_configuration_is_usable),
    ("command surface", check_the_command_surface_builds),
    ("document corpora", check_the_document_corpora_are_present),
    ("metrics server", check_the_metrics_server_answers),
)


def main(checks=CHECKS, out=print) -> int:
    """Run every check, report each one, and exit non-zero if any failed.

    Every check runs even after one fails: a CI log that says only "the first
    thing broke" costs another twenty-minute image build to learn the rest.
    """
    failures = []
    out(f"preflight: {len(checks)} checks")
    for name, check in checks:
        try:
            out(f"  ok    {name}: {check()}")
        except Exception as error:
            failures.append(name)
            out(f"  FAIL  {name}: {type(error).__name__}: {error}")
            out("        " + traceback.format_exc().strip().replace("\n", "\n        "))
    if failures:
        out(f"preflight FAILED: {', '.join(failures)}")
        return 1
    out("preflight passed")
    return 0
