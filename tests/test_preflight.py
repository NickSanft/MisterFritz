"""The boot check the image's entrypoint runs, and the arguments it refuses.

Nothing ever booted this image. The release workflow's smoke test asked for
`docker run <image> python local-sim/mock_fritz.py`, and an exec-form
ENTRYPOINT appends what `docker run` is given rather than replacing it — so the
real command was `python main_discord.py python local-sim/mock_fritz.py`, and
main_discord read no arguments at all. The step booted the actual bot, failed
the Discord login on a placeholder token, and died; the curl that followed
could only ever fail. Nobody could tell that from reading the step.

So two things are tested here: that the entrypoint now refuses arguments it
does not understand, and that `--preflight` proves the things a container boot
would otherwise discover in production.
"""
import ast
import json
import os
import socket
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import preflight

REPO = Path(__file__).resolve().parents[1]


def _os_environ() -> dict:
    """The parent environment, so a subprocess still finds PATH and HOME."""
    return dict(os.environ)


def _corpora_present() -> bool:
    """Whether the two corpora the Dockerfile bakes are findable here.

    They are an image artefact, not a dependency: `pip install` does not bring
    them, and ci.yml's `core + dev` job runs with AUTO_DOWNLOAD_NLTK=false and
    no download step. So a check that needs them is a check about the image.
    """
    try:
        import nltk
        nltk.data.find("tokenizers/punkt_tab")
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except Exception:
        return False
    return True


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestWhatTheEntrypointDoesWithArguments(unittest.TestCase):
    """The failure this whole file exists for: arguments that vanish."""

    # Long enough for a cold import of the whole graph (~8s warm here), short
    # enough that "it booted the bot" fails instead of hanging. discord.py
    # retries a failed connection forever, so without a bound this test caught
    # a mutation by timing out after five minutes rather than by failing.
    BOOT_TIMEOUT = 90

    def _run(self, *args):
        try:
            return subprocess.run(
                [sys.executable, "main_discord.py", *args],
                cwd=REPO, capture_output=True, text=True,
                timeout=self.BOOT_TIMEOUT,
                env={**_os_environ(), "DISCORD_BOT_TOKEN": "test-placeholder",
                     "ROOT_USER": "discord-000000000000000000",
                     "AUTO_DOWNLOAD_NLTK": "false"})
        except subprocess.TimeoutExpired:
            self.fail(f"main_discord.py {args} did not exit within "
                      f"{self.BOOT_TIMEOUT}s, which is what booting the bot "
                      "looks like: discord.py retries a connection forever.")

    def test_an_argument_it_does_not_understand_is_an_error(self):
        """Verbatim the command the release smoke test used. It exited 0 and
        booted the bot; it must now exit non-zero and say why."""
        result = self._run("python", "local-sim/mock_fritz.py")
        self.assertEqual(result.returncode, 1)
        self.assertIn("--preflight", result.stdout + result.stderr)
        self.assertIn("mock_fritz", result.stdout + result.stderr)

    def test_preflight_actually_reaches_preflight(self):
        """The dispatch itself, through a real process, because the AST test
        below only proves the source has the right shape — it held while
        `--preflight` was wired to a dead branch.

        It asserts the REPORT, not success: the corpora check needs the files
        the Dockerfile bakes, which a developer's checkout and the `core + dev`
        job do not have. Whether every check passes is what the image build's
        own boot step is for.
        """
        result = self._run("--preflight")
        self.assertIn("preflight:", result.stdout,
                      "--preflight did not reach preflight.main()")
        self.assertIn("ok    configuration", result.stdout)

    def test_the_bot_can_only_start_when_no_arguments_were_given(self):
        """Read from the source, because the alternative is a test that logs
        in to Discord. If `client.run` ever moves out from behind the argument
        check, the next smoke test to pass junk boots the bot again."""
        tree = ast.parse((REPO / "main_discord.py").read_text(encoding="utf-8"))
        [guard] = [node for node in tree.body
                   if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)]
        body = ast.unparse(guard)
        self.assertIn("sys.argv[1:]", body)
        # Every exit path before the boot, in order: --preflight, then refuse.
        self.assertLess(body.index("preflight.main()"), body.index("client.run"))
        self.assertLess(body.index("takes --preflight or nothing"), body.index("client.run"))


class TestPreflightCoversWhatABootWouldHit(unittest.TestCase):
    def test_the_modules_a_boot_never_loads_are_exactly_the_ones_it_lists(self):
        """preflight's coverage claim is that `import main_discord` reaches
        every core module except these. Derived here by walking the import
        graph statically from main_discord, so adding a module that nothing
        imports fails this rather than going unchecked for a year."""
        own = {path.stem for path in REPO.glob("*.py")}
        reached, frontier = set(), ["main_discord"]
        while frontier:
            name = frontier.pop()
            if name in reached or name not in own:
                continue
            reached.add(name)
            tree = ast.parse((REPO / f"{name}.py").read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    frontier += [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    frontier.append(node.module.split(".")[0])

        core = set(_check_imports_core())
        missed = sorted(core - reached)
        self.assertEqual(
            missed, sorted(preflight.NEVER_REACHED_BY_A_BOOT),
            "scripts/check_imports.py lists a core module that booting the "
            "entrypoint does not reach, and preflight does not import it "
            "either. Either add it to NEVER_REACHED_BY_A_BOOT or stop calling "
            "it core.")

    def test_every_check_is_in_the_run_list(self):
        """A check function nobody runs is worse than no check, because the
        name in the file reads like coverage."""
        defined = {name for name in dir(preflight) if name.startswith("check_")}
        wired = {check.__name__ for _label, check in preflight.CHECKS}
        self.assertEqual(defined, wired)

    def test_the_addresses_a_container_binds_are_turned_into_connectable_ones(self):
        """The image sets METRICS_HOST=0.0.0.0 because Prometheus and the
        kubelet cannot see a listener on the container's loopback. 0.0.0.0 is
        not an address you can connect TO."""
        for bound, expected in (("0.0.0.0", "127.0.0.1"), ("", "127.0.0.1"),
                                ("::", "[::1]"), ("::0", "[::1]"),
                                ("127.0.0.1", "127.0.0.1"), ("fritz", "fritz")):
            with self.subTest(bound=bound):
                self.assertEqual(preflight._connectable(bound), expected)


def _check_imports_core():
    """scripts/check_imports.py's CORE list, read without importing it."""
    tree = ast.parse((REPO / "scripts/check_imports.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "CORE":
            return [element.value for element in node.value.elts]
    raise AssertionError("CORE not found in scripts/check_imports.py")


class TestTheChecksThemselves(unittest.TestCase):
    def test_the_unreached_modules_import(self):
        detail = preflight.check_the_modules_a_boot_never_loads()
        for name in preflight.NEVER_REACHED_BY_A_BOOT:
            self.assertIn(name, detail)

    def test_the_configuration_check_runs_the_real_validator(self):
        with patch("fritz_utils.validate_config") as validate:
            detail = preflight.check_the_configuration_is_usable()
        validate.assert_called_once_with()
        self.assertIn("version", detail)

    def test_the_configuration_check_fails_when_the_config_does(self):
        with patch("fritz_utils.validate_config", side_effect=ValueError("no token")):
            with self.assertRaises(ValueError):
                preflight.check_the_configuration_is_usable()

    @unittest.skipUnless(_corpora_present(),
                         "the nltk corpora are not installed here — they are "
                         "baked by the Dockerfile, so this check belongs to "
                         "the image and runs in ci.yml's build-image job")
    def test_the_corpora_check_tokenises_for_real(self):
        """Offline: with the corpora present this needs no network, which is
        the whole point of the Dockerfile baking them.

        Skipped rather than faked when they are absent, following
        tests/test_packaging.py's `find_spec -> skipTest` idiom. Asserting the
        real tokenizer here without that guard is what turned CI red on the
        commit that added this file: this machine has the corpora and a
        `core + dev` runner does not."""
        detail = preflight.check_the_document_corpora_are_present()
        self.assertIn("punkt_tab", detail)

    def test_the_corpora_check_fails_when_the_tokenizer_misbehaves(self):
        """What a missing corpus looks like once nltk's own LookupError is out
        of the way: a tokenizer that returns the wrong shape. Both functions
        are patched so this needs no corpora anywhere."""
        with patch("unstructured.nlp.tokenize.sent_tokenize", return_value=["one"]), \
             patch("unstructured.nlp.tokenize.pos_tag", return_value=[("a", "DT")]):
            with self.assertRaises(AssertionError):
                preflight.check_the_document_corpora_are_present()

    def test_the_corpora_check_lets_a_missing_corpus_through_as_itself(self):
        """nltk raises LookupError with a long diagnostic naming the resource
        and where it looked. preflight must not swallow or reshape it: that
        message is how an operator finds out the bake did not happen."""
        with patch("unstructured.nlp.tokenize.sent_tokenize",
                   side_effect=LookupError("Resource 'punkt_tab' not found")):
            with self.assertRaises(LookupError) as caught:
                preflight.check_the_document_corpora_are_present()
        self.assertIn("punkt_tab", str(caught.exception))

    def test_the_command_surface_check_builds_every_command(self):
        detail = preflight.check_the_command_surface_builds()
        for expected in ("tell", "relay", "schedule", "forget", "workspace"):
            with self.subTest(command=expected):
                self.assertIn(expected, detail)

    def test_the_command_surface_check_is_runnable_twice(self):
        """add_cog raises ClientException on a duplicate cog name, so a check
        that is not idempotent fails the second time anything calls it."""
        preflight.check_the_command_surface_builds()
        preflight.check_the_command_surface_builds()

    def test_the_command_surface_check_notices_a_nearly_empty_tree(self):
        import main_discord
        with patch.object(main_discord.client.tree, "get_commands", return_value=[]):
            with self.assertRaises(AssertionError) as caught:
                preflight.check_the_command_surface_builds()
        self.assertIn("0 commands", str(caught.exception))

    def test_the_metrics_check_probes_a_real_server(self):
        port = _free_port()
        with patch.dict("os.environ", {"METRICS_PORT": str(port),
                                       "METRICS_HOST": "127.0.0.1"}):
            detail = preflight.check_the_metrics_server_answers()
        self.assertIn(str(port), detail)
        self.assertIn("/health ok", detail)

    def test_the_metrics_check_fails_when_prometheus_is_missing(self):
        """It is pinned in the lock and declared core, so its absence means a
        broken install rather than an unselected extra — and /metrics would
        serve 503 to a scraper that reports the bot as up."""
        import observability
        with patch.object(observability, "_PROMETHEUS_AVAILABLE", False):
            with self.assertRaises(AssertionError) as caught:
                preflight.check_the_metrics_server_answers()
        self.assertIn("pinned", str(caught.exception))

    def test_the_metrics_check_fails_on_a_health_payload_that_is_not_ok(self):
        with patch.object(preflight, "_get", return_value=(200, json.dumps(
                {"status": "degraded"}).encode())):
            with patch("observability.start_metrics_server", return_value=1):
                with self.assertRaises(AssertionError) as caught:
                    preflight.check_the_metrics_server_answers()
        self.assertIn("degraded", str(caught.exception))

    def test_the_metrics_check_fails_when_a_dashboard_metric_is_gone(self):
        """The names the Grafana dashboards and the canary analysis template
        query by. "Some text came back" is not the assertion worth making."""
        answers = {"/health": (200, b'{"status": "ok", "uptime_sec": 1.0}'),
                   "/metrics": (200, b"# nothing useful here\\n"),
                   "/nope": (404, b"")}
        with patch.object(preflight, "_get",
                          side_effect=lambda h, p, path: answers[path]):
            with patch("observability.start_metrics_server", return_value=1):
                with self.assertRaises(AssertionError) as caught:
                    preflight.check_the_metrics_server_answers()
        self.assertIn("misterfritz_discord_messages_total", str(caught.exception))


class TestTheReport(unittest.TestCase):
    """What a CI log has to contain for a twenty-minute image build to be
    worth having failed."""

    def test_every_check_runs_even_after_one_fails(self):
        ran, lines = [], []

        def _ok():
            ran.append("second")
            return "fine"

        def _boom():
            ran.append("first")
            raise RuntimeError("the apt layer is missing ffmpeg")

        code = preflight.main(checks=(("first", _boom), ("second", _ok)),
                              out=lines.append)
        self.assertEqual(code, 1)
        self.assertEqual(ran, ["first", "second"])

    def test_a_failure_names_the_check_the_error_and_the_traceback(self):
        lines = []

        def _boom():
            raise RuntimeError("the apt layer is missing ffmpeg")

        preflight.main(checks=(("apt", _boom),), out=lines.append)
        report = "\n".join(lines)
        self.assertIn("FAIL  apt", report)
        self.assertIn("RuntimeError", report)
        self.assertIn("missing ffmpeg", report)
        self.assertIn("preflight FAILED: apt", report)
        self.assertIn("test_preflight.py", report)      # the traceback

    def test_success_is_zero_and_says_so(self):
        lines = []
        code = preflight.main(checks=(("one", lambda: "detail here"),), out=lines.append)
        self.assertEqual(code, 0)
        self.assertIn("ok    one: detail here", "\n".join(lines))
        self.assertIn("preflight passed", "\n".join(lines))

    def test_the_real_check_list_is_not_empty(self):
        """So main() with the default list cannot pass by doing nothing."""
        self.assertGreaterEqual(len(preflight.CHECKS), 5)


if __name__ == "__main__":                    # pragma: no cover
    unittest.main()
