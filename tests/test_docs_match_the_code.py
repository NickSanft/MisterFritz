"""Every knob the code reads, and every command it registers, is documented.

CONTRIBUTING asks for a README row per setting, and the README had drifted
badly: 49 of the 77 environment variables the code reads had no row, `.env.example`
was itself missing four, and the whole direct-message relay — six PRs, seven
knobs, a slash-command group and a context-menu entry — appeared nowhere. A
feature nobody can discover is close to one that does not exist.

Documentation rots quietly, so this checks it rather than asking anybody to
remember. The discovery rule is deliberately blunt: any SCREAMING_CASE string
literal in a call's first argument position, anywhere in a top-level module.
That catches `os.getenv`, `os.environ[...]` and every private helper that wraps
them — `_env_or_json`, `_at_least_one`, `_positive_int_env` — without this file
having to know their names, which is exactly how an earlier audit of mine
missed two.

The cost is a handful of false positives, listed in NOT_SETTINGS with reasons.
"""
import ast
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCREAMING = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")

# SCREAMING_CASE first arguments that are not settings.
NOT_SETTINGS = {
    "COMMIT": "a SQL statement passed to sqlite3",
    "ROLLBACK": "likewise",
    "PATH": "the OS variable the exec scrub always forces through, not a knob",
    "RGB": "a PIL image mode",
}

# Documented, but read by a dependency rather than by this code. Both are real
# settings an operator may need to set, so they get a README row anyway.
READ_BY_DEPENDENCIES = {
    "ANONYMIZED_TELEMETRY": "chromadb reads it from the environment itself",
    "AUTO_DOWNLOAD_NLTK": "unstructured reads it at import",
}


def _settings_the_code_reads() -> dict:
    found: dict[str, set] = {}
    for path in sorted(REPO.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Call) and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    name = first.value
            elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                    and isinstance(node.slice.value, str) \
                    and "environ" in ast.unparse(node.value):
                name = node.slice.value
            if name and SCREAMING.match(name) and name not in NOT_SETTINGS:
                found.setdefault(name, set()).add(path.name)
    return found


def _documented_in_readme() -> set:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    return set(re.findall(r"^\|\s*`([A-Z][A-Z0-9_]*)`", readme, re.MULTILINE))


def _in_env_example() -> set:
    text = (REPO / ".env.example").read_text(encoding="utf-8")
    return set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", text, re.MULTILINE))


class TestEverySettingIsDocumented(unittest.TestCase):
    def test_the_discovery_rule_still_finds_things(self):
        """So a refactor that moves every read behind something this rule
        cannot see fails here rather than silently passing."""
        found = _settings_the_code_reads()
        self.assertGreater(len(found), 50, sorted(found))
        for expected in ("DISCORD_BOT_TOKEN", "OLLAMA_HOST", "RELAY_ENABLED",
                         "AUDIT_LOG_MAX_BYTES", "LOG_LEVEL"):
            with self.subTest(setting=expected):
                self.assertIn(expected, found)

    def test_every_setting_has_a_readme_row(self):
        documented = _documented_in_readme()
        for name, modules in sorted(_settings_the_code_reads().items()):
            with self.subTest(setting=name, read_in=",".join(sorted(modules))):
                self.assertIn(
                    name, documented,
                    f"{name} is read by {sorted(modules)} and has no README row. "
                    "CONTRIBUTING asks for one per setting; if it is not a "
                    "setting, add it to NOT_SETTINGS with the reason.")

    def test_every_setting_is_in_env_example(self):
        """The file people copy to .env. A knob documented only in the README
        is one nobody discovers while editing their config."""
        in_example = _in_env_example()
        for name, modules in sorted(_settings_the_code_reads().items()):
            with self.subTest(setting=name):
                self.assertIn(name, in_example,
                              f"{name} is read by {sorted(modules)} and is not in "
                              ".env.example")

    def test_the_readme_documents_nothing_imaginary(self):
        """The other direction: a row for a setting nothing reads is worse than
        no row, because somebody will set it and expect an effect."""
        read = set(_settings_the_code_reads())
        for name in sorted(_documented_in_readme()):
            with self.subTest(setting=name):
                self.assertTrue(
                    name in read or name in READ_BY_DEPENDENCIES,
                    f"{name} has a README row but nothing reads it. If a "
                    "dependency does, say so in READ_BY_DEPENDENCIES.")


class TestEveryCommandIsDocumented(unittest.TestCase):
    """`/tell`, the whole `/relay` group and all of `/schedule` were missing
    from the command table. `/schedule` at least appeared in prose; the relay
    appeared nowhere at all."""

    @staticmethod
    def _registered() -> set:
        """Top-level command and group names, read from the decorators."""
        source = (REPO / "bot_commands.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = set()
        for node in ast.walk(tree):
            # app_commands.Group(name="relay", ...) at class scope
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "Group":
                for keyword in node.keywords:
                    if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                        names.add(keyword.value.value)
            # @app_commands.command(name="tell", ...) on a method
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    if not isinstance(decorator, ast.Call):
                        continue
                    attribute = getattr(decorator.func, "attr", None)
                    if attribute != "command":
                        continue
                    owner = getattr(decorator.func.value, "id", None)
                    if owner not in (None, "app_commands"):
                        continue        # a subcommand of a group
                    for keyword in decorator.keywords:
                        if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                            names.add(keyword.value.value)
        return names

    def test_the_reader_finds_the_commands(self):
        registered = self._registered()
        self.assertGreater(len(registered), 10, sorted(registered))
        for expected in ("tell", "relay", "schedule", "forget", "workspace"):
            with self.subTest(command=expected):
                self.assertIn(expected, registered)

    def test_each_one_appears_in_the_readme(self):
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        mentioned = set(re.findall(r"`/([a-z_-]+)", readme))
        for name in sorted(self._registered()):
            with self.subTest(command=name):
                self.assertIn(name, mentioned,
                              f"/{name} is registered and never mentioned in the "
                              "README")

    def test_the_command_table_lists_them_rather_than_only_the_prose(self):
        """/schedule was in the troubleshooting text and the ROOT_USER row, but
        not in the table somebody reads to find out what the bot does."""
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        table = readme[readme.index("## Discord Slash Commands"):
                       readme.index("## System Requirements")]
        listed = set(re.findall(r"`/([a-z_-]+)", table))
        for name in sorted(self._registered()):
            with self.subTest(command=name):
                self.assertIn(name, listed)


class TestTheRelayIsDiscoverable(unittest.TestCase):
    """Six PRs shipped it and the README did not contain the word. The point of
    these is not word-counting: each names something an operator cannot work
    out from the code without reading all of it."""

    @classmethod
    def setUpClass(cls):
        cls.readme = (REPO / "README.md").read_text(encoding="utf-8")

    def _section(self) -> str:
        """The feature section, sliced on WHOLE headings.

        Not `index("## Direct-message relay")`: that also matches inside the
        `### Relay settings` sub-heading of the configuration table, which made
        the slice start there and run through Identity. Three assertions below
        then passed against configuration rows rather than the section they
        were about — found by mutating the section away and watching them hold.
        """
        start = re.search(r"^## Direct-message relay$", self.readme, re.MULTILINE)
        self.assertIsNotNone(start, "the relay section heading is gone")
        rest = self.readme[start.end():]
        end = re.search(r"^## ", rest, re.MULTILINE)
        self.assertIsNotNone(end, "no heading follows the relay section")
        return rest[:end.start()]

    def test_it_has_a_section_of_its_own(self):
        self.assertRegex(self.readme, r"(?m)^## Direct-message relay$")

    def test_the_features_table_points_at_it(self):
        features = self.readme[self.readme.index("## Features"):
                               self.readme.index("## Discord Slash Commands")]
        self.assertIn("/tell", features)
        self.assertIn("#direct-message-relay", features)

    def test_it_says_that_it_is_off_by_default_and_why(self):
        section = self._section()
        self.assertIn("off by default", section.lower())
        self.assertIn("RELAY_ENABLED", section)

    def test_it_explains_the_recipients_side(self):
        """The half a sender cannot discover: blocks, and that they outlast
        /forget all."""
        section = self._section()
        for claim in ("/relay block", "/forget all", "forget-blocks",
                      # Each of these is a specific promise the code makes and
                      # a sender cannot infer: that a block is resolved before
                      # anything is sent, that /tell answers identically either
                      # way so it cannot probe for one, and that deleting your
                      # own data does not re-arm somebody you refused.
                      "before anything is sent", "cannot be used to detect",
                      "does not re-arm"):
            with self.subTest(claim=claim):
                self.assertIn(claim, section)

    def test_it_names_the_retention_trap(self):
        """Retention shorter than the reply window leaves a message answerable
        after its body is gone, which validate_config warns about."""
        section = self._section()
        self.assertIn("RELAY_RETENTION_DAYS", section)
        self.assertIn("RELAY_REPLY_WINDOW_MIN", section)
        # Naming both is not enough: the trap is the RELATIONSHIP between them,
        # and that validate_config says so when they are inverted.
        self.assertIn("longer than the window", section)
        self.assertIn("validate_config", section)

    def test_the_master_switch_no_longer_claims_the_feature_is_unfinished(self):
        """.env.example said reply routing and /relay block "have not landed
        yet". Both shipped."""
        env = (REPO / ".env.example").read_text(encoding="utf-8")
        self.assertNotIn("have not landed yet", env)


if __name__ == "__main__":                    # pragma: no cover
    unittest.main()
