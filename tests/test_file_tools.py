"""
Tests for file_tools.py — all six tools, authorization, and path safety.

Tools are LangChain @tool-decorated functions; invoked via .invoke(input, config=config).
Authorization runs through fritz_utils.is_admin; we patch fritz_utils.ROOT_USER
in each test to control who counts as admin.
"""
import os
import tempfile
import unittest
from unittest.mock import patch

import fritz_utils
import file_tools  # noqa: F401  — needed so its module-level state is available for tests
from file_tools import (
    list_directory, read_file, write_file,
    edit_file, search_files, execute_command,
)

_ROOT = "_test_root_user_"


def _config(workspace: str, user: str = _ROOT) -> dict:
    return {
        "configurable": {},
        "metadata": {"user_id": user, "workspace_root": workspace},
    }


def _echo(text: str) -> str:
    """A portable `echo` for tests that need a command to actually run.

    `echo` is a cmd.exe builtin on Windows with no executable anywhere on PATH,
    so every test built on it passed on the Linux CI runner and failed on a
    Windows dev box. `python` is on the allowlist and on PATH on both.
    """
    return 'python -c "print(\'%s\')"' % text


def _echo_argv(text: str) -> list[str]:
    """The argv `_echo(text)` parses into — for audit-log assertions."""
    return ["python", "-c", "print('%s')" % text]


class TestAuthorization(unittest.TestCase):
    """Phase 7b: file tools are per-user-sandboxed, not admin-gated.
    Any user with workspace_root set in their config can use them.
    """
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_user_with_workspace_is_allowed(self):
        # Non-admin user with a workspace gets in.
        result = list_directory.invoke(
            {"path": "."},
            config=_config(self.tmp, user="regular_user"),
        )
        # Empty dir returns a "Directory '.' is empty." message, not an error.
        self.assertNotIn("Error", result)
        self.assertNotIn("permission", result.lower())

    def test_no_workspace_raises_permission_error(self):
        # Without workspace_root set, _authorize should reject.
        cfg = {"configurable": {}, "metadata": {"user_id": "anyone", "workspace_root": None}}
        with self.assertRaises(PermissionError):
            list_directory.invoke({"path": "."}, config=cfg)

    def test_missing_workspace_key_raises_permission_error(self):
        # Metadata exists but workspace_root key is absent.
        cfg = {"configurable": {}, "metadata": {"user_id": "anyone"}}
        with self.assertRaises(PermissionError):
            list_directory.invoke({"path": "."}, config=cfg)


class TestPathTraversal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_path_traversal_attempt_raises(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            with self.assertRaises(ValueError):
                file_tools._resolve_safe_path(self.tmp, "../../etc/passwd")

    def test_normal_subpath_is_allowed(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = file_tools._resolve_safe_path(self.tmp, "subdir/file.txt")
        self.assertTrue(result.startswith(self.tmp))


class TestListDirectory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        open(os.path.join(self.tmp, "hello.txt"), "w").close()
        os.makedirs(os.path.join(self.tmp, "mydir"), exist_ok=True)

    def test_lists_files_and_dirs(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = list_directory.invoke({"path": ".", "include_hidden": False},
                                           config=_config(self.tmp))
        self.assertIn("hello.txt", result)
        self.assertIn("mydir", result)

    def test_hidden_files_excluded_by_default(self):
        open(os.path.join(self.tmp, ".hidden"), "w").close()
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = list_directory.invoke({"path": ".", "include_hidden": False},
                                           config=_config(self.tmp))
        self.assertNotIn(".hidden", result)

    def test_hidden_files_included_when_flag_set(self):
        open(os.path.join(self.tmp, ".hidden"), "w").close()
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = list_directory.invoke({"path": ".", "include_hidden": True},
                                           config=_config(self.tmp))
        self.assertIn(".hidden", result)

    def test_nonexistent_path_returns_error_string(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = list_directory.invoke({"path": "no_such_dir"},
                                           config=_config(self.tmp))
        self.assertIn("Error", result)


class TestReadFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.file = os.path.join(self.tmp, "sample.txt")
        with open(self.file, "w") as f:
            f.writelines([f"Line {i}\n" for i in range(1, 21)])

    def test_reads_file_content(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = read_file.invoke({"path": "sample.txt"}, config=_config(self.tmp))
        self.assertIn("Line 1", result)
        self.assertIn("Line 20", result)

    def test_offset_and_limit(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = read_file.invoke({"path": "sample.txt", "offset": 5, "limit": 3},
                                      config=_config(self.tmp))
        self.assertIn("Line 6", result)
        self.assertNotIn("Line 5", result)
        self.assertNotIn("Line 9", result)

    def test_file_not_found_returns_error(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = read_file.invoke({"path": "ghost.txt"}, config=_config(self.tmp))
        self.assertIn("Error", result)

    def test_large_file_returns_error(self):
        big = os.path.join(self.tmp, "big.bin")
        with open(big, "wb") as f:
            f.write(b"x" * (file_tools.MAX_FILE_SIZE + 1))
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = read_file.invoke({"path": "big.bin"}, config=_config(self.tmp))
        self.assertIn("too large", result)


class TestWriteFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_creates_new_file(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = write_file.invoke({"path": "new.txt", "content": "hello"},
                                       config=_config(self.tmp))
        self.assertIn("Created", result)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "new.txt")))

    def test_overwrites_existing_file(self):
        target = os.path.join(self.tmp, "existing.txt")
        with open(target, "w") as f:
            f.write("old content")
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = write_file.invoke({"path": "existing.txt", "content": "new content"},
                                       config=_config(self.tmp))
        self.assertIn("Updated", result)
        self.assertEqual(open(target).read(), "new content")

    def test_creates_parent_directories(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            write_file.invoke({"path": "deep/nested/file.txt", "content": "data"},
                              config=_config(self.tmp))
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "deep", "nested", "file.txt")))


class TestEditFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.file = os.path.join(self.tmp, "edit_me.txt")
        with open(self.file, "w") as f:
            f.write("Hello world\nGoodbye world\n")

    def test_replaces_text(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = edit_file.invoke(
                {"path": "edit_me.txt", "old_text": "Hello world", "new_text": "Hi earth"},
                config=_config(self.tmp),
            )
        self.assertIn("Edited", result)
        content = open(self.file).read()
        self.assertIn("Hi earth", content)
        self.assertNotIn("Hello world", content)

    def test_text_not_found_returns_error(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = edit_file.invoke(
                {"path": "edit_me.txt", "old_text": "nonexistent text", "new_text": "x"},
                config=_config(self.tmp),
            )
        self.assertIn("Error", result)

    def test_file_not_found_returns_error(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = edit_file.invoke(
                {"path": "ghost.txt", "old_text": "x", "new_text": "y"},
                config=_config(self.tmp),
            )
        self.assertIn("Error", result)


class TestSearchFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        with open(os.path.join(self.tmp, "a.py"), "w") as f:
            f.write("def hello(): pass\n")
        with open(os.path.join(self.tmp, "b.py"), "w") as f:
            f.write("def world(): pass\n")

    def test_finds_matching_pattern(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = search_files.invoke(
                {"pattern": "def hello", "path": ".", "file_glob": "*.py"},
                config=_config(self.tmp),
            )
        self.assertIn("a.py", result)
        self.assertNotIn("b.py", result)

    def test_no_match_returns_no_match_message(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = search_files.invoke(
                {"pattern": "ZZZNOMATCH", "path": ".", "file_glob": "*.py"},
                config=_config(self.tmp),
            )
        self.assertIn("No matches", result)

    def test_invalid_regex_returns_error(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = search_files.invoke(
                {"pattern": "[invalid(", "path": "."},
                config=_config(self.tmp),
            )
        self.assertIn("Error", result)


class TestExecuteCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_runs_simple_command(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = execute_command.invoke(
                {"command": _echo("hello"), "timeout": 5},
                config=_config(self.tmp),
            )
        self.assertIn("hello", result)
        self.assertIn("Exit code: 0", result)

    def test_captures_stderr(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = execute_command.invoke(
                # Use python to write to stderr portably
                {"command": 'python -c "import sys; sys.stderr.write(\'err_msg\')"',
                 "timeout": 5},
                config=_config(self.tmp),
            )
        self.assertIn("err_msg", result)

    def test_timeout_returns_timeout_message(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = execute_command.invoke(
                # python -c "import time; time.sleep(60)" with 1s timeout
                {"command": 'python -c "import time; time.sleep(60)"', "timeout": 1},
                config=_config(self.tmp),
            )
        self.assertIn("timed out", result.lower())

    def test_max_timeout_capped(self):
        # Passing timeout > MAX_EXEC_TIMEOUT should be silently capped
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = execute_command.invoke(
                {"command": _echo("capped"), "timeout": 9999},
                config=_config(self.tmp),
            )
        self.assertIn("capped", result)

    def test_command_not_in_allowlist_rejected(self):
        # `rm` is not in the default allowlist.
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = execute_command.invoke(
                {"command": "rm -rf /tmp/foo", "timeout": 5},
                config=_config(self.tmp),
            )
        self.assertIn("allowlist", result.lower())

    def test_argument_with_parent_traversal_rejected(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = execute_command.invoke(
                {"command": "cat ../../../etc/passwd", "timeout": 5},
                config=_config(self.tmp),
            )
        self.assertIn("..", result)
        self.assertIn("not allowed", result.lower())

    def test_absolute_path_outside_workspace_rejected(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = execute_command.invoke(
                {"command": "cat /etc/passwd", "timeout": 5},
                config=_config(self.tmp),
            )
        self.assertIn("outside the workspace", result.lower())

    def test_shell_metacharacters_not_interpreted(self):
        # With argv mode, `<echo> a && echo b` parses as one command whose
        # arguments happen to include the literal "&&" — no chained second call.
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = execute_command.invoke(
                {"command": 'python -c "import sys; print(\' \'.join(sys.argv[1:]))"'
                            " a && echo b",
                 "timeout": 5},
                config=_config(self.tmp),
            )
        # The literal "&&" is echoed back, proving no shell evaluation happened.
        self.assertIn("&&", result)


class TestFileToolAuditLog(unittest.TestCase):
    """Item #8: write_file, edit_file, and execute_command write to the audit
    log so admins can reconstruct who-did-what after the fact."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _invoke_capturing_audit(self, fn, payload):
        # ROOT_USER must be patched too: with EXEC_REQUIRE_ADMIN on, _ROOT is
        # not an admin by default and execute_command returns the denial before
        # it ever reaches the path under test.
        with patch.object(fritz_utils, "ROOT_USER", _ROOT), \
             patch.object(file_tools, "audit_log") as audit:
            fn.invoke(payload, config=_config(self.tmp))
        return audit

    def test_write_file_success_emits_audit_entry(self):
        audit = self._invoke_capturing_audit(
            write_file,
            {"path": "new.txt", "content": "hello world"},
        )
        audit.assert_called_once()
        args, kwargs = audit.call_args
        self.assertEqual(args[0], "file_write")
        self.assertEqual(kwargs["user_id"], _ROOT)
        self.assertEqual(kwargs["path"], "new.txt")
        self.assertEqual(kwargs["bytes"], len("hello world"))
        self.assertEqual(kwargs["result"], "ok")
        self.assertFalse(kwargs["existed"])  # fresh file

    def test_write_file_records_existed_flag_on_overwrite(self):
        target = os.path.join(self.tmp, "exists.txt")
        with open(target, "w") as f:
            f.write("old")
        audit = self._invoke_capturing_audit(
            write_file,
            {"path": "exists.txt", "content": "new"},
        )
        kwargs = audit.call_args.kwargs
        self.assertTrue(kwargs["existed"])

    def test_edit_file_success_emits_audit_entry(self):
        target = os.path.join(self.tmp, "src.txt")
        with open(target, "w") as f:
            f.write("alpha beta gamma")
        audit = self._invoke_capturing_audit(
            edit_file,
            {"path": "src.txt", "old_text": "beta", "new_text": "BETA"},
        )
        audit.assert_called_once()
        args, kwargs = audit.call_args
        self.assertEqual(args[0], "file_edit")
        self.assertEqual(kwargs["path"], "src.txt")
        self.assertEqual(kwargs["old_len"], 4)
        self.assertEqual(kwargs["new_len"], 4)
        self.assertEqual(kwargs["replaced"], 1)
        self.assertEqual(kwargs["result"], "ok")

    def test_execute_command_success_records_argv_and_exit_code(self):
        audit = self._invoke_capturing_audit(
            execute_command,
            {"command": _echo("hello"), "timeout": 5},
        )
        audit.assert_called_once()
        args, kwargs = audit.call_args
        self.assertEqual(args[0], "exec")
        self.assertEqual(kwargs["argv"], _echo_argv("hello"))
        self.assertEqual(kwargs["exit_code"], 0)
        self.assertEqual(kwargs["result"], "ok")

    def test_execute_command_rejection_is_audited(self):
        audit = self._invoke_capturing_audit(
            execute_command,
            {"command": "rm -rf /tmp", "timeout": 5},  # rm not in allowlist
        )
        audit.assert_called_once()
        args, kwargs = audit.call_args
        self.assertEqual(args[0], "exec")
        self.assertEqual(kwargs["result"], "rejected")
        self.assertIn("allowlist", kwargs["reason"].lower())

    def test_execute_command_parse_error_is_audited(self):
        # An unbalanced quote makes shlex.split raise ValueError.
        audit = self._invoke_capturing_audit(
            execute_command,
            {"command": 'echo "unclosed', "timeout": 5},
        )
        audit.assert_called_once()
        kwargs = audit.call_args.kwargs
        self.assertEqual(kwargs["result"], "parse_error")

    def test_read_file_does_not_emit_audit_entry(self):
        # Reads are intentionally not audited (would be too noisy).
        target = os.path.join(self.tmp, "rfile.txt")
        with open(target, "w") as f:
            f.write("content")
        with patch.object(file_tools, "audit_log") as audit:
            from file_tools import read_file as read_file_tool
            read_file_tool.invoke({"path": "rfile.txt"}, config=_config(self.tmp))
        audit.assert_not_called()


class TestExecEnvScrub(unittest.TestCase):
    """Children must not inherit the bot's secrets from os.environ."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_child_cannot_read_bot_secrets(self):
        # The direct regression for the verified exploit: before the scrub this
        # printed the real token and admin password back into the chat reply.
        secrets = {
            "DISCORD_BOT_TOKEN": "leak-sentinel-abc",
            "ADMIN_PANEL_PASSWORD": "pw-sentinel-xyz",
            "CHAT_COOKIE_SECRET": "cookie-sentinel",
        }
        probe = (
            'python -c "import os;print(os.environ.get(\'DISCORD_BOT_TOKEN\'),'
            "os.environ.get('ADMIN_PANEL_PASSWORD'),"
            'os.environ.get(\'CHAT_COOKIE_SECRET\'))"'
        )
        with patch.dict(os.environ, secrets), \
             patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = execute_command.invoke(
                {"command": probe, "timeout": 10},
                config=_config(self.tmp),
            )
        for sentinel in secrets.values():
            self.assertNotIn(sentinel, result)
        self.assertIn("None None None", result)

    def test_build_exec_env_keeps_only_allowlisted(self):
        with patch.object(file_tools, "EXEC_ENV_PASSTHROUGH", frozenset({"PATH"})), \
             patch.dict(os.environ, {"SOME_JUNK_VAR": "junk"}):
            env = file_tools._build_exec_env()
        self.assertEqual(set(env), {"PATH", "MISTERFRITZ_SANDBOX"})

    def test_build_exec_env_forces_path_when_absent(self):
        with patch.object(file_tools, "EXEC_ENV_PASSTHROUGH", frozenset()):
            env = file_tools._build_exec_env()
        self.assertEqual(env["PATH"], os.defpath)

    def test_scrubbed_child_still_runs(self):
        # Proves PATH passthrough did not break program resolution.
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = execute_command.invoke(
                {"command": _echo("scrub_ok"), "timeout": 10},
                config=_config(self.tmp),
            )
        self.assertIn("scrub_ok", result)
        self.assertIn("Exit code: 0", result)

    def test_sandbox_marker_is_injected(self):
        env = file_tools._build_exec_env()
        self.assertEqual(env["MISTERFRITZ_SANDBOX"], "1")


class TestExecAdminGate(unittest.TestCase):
    """EXEC_REQUIRE_ADMIN restricts only execute_command, not the other tools."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_non_admin_is_denied_when_gate_on(self):
        with patch.object(file_tools, "EXEC_REQUIRE_ADMIN", True), \
             patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = execute_command.invoke(
                {"command": _echo("hi"), "timeout": 5},
                config=_config(self.tmp, user="regular_user"),
            )
        self.assertIn("administrators", result)
        self.assertNotIn("Exit code: 0", result)

    def test_admin_is_allowed_when_gate_on(self):
        with patch.object(file_tools, "EXEC_REQUIRE_ADMIN", True), \
             patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = execute_command.invoke(
                {"command": _echo("hi"), "timeout": 10},
                config=_config(self.tmp),
            )
        self.assertIn("Exit code: 0", result)

    def test_non_admin_allowed_when_gate_off(self):
        with patch.object(file_tools, "EXEC_REQUIRE_ADMIN", False), \
             patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = execute_command.invoke(
                {"command": _echo("hi"), "timeout": 10},
                config=_config(self.tmp, user="regular_user"),
            )
        self.assertIn("Exit code: 0", result)

    def test_denial_is_audited(self):
        with patch.object(file_tools, "EXEC_REQUIRE_ADMIN", True), \
             patch.object(fritz_utils, "ROOT_USER", _ROOT), \
             patch.object(file_tools, "audit_log") as audit:
            execute_command.invoke(
                {"command": _echo("hi"), "timeout": 5},
                config=_config(self.tmp, user="regular_user"),
            )
        audit.assert_called_once()
        args, kwargs = audit.call_args
        self.assertEqual(args[0], "exec")
        self.assertEqual(kwargs["result"], "denied")
        self.assertIn("administrators", kwargs["reason"])

    def test_other_file_tools_unaffected_by_gate(self):
        # Guards the Phase 7b promise: the gate is on execute_command only.
        cfg = _config(self.tmp, user="regular_user")
        with patch.object(file_tools, "EXEC_REQUIRE_ADMIN", True), \
             patch.object(fritz_utils, "ROOT_USER", _ROOT):
            written = write_file.invoke({"path": "note.txt", "content": "kept"}, config=cfg)
            read_back = read_file.invoke({"path": "note.txt"}, config=cfg)
        self.assertNotIn("administrators", written)
        self.assertIn("kept", read_back)


class TestExecArgv0IsBareName(unittest.TestCase):
    """argv[0] must resolve from PATH, never from a workspace-authored file."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_absolute_path_to_workspace_script_rejected(self):
        # The Windows bypass: git.bat passes the basename + suffix-strip
        # allowlist check, sits inside the workspace so the path check accepts
        # it, and CreateProcess runs .bat through cmd.exe.
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            write_file.invoke(
                {"path": "git.bat", "content": "@echo off\r\necho PWNED_VIA_BAT\r\n"},
                config=_config(self.tmp),
            )
            bat = os.path.join(self.tmp, "git.bat").replace("\\", "/")
            result = execute_command.invoke(
                {"command": '"%s"' % bat, "timeout": 5},
                config=_config(self.tmp),
            )
        self.assertIn("bare program name", result)
        self.assertNotIn("PWNED_VIA_BAT", result)

    def test_relative_path_argv0_rejected(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = execute_command.invoke(
                {"command": './python -c "print(1)"', "timeout": 5},
                config=_config(self.tmp),
            )
        self.assertIn("bare program name", result)

    def test_bare_name_still_accepted(self):
        # Proves the check did not over-reject.
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = execute_command.invoke(
                {"command": _echo("ok"), "timeout": 10},
                config=_config(self.tmp),
            )
        self.assertIn("Exit code: 0", result)


class TestExecTimeoutClamp(unittest.TestCase):
    """timeout is clamped at both ends — run(timeout=0) would always expire."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_zero_timeout_is_clamped_to_one(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = execute_command.invoke(
                {"command": _echo("t"), "timeout": 0},
                config=_config(self.tmp),
            )
        self.assertIn("Exit code: 0", result)

    def test_negative_timeout_is_clamped(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = execute_command.invoke(
                {"command": _echo("t"), "timeout": -5},
                config=_config(self.tmp),
            )
        self.assertIn("Exit code: 0", result)


if __name__ == "__main__":
    unittest.main()
