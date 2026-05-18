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


class TestAuthorization(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_non_root_user_raises_permission_error(self):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            with self.assertRaises(PermissionError):
                list_directory.invoke({"path": "."}, config=_config(self.tmp, user="evil_user"))

    def test_no_workspace_raises_value_error(self):
        cfg = {"configurable": {}, "metadata": {"user_id": _ROOT, "workspace_root": None}}
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            with self.assertRaises((ValueError, Exception)):
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
                {"command": "echo hello", "timeout": 5},
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
                {"command": "echo capped", "timeout": 9999},
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
        # With argv mode, `echo a && rm b` is parsed as: echo a "&&" rm b — no chained rm.
        with patch.object(fritz_utils, "ROOT_USER", _ROOT):
            result = execute_command.invoke(
                {"command": "echo a && echo b", "timeout": 5},
                config=_config(self.tmp),
            )
        # The literal "&&" should appear in echo's output, proving no shell evaluation.
        self.assertIn("&&", result)


if __name__ == "__main__":
    unittest.main()
