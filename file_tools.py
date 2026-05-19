import fnmatch
import functools
import logging
import os
import re
import shlex
import subprocess
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from fritz_utils import (
    EXEC_ALLOWED_COMMANDS,
    EXEC_OUTPUT_TRUNCATE,
    MAX_FILE_SIZE_BYTES,
    MAX_READ_LINES,
)
from observability import METRICS, time_tool

logger = logging.getLogger(__name__)

# Backwards-compatible aliases — the test suite and other modules may still
# import these names. The values are now sourced from fritz_utils.
MAX_FILE_SIZE = MAX_FILE_SIZE_BYTES


def _timed(name: str):
    """Decorator: wrap the function in time_tool so counter + latency are
    both recorded. Stack BEFORE @tool so LangChain wraps an already-timed fn.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with time_tool(name):
                return fn(*args, **kwargs)
        return wrapper
    return deco
# File extensions to include in search by default
TEXT_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp',
    '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.sh',
    '.bash', '.zsh', '.ps1', '.bat', '.cmd',
    '.html', '.css', '.scss', '.sass', '.less',
    '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    '.xml', '.csv', '.sql', '.graphql',
    '.md', '.txt', '.rst', '.log',
    '.env', '.gitignore', '.dockerignore',
    '.dockerfile', '.makefile',
}


def _authorize(config: RunnableConfig) -> None:
    """Verify the request has a workspace_root set.

    Per-user workspaces (Phase 7b) mean any user can use file tools — they're
    sandboxed to their own dir. The actual authorisation happens at /workspace
    enable time and is enforced by _resolve_safe_path keeping operations
    inside the workspace.
    """
    metadata = config.get("metadata", {})
    if not metadata.get("workspace_root"):
        raise PermissionError(
            "You do not have a workspace. Run /workspace enable to create one."
        )


def _get_workspace(config: RunnableConfig) -> str:
    """Extract and validate the workspace root from config metadata, with authorization."""
    _authorize(config)
    metadata = config.get("metadata", {})
    workspace = metadata.get("workspace_root")
    if not workspace:
        raise ValueError(
            "No workspace directory is set. The user must first use the /workspace command "
            "to set a directory before file operations can be used."
        )
    workspace = os.path.abspath(workspace)
    if not os.path.isdir(workspace):
        raise ValueError(f"Workspace directory does not exist: {workspace}")
    return workspace


def _resolve_safe_path(workspace: str, relative_path: str) -> str:
    """Resolve a relative path within the workspace, preventing traversal attacks."""
    # Normalize and join
    joined = os.path.normpath(os.path.join(workspace, relative_path))
    # Ensure the resolved path is within the workspace
    if not joined.startswith(os.path.normpath(workspace) + os.sep) and joined != os.path.normpath(workspace):
        raise ValueError(f"Path '{relative_path}' resolves outside the workspace. Access denied.")
    return joined


def _is_text_file(file_path: str) -> bool:
    """Check if a file is likely a text file based on extension."""
    _, ext = os.path.splitext(file_path)
    if ext.lower() in TEXT_EXTENSIONS:
        return True
    # Files without extensions (Makefile, Dockerfile, etc.)
    basename = os.path.basename(file_path).lower()
    return basename in {'makefile', 'dockerfile', 'vagrantfile', 'gemfile', 'rakefile', 'procfile'}


@tool(parse_docstring=True)
@_timed("list_directory")
def list_directory(config: RunnableConfig, path: str = ".", include_hidden: bool = False) -> str:
    """Lists files and directories at the given path within the workspace.

    Args:
        config: The RunnableConfig containing workspace_root in metadata.
        path: Relative path within the workspace to list. Defaults to workspace root.
        include_hidden: Whether to include hidden files/directories (starting with .).

    Returns:
        A formatted listing of directory contents with type indicators.
    """
    workspace = _get_workspace(config)
    target = _resolve_safe_path(workspace, path)

    if not os.path.isdir(target):
        return f"Error: '{path}' is not a directory."

    try:
        entries = sorted(os.listdir(target))
    except PermissionError:
        return f"Error: Permission denied reading '{path}'."

    if not include_hidden:
        entries = [e for e in entries if not e.startswith('.')]

    if not entries:
        return f"Directory '{path}' is empty."

    lines = []
    dirs = []
    files = []
    for entry in entries:
        full = os.path.join(target, entry)
        if os.path.isdir(full):
            dirs.append(f"  {entry}/")
        else:
            size = os.path.getsize(full)
            files.append(f"  {entry}  ({_format_size(size)})")

    if dirs:
        lines.append("Directories:")
        lines.extend(dirs)
    if files:
        lines.append("Files:")
        lines.extend(files)

    rel = os.path.relpath(target, workspace)
    header = f"Contents of '{rel}':" if rel != '.' else "Contents of workspace root:"
    return header + "\n" + "\n".join(lines)


@tool(parse_docstring=True)
@_timed("read_file")
def read_file(config: RunnableConfig, path: str, offset: int = 0, limit: Optional[int] = None) -> str:
    """Reads the contents of a file within the workspace.

    Args:
        config: The RunnableConfig containing workspace_root in metadata.
        path: Relative path to the file within the workspace.
        offset: Line number to start reading from (0-indexed). Defaults to 0.
        limit: Maximum number of lines to read. Defaults to 500.

    Returns:
        The file contents with line numbers.
    """
    workspace = _get_workspace(config)
    target = _resolve_safe_path(workspace, path)

    if not os.path.isfile(target):
        return f"Error: '{path}' is not a file or does not exist."

    file_size = os.path.getsize(target)
    if file_size > MAX_FILE_SIZE:
        return f"Error: File is too large ({_format_size(file_size)}). Maximum is {_format_size(MAX_FILE_SIZE)}."

    if limit is None:
        limit = MAX_READ_LINES

    try:
        with open(target, 'r', encoding='utf-8', errors='replace') as f:
            all_lines = f.readlines()
    except PermissionError:
        return f"Error: Permission denied reading '{path}'."

    total_lines = len(all_lines)
    selected = all_lines[offset:offset + limit]

    if not selected:
        return f"No content at offset {offset}. File has {total_lines} lines."

    # Format with line numbers
    numbered = []
    for i, line in enumerate(selected, start=offset + 1):
        numbered.append(f"{i:>5} | {line.rstrip()}")

    header = f"File: {path} ({total_lines} lines total)"
    if offset > 0 or offset + limit < total_lines:
        header += f" [showing lines {offset + 1}-{min(offset + limit, total_lines)}]"

    return header + "\n" + "\n".join(numbered)


@tool(parse_docstring=True)
@_timed("write_file")
def write_file(config: RunnableConfig, path: str, content: str) -> str:
    """Writes content to a file within the workspace. Creates the file if it doesn't exist,
    or overwrites it if it does. Creates parent directories as needed.

    Args:
        config: The RunnableConfig containing workspace_root in metadata.
        path: Relative path to the file within the workspace.
        content: The full content to write to the file.

    Returns:
        A confirmation message.
    """
    workspace = _get_workspace(config)
    target = _resolve_safe_path(workspace, path)

    # Create parent directories if needed
    parent = os.path.dirname(target)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    existed = os.path.isfile(target)
    try:
        with open(target, 'w', encoding='utf-8') as f:
            f.write(content)
    except PermissionError:
        return f"Error: Permission denied writing to '{path}'."

    line_count = content.count('\n') + (1 if content and not content.endswith('\n') else 0)
    action = "Updated" if existed else "Created"
    return f"{action} '{path}' ({line_count} lines, {_format_size(len(content.encode('utf-8')))})"


@tool(parse_docstring=True)
@_timed("edit_file")
def edit_file(config: RunnableConfig, path: str, old_text: str, new_text: str) -> str:
    """Performs a targeted text replacement in a file within the workspace.
    Replaces the first occurrence of old_text with new_text.

    Args:
        config: The RunnableConfig containing workspace_root in metadata.
        path: Relative path to the file within the workspace.
        old_text: The exact text to find and replace. Must match exactly.
        new_text: The text to replace it with.

    Returns:
        A confirmation message showing the edit was applied.
    """
    workspace = _get_workspace(config)
    target = _resolve_safe_path(workspace, path)

    if not os.path.isfile(target):
        return f"Error: '{path}' does not exist."

    try:
        with open(target, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except PermissionError:
        return f"Error: Permission denied reading '{path}'."

    if old_text not in content:
        return f"Error: The specified text was not found in '{path}'. Make sure the text matches exactly, including whitespace and indentation."

    occurrences = content.count(old_text)
    new_content = content.replace(old_text, new_text, 1)

    try:
        with open(target, 'w', encoding='utf-8') as f:
            f.write(new_content)
    except PermissionError:
        return f"Error: Permission denied writing to '{path}'."

    info = f"Edited '{path}': replaced 1 occurrence"
    if occurrences > 1:
        info += f" ({occurrences - 1} additional occurrence(s) remain)"
    return info


@tool(parse_docstring=True)
@_timed("search_files")
def search_files(config: RunnableConfig, pattern: str, path: str = ".", file_glob: str = "*") -> str:
    """Searches for a text pattern across files in the workspace, similar to grep.

    Args:
        config: The RunnableConfig containing workspace_root in metadata.
        pattern: The regex pattern to search for in file contents.
        path: Relative directory path to search in. Defaults to workspace root.
        file_glob: Glob pattern to filter files (e.g. "*.py", "*.js"). Defaults to all files.

    Returns:
        Matching lines with file paths and line numbers.
    """
    workspace = _get_workspace(config)
    target = _resolve_safe_path(workspace, path)

    if not os.path.isdir(target):
        return f"Error: '{path}' is not a directory."

    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    matches = []
    files_searched = 0
    max_matches = 100

    for root, dirs, files_list in os.walk(target):
        # Skip hidden directories and common non-code directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {
            'node_modules', '__pycache__', '.git', 'venv', '.venv', 'env',
            'dist', 'build', '.idea', '.vscode'
        }]

        for filename in files_list:
            if not fnmatch.fnmatch(filename, file_glob):
                continue
            full_path = os.path.join(root, filename)
            if not _is_text_file(full_path) and file_glob == "*":
                continue
            if os.path.getsize(full_path) > MAX_FILE_SIZE:
                continue

            files_searched += 1
            try:
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    for line_num, line in enumerate(f, 1):
                        if compiled.search(line):
                            rel_path = os.path.relpath(full_path, workspace)
                            matches.append(f"  {rel_path}:{line_num}: {line.rstrip()}")
                            if len(matches) >= max_matches:
                                break
            except (PermissionError, OSError):
                continue

            if len(matches) >= max_matches:
                break
        if len(matches) >= max_matches:
            break

    if not matches:
        return f"No matches found for '{pattern}' in {files_searched} files."

    header = f"Found {len(matches)} match(es) across {files_searched} files:"
    if len(matches) >= max_matches:
        header += f" (results truncated at {max_matches})"
    return header + "\n" + "\n".join(matches)


# Maximum command execution timeout in seconds
MAX_EXEC_TIMEOUT = 30


def _validate_command_argv(argv: list[str], workspace: str) -> Optional[str]:
    """Validate a parsed argv list. Returns an error message if rejected, else None.

    Enforces:
      - argv[0] (basename, lowercased) must be in EXEC_ALLOWED_COMMANDS.
      - No argument may contain '..' as a path component or be an absolute path
        that escapes the workspace.
    """
    if not argv:
        return "Error: Empty command."

    program = os.path.basename(argv[0]).lower()
    # Strip Windows .exe / .cmd / .bat suffixes for the allowlist check.
    for suffix in (".exe", ".cmd", ".bat", ".ps1"):
        if program.endswith(suffix):
            program = program[: -len(suffix)]
            break

    if program not in EXEC_ALLOWED_COMMANDS:
        return (
            f"Error: Command '{program}' is not in the allowlist. "
            f"Allowed: {', '.join(sorted(EXEC_ALLOWED_COMMANDS))}."
        )

    workspace_norm = os.path.normpath(os.path.abspath(workspace))
    for arg in argv[1:]:
        # Reject explicit parent traversal as a path segment.
        # Split on both separators so we catch ../ and ..\ on either platform.
        segments = re.split(r"[\\/]+", arg)
        if ".." in segments:
            return f"Error: Argument '{arg}' contains '..' which is not allowed."

        # If the arg looks like an absolute path, ensure it stays within the workspace.
        if os.path.isabs(arg):
            resolved = os.path.normpath(os.path.abspath(arg))
            if resolved != workspace_norm and not resolved.startswith(workspace_norm + os.sep):
                return f"Error: Argument '{arg}' resolves outside the workspace."

    return None


@tool(parse_docstring=True)
@_timed("execute_command")
def execute_command(config: RunnableConfig, command: str, timeout: int = 30) -> str:
    """Executes a command in the workspace directory and returns the output.
    The command runs with the workspace as the current working directory.

    Only commands whose program name is in the allowlist (see EXEC_ALLOWED_COMMANDS
    env var) are permitted. Shell metacharacters are not interpreted — this runs
    via argv, not a shell. Arguments with '..' or absolute paths outside the
    workspace are rejected.

    Args:
        config: The RunnableConfig containing workspace_root and user_id in metadata.
        command: The command to execute (parsed with shlex; no shell interpretation).
        timeout: Maximum seconds to wait for the command to finish. Defaults to 30, max 30.

    Returns:
        The combined stdout and stderr output from the command.
    """
    workspace = _get_workspace(config)

    timeout = min(timeout, MAX_EXEC_TIMEOUT)

    try:
        # posix=True works on Windows too for our purposes; we don't want cmd.exe semantics.
        argv = shlex.split(command, posix=True)
    except ValueError as e:
        return f"Error: Could not parse command: {e}"

    rejection = _validate_command_argv(argv, workspace)
    if rejection is not None:
        logger.warning("Rejected command in %s: %s (%s)", workspace, command, rejection)
        return rejection

    logger.info("Executing command in %s: %s", workspace, argv)
    try:
        result = subprocess.run(
            argv,
            shell=False,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"STDERR:\n{result.stderr}")
        output_parts.append(f"Exit code: {result.returncode}")

        output = "\n".join(output_parts)

        # Truncate if too long for the LLM context
        if len(output) > EXEC_OUTPUT_TRUNCATE:
            output = output[:EXEC_OUTPUT_TRUNCATE] + "\n... (output truncated)"

        return output

    except FileNotFoundError:
        return f"Error: Program '{argv[0]}' not found in PATH."
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds."
    except Exception as e:
        METRICS.record_error("execute_command", e)
        return f"Error executing command: {e}"


def _format_size(size_bytes: int) -> str:
    """Format a file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def get_file_tools():
    """Returns the list of file operation tools."""
    return [list_directory, read_file, write_file, edit_file, search_files, execute_command]


def get_file_tools_description():
    """Returns a dictionary of file tools and their descriptions for the system prompt."""
    return {
        "list_directory": (list_directory, "List files and directories in the workspace. Provide a relative path."),
        "read_file": (read_file, "Read the contents of a file in the workspace. Supports offset and line limits."),
        "write_file": (write_file, "Write or create a file in the workspace. Provide relative path and full content."),
        "edit_file": (edit_file, "Edit a file by replacing a specific text string with new text. Targeted and safe."),
        "search_files": (search_files, "Search for a regex pattern across files in the workspace, like grep."),
        "execute_command": (execute_command, "Execute a shell command in the workspace directory and return the output. Use for running code, tests, builds, git commands, etc."),
    }
