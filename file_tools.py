import fnmatch
import logging
import os
import re
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from fritz_utils import ROOT_USER
from observability import METRICS

logger = logging.getLogger(__name__)

# Maximum lines to return in a single read
MAX_READ_LINES = 500
# Maximum file size in bytes to read (1MB)
MAX_FILE_SIZE = 1_048_576
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
    """Verify the requesting user matches the configured root_user."""
    metadata = config.get("metadata", {})
    user_id = metadata.get("user_id", "")
    if not ROOT_USER or user_id != ROOT_USER:
        raise PermissionError("You do not have permission to use file operations.")


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
def list_directory(config: RunnableConfig, path: str = ".", include_hidden: bool = False) -> str:
    """Lists files and directories at the given path within the workspace.

    Args:
        config: The RunnableConfig containing workspace_root in metadata.
        path: Relative path within the workspace to list. Defaults to workspace root.
        include_hidden: Whether to include hidden files/directories (starting with .).

    Returns:
        A formatted listing of directory contents with type indicators.
    """
    METRICS.increment("tool.list_directory")
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
    METRICS.increment("tool.read_file")
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
    METRICS.increment("tool.write_file")
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
    METRICS.increment("tool.edit_file")
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
    METRICS.increment("tool.search_files")
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
    return [list_directory, read_file, write_file, edit_file, search_files]


def get_file_tools_description():
    """Returns a dictionary of file tools and their descriptions for the system prompt."""
    return {
        "list_directory": (list_directory, "List files and directories in the workspace. Provide a relative path."),
        "read_file": (read_file, "Read the contents of a file in the workspace. Supports offset and line limits."),
        "write_file": (write_file, "Write or create a file in the workspace. Provide relative path and full content."),
        "edit_file": (edit_file, "Edit a file by replacing a specific text string with new text. Targeted and safe."),
        "search_files": (search_files, "Search for a regex pattern across files in the workspace, like grep."),
    }
