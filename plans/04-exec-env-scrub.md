# 4. Contain execute_command

[← back to index](README.md)

**Effort:** M (half day)  
**Depends on:** nothing

## Goal
Today any Discord user who runs `/workspace enable` can make Mister Fritz run `python -c "<anything>"` as the bot's OS user, with the bot's full environment inherited — I verified live that this prints DISCORD_BOT_TOKEN and ADMIN_PANEL_PASSWORD back into the chat. When this item is done, three things are true that are not true today. (1) `execute_command` children get an explicit, platform-aware environment allowlist, so no secret the bot holds is reachable through a subprocess even if the command itself is arbitrary code. (2) `argv[0]` must be a bare program name resolved from PATH, closing a verified Windows bypass where a user writes `git.bat` with `write_file` and then invokes it by absolute path — the basename+suffix-strip check passes it and CreateProcess runs it through cmd.exe. (3) The owner has made a conscious, documented choice between two configurations — "interpreters allowed, admin-gated" vs "inspection-only allowlist, open to all workspace holders" — instead of the current accidental state, where the code comment at fritz_utils.py:177 still claims ROOT_USER gating that Phase 7b removed.

## Definition of done

- [ ] `execute_command` calls `subprocess.run(..., env=_build_exec_env())`; a test asserts a sentinel value placed in `os.environ['DISCORD_BOT_TOKEN']` is NOT visible to the child process.
- [ ] `_build_exec_env()` returns only vars whose upper-cased name is in `fritz_utils.EXEC_ENV_PASSTHROUGH`, always includes a non-empty PATH (falling back to `os.defpath`), and the Windows default set includes SYSTEMROOT/COMSPEC/PATHEXT/TEMP/TMP.
- [ ] `_validate_command_argv` rejects any `argv[0]` that is absolute or contains `/` or `\\`; a regression test writes `git.bat` into the workspace, invokes it by absolute path, and asserts the rejection message and that the batch file's output does not appear.
- [ ] `EXEC_REQUIRE_ADMIN` exists as an env knob wired to `fritz_utils.is_admin`; when true a non-admin workspace holder gets a refusal string (not an exception) from `execute_command` while `read_file`/`write_file`/`edit_file`/`search_files`/`list_directory` keep working for them.
- [ ] The denial path emits `audit_log("exec", ..., result="denied", reason=...)`.
- [ ] The owner has picked Config A or Config B; `_DEFAULT_EXEC_ALLOWED` and the `EXEC_REQUIRE_ADMIN` default in fritz_utils.py match that choice, and the stale comment at fritz_utils.py:177 is corrected.
- [ ] `timeout` is clamped to `max(1, min(timeout, EXEC_TIMEOUT_MAX))`; `MAX_EXEC_TIMEOUT` becomes an alias of the new env-configurable `fritz_utils.EXEC_TIMEOUT_MAX`, matching the existing `MAX_FILE_SIZE = MAX_FILE_SIZE_BYTES` pattern at file_tools.py:25.
- [ ] Every new env var (`EXEC_ENV_PASSTHROUGH`, `EXEC_REQUIRE_ADMIN`, `EXEC_TIMEOUT_MAX`) is documented in `.env.example` with its default shown, per the CONTRIBUTING.md:86 convention.
- [ ] README.md:409-410 troubleshooting entry and the `execute_command` docstring (file_tools.py:437-451) describe the scrubbed env, the bare-name rule, and the admin gate.
- [ ] CHANGELOG.md gets a phase-style entry under `## [Unreleased]` → `### Security` (a new subsection; the file currently has Performance/Added/Changed).
- [ ] `pytest tests/` and `ruff check .` both pass.

## Current state (verified against the working tree)
VERIFIED BY EXECUTION, not just reading. Running `execute_command` with `metadata={"user_id": "regular_user", "workspace_root": <tmpdir>}` and `command='python -c "import os;print(os.environ.get(\'DISCORD_BOT_TOKEN\'), os.environ.get(\'ADMIN_PANEL_PASSWORD\'))"'` returned `leak-me-123 hunter2\nExit code: 0`. The audit finding is correct on every point; line numbers have NOT drifted.

Confirmed call chain and line numbers:
- `file_tools.py:64-77` `_authorize` gates only on `metadata["workspace_root"]` being truthy. No admin check. Docstring at 66-71 explicitly says Phase 7b made this per-user rather than admin.
- `file_tools.py:79-92` `_get_workspace` calls `_authorize`, abspaths, and checks `isdir`.
- `file_tools.py:389-390` `MAX_EXEC_TIMEOUT = 30` — a bare module constant, NOT env-configurable (inconsistent with every other tunable, which lives in fritz_utils).
- `file_tools.py:393-431` `_validate_command_argv`. Line 404: `program = os.path.basename(argv[0]).lower()`. Lines 406-409 strip `.exe/.cmd/.bat/.ps1`. Lines 411-415 allowlist membership. Lines 417-431 loop over `argv[1:]` only — rejecting `..` segments and absolute paths outside the workspace. **`argv[0]` itself is never subjected to the path checks.** Nothing inspects flags.
- `file_tools.py:456` `timeout = min(timeout, MAX_EXEC_TIMEOUT)` — capped above, unbounded below (0 or negative is accepted).
- `file_tools.py:479-486` `subprocess.run(argv, shell=False, cwd=workspace, capture_output=True, text=True, timeout=timeout)` — **no `env=` argument, so `os.environ` is inherited wholesale.**
- `file_tools.py:498-499` `EXEC_OUTPUT_TRUNCATE` is applied AFTER `capture_output=True` has already buffered the entire child output into RAM. The cap protects LLM context, not the bot's memory.
- `fritz_utils.py:171-191` the sandbox block. **Line 177 is stale: "Keep it tight by default — this is ROOT_USER-gated, but defence-in-depth."** Lines 178-186 `_DEFAULT_EXEC_ALLOWED` includes `python,python3,pip,pytest,node,npm,npx,git,go,cargo,make,find`.
- `fritz_utils.py:221-231` `is_admin(user_id)` reads `ROOT_USER`/`ADMIN_USERS` from module scope at call time (so `patch.object(fritz_utils, "ROOT_USER", ...)` works from any caller).
- `bot_commands.py:497-517` `/workspace enable` has NO admin gate — contrast `/workspace set` at 534-554 which calls `_require_admin`. So `enable` is the reachability path for any Discord user.
- `main_discord.py:193-203` passes `workspace_store.get(author)` into `ask_stuff`; `mister_fritz.py:290-293` and `543` set `include_file_tools = workspace_root is not None`; `agent_tools.py:539-540` merges `get_file_tools_description()` (file_tools.py:543-552) into the agent toolset.
- Reachability limits worth knowing: `admin_panel.py:448-454` and `522-530` (web chat send/stream) call `ask_stuff` WITHOUT `workspace_root`, and `main_telegram.py:28,54` likewise. So file tools are Discord-only today. The audit's mention of CHAT_COOKIE_SECRET is still valid as a leaked secret (it is in `os.environ` if set), just not as an entry point.

TWO ADDITIONAL VERIFIED HOLES the audit did not name:
1. **Windows allowlist bypass via workspace-authored batch file.** I created `git.bat` in a temp dir and ran `subprocess.run([<absolute path to git.bat>], cwd=d)` — exit 0, stdout `PWNED_VIA_BAT`. `_validate_command_argv(["<abs>\\git.bat"], d)` returns `None` (accepted): basename → `git.bat` → suffix-strip → `git` → in allowlist; and the absolute-path check at 426-429 only runs over `argv[1:]`, and even if it ran on argv[0] the path is *inside* the workspace so it would pass. Bare `git.bat` and `.\git.bat` were NOT found (CreateProcess searches the parent's cwd, not `lpCurrentDirectory`), so the absolute-path form is the working variant. On POSIX this needs the exec bit, which `write_file` does not set — so this is Windows-specific, which is this deployment's platform.
2. **A flag denylist would be security theatre while interpreters are allowlisted.** `write_file("x.py", <code>)` + `execute_command("python x.py")` is RCE with no inline-eval flag at all. Same for `make` (user-authored Makefile), `git` (`-c core.pager=...`, `!alias`), and `find -exec`. This is why the plan below offers admin-gating and allowlist reduction rather than argument parsing.

Empirical Windows env note (measured on this box, Win11 + Python 3.12): a child launched with `env={"PATH": ...}` alone still imported `socket`, `ssl`, `tempfile` and exited 0 — SYSTEMROOT was not strictly required for that case. But `tempfile.gettempdir()` fell back to `c:\temp` (PATH-only) / `C:\WINDOWS\Temp` (PATH+SYSTEMROOT), both of which a non-elevated user may not be able to write to. So TEMP/TMP are needed for anything that writes a temp file, and SYSTEMROOT/COMSPEC/PATHEXT belong in the set for DLL-loading, `os.system` inside children, and `.cmd` shim resolution (npm). Keep them.

## Change sites

### `fritz_utils.py:171-191`

Correct the stale ROOT_USER comment; add EXEC_REQUIRE_ADMIN, EXEC_ENV_PASSTHROUGH (platform-aware default), and EXEC_TIMEOUT_MAX; set _DEFAULT_EXEC_ALLOWED per the chosen config.

# --- BEFORE (fritz_utils.py:175-191) ---
# Allowlist for the `execute_command` file tool. Only argv[0] values listed here
# are permitted. Override with EXEC_ALLOWED_COMMANDS as a comma-separated list.
# Keep it tight by default — this is ROOT_USER-gated, but defence-in-depth.
_DEFAULT_EXEC_ALLOWED = (
    "ls,dir,pwd,cd,echo,cat,type,head,tail,wc,"
    "python,python3,pip,pytest,"
    "node,npm,npx,"
    "git,"
    "go,cargo,"
    "make,"
    "grep,find,where"
)
EXEC_ALLOWED_COMMANDS: frozenset[str] = frozenset(
    cmd.strip().lower()
    for cmd in os.environ.get("EXEC_ALLOWED_COMMANDS", _DEFAULT_EXEC_ALLOWED).split(",")
    if cmd.strip()
)

# --- AFTER (shared by both configs) ---
# Allowlist for the `execute_command` file tool. Only argv[0] values listed here
# are permitted. Override with EXEC_ALLOWED_COMMANDS as a comma-separated list.
#
# NOT ROOT_USER-gated since Phase 7b: any user who runs `/workspace enable`
# reaches this tool. Every entry here that can evaluate caller-supplied code —
# python/node (-c/-e), pip/npm (install scripts), make (user Makefile),
# git (-c, ! aliases), find (-exec) — is arbitrary code execution for that
# user. Pair the list with EXEC_REQUIRE_ADMIN below; pick one of the two
# configurations documented in .env.example.

# >>> CONFIG A — interpreters allowed, contained by identity + env scrub <<<
_DEFAULT_EXEC_ALLOWED = (
    "ls,dir,pwd,echo,cat,type,head,tail,wc,"
    "python,python3,pip,pytest,"
    "node,npm,npx,"
    "git,"
    "go,cargo,"
    "make,"
    "grep,find,where"
)
_DEFAULT_EXEC_REQUIRE_ADMIN = "true"

# >>> CONFIG B — inspection only, safe for every workspace holder <<<
# _DEFAULT_EXEC_ALLOWED = "ls,dir,pwd,echo,cat,type,head,tail,wc,grep,where"
# _DEFAULT_EXEC_REQUIRE_ADMIN = "false"
# Removed vs. Config A: python, python3, pip, pytest, node, npm, npx, go,
# cargo, make, git, find, cd. `cd` was always a no-op (no shell; cwd is always
# the workspace). `find` goes because of -exec/-delete; `git` because of
# `git -c core.pager=<cmd>` and `!`-aliases. Re-add any of them via
# EXEC_ALLOWED_COMMANDS and you are back to Config A's threat model.

EXEC_ALLOWED_COMMANDS: frozenset[str] = frozenset(
    cmd.strip().lower()
    for cmd in os.environ.get("EXEC_ALLOWED_COMMANDS", _DEFAULT_EXEC_ALLOWED).split(",")
    if cmd.strip()
)

# execute_command only. The other five file tools stay open to any workspace
# holder (Phase 7b) — they are confined by _resolve_safe_path. Running programs
# is not confined by anything the bot controls.
EXEC_REQUIRE_ADMIN: bool = os.environ.get(
    "EXEC_REQUIRE_ADMIN", _DEFAULT_EXEC_REQUIRE_ADMIN
).strip().lower() not in ("0", "false", "no", "off", "")

# Environment variables handed to execute_command children. Everything else is
# dropped, so DISCORD_BOT_TOKEN / ADMIN_PANEL_PASSWORD / CHAT_COOKIE_SECRET /
# OLLAMA_HOST never reach a subprocess. Names are matched case-insensitively
# (os.environ upper-cases keys on Windows).
_DEFAULT_EXEC_ENV_PASSTHROUGH_NT = (
    "PATH,PATHEXT,SYSTEMROOT,WINDIR,COMSPEC,SYSTEMDRIVE,"
    "TEMP,TMP,USERPROFILE,HOMEDRIVE,HOMEPATH,APPDATA,LOCALAPPDATA,"
    "NUMBER_OF_PROCESSORS,PROCESSOR_ARCHITECTURE,OS"
)
_DEFAULT_EXEC_ENV_PASSTHROUGH_POSIX = "PATH,HOME,TMPDIR,LANG,LC_ALL,TZ,TERM,USER,LOGNAME"
EXEC_ENV_PASSTHROUGH: frozenset[str] = frozenset(
    name.strip().upper()
    for name in os.environ.get(
        "EXEC_ENV_PASSTHROUGH",
        _DEFAULT_EXEC_ENV_PASSTHROUGH_NT if os.name == "nt"
        else _DEFAULT_EXEC_ENV_PASSTHROUGH_POSIX,
    ).split(",")
    if name.strip()
)

# Hard ceiling on the execute_command timeout (seconds). Was a bare constant in
# file_tools; env-configurable now to match every other tunable.
EXEC_TIMEOUT_MAX: int = int(os.environ.get("EXEC_TIMEOUT_MAX", "30"))

### `file_tools.py:13-18, 25`

Import the new constants and is_admin; alias MAX_EXEC_TIMEOUT next to the existing MAX_FILE_SIZE alias.

from fritz_utils import (
    EXEC_ALLOWED_COMMANDS,
    EXEC_ENV_PASSTHROUGH,
    EXEC_OUTPUT_TRUNCATE,
    EXEC_REQUIRE_ADMIN,
    EXEC_TIMEOUT_MAX,
    MAX_FILE_SIZE_BYTES,
    MAX_READ_LINES,
    is_admin,
)
...
# Backwards-compatible aliases — the test suite and other modules may still
# import these names. The values are now sourced from fritz_utils.
MAX_FILE_SIZE = MAX_FILE_SIZE_BYTES
MAX_EXEC_TIMEOUT = EXEC_TIMEOUT_MAX

# NOTE for tests: these are from-imports, so they are patched as
# file_tools.EXEC_ALLOWED_COMMANDS / file_tools.EXEC_REQUIRE_ADMIN, NOT as
# fritz_utils.<name>. (fritz_utils.ROOT_USER is different — is_admin reads it
# from its own module globals at call time, so patching fritz_utils works there.)

### `file_tools.py:389-390`

Delete the hardcoded `MAX_EXEC_TIMEOUT = 30` (now the alias defined near line 25).

# DELETE:
# # Maximum command execution timeout in seconds
# MAX_EXEC_TIMEOUT = 30

### `file_tools.py:389-392 (new helpers, inserted where MAX_EXEC_TIMEOUT was)`

Add `_build_exec_env()` and `_exec_denied_reason()`.

def _build_exec_env() -> dict[str, str]:
    """Explicit environment for execute_command children.

    subprocess inherits os.environ by default, which hands every child the
    bot's secrets — DISCORD_BOT_TOKEN, ADMIN_PANEL_PASSWORD, CHAT_COOKIE_SECRET,
    OLLAMA_HOST. Pass an allowlist instead (EXEC_ENV_PASSTHROUGH).
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in EXEC_ENV_PASSTHROUGH
    }
    # POSIX resolves argv[0] via os.get_exec_path(env); with no PATH in the
    # child env nothing is findable at all. On Windows CreateProcess searches
    # the *parent's* PATH, but children that spawn their own tools still need it.
    if not env.get("PATH"):
        env["PATH"] = os.defpath
    # Lets a workspace script tell it is running under the bot sandbox.
    env["MISTERFRITZ_SANDBOX"] = "1"
    return env


def _exec_denied_reason(user_id: str) -> Optional[str]:
    """Return a refusal message if this user may not run programs, else None.

    Returns a string rather than raising (unlike _authorize) so the refusal
    reaches the LLM as a normal tool result and lands in the audit log.
    """
    if not EXEC_REQUIRE_ADMIN:
        return None
    if is_admin(user_id):
        return None
    return (
        "Error: Running commands is restricted to administrators. "
        "The other file tools (read, write, edit, search, list) are still "
        "available in your workspace."
    )

### `file_tools.py:401-410`

Require argv[0] to be a bare program name. This is the fix for the verified Windows `git.bat` bypass.

    if not argv:
        return "Error: Empty command."

    # argv[0] must be a bare program name resolved from PATH. Without this,
    # an absolute path to a workspace file the user just wrote with write_file
    # (e.g. <workspace>/git.bat) passes the basename + suffix-strip check below
    # and executes — CreateProcess runs .bat/.cmd through cmd.exe. Verified.
    if os.path.isabs(argv[0]) or re.search(r"[\\/]", argv[0]):
        return (
            f"Error: '{argv[0]}' must be a bare program name, not a path. "
            "Commands are resolved from PATH only."
        )

    program = os.path.basename(argv[0]).lower()

### `file_tools.py:437-451`

Update the execute_command docstring — it is the tool description the LLM sees, so it must state the new rules.

    """Executes a command in the workspace directory and returns the output.
    The command runs with the workspace as the current working directory.

    Only commands whose program name is in the allowlist (see EXEC_ALLOWED_COMMANDS
    env var) are permitted, and the program name must be bare — no paths. Shell
    metacharacters are not interpreted — this runs via argv, not a shell.
    Arguments with '..' or absolute paths outside the workspace are rejected.
    The command runs with a scrubbed environment: only PATH and a small set of
    platform basics are passed through, so the bot's secrets are not visible.
    Depending on configuration this tool may be restricted to administrators.
    """

### `file_tools.py:456, 468-475`

Clamp timeout at both ends; insert the admin gate after the shlex parse and before allowlist validation.

    timeout = max(1, min(timeout, MAX_EXEC_TIMEOUT))
    ...
    except ValueError as e:
        ... (unchanged parse_error audit)

    # Placed AFTER the parse and BEFORE _validate_command_argv on purpose:
    # parsing is side-effect-free, and keeping it first preserves the existing
    # parse_error audit path (tests/test_file_tools.py:396).
    denial = _exec_denied_reason(user_id)
    if denial is not None:
        logger.warning("Denied exec for non-admin %s in %s", user_id, workspace)
        audit_log(
            "exec", user_id=user_id, workspace=workspace_root,
            argv=argv, result="denied", reason=denial,
        )
        return denial

    rejection = _validate_command_argv(argv, workspace)

### `file_tools.py:479-486`

The core fix: pass an explicit env to subprocess.run.

        result = subprocess.run(
            argv,
            shell=False,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_build_exec_env(),
        )

### `file_tools.py:479-499 (OPTIONAL step 6 — memory-bounded output)`

Replace capture_output pipes with temp files so EXEC_OUTPUT_TRUNCATE bounds RAM, not just LLM context. Also removes the pipe-drain hazard when run() kills a child on timeout. Requires `import tempfile` at the top.

    try:
        # capture_output=True buffers the child's entire stdout in RAM before
        # the truncation below ever runs — `python -c "print('x'*10**10)"` or
        # 30s of `yes` OOMs the bot. Redirect to temp files and read back only
        # the cap. Also avoids run() blocking on pipe drain after a timeout kill.
        with tempfile.TemporaryFile("w+b") as out_f, tempfile.TemporaryFile("w+b") as err_f:
            result = subprocess.run(
                argv, shell=False, cwd=workspace,
                stdout=out_f, stderr=err_f,
                timeout=timeout, env=_build_exec_env(),
            )
            out_f.seek(0)
            err_f.seek(0)
            cap = EXEC_OUTPUT_TRUNCATE
            stdout = out_f.read(cap + 1).decode("utf-8", "replace")
            stderr = err_f.read(cap + 1).decode("utf-8", "replace")

        output_parts = []
        if stdout:
            output_parts.append(stdout)
        if stderr:
            output_parts.append(f"STDERR:\n{stderr}")
        output_parts.append(f"Exit code: {result.returncode}")
        output = "\n".join(output_parts)
        if len(output) > EXEC_OUTPUT_TRUNCATE:
            output = output[:EXEC_OUTPUT_TRUNCATE] + "\n... (output truncated)"
        # NOTE: result.stdout / result.stderr are None on this path — the rest
        # of the function must use the local `stdout` / `stderr` names.

### `.env.example:67-69`

Move the exec-sandbox block out of the '----- Admin panel -----' section (where it currently sits by accident) into its own section, and document the three new knobs plus the two configurations.

# ----- execute_command sandbox -----
# Two supported configurations. Pick one.
#
# CONFIG A — interpreters allowed, admin-gated (repo default).
#   `python -c`, `node -e`, a user-written script run via `python x.py`, `make`,
#   and `git -c` are all arbitrary code execution as the bot's OS user. That is
#   accepted, and contained by restricting the tool to admins.
# EXEC_REQUIRE_ADMIN=true
# EXEC_ALLOWED_COMMANDS=ls,dir,pwd,echo,cat,type,head,tail,wc,python,python3,pip,pytest,node,npm,npx,git,go,cargo,make,grep,find,where
#
# CONFIG B — inspection only, open to every /workspace enable user.
#   Nothing on the list can evaluate caller-supplied code, so no admin gate is
#   needed. The agent can look around a workspace but cannot build or test in it.
# EXEC_REQUIRE_ADMIN=false
# EXEC_ALLOWED_COMMANDS=ls,dir,pwd,echo,cat,type,head,tail,wc,grep,where
#
# Environment variables passed through to child processes. Everything else —
# DISCORD_BOT_TOKEN, ADMIN_PANEL_PASSWORD, CHAT_COOKIE_SECRET, OLLAMA_HOST — is
# dropped. Defaults differ by platform; PATH is always forced in.
# Windows default:
# EXEC_ENV_PASSTHROUGH=PATH,PATHEXT,SYSTEMROOT,WINDIR,COMSPEC,SYSTEMDRIVE,TEMP,TMP,USERPROFILE,HOMEDRIVE,HOMEPATH,APPDATA,LOCALAPPDATA,NUMBER_OF_PROCESSORS,PROCESSOR_ARCHITECTURE,OS
# POSIX default:
# EXEC_ENV_PASSTHROUGH=PATH,HOME,TMPDIR,LANG,LC_ALL,TZ,TERM,USER,LOGNAME
#
# Hard ceiling on the per-command timeout (seconds). Requests are clamped to [1, this].
# EXEC_TIMEOUT_MAX=30

### `README.md:409-410`

Rewrite the troubleshooting entry to cover the bare-name rule, the admin gate, and the scrubbed env; add a pointer to the Config A/B choice.

**`execute_command` rejects a command**
The file-tools shell sandbox uses an allowlist. Allowed programs are listed in
`EXEC_ALLOWED_COMMANDS` (see `.env.example`, which documents the two supported
configurations). Three other rules can reject a command: the program name must
be bare (`python`, not `./python` or `C:\...\python.exe`); arguments may not
contain `..` or absolute paths outside the workspace; and if
`EXEC_REQUIRE_ADMIN=true` only admins may run programs at all. Shell features
(pipes, `&&`, redirects) are not interpreted. Commands run with a scrubbed
environment — only `PATH` plus a few platform basics (`EXEC_ENV_PASSTHROUGH`)
are passed through, so a script cannot read the bot's tokens from `os.environ`.

### `CHANGELOG.md:8-10 (insert a new `### Security` subsection under `## [Unreleased]`)`

Phase-style entry matching the existing prose style.

### Security
- **Phase 15 — `execute_command` containment.**
  - Child processes now get an explicit environment (`EXEC_ENV_PASSTHROUGH`, platform-aware default) instead of inheriting `os.environ`. Previously any user with a workspace could run `python -c "import os; print(os.environ['DISCORD_BOT_TOKEN'])"` and read the bot token, `ADMIN_PANEL_PASSWORD`, and `CHAT_COOKIE_SECRET` straight out of the chat reply. `PATH` is always present (falling back to `os.defpath`); `MISTERFRITZ_SANDBOX=1` is injected so workspace scripts can detect the sandbox.
  - `argv[0]` must now be a bare program name. The allowlist check strips `.exe/.cmd/.bat/.ps1` before matching, so an absolute path to a workspace-authored `git.bat` used to pass validation and execute through `cmd.exe` on Windows. Paths in `argv[0]` are rejected outright.
  - New `EXEC_REQUIRE_ADMIN` (default `true`) restricts *only* `execute_command` to `fritz_utils.is_admin` users. Phase 7b correctly opened `read/write/edit/search/list` to every workspace holder, but it also opened program execution — and everything on the default allowlist that can evaluate caller-supplied code (`python -c`, `node -e`, `make`, `git -c`, `find -exec`, or simply `write_file` + `python script.py`) is arbitrary code execution. Denials are audited as `result="denied"`.
  - `MAX_EXEC_TIMEOUT` moved to `fritz_utils.EXEC_TIMEOUT_MAX` (env-configurable, default 30) and requested timeouts are now clamped to `[1, max]` rather than only capped above.
  - The stale "this is ROOT_USER-gated" comment above the allowlist in `fritz_utils.py` has been corrected — it had been wrong since Phase 7b.

### `tests/test_file_tools.py:241-414`

Add four new test classes; fix two existing python-dependent tests and the shared _invoke_capturing_audit helper. Detail in testPlan.

# Minimal fix that repairs three audit tests at once (line 326):
    def _invoke_capturing_audit(self, fn, payload):
        with patch.object(fritz_utils, "ROOT_USER", _ROOT), \
             patch.object(file_tools, "audit_log") as audit:
            fn.invoke(payload, config=_config(self.tmp))
        return audit

# Helper for tests that legitimately need an interpreter, so they survive
# Config B (EXEC_ALLOWED_COMMANDS is a from-import — patch it on file_tools):
def _with_python_allowed():
    return patch.object(
        file_tools, "EXEC_ALLOWED_COMMANDS",
        frozenset(file_tools.EXEC_ALLOWED_COMMANDS | {"python"}),
    )

## Steps

1. Step 1 — DECISION GATE (do this first; everything else is mechanical). Get the owner to pick Config A or Config B. Config A = keep the current allowlist, set `_DEFAULT_EXEC_REQUIRE_ADMIN = "true"`. Config B = reduce `_DEFAULT_EXEC_ALLOWED` to `ls,dir,pwd,echo,cat,type,head,tail,wc,grep,where` and set `_DEFAULT_EXEC_REQUIRE_ADMIN = "false"`. The rest of the plan is identical either way — only the two default strings in fritz_utils.py differ. If no answer is available, ship Config A: it preserves today's agent capability, breaks the fewest tests, and the admin gate is a one-line env flip to undo.
2. Step 2 — fritz_utils.py: rewrite the `# File-tool sandbox` block (lines 171-191) per the code sketch. Fix the stale line-177 comment, add `EXEC_REQUIRE_ADMIN`, `EXEC_ENV_PASSTHROUGH`, `EXEC_TIMEOUT_MAX`. Commit alone — it is additive and cannot break anything yet.
3. Step 3 — file_tools.py: extend the `from fritz_utils import (...)` block (lines 13-18) and add `MAX_EXEC_TIMEOUT = EXEC_TIMEOUT_MAX` beside the existing `MAX_FILE_SIZE` alias at line 25. Delete the hardcoded `MAX_EXEC_TIMEOUT = 30` at lines 389-390. Add `_build_exec_env()` and `_exec_denied_reason()` in its place. Add `import tempfile` only if you are doing the optional Step 6.
4. Step 4 — file_tools.py: wire in the containment. (a) `subprocess.run(..., env=_build_exec_env())` at line 479. (b) `timeout = max(1, min(timeout, MAX_EXEC_TIMEOUT))` at line 456. (c) the `_exec_denied_reason` gate + audit between the shlex block and `_validate_command_argv` (after line 466, before line 468). (d) the bare-name check at the top of `_validate_command_argv` (after line 402). (e) refresh the `execute_command` docstring at lines 437-451 — it is the tool description the model reads.
5. Step 5 — tests: add `TestExecEnvScrub`, `TestExecAdminGate`, `TestExecArgv0IsBareName`, `TestExecTimeoutClamp`; patch `_invoke_capturing_audit` (line 326) to also patch `fritz_utils.ROOT_USER`; wrap `test_captures_stderr` (line 254) and `test_timeout_returns_timeout_message` (line 264) in the python-allowlist patch so they survive Config B. Run `pytest tests/test_file_tools.py -v` — everything green before moving on.
6. Step 6 (OPTIONAL, separately committable, defer freely) — memory-bounded output: swap `capture_output=True` for `tempfile.TemporaryFile` redirection so `EXEC_OUTPUT_TRUNCATE` caps RAM rather than only LLM context. Do this only if the owner cares about a `python -c "print('x'*10**10)"` OOM; it is orthogonal to the env-scrub goal and touches the output-assembly block at lines 488-505 (note: `result.stdout` becomes `None` on that path).
7. Step 7 — docs: `.env.example` new `----- execute_command sandbox -----` section (moving the stray `EXEC_ALLOWED_COMMANDS` lines 67-69 out of the admin-panel section); README.md:409-410 troubleshooting rewrite; CHANGELOG.md new `### Security` subsection under `## [Unreleased]`.
8. Step 8 — verify: `ruff check .` and `pytest tests/ --cov=. --cov-fail-under=60` (matching .github/workflows/ci.yml:29-39). Then run the manual exploit re-check from the testPlan and confirm the child now prints `None None`.

## Config and env changes

- EXEC_REQUIRE_ADMIN (new) — default "true" under Config A, "false" under Config B. Restricts only `execute_command` to `fritz_utils.is_admin` users; the other five file tools stay open to any workspace holder. Falsy values: 0/false/no/off/empty.
- EXEC_ENV_PASSTHROUGH (new) — comma-separated, case-insensitive. Windows default: PATH,PATHEXT,SYSTEMROOT,WINDIR,COMSPEC,SYSTEMDRIVE,TEMP,TMP,USERPROFILE,HOMEDRIVE,HOMEPATH,APPDATA,LOCALAPPDATA,NUMBER_OF_PROCESSORS,PROCESSOR_ARCHITECTURE,OS. POSIX default: PATH,HOME,TMPDIR,LANG,LC_ALL,TZ,TERM,USER,LOGNAME. PATH is force-added (os.defpath) even if the operator omits it.
- EXEC_TIMEOUT_MAX (new) — default "30". Replaces the hardcoded `file_tools.MAX_EXEC_TIMEOUT`, which stays as a backwards-compatible alias.
- EXEC_ALLOWED_COMMANDS (existing, default may change) — under Config B the shipped default drops python, python3, pip, pytest, node, npm, npx, go, cargo, make, git, find, and cd. This is a behaviour change for anyone relying on the default; call it out in the CHANGELOG as such.
- MISTERFRITZ_SANDBOX=1 is injected into every child environment (not read by the bot; a marker for workspace scripts).
- .env.example must document all of the above per CONTRIBUTING.md:86. No CI env changes needed — .github/workflows/ci.yml:40-48 already sets ROOT_USER=ci-root, which the tests override via patch.

## Tests
### New

- tests/test_file_tools.py::TestExecEnvScrub::test_child_cannot_read_bot_secrets — the direct regression for the verified exploit. `patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "leak-sentinel-abc", "ADMIN_PANEL_PASSWORD": "pw-sentinel-xyz", "CHAT_COOKIE_SECRET": "cookie-sentinel"})`, patch `file_tools.EXEC_ALLOWED_COMMANDS` to include "python" and `file_tools.EXEC_REQUIRE_ADMIN` to False, then invoke `python -c "import os;print(os.environ.get('DISCORD_BOT_TOKEN'), os.environ.get('ADMIN_PANEL_PASSWORD'), os.environ.get('CHAT_COOKIE_SECRET'))"`. Assert none of the three sentinels appear in the result and that "None None None" does.
- tests/test_file_tools.py::TestExecEnvScrub::test_build_exec_env_keeps_only_allowlisted — patch `file_tools.EXEC_ENV_PASSTHROUGH` to frozenset({"PATH"}) and `os.environ` with a junk var; assert `set(_build_exec_env()) == {"PATH", "MISTERFRITZ_SANDBOX"}`.
- tests/test_file_tools.py::TestExecEnvScrub::test_build_exec_env_forces_path_when_absent — patch `file_tools.EXEC_ENV_PASSTHROUGH` to frozenset(); assert `_build_exec_env()["PATH"] == os.defpath`.
- tests/test_file_tools.py::TestExecEnvScrub::test_scrubbed_child_still_runs — invoke `echo scrub_ok` (the platform-portable command already used by test_runs_simple_command) and assert "Exit code: 0"; proves PATH passthrough did not break program resolution.
- tests/test_file_tools.py::TestExecAdminGate::test_non_admin_is_denied_when_gate_on — patch `file_tools.EXEC_REQUIRE_ADMIN=True` + `fritz_utils.ROOT_USER=_ROOT`; invoke `echo hi` with `_config(self.tmp, user="regular_user")`; assert "administrators" in result and "hi" not in result.
- tests/test_file_tools.py::TestExecAdminGate::test_admin_is_allowed_when_gate_on — same patches, user=_ROOT; assert "Exit code: 0".
- tests/test_file_tools.py::TestExecAdminGate::test_non_admin_allowed_when_gate_off — patch `file_tools.EXEC_REQUIRE_ADMIN=False`; assert the command runs. This is the Config B path.
- tests/test_file_tools.py::TestExecAdminGate::test_denial_is_audited — patch `file_tools.audit_log`; assert `args[0] == "exec"` and `kwargs["result"] == "denied"`.
- tests/test_file_tools.py::TestExecAdminGate::test_other_file_tools_unaffected_by_gate — with the gate on and a non-admin user, `write_file` then `read_file` still succeed. Guards the Phase 7b product promise.
- tests/test_file_tools.py::TestExecArgv0IsBareName::test_absolute_path_to_workspace_script_rejected — THE regression test for the second verified hole. `write_file.invoke({"path": "git.bat", "content": "@echo off\r\necho PWNED_VIA_BAT\r\n"})`, then `execute_command.invoke({"command": shlex.quote-safe absolute path to that file})`. Assert "bare program name" in result and "PWNED_VIA_BAT" not in result. Note: on Windows use forward slashes or a raw string so shlex.split(posix=True) does not eat the backslashes — write the command as `f'"{path}"'`.
- tests/test_file_tools.py::TestExecArgv0IsBareName::test_relative_path_argv0_rejected — `./python -c "print(1)"` → "bare program name".
- tests/test_file_tools.py::TestExecArgv0IsBareName::test_bare_name_still_accepted — `echo ok` still runs; proves the check did not over-reject.
- tests/test_file_tools.py::TestExecTimeoutClamp::test_zero_timeout_is_clamped_to_one — invoke `echo t` with timeout=0; assert "Exit code: 0" (before the fix, subprocess.run(timeout=0) raises TimeoutExpired immediately).
- tests/test_file_tools.py::TestExecTimeoutClamp::test_negative_timeout_is_clamped — timeout=-5, same assertion.

### Existing tests affected

- tests/test_file_tools.py::TestFileToolAuditLog::_invoke_capturing_audit (helper, line 326) — MUST be updated. It does not patch `fritz_utils.ROOT_USER`, so under `EXEC_REQUIRE_ADMIN=true` the `_ROOT` user is not an admin (CI sets ROOT_USER="ci-root" per ci.yml:44) and `execute_command` returns the denial. Add `patch.object(fritz_utils, "ROOT_USER", _ROOT)` alongside the existing `patch.object(file_tools, "audit_log")`.
- tests/test_file_tools.py::TestFileToolAuditLog::test_execute_command_success_records_argv_and_exit_code (line 373) — WILL FAIL under Config A without the helper fix: asserts `kwargs["result"] == "ok"` and `kwargs["exit_code"] == 0`, gets `result="denied"` and no exit_code (KeyError).
- tests/test_file_tools.py::TestFileToolAuditLog::test_execute_command_rejection_is_audited (line 385) — WILL FAIL under Config A without the helper fix: asserts `"allowlist" in kwargs["reason"].lower()`, gets the admin-denial reason instead.
- tests/test_file_tools.py::TestFileToolAuditLog::test_execute_command_parse_error_is_audited (line 396) — SURVIVES, but only because the plan deliberately places the admin gate after the shlex parse. If you move the gate earlier this test starts asserting "parse_error" against "denied". Re-run it after any reordering.
- tests/test_file_tools.py::TestExecuteCommand::test_captures_stderr (line 254) — WILL FAIL under Config B: the command is `python -c "import sys; sys.stderr.write('err_msg')"` and python is no longer allowlisted, so the result is the allowlist rejection and `assertIn("err_msg", result)` fails. Wrap in `patch.object(file_tools, "EXEC_ALLOWED_COMMANDS", frozenset(file_tools.EXEC_ALLOWED_COMMANDS | {"python"}))`. Passes unchanged under Config A (it already patches fritz_utils.ROOT_USER to _ROOT, so the admin gate lets it through).
- tests/test_file_tools.py::TestExecuteCommand::test_timeout_returns_timeout_message (line 264) — same `python -c` dependency, same failure mode under Config B, same fix. Also confirm the timeout clamp change did not affect it: it passes timeout=1, which `max(1, min(1, 30))` leaves at 1.
- tests/test_file_tools.py::TestExecuteCommand::test_runs_simple_command (245), test_max_timeout_capped (273), test_shell_metacharacters_not_interpreted (308) — all use `echo`. Unaffected by the env scrub (PATH is preserved) and by the bare-name rule. Caveat: these already fail on a Windows dev box (there is no `echo.exe`; `echo` is a cmd.exe builtin) and pass only on the ubuntu-latest CI runner. Do not mistake that pre-existing local failure for a regression from this change.
- tests/test_file_tools.py::TestAuthorization (lines 30-57) — unaffected. The admin gate lives inside `execute_command`, not in `_authorize`, so `test_user_with_workspace_is_allowed` (list_directory as a non-admin) still passes. This is deliberate: it is the Phase 7b behaviour the gate must not regress.
- tests/test_fritz_utils.py — no changes required. TestConstantDefaults (line 99) does not enumerate the sandbox constants, and no test asserts the contents of EXEC_ALLOWED_COMMANDS. Optionally add a `test_exec_env_passthrough_includes_path` there for symmetry with the other default-value tests.
- No other test file touches execute_command — verified by grepping `execute_command|get_file_tools` across tests/.

### Manual verification

- Re-run the exploit exactly as it was verified. From the repo root: `DISCORD_BOT_TOKEN=leak-me-123 ADMIN_PANEL_PASSWORD=hunter2 python -c "import sys,tempfile;sys.path.insert(0,'.');from file_tools import execute_command;ws=tempfile.mkdtemp();print(execute_command.invoke({'command':'python -c \"import os;print(os.environ.get(1st secret), os.environ.get(2nd secret))\"','timeout':10}, config={'configurable':{},'metadata':{'user_id':'regular_user','workspace_root':ws}}))"`. Before: `leak-me-123 hunter2 / Exit code: 0`. After Config A: the administrators refusal. After Config B: the allowlist rejection. With the gate/allowlist patched open: `None None`.
- Windows batch bypass, live: `/workspace enable` as a non-admin, ask Fritz to write `git.bat` containing `@echo off` + `echo PWNED`, then ask it to run that file by absolute path. Expect "must be a bare program name, not a path."
- Smoke the tools that legitimately need the passthrough vars under Config A on the real host: `git status`, `python -c "import tempfile;print(tempfile.gettempdir())"` (must land in the user TEMP, not `C:\WINDOWS\Temp` — that is what proves TEMP/TMP passthrough works), and `npm --version` if node is installed. If any of these break, the fix is to add the missing var to EXEC_ENV_PASSTHROUGH, not to revert the scrub.
- Confirm the LLM behaves sensibly on a denial: as a non-admin under Config A, ask Fritz to "run the tests in my workspace" and check it relays the refusal rather than looping on the tool. The tool is still advertised in the prompt (see the deferral note in risks).
- Tail `audit.log` (AUDIT_LOG_PATH) after the denial and confirm one NDJSON line with `"event": "exec"` and `"result": "denied"`.

## Risks

- The scrubbed env breaks a legitimate command on the owner's host. Most likely culprits: a tool that needs a var not on the list (GIT_*, GOPATH, CARGO_HOME, NODE_PATH, VIRTUAL_ENV, JAVA_HOME, HTTP_PROXY/HTTPS_PROXY/NO_PROXY, SSL_CERT_FILE, PIP_INDEX_URL). Detection: the command works from a terminal but fails via the bot with a 'not found' / 'cannot find home' / TLS error. Mitigation: EXEC_ENV_PASSTHROUGH is an env var precisely so this is a config fix, not a code change. Do NOT add proxy/credential-bearing vars back reflexively — HTTP_PROXY with embedded credentials and PIP_INDEX_URL with a token are exactly the kind of secret this change exists to withhold.
- Windows: the empirical check on this box showed a PATH-only child still imported socket/ssl/tempfile fine, so the SYSTEMROOT/COMSPEC/PATHEXT entries are precautionary rather than proven-required for the simple case. They are cheap and non-secret; keeping them costs nothing and removing them risks obscure DLL-load and `.cmd`-shim failures. Do not 'simplify' the list down to PATH.
- The admin gate (Config A default) is a visible product regression for non-admin Discord users who already use `/workspace enable` + build/test workflows. Detection: user complaints, or a spike of `result="denied"` lines in audit.log. Mitigation: `EXEC_REQUIRE_ADMIN=false` reverts it instantly without a deploy, though that re-accepts full RCE for those users.
- Config B silently removes capability the agent's system prompt still advertises (`get_file_tools_description` at file_tools.py:551 says 'Use for running code, tests, builds, git commands'). The model will keep proposing `pytest` / `npm test` and keep getting rejected, burning turns. Detection: repeated allowlist rejections in audit.log for the same user. Mitigation: if Config B is chosen, also reword that description string — a one-line change worth doing in the same commit.
- The bare-name rule could over-reject a legitimate workflow that invokes a tool by path (e.g. `./node_modules/.bin/jest`). Nothing in the default allowlist matches such a path, so this is unlikely, but detection is a rejection mentioning 'bare program name' for something that used to work. Mitigation: the correct fix is to add the program to EXEC_ALLOWED_COMMANDS and put its directory on the bot's PATH, not to relax the rule.
- Timeout still only kills the direct child, not its descendants. `subprocess.run` has no process-group semantics, so `python -c "subprocess.Popen(['sleep','9999'])"` (Config A) leaves an orphan running forever, and with pipes the run() call can block draining a pipe an orphan still holds. This change does not fix that; see rollback/deferrals.
- Merge risk: there is a git worktree at .claude/worktrees/nifty-hoover-6fe3d6/ containing an identical copy of file_tools.py and fritz_utils.py. Edit the repo-root files only; do not let a search-and-replace touch the worktree copy.

## Rollback
"No feature flag is needed beyond the env vars this item introduces, which are themselves the rollback: `EXEC_REQUIRE_ADMIN=false` restores Phase 7b's open access, and `EXEC_ALLOWED_COMMANDS=<old default>` restores the pre-change allowlist. Both take effect on restart, no code deploy. The env scrub itself is deliberately NOT flag-guarded — it is the security fix and there is no legitimate reason to want the bot token in a subprocess; if a command genuinely needs a variable, add it to EXEC_ENV_PASSTHROUGH rather than reverting. Full code revert is a clean `git revert` of the file_tools.py + fritz_utils.py commits: the change is additive (two new helpers, one new subprocess kwarg, one new validator guard) and touches no persisted state, no DB schema, and no agent graph structure. The optional Step 6 (temp-file output capture) is a separate commit so it can be reverted independently if a platform quirk shows up.\n\nEXPLICITLY DEFERRED — tempting bigger refactors that are out of scope here:\n(1) Real OS-level sandboxing (dedicated low-privilege user, container-per-exec, seccomp/AppContainer, network namespace). This is the only thing that makes Config A genuinely 'contained' rather than 'contained against env-var exfiltration'. With Config A a user can still read `~/.ssh`, `../.env`, and `fritz.db`, and can make outbound network calls. Say so honestly in the CHANGELOG rather than overclaiming.\n(2) Process-group kill on timeout (`start_new_session=True` / `CREATE_NEW_PROCESS_GROUP` + Popen + killpg). Requires replacing `subprocess.run` with a Popen/communicate loop — a real rewrite of lines 478-505 and its own set of platform tests.\n(3) Filtering `execute_command` out of the toolset at registration time (threading `is_admin(user_id)` into `get_file_tools_description()` at file_tools.py:543 and `get_conversation_tools_description()` at agent_tools.py:516-541, called from mister_fritz.py:315-321 and 544). Cleaner for prompt hygiene and saves a wasted tool call, but it changes three function signatures across two modules and would need test_agent_tools/test_mister_fritz updates. The in-tool check is the security boundary; registration-time filtering is only an optimisation. Do it as a follow-up.\n(4) A dangerous-flag denylist (`-c`, `-e`, `--eval`, `find -exec`, `git -c`). Deliberately NOT in this plan: it is bypassable in one step via `write_file(\"x.py\", ...)` + `python x.py`, so it would create a false sense of safety while breaking legitimate use. Argument inspection is only worth building if the owner later wants a per-program subcommand allowlist (e.g. git: status/log/diff/show only), which is a much larger design."

## Open questions for you to decide

- THE decision: Config A or Config B? Config A keeps the agent able to build and test in a workspace, accepts that this is RCE, and contains it by restricting the tool to admins — costs non-admin users the capability. Config B keeps `execute_command` open to everyone but reduces it to filesystem inspection — costs everyone the build/test capability. There is no third option that keeps interpreters and stays safe for untrusted users without OS-level sandboxing. My recommendation if you want one: Config A, because Phase 7b's per-user workspaces were about giving people a place to keep files, not a shared build farm, and `EXEC_REQUIRE_ADMIN=false` is a one-line escape hatch.
- Should `git` be re-added under Config B? It is by far the most useful non-interpreter command, but `git -c core.pager=<cmd> log` and `git config alias.x '!sh -c ...'` are code execution, so adding it back collapses Config B into Config A for anyone who thinks to try. If the answer is yes, the honest move is Config A with the admin gate rather than a Config B that pretends.
- Who counts as an admin here — is `ADMIN_USERS` (fritz_utils.py:214-218) the right blast radius for arbitrary code execution as the bot's OS user? It currently grants document upload, `/workspace set`, and admin-panel powers. Executing arbitrary code is a strictly larger privilege than any of those. If the ADMIN_USERS list contains anyone the owner would not hand an unrestricted shell to, this item needs a separate `EXEC_ADMIN_USERS` rather than reusing `is_admin`.
- Should `/workspace enable` (bot_commands.py:497-517) stay ungated? It is currently open to every Discord user in every guild the bot joins, which means unbounded disk use under WORKSPACES_ROOT and — today — the exec path. Gating it, or capping workspace size, is a separate item but the decision interacts with this one: if `/workspace enable` becomes admin-gated, `EXEC_REQUIRE_ADMIN` matters much less.
- Is the optional Step 6 (bounded output) wanted now or later? It is the difference between `EXEC_OUTPUT_TRUNCATE` being a context-window courtesy and being an actual memory guard. Cheap (~15 lines) but orthogonal to the env-scrub goal.
- Cannot be settled statically: whether the scrubbed environment breaks a real command on the owner's actual host. The experiment that settles it is the third manual-verification bullet — run `git status`, `python -c "import tempfile;print(tempfile.gettempdir())"`, and `npm --version` through the bot on the production host after deploying, and read the failures. Anything that breaks is an EXEC_ENV_PASSTHROUGH addition, not a code bug.
