# 9. Stable identity and per-channel threads

[← back to index](README.md)

**Effort:** XL (>3 days)  
**Depends on:** nothing

## Goal
Today every store in MisterFritz is keyed off a mutable, platform-ambiguous display string, and three different transformations of that string are applied at three different layers, so write paths and delete paths disagree. After this item, there is exactly one identity transformation — `fritz_utils.canonical_user_id(platform, raw_id)` producing `discord:<snowflake>` / `telegram:<id>` / `web:<name>` — minted at each of the four adapter boundaries (`main_discord.on_message`, `bot_commands.FritzCommands`, `main_telegram.handle_text/handle_voice`, `admin_panel` chat routes) and passed verbatim into Chroma namespaces, LangGraph `thread_id`, the `schedules` and `workspaces` tables, and the admin gate. A Discord rename no longer orphans memories/workspace/schedules/admin rights; `/forget memories` actually deletes for usernames containing punctuation; and `thread_id` becomes identity+channel (behind a flag) so a conversation in #general no longer bleeds into DMs. Human-readable names survive in prompts and the admin panel via a new `user_aliases` table rather than by being the key. A one-time `migrate_identity.py` (dry-run by default) rewrites existing rows in all five stores.

## Definition of done

- [ ] `fritz_utils.canonical_user_id()` is the only place in the repo that transforms a platform id into a storage key. `grep -n 'a-zA-Z0-9' mister_fritz.py privacy.py admin_panel.py` returns no user-id-stripping regex — the three duplicates at `mister_fritz.py:536`, `privacy.py:25`, and `admin_panel.py:353` are gone.
- [ ] All four adapters mint a namespaced id: `main_discord.on_message` from `ctx.author.id`, `bot_commands._identity` from `interaction.user.id`, `main_telegram` from `update.effective_user.id`, `admin_panel._chat_identity` from the cookie name. Renaming a Discord account changes nothing in any store.
- [ ] `fritz_utils.is_admin(user_id, display_name=None)` gates on the canonical id; the display-name path is reachable only when `ADMIN_LEGACY_NAME_MATCH` is true, and `validate_config()` logs a warning naming every non-canonical entry in `ROOT_USER`/`ADMIN_USERS`.
- [ ] `privacy.forget_memories(uid)` and `agent_tools.add_memory(uid, ...)` use byte-identical namespace tuples — proven by a test that writes then deletes with a punctuated id and asserts the count is non-zero. The `/forget memories` bug from the audit is closed.
- [ ] `privacy.forget_conversation` removes both `thread_id = <identity>` and every `thread_id LIKE '<identity>#%'` row, and does not touch a sibling identity that shares a prefix (`discord:1` vs `discord:10`).
- [ ] Every filesystem path and Discord attachment filename that embeds an identity goes through `fritz_utils.safe_user_token()`; no file created on Windows contains a `:`.
- [ ] `identity_store` supplies the human name to the prompt builder (`get_source_info`), the admin user list, and `scheduler._run_task`. A user whose id is `discord:123456789` is still addressed by name in Fritz's replies.
- [ ] `python migrate_identity.py --dry-run` runs against the live `fritz.db` + `chroma_store/` with Ollama stopped, mutates nothing, reports per-store counts, tolerates the missing `workspaces` table, and flags any discovered key lacking a `--map` entry. `--apply` is idempotent, writes its mapping JSON before mutating, and `--reverse` restores the original state exactly.
- [ ] The Chroma migration rewrites `metadata['namespace']` in place without re-embedding — a sampled document's embedding vector is byte-identical before and after.
- [ ] With `THREADS_PER_CHANNEL=false` (default), upgrading + migrating preserves the existing conversation: the first post-migration DM still has prior context. With it `true`, a guild channel and a DM maintain independent contexts.
- [ ] `IDENTITY_LINKS` restores the web↔Discord shared thread that `README.md:318` currently promises, and the README says so instead of claiming it happens automatically.
- [ ] `ruff check .` is clean and `pytest tests/` passes with coverage at or above the existing 60% CI gate. Every test named in `existingTestsAffected` is updated (not deleted-to-green), and `tests/test_scheduler.py` and `tests/test_workspace_store.py` pass **without modification** — proof that the store layer stayed identity-agnostic.
- [ ] `.env.example` documents all four knobs, `CHANGELOG.md` has an `[Unreleased]` entry in the existing phase style, and `README.md` lines 196-197 and 318 are corrected.

## Current state (verified against the working tree)
Verified by reading the files in this session. Corrections to the audit findings are marked **[CORRECTION]**.

**Identity minting (4 adapters, 3 different shapes):**
- `main_discord.py:135` — `author = ctx.author.name`; that same string is used as the agent identity (`:199` `workspace_store.get(author)`, `:195-202` `ask_stuff(..., author, ...)`) *and* as a filesystem path component at `:161` and `:170` (`os.path.join("temp_images", f"{author}_{attachment.id}_{attachment.filename}")`).
- `bot_commands.py` — `interaction.user.name` at `:94` (admin gate), `:135` (schedule owner), `:160`, `:180`, `:227`, `:237`, `:248`, `:261`, `:280`, `:308`/`:313`/`:318` (cards), `:395` (`/voice` → `ask_stuff`), `:485`, `:505`, `:523`, `:543`/`:550` (workspace).
- `main_telegram.py:21` and `:36` — `str(update.effective_user.id)`. Already immutable but not namespaced, so it shares a keyspace with Discord names.
- `admin_panel.py:404` — `chat_login` sanitises the self-asserted username with `re.sub(r"[^a-zA-Z0-9_-]", "", username)[:64]` (note: **keeps** `_` and `-`), signs it into the cookie, and `_chat_user()` (`:307-310`) returns it.

**The single transformation, and the three places it is duplicated / diverges:**
- `mister_fritz.py:536` — `user_id_clean = _re.sub(r'[^a-zA-Z0-9]', '', user_id)`; `:549` `"configurable": {"user_id": user_id_clean, "thread_id": user_id_clean}` and `:551-552` the same in `metadata`. One global thread per user across every channel *and* platform.
- `admin_panel.py:353` — `_load_chat_history` re-derives it independently: `thread_id = re.sub(r"[^a-zA-Z0-9]", "", user_id)`.
- `privacy.py:22-25` — `_sanitise_thread_id` re-derives it a third time.
- `workspace_store.py:46-50` — `_safe_user_dir` uses a *fourth* charset (`[^a-zA-Z0-9_-]` → `_`) for the directory name only; the DB key at `:60`/`:79`/`:105` stays raw.

**[CORRECTION] to the audit's `/forget memories` finding.** The audit says "privacy.py:22-25 rebuilds the namespace from the raw name". That is not what the code does. `privacy._sanitise_thread_id` is used **only** by `forget_conversation` (`:69`) and `count_conversation_checkpoints` (`:95`) — and those two are *correct*, because they match `ask_stuff`'s stripped `thread_id`. The actual bug is one line lower: `privacy.forget_memories:41` calls `get_default_chroma_store().delete_namespace((str(user_id),))` with the **raw** id, while the write path `agent_tools.add_memory:90` does `put((str(user_id),), ...)` with the **stripped** id (because `mister_fritz.executor:292` reads `user_id` out of metadata, which `ask_stuff:551` populated with `user_id_clean`). `privacy.export_memories:53` has the identical raw/stripped mismatch, which is why the admin panel's memory counts are also wrong. Net effect is what the audit claims — write and delete disagree — but the fix is in `forget_memories`/`export_memories`, not in `_sanitise_thread_id`.

**[CORRECTION] on decks.** `cards.py:27` `USER_DECKS = {}` is a plain in-process dict, keyed by `interaction.user.name`, never persisted. There is nothing to migrate; decks simply reset on restart today and will continue to. Drop "decks" from the migration scope.

**Store key inventory (what the migration must actually touch), verified against the live `fritz.db` and `chroma_store/`:**
- `checkpoints.thread_id` / `writes.thread_id` — stripped id. Live DB currently holds exactly one thread: `divora`.
- `schedules.user_id` (`scheduler.py:154`, index at `:56`) — raw name.
- `workspaces.user_id` (`workspace_store.py:36-41`, `:79`) — raw name. **The table does not exist yet in the live `fritz.db`** (never created — `_init_db` runs lazily on first `/workspace` use), so the migration must tolerate its absence.
- Chroma collection `langchain_store` in `./chroma_store` — metadata key `namespace` = stripped id (11 docs, all `namespace='divora'`). Confirmed reachable via raw `chromadb.PersistentClient(path='chroma_store').get_collection('langchain_store')` with `include=['metadatas','documents','embeddings']` and **no Ollama running** — so metadata can be rewritten in place with `collection.update(ids=..., metadatas=...)` without re-embedding.
- Chroma document **ids**: memory ids are `uuid4` (`agent_tools.py:90`) — identity-free. Only the profile doc has an identity-derived id: `agent_tools.py:167` `get(f"profile_{user_id}")` and `:204` `put(..., f"profile_{user_id}", ...)`. That one id per user needs a delete+re-add carrying the existing embedding vector.
- `store` table (`storage.SQLiteStore`, `namespace` column) — wired as LangGraph's `store` at `mister_fritz.py:640` but nothing in the codebase writes to it; live table is empty. Include in the migration as a cheap `UPDATE` for completeness.

**Admin gate:** `fritz_utils.is_admin:221-231` is pure string equality against `ROOT_USER` (`:210`) and `ADMIN_USERS` (`:214-218`). Callers: `bot_commands._require_admin:94` (`interaction.user.name`), `admin_panel.py:394` (`chat_page`, cookie name) and `:732` (`chat_upload_document`, cookie name). A Discord rename revokes admin; anyone who claims the owner's name at `/chat/login` gains document-upload rights today.

**Admin panel user list:** `_collect_users:109-127` unions `schedule_manager.list_all_schedules()` (raw-keyed) with `workspace_store.list_all()` (raw-keyed) and `users_list:147-160` hands those raw ids to `privacy.export_memories` — which queries the raw namespace, so memory counts read 0 for any user whose name contains punctuation. Audit finding confirmed.

**Migration precedent:** `migrate_db.py` exists and is the template to follow (`_table_exists`, `_migrate_table`, `main()` returning an exit code). Its f-string `ATTACH DATABASE '{source}'` at `:144` and `:158` is a SQL-injection-shaped pattern; the new script must not copy it — the new script does not attach anything, so this is easy to avoid.

**Documented behaviour this change breaks:** `README.md:318` explicitly promises "Use the **same Discord username** you normally use and you'll continue the same conversation thread — the web UI and Discord share the LangGraph conversation state". Namespacing (`web:alice` ≠ `discord:1234`) removes that. It needs an explicit link mechanism, not silence.

## Change sites

### `fritz_utils.py:210-231 (insert new block after 218; rewrite is_admin at 221-231)`

Add the canonical identity helpers — the single transformation. Extend is_admin to take an optional display_name so legacy ROOT_USER=<discord username> configs keep working for one release behind a flag.

# ── Identity ─────────────────────────────────────────────────────────────
# One transformation, one place. Everything downstream (Chroma namespace,
# LangGraph thread_id, schedules.user_id, workspaces.user_id, admin gate)
# consumes the output of canonical_user_id() verbatim.

import re as _re

IDENTITY_SEPARATOR = ":"
KNOWN_PLATFORMS: frozenset[str] = frozenset({"discord", "telegram", "web", "local"})
_IDENT_STRIP_RE = _re.compile(r"[^a-zA-Z0-9_-]")   # same charset admin_panel.chat_login already uses
_TOKEN_SAFE_RE = _re.compile(r"[^a-zA-Z0-9_-]")


def canonical_user_id(platform: str, raw_id) -> str:
    """Return the stable, namespaced identity for a user: '<platform>:<id>'.

    platform: 'discord' | 'telegram' | 'web' | 'local'
    raw_id:   the platform's IMMUTABLE id where one exists (Discord/Telegram
              numeric snowflake), otherwise the self-asserted name (web).
    """
    plat = (platform or "local").strip().lower()
    ident = _IDENT_STRIP_RE.sub("", str(raw_id or "").strip())[:64]
    if not ident:
        raise ValueError(f"cannot build a canonical id from platform={platform!r} raw_id={raw_id!r}")
    return f"{plat}{IDENTITY_SEPARATOR}{ident}"


def split_user_id(user_id: str | None) -> tuple[str | None, str]:
    """('discord', '123') for canonical ids; (None, <as-is>) for legacy ones."""
    if not user_id or IDENTITY_SEPARATOR not in user_id:
        return None, (user_id or "")
    plat, _, ident = user_id.partition(IDENTITY_SEPARATOR)
    plat = plat.lower()
    return (plat, ident) if plat in KNOWN_PLATFORMS else (None, user_id)


def is_canonical_user_id(user_id: str | None) -> bool:
    return split_user_id(user_id)[0] is not None


def safe_user_token(user_id: str | None) -> str:
    """Filesystem- and URL-safe rendering of an identity. ':' is illegal in
    Windows filenames, so every path that embeds an identity MUST go through
    this: 'discord:123' -> 'discord_123'."""
    return _TOKEN_SAFE_RE.sub("_", user_id or "") or "anonymous"


def thread_id_for(user_id: str, channel_key: str | None = None) -> str:
    """LangGraph thread id. With THREADS_PER_CHANNEL off (default) this is the
    identity alone, preserving today's one-thread-per-user behaviour."""
    if not THREADS_PER_CHANNEL or not channel_key:
        return user_id
    return f"{user_id}#{safe_user_token(str(channel_key))}"


# Per-channel conversation threads. Turning this on BRANCHES every existing
# conversation (see CHANGELOG / README): the identity-only thread stays in the
# DB untouched, but new messages start a fresh per-channel thread.
THREADS_PER_CHANNEL: bool = os.environ.get("THREADS_PER_CHANNEL", "false").lower() in ("1", "true", "yes")

# Transitional: match ROOT_USER / ADMIN_USERS against the human display name as
# well as the canonical id. Lets an existing ROOT_USER=<discord username> keep
# working across the upgrade. Set to false once ROOT_USER is a canonical id --
# while true, anyone who takes the owner's freed-up username inherits admin.
ADMIN_LEGACY_NAME_MATCH: bool = os.environ.get("ADMIN_LEGACY_NAME_MATCH", "true").lower() in ("1", "true", "yes")

# Explicit cross-platform identity links, e.g. so web:alice and discord:123
# share one conversation thread and one memory namespace.
#   IDENTITY_LINKS=web:alice=discord:123456789,web:bob=discord:987654321
def _parse_identity_links() -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in os.environ.get("IDENTITY_LINKS", "").split(","):
        alias, sep, primary = pair.strip().partition("=")
        if sep and alias.strip() and primary.strip():
            out[alias.strip()] = primary.strip()
    return out

IDENTITY_LINKS: dict[str, str] = _parse_identity_links()


def resolve_identity(user_id: str) -> str:
    """Follow IDENTITY_LINKS one hop. Deliberately not transitive."""
    return IDENTITY_LINKS.get(user_id, user_id)


# --- rewrite of the existing is_admin (currently lines 221-231) ---
def is_admin(user_id: str | None, display_name: str | None = None) -> bool:
    """True if the caller is ROOT_USER or in ADMIN_USERS.

    user_id should be a canonical id. display_name is only consulted when
    ADMIN_LEGACY_NAME_MATCH is on, so pre-migration configs keep working.
    Reads module scope at call time so tests can patch without re-importing.
    """
    candidates = [user_id]
    if ADMIN_LEGACY_NAME_MATCH and display_name:
        candidates.append(display_name)
    for candidate in candidates:
        if not candidate:
            continue
        if ROOT_USER and candidate == ROOT_USER:
            return True
        if candidate in ADMIN_USERS:
            return True
    return False

### `fritz_utils.py:273-286`

validate_config() warns loudly when the configured admins are still legacy display names, and tells the owner exactly what to change.

def validate_config() -> None:
    missing = []
    if not DISCORD_BOT_TOKEN:
        missing.append("DISCORD_BOT_TOKEN  (or 'discord_bot_token' in config.json)")
    if not ROOT_USER:
        missing.append("ROOT_USER  (or 'root_user' in config.json)")
    if missing:
        raise RuntimeError(...)   # unchanged

    legacy = [u for u in ([ROOT_USER] if ROOT_USER else []) + sorted(ADMIN_USERS)
              if not is_canonical_user_id(u)]
    if legacy:
        logging.getLogger(__name__).warning(
            "ROOT_USER/ADMIN_USERS still use display names (%s). Admin rights are "
            "currently matched on a mutable username. Run `python migrate_identity.py "
            "--dry-run` to get the canonical ids, set e.g. ROOT_USER=discord:123456789, "
            "then set ADMIN_LEGACY_NAME_MATCH=false.",
            ", ".join(legacy),
        )

### `identity_store.py:new file`

New SQLite-backed display-name map so prompts and the admin panel can still show a human name once the key is an opaque id. Deliberately mirrors workspace_store.py's shape (module-level functions, SCHEDULE_DB, lazy _init_db under a lock).

"""Display-name map for canonical user identities.

Once user_id is 'discord:1234...', nothing else knows the human's name. This
table is the alias side-channel: written on every inbound message, read by the
prompt builder, the admin panel, and the scheduler (which has no live user
object when a cron job fires).

Lives in the same fritz.db as schedules/workspaces (SCHEDULE_DB).
"""
from __future__ import annotations

import logging, sqlite3, threading
from datetime import datetime, timezone

from fritz_utils import SCHEDULE_DB

logger = logging.getLogger(__name__)
_INIT_LOCK = threading.Lock()
_INITIALISED = False
# Skip the write when the name hasn't changed -- one turn should not cost a
# gratuitous SQLite round-trip.
_NAME_CACHE: dict[str, str] = {}
_NAME_CACHE_LOCK = threading.Lock()


def _init_db() -> None:
    global _INITIALISED
    with _INIT_LOCK:
        if _INITIALISED:
            return
        with sqlite3.connect(SCHEDULE_DB) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_aliases (
                    user_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    platform TEXT,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()
        _INITIALISED = True


def record(user_id: str, display_name: str | None, platform: str | None = None) -> None:
    """Upsert the human-readable name for an identity. Best-effort: a failure
    here must never break a conversation turn."""
    if not user_id or not display_name:
        return
    with _NAME_CACHE_LOCK:
        if _NAME_CACHE.get(user_id) == display_name:
            return
    try:
        _init_db()
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(SCHEDULE_DB) as conn:
            conn.execute(
                "INSERT INTO user_aliases (user_id, display_name, platform, updated_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET "
                "display_name = excluded.display_name, updated_at = excluded.updated_at",
                (user_id, display_name, platform, now),
            )
            conn.commit()
        with _NAME_CACHE_LOCK:
            _NAME_CACHE[user_id] = display_name
    except Exception as e:
        logger.debug("identity_store.record failed for %s (non-fatal): %s", user_id, e)


def display_name(user_id: str, default: str | None = None) -> str:
    """Human name for an identity, falling back to the id's bare part."""
    ...  # SELECT display_name FROM user_aliases WHERE user_id = ?


def list_all() -> list[dict]:
    """Every known identity + display name. Admin panel use."""
    ...

### `mister_fritz.py:109-131, 510-518, 523-593`

ask_stuff stops transforming the id (it now receives a canonical one), takes display_name and channel_key, and derives thread_id via thread_id_for(). The prompt shows the human name instead of an opaque snowflake.

def get_source_info(source: MessageSource, user_id: str, display_name: str | None = None) -> str:
    who = display_name or user_id
    if source == MessageSource.DISCORD_TEXT:
        return f"User is texting from Discord (name: {who}, User ID: {user_id})"
    ...  # same shape for the other five branches


def format_prompt(prompt: str, source: MessageSource, user_id: str,
                  additional_info: str = "", display_name: str | None = None) -> str:
    # NOTE: display_name is keyword-with-default so tests/test_agent_tools.py's
    # TestFormatPromptAndSource 3-arg calls keep working unchanged.
    ...


def ask_stuff(
    base_prompt: str,
    source: MessageSource,
    user_id: str,                       # now a CANONICAL id from the adapter
    progress_callback=None,
    streaming_callback=None,
    user_image_paths: list[str] = None,
    workspace_root: str = None,
    channel_id: int | None = None,
    schedule_manager=None,
    display_name: str | None = None,    # NEW
    channel_key: str | None = None,     # NEW
) -> dict:
    import fritz_utils, identity_store

    # OLD (line 536), deleted:
    #   user_id_clean = _re.sub(r'[^a-zA-Z0-9]', '', user_id)
    # The adapter already produced the canonical id; there is nothing to strip.
    identity = fritz_utils.resolve_identity(user_id)
    identity_store.record(identity, display_name, fritz_utils.split_user_id(identity)[0])
    thread_id = fritz_utils.thread_id_for(identity, channel_key or (str(channel_id) if channel_id else None))

    if user_image_paths:
        full_prompt = format_prompt(base_prompt, source, identity,
                                    f" User has attached images: {user_image_paths}",
                                    display_name=display_name)
    else:
        user_image_paths = []
        full_prompt = format_prompt(base_prompt, source, identity, display_name=display_name)
    ...
    config = {
        "configurable": {"user_id": identity, "thread_id": thread_id},
        "metadata": {
            "user_id": identity,
            "thread_id": thread_id,
            "display_name": display_name,
            ...  # unchanged keys
        },
    }
    ...
    if final_text and base_prompt:
        extract_memories_background(identity, base_prompt, final_text)   # was user_id_clean (line 587)

### `privacy.py:22-25, 30-56, 61-105`

Fix the raw/stripped mismatch in forget_memories/export_memories (the actual /forget bug), and make forget_conversation/count_conversation_checkpoints cover every per-channel thread belonging to an identity.

# _sanitise_thread_id (22-25) is DELETED. It existed only to mirror
# ask_stuff's regex, and ask_stuff no longer has one.

def _thread_predicate(user_id: str) -> tuple[str, tuple[str, str]]:
    """SQL fragment matching the identity's base thread AND every per-channel
    branch ('<identity>#<channel>')."""
    return "(thread_id = ? OR thread_id LIKE ?)", (user_id, f"{user_id}#%")


def forget_memories(user_id: str) -> int:
    if not user_id:
        return 0
    from storage import get_default_chroma_store
    try:
        # BUGFIX: this used the raw id while agent_tools.add_memory wrote the
        # stripped one, so /forget memories deleted nothing for any username
        # containing punctuation. Post-change both sides use the canonical id
        # verbatim, so they cannot drift again.
        return get_default_chroma_store().delete_namespace((str(user_id),))
    except Exception as e:
        logger.warning("forget_memories failed for %s: %s", user_id, e)
        return 0


def forget_conversation(user_id: str) -> int:
    if not user_id:
        return 0
    pred, params = _thread_predicate(user_id)
    try:
        with sqlite3.connect(CHAT_DB_NAME) as conn:
            cur1 = conn.execute(f"DELETE FROM checkpoints WHERE {pred}", params)
            cur2 = conn.execute(f"DELETE FROM writes WHERE {pred}", params)
            conn.commit()
            return cur1.rowcount + cur2.rowcount
    except sqlite3.OperationalError as e:
        logger.debug("forget_conversation: no checkpoint tables yet (%s)", e)
        return 0
    ...
# count_conversation_checkpoints (91-105) gets the same _thread_predicate treatment.

### `main_discord.py:133-203 (specifically 135, 148, 161, 170, 193-203)`

Mint the canonical identity from ctx.author.id; keep the display name for the prompt and the audit log; use safe_user_token for the temp-file names (':' is illegal in Windows filenames — this is the trap in changing `author` in place).

@client.event
async def on_message(ctx):
    # was: author = ctx.author.name   (line 135)
    identity = fritz_utils.canonical_user_id("discord", ctx.author.id)
    display_name = getattr(ctx.author, "display_name", None) or ctx.author.name
    file_token = fritz_utils.safe_user_token(identity)   # 'discord_123456789'
    channel_key = str(ctx.channel.id)
    ...
    logger.info("Incoming message %s from %s (%s)", request_id, display_name, identity)   # was line 148
    ...
            # lines 161 and 170 -- MUST NOT interpolate `identity` directly:
            file_path = os.path.join("temp_images", f"{file_token}_{attachment.id}_{attachment.filename}")
            ...
            file_path = os.path.join("temp_audio", f"{file_token}_{attachment.id}_{attachment.filename}")
    ...
        response_data = await loop.run_in_executor(
            None,
            lambda: ask_stuff(
                message_clean, source, identity,
                progress_callback, streaming_callback,
                user_image_paths,
                workspace_store.get(identity),      # was workspace_store.get(author)
                ctx.channel.id,
                schedule_manager,
                display_name=display_name,
                channel_key=channel_key,
            )
        )

### `bot_commands.py:54-99 and every interaction.user.name site (135, 160, 180, 227, 237, 248, 261, 280, 308, 313, 318, 395, 485, 505, 523, 543, 550)`

One helper mints identity+display name from an Interaction; every call site switches to it. Admin gate keys on the immutable snowflake with the display name only as the legacy fallback. The /forget all confirm view checks user.id, not user.name.

def _identity(interaction: discord.Interaction) -> tuple[str, str]:
    """(canonical_id, display_name) for the interaction's caller."""
    user = interaction.user
    return (
        fritz_utils.canonical_user_id("discord", user.id),
        getattr(user, "display_name", None) or user.name,
    )


async def _require_admin(interaction: discord.Interaction) -> bool:
    identity, display_name = _identity(interaction)
    if fritz_utils.is_admin(identity, display_name):        # was is_admin(interaction.user.name)
        return True
    await interaction.response.send_message(
        "You do not have permission to use this command.", ephemeral=True
    )
    return False


class _ForgetConfirmView(discord.ui.View):
    def __init__(self, requester_id: int, identity: str, schedule_manager):
        super().__init__(timeout=30.0)
        self.requester_id = requester_id      # immutable snowflake
        self.identity = identity
        self.schedule_manager = schedule_manager

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.requester_id:       # was interaction.user.name != self.requester
            await interaction.response.send_message("This confirmation isn't for you.", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="Confirm", style=discord.ButtonStyle.danger)
    async def confirm(self, interaction, button):
        result = privacy.forget_all(self.identity, self.schedule_manager)
        audit_log("forget", user_id=self.identity, scope="all", result=result)
        ...

# Representative call-site rewrites:
#   :135  user_id=interaction.user.name              -> user_id=identity
#   :160  list_schedules(interaction.user.name)      -> list_schedules(identity)
#   :180  remove_schedule(schedule_id, ...user.name) -> remove_schedule(schedule_id, identity)
#   :227/:237/:248/:261/:280  user_id = interaction.user.name -> identity, _ = _identity(interaction)
#   :295  filename=f"misterfritz_export_{user_id}.json" -> f"...{fritz_utils.safe_user_token(identity)}.json"
#   :308/:313/:318  cards keep the DISPLAY name (in-memory only, user-facing text)
#   :395  ask_stuff(message, MessageSource.DISCORD_VOICE, identity, display_name=display_name,
#                   channel_key=str(interaction.channel_id))
#   :485/:505/:523/:550  workspace_store.*(identity)

### `main_telegram.py:20-32, 35-60`

Namespace the already-immutable Telegram id and pass the chat id as the channel key.

async def handle_text(update: Update, context) -> None:
    identity = canonical_user_id("telegram", update.effective_user.id)   # was str(update.effective_user.id)
    display_name = update.effective_user.full_name or update.effective_user.username or ""
    channel_key = str(update.effective_chat.id)
    ...
        lambda: ask_stuff(text, MessageSource.TELEGRAM_TEXT, identity,
                          display_name=display_name, channel_key=channel_key),
# handle_voice (35-60) gets the identical treatment at :36 and :54.

### `admin_panel.py:307-310, 344-368, 380-397, 425-480, 487-579, 636-646, 653-664, 707-716, 726-737, 109-160`

Web identity becomes web:<name>; the duplicated thread_id regex at :353 is deleted in favour of thread_id_for(); the pending-image registry and upload filenames key off the identity; the admin panel user list gains display names.

def _chat_identity(request: Request) -> tuple[str, str] | tuple[None, None]:
    """(canonical_id, display_name) from the signed cookie, or (None, None)."""
    name = chat_auth.verify_cookie(request.cookies.get(chat_auth.COOKIE_NAME), CHAT_COOKIE_SECRET)
    if not name:
        return None, None
    return fritz_utils.resolve_identity(fritz_utils.canonical_user_id("web", name)), name


def _load_chat_history(user_id: str, limit: int = 40) -> list[dict]:
    if not user_id:
        return []
    try:
        from mister_fritz import app as agent_app, get_config_values
        # DELETED: thread_id = re.sub(r"[^a-zA-Z0-9]", "", user_id)   (line 353)
        thread_id = fritz_utils.thread_id_for(user_id, "web")
        config = get_config_values({"metadata": {"user_id": user_id, "thread_id": thread_id}})
        snapshot = agent_app.get_state(config)
        ...


async def chat_page(request: Request) -> HTMLResponse:
    identity, display_name = _chat_identity(request)
    if not identity:
        return templates.TemplateResponse(request, "chat_login.html", {})
    history = await ...run_in_executor(None, _load_chat_history, identity, 40)
    response = templates.TemplateResponse(request, "chat.html", {
        "username": display_name,
        "messages": history,
        "is_admin": fritz_utils.is_admin(identity, display_name),   # was is_admin(user) at :394
    })
    _set_chat_cookie(response, display_name)
    return response

# chat_send (:448-454) / chat_stream (:522-530): ask_stuff(message, source, identity,
#     ..., display_name=display_name, channel_key="web")
# chat_forget (:643): privacy.forget_conversation(identity)
# chat_upload_image (:711): safe_user = fritz_utils.safe_user_token(identity)
# chat_upload_document (:732): fritz_utils.is_admin(identity, display_name)
# _stash_pending_image / _drain_pending_images: keyed by identity, not raw name
# _collect_users (:109-127) also unions identity_store.list_all()
# users_list (:147-160) adds "display_name": identity_store.display_name(uid) to each row

### `admin_templates/users.html:table header + row cells`

Show the human name next to the opaque identity so the admin list stays legible, and URL-encode the identity in the link (the ':' is legal in a path segment but should not be hand-rolled).

<tr><th>User</th><th>Name</th><th>Memories</th><th>Schedules</th><th>Workspace</th></tr>
...
<td><a href="/users/{{ u.user_id|urlencode }}"><code>{{ u.user_id }}</code></a></td>
<td>{{ u.display_name or '—' }}</td>

### `scheduler.py:89-122, 124-166`

No schema change — user_id stays an opaque TEXT column, so scheduler.py is almost untouched. The one change is passing the channel through to ask_stuff so a scheduled run lands in that channel's thread rather than the identity-wide one.

    async def _run_task(self, schedule_id: str, user_id: str, channel_id: int, prompt: str):
        from mister_fritz import ask_stuff
        import identity_store
        ...
            response_data = await loop.run_in_executor(
                None,
                lambda: ask_stuff(prompt, MessageSource.LOCAL, user_id,
                                  display_name=identity_store.display_name(user_id),
                                  channel_key=str(channel_id)),
            )
# add_schedule / remove_schedule / list_schedules / remove_all_for_user are
# UNCHANGED -- they already treat user_id as an opaque key.

### `migrate_identity.py:new file`

One-time, dry-run-by-default migration across all five stores. Follows migrate_db.py's shape but does NOT copy its f-string ATTACH DATABASE pattern (it never attaches). Writes a JSON mapping file so --reverse can undo it.

"""One-time migration: legacy display-name keys -> canonical namespaced ids.

    python migrate_identity.py --dry-run                       # default; shows the plan
    python migrate_identity.py --map divora=discord:123456789 --apply
    python migrate_identity.py --reverse identity_migration_20260801.json --apply

Rewrites, in one transaction per store:
  fritz.db  schedules.user_id, workspaces.user_id (may not exist),
            checkpoints.thread_id, writes.thread_id, store.namespace
  chroma    collection 'langchain_store': metadata['namespace'] in place
            (no re-embedding), plus the one 'profile_<old>' document id per user
"""
import argparse, json, os, sqlite3, sys, time


def discover(db_path: str) -> dict[str, list[str]]:
    """Every distinct legacy key per store, so --dry-run can print a checklist
    of ids the operator still needs to supply a --map entry for."""
    found = {"schedules": [], "workspaces": [], "checkpoints": [], "writes": [], "store": []}
    with sqlite3.connect(db_path) as conn:
        for table, col in (("schedules", "user_id"), ("workspaces", "user_id"),
                           ("checkpoints", "thread_id"), ("writes", "thread_id"),
                           ("store", "namespace")):
            if not _table_exists(conn, table):     # workspaces genuinely may not exist
                continue
            found[table] = [r[0] for r in conn.execute(f"SELECT DISTINCT {col} FROM {table}")]
    return found


def migrate_sqlite(db_path: str, mapping: dict[str, str], apply: bool) -> dict[str, int]:
    counts = {}
    with sqlite3.connect(db_path) as conn:
        for table, col in (("schedules", "user_id"), ("workspaces", "user_id"),
                           ("checkpoints", "thread_id"), ("writes", "thread_id"),
                           ("store", "namespace")):
            if not _table_exists(conn, table):
                continue
            n = 0
            for old, new in mapping.items():
                (hits,) = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {col} = ?", (old,)
                ).fetchone()
                if hits and apply:
                    conn.execute(f"UPDATE {table} SET {col} = ? WHERE {col} = ?", (new, old))
                n += hits
            counts[table] = n
        if apply:
            conn.commit()
    return counts


def migrate_chroma(chroma_path: str, collection: str, mapping: dict[str, str], apply: bool) -> int:
    """Rewrite metadata['namespace'] in place. Verified against chromadb 1.3.7:
    collection.update(ids=..., metadatas=...) needs no embedding function and
    no running Ollama, so this does NOT re-embed."""
    import chromadb
    col = chromadb.PersistentClient(path=chroma_path).get_collection(collection)
    touched = 0
    for old, new in mapping.items():
        got = col.get(where={"namespace": old}, include=["metadatas", "documents", "embeddings"])
        ids, metas = got["ids"], got["metadatas"]
        if not ids:
            continue
        touched += len(ids)
        if not apply:
            continue
        new_metas = [{**m, "namespace": new} for m in metas]
        col.update(ids=ids, metadatas=new_metas)
        # The profile doc is the ONLY id derived from the user id
        # (agent_tools.get_user_profile: get(f"profile_{user_id}")). Renaming an
        # id requires delete+add, so carry the existing vector across rather
        # than re-embedding.
        old_pid, new_pid = f"profile_{old}", f"profile_{new}"
        prof = col.get(ids=[old_pid], include=["metadatas", "documents", "embeddings"])
        if prof["ids"]:
            meta = {**prof["metadatas"][0], "namespace": new, "original_key": new_pid}
            col.add(ids=[new_pid], embeddings=prof["embeddings"],
                    documents=prof["documents"], metadatas=[meta])
            col.delete(ids=[old_pid])
    return touched

## Steps

1. **Step 1 (no behaviour change, safe to land alone).** Add the identity block to `fritz_utils.py` after line 218: `canonical_user_id`, `split_user_id`, `is_canonical_user_id`, `safe_user_token`, `thread_id_for`, `resolve_identity`, and the three new config flags `THREADS_PER_CHANNEL`, `ADMIN_LEGACY_NAME_MATCH`, `IDENTITY_LINKS`. Extend `is_admin` at 221-231 to `is_admin(user_id, display_name=None)`. Add the legacy-name warning to `validate_config` (273-286). Add `tests/test_identity.py`. All existing tests must still pass at this point — verify `tests/test_fritz_utils.py::TestIsAdmin` (5 tests) is green untouched, since it calls `is_admin` positionally with one arg.
2. **Step 2 (no behaviour change).** Add `identity_store.py` with `record()`, `display_name()`, `list_all()`, and the `user_aliases` table. Add `tests/test_identity_store.py`. Nothing imports it yet.
3. **Step 3 (bug fix, lands independently and is worth having on its own).** In `privacy.py`: delete `_sanitise_thread_id` (22-25), add `_thread_predicate`, and rewrite `forget_conversation` (61-88) and `count_conversation_checkpoints` (91-105) to match both `thread_id = ?` and `thread_id LIKE '<id>#%'`. Leave `forget_memories`/`export_memories` passing their argument through unchanged — the raw/stripped mismatch is closed in Step 4 when both sides start receiving the same canonical id. Update `tests/test_privacy.py::TestSanitiseThreadId` (delete it or repoint it at `fritz_utils.thread_id_for`).
4. **Step 4a (the cut-over — Discord).** `main_discord.on_message`: split `author` into `identity` / `display_name` / `file_token` and fix lines 161 and 170 to use `file_token`. Do this line *before* anything else in the function body — a colon in a Windows path is the single most likely way to break this change. Update the `ask_stuff` call at 193-203.
5. **Step 4b (cut-over — slash commands).** Add `_identity(interaction)` to `bot_commands.py`; rewrite `_require_admin` (87-99), `_ForgetConfirmView` (54-80) to key on `interaction.user.id`, and every `interaction.user.name` site listed in Change Sites. Keep `interaction.user.name` for the *cards* commands (308/313/318) and `/hello` (325) — those are display text over an in-memory dict, not stored state.
6. **Step 4c (cut-over — Telegram and web).** `main_telegram.py` lines 21/28 and 36/54. `admin_panel.py`: add `_chat_identity`, delete the duplicate regex at line 353, and update `chat_page`, `chat_send`, `chat_stream`, `chat_forget`, `chat_history`, `chat_upload_image` (711), `chat_upload_document` (732), and the pending-image registry (653-664).
7. **Step 4d (cut-over — agent core).** `mister_fritz.ask_stuff` (523-593): delete the `user_id_clean` regex at 536, add `display_name` / `channel_key` parameters, derive `thread_id` via `thread_id_for`, call `identity_store.record`. Thread `display_name` into `format_prompt` / `get_source_info` (109-131) as a keyword-with-default so `tests/test_agent_tools.py::TestFormatPromptAndSource` keeps passing. `scheduler._run_task` (89-122) passes `channel_key` and the looked-up display name.
8. **Step 5 (migration).** Write `migrate_identity.py` with `--dry-run` (default), `--apply`, `--map old=new` (repeatable), `--map-file`, `--skip-chroma`, and `--reverse <mapping.json>`. `--dry-run` prints every distinct legacy key found in each store and flags any that has no `--map` entry. `--apply` writes `identity_migration_<ts>.json` before touching anything. Add `tests/test_migrate_identity.py`.
9. **Step 6 (admin panel polish).** `_collect_users` (109-127) also unions `identity_store.list_all()`; `users_list` (147-160) adds a `display_name` field; `admin_templates/users.html` gains the column and `|urlencode`s the link.
10. **Step 7 (per-channel threads — the branching change, last and flag-gated).** Confirm every adapter passes `channel_key`. Ship with `THREADS_PER_CHANNEL=false` so `thread_id_for` returns the identity alone and no conversation branches on upgrade. Document in the CHANGELOG and README that flipping it to `true` starts fresh threads per channel (old context stays in the DB under the identity-only thread and is simply no longer read).
11. **Step 8 (docs + config).** Update `.env.example` with the four knobs; update `README.md` line 196-197 (`ROOT_USER`/`ADMIN_USERS` now accept canonical ids) and line 318 (the cross-platform thread-sharing promise is now opt-in via `IDENTITY_LINKS`); add a `## [Unreleased] / Changed` CHANGELOG entry in the existing phase style ("**Phase 15 — stable namespaced identity.**"); update `scripts/setup.py::gather_root_user` (269-276) to offer the canonical form.
12. **Step 9 (owner runbook).** Stop the bot → `python migrate_identity.py --dry-run` → note the canonical id for `divora` → `python migrate_identity.py --map divora=discord:<snowflake> --apply` → set `ROOT_USER=discord:<snowflake>` and `ADMIN_LEGACY_NAME_MATCH=false` in `.env` → start the bot → send a DM and confirm the reply still knows prior context.

## Config and env changes

- `.env.example`: `# THREADS_PER_CHANNEL=false` — one conversation thread per (user, channel) instead of one per user. Turning this on branches every existing conversation; old context stays in the DB but is no longer read. Leave false unless you actually want #general and DMs kept apart.
- `.env.example`: `# ADMIN_LEGACY_NAME_MATCH=true` — transitional. While true, `ROOT_USER`/`ADMIN_USERS` are matched against the Discord/web *display name* as well as the canonical id, so an existing `ROOT_USER=divora` keeps working across the upgrade. Set to `false` once `ROOT_USER` is a canonical id; while it is true, anyone who claims the owner's freed-up username inherits admin.
- `.env.example`: `# IDENTITY_LINKS=web:alice=discord:123456789` — comma-separated `alias=primary` pairs. Restores the pre-change behaviour where the web chat and Discord shared one conversation thread and one memory namespace. Resolution is one hop, not transitive.
- `.env.example`: update the `ROOT_USER` comment (currently line 11-12) — it now accepts either a legacy Discord username or a canonical id `discord:<numeric user id>`; the canonical form is required once `ADMIN_LEGACY_NAME_MATCH=false`. Same for the `ADMIN_USERS` comment at lines 14-17.
- No new env var is needed for the alias table — `identity_store` reuses `SCHEDULE_DB`, exactly as `workspace_store` does.

## Tests
### New

- `tests/test_identity.py::TestCanonicalUserId` — `canonical_user_id('discord', 123456789) == 'discord:123456789'`; `canonical_user_id('web', 'Alice.Smith!') == 'web:AliceSmith'` (the `.` and `!` are stripped, `_`/`-` survive); `canonical_user_id('DISCORD', 1) == 'discord:1'`; raises `ValueError` for `('discord', '')` and `('discord', '!!!')`; ids longer than 64 chars are truncated.
- `tests/test_identity.py::TestSplitUserId` — round-trips `discord:123`; returns `(None, 'divora')` for a legacy bare name; returns `(None, 'weird:thing')` for an unknown platform prefix so legacy keys containing a colon are never misparsed.
- `tests/test_identity.py::TestSafeUserToken` — `safe_user_token('discord:123') == 'discord_123'`; empty input yields `'anonymous'`. This is the Windows-filename guard for `main_discord.py:161`/`:170`, `bot_commands.py:295`, and `admin_panel.py:711`.
- `tests/test_identity.py::TestThreadIdFor` — with `THREADS_PER_CHANNEL` patched false, `thread_id_for('discord:1', '999') == 'discord:1'`; patched true it is `'discord:1#999'`; a `None` channel_key always yields the bare identity.
- `tests/test_identity.py::TestResolveIdentity` — `IDENTITY_LINKS={'web:alice': 'discord:1'}` resolves `web:alice` → `discord:1`, leaves `discord:1` alone, and does not chain through a second hop.
- `tests/test_fritz_utils.py::TestIsAdminCanonical` (new class alongside the existing `TestIsAdmin`) — canonical `ROOT_USER='discord:1'` admits `is_admin('discord:1')` and rejects `is_admin('discord:2', 'divora')`; with `ROOT_USER='divora'` and `ADMIN_LEGACY_NAME_MATCH=True`, `is_admin('discord:1', 'divora')` is True; with the flag False the same call is False. Plus a `test_validate_config_warns_on_legacy_admin_names` using `assertLogs`.
- `tests/test_identity_store.py` — `record()` creates the table on a fresh temp DB; a second `record()` with the same name performs no write (assert via a `sqlite3.connect` spy or by checking `updated_at` is unchanged); a changed name upserts; `display_name()` falls back to the supplied default for an unknown id; `record()` swallows a DB error without raising (patch `sqlite3.connect` to raise).
- `tests/test_migrate_identity.py` — build a temp `fritz.db` with rows in `schedules`, `checkpoints`, `writes`, `store` but *no* `workspaces` table (matching the real DB), then: `--dry-run` reports the correct counts and mutates nothing; `--apply` rewrites every store; a second `--apply` is a no-op (idempotent); an unmapped key is left untouched; the mapping JSON is written before any mutation; `--reverse` restores the original keys. Chroma is exercised in a separate case against a `chromadb.EphemeralClient` collection asserting `metadata['namespace']` changed and the embedding vector is byte-identical (i.e. nothing was re-embedded).
- `tests/test_privacy.py::TestForgetConversationPerChannel` — seed `checkpoints` with `discord:1`, `discord:1#111`, `discord:1#222`, `discord:10` and assert `forget_conversation('discord:1')` removes the first three and leaves `discord:10` (the `LIKE 'discord:1#%'` must not swallow `discord:10`).
- `tests/test_privacy.py::TestForgetMemoriesPunctuatedId` — the regression the audit found: `forget_memories('web:alice_smith')` must call `delete_namespace(('web:alice_smith',))` with no transformation, i.e. exactly the namespace `agent_tools.add_memory` would have written.
- `tests/test_bot_commands.py::TestIdentityFromSnowflake` — `_identity(interaction)` returns `('discord:424242', 'Nick')` and is unaffected by changing `interaction.user.name`; a renamed user with the same `.id` still passes `_require_admin` when `ROOT_USER='discord:424242'`.
- `tests/test_admin_panel.py::TestWebIdentityNamespacing` — after `/chat/login` as `alice`, `ask_stuff` receives `'web:alice'`; `_load_chat_history` is called with `'web:alice'`; the uploaded image filename starts with `web_alice_` (no colon anywhere in the path).

### Existing tests affected

- `tests/test_privacy.py::TestSanitiseThreadId::test_strips_non_alphanumeric` (line 34) — asserts `_sanitise_thread_id('alice.smith_42') == 'alicesmith42'`. The function is deleted in Step 3. Delete this test.
- `tests/test_privacy.py::TestSanitiseThreadId::test_empty_input_returns_empty` (lines 37-38) — same class, same fate. Delete the class.
- `tests/test_bot_commands.py::_fake_interaction` (lines 37-46) — only sets `interaction.user.name`; `interaction.user.id` is an auto-created `MagicMock`, which `canonical_user_id` would stringify into garbage. Add `interaction.user.id = 424242` and change the signature to `_fake_interaction(username, user_id=424242)`.
- `tests/test_bot_commands.py::TestScheduleAddOpenToAll::test_non_admin_can_schedule_add` (line 106) — `assertEqual(kwargs.get('user_id'), 'regular_user')` becomes `'discord:424242'`.
- `tests/test_bot_commands.py::TestScheduleRemoveOpenToAll::test_non_admin_can_remove_own_schedule` (line 134) — `remove_schedule.assert_called_once_with('abc12345', 'regular_user')` becomes `('abc12345', 'discord:424242')`.
- `tests/test_bot_commands.py::TestScheduleListOpenToAll::test_non_admin_can_list_their_own_schedules` (line 156) — `list_schedules.assert_called_once_with('regular_user')` becomes `'discord:424242'`.
- `tests/test_bot_commands.py::TestWorkspaceEnableOpenToAll::test_non_admin_can_enable_workspace` (line 197) — `enable_mock.assert_called_once_with('regular_user')` becomes `'discord:424242'`.
- `tests/test_bot_commands.py::TestRequireAdmin` (all three tests, 67-88) — these currently pass by display-name equality and would keep passing *only* via `ADMIN_LEGACY_NAME_MATCH`. Rewrite `_patch_admins` (57-63) to also patch `ADMIN_LEGACY_NAME_MATCH`, and add explicit id-based and name-fallback cases so the tests assert the intended gate rather than the compatibility shim.
- `tests/test_bot_commands.py::TestWorkspaceSetAdminOnly::test_non_admin_blocked_from_workspace_set` (224-231) — passes either way, but re-verify once `_fake_interaction` gains an `.id`.
- `tests/test_admin_panel.py::TestChatSend::test_authed_send_invokes_ask_stuff_with_username` (line 429) — `assertEqual(args[2], 'alice')` becomes `'web:alice'`.
- `tests/test_admin_panel.py::TestChatSend::test_send_audit_log_records_message_chars` (line 445) — `kwargs['user_id'] == 'alice'` becomes `'web:alice'` (decide once and apply consistently: audit entries record the identity, not the display name).
- `tests/test_admin_panel.py::TestChatStreamSuccess::test_audit_log_records_streamed_message` (line 541) — same `user_id` assertion.
- `tests/test_admin_panel.py::TestChatForget::test_authed_post_calls_forget_conversation_and_redirects` (lines 675 and 679) — `fc.assert_called_once_with('alice')` and `calls[0].kwargs['user_id'] == 'alice'` both become `'web:alice'`.
- `tests/test_admin_panel.py::TestChatUploadImage::test_happy_path_saves_file_and_stashes_pending` (line 781) — `_pending_images.get('alice')` becomes `.get('web:alice')`. Also update the module-level `_drain_for_user` helper (lines 741-743) and this class's `tearDown` filename prefix filter (line 755, `f.startswith('alice_')` → `'web_alice_'`).
- `tests/test_admin_panel.py::TestPendingImagePlumbing::test_send_picks_up_pending_image_and_clears` (lines 879 and 903) — `_stash_pending_image('alice', ...)` and `assertNotIn('alice', _pending_images)` both become `'web:alice'`.
- `tests/test_admin_panel.py::TestChatUploadDocument::setUp` (lines 816-817) — `patch.object(admin_panel.fritz_utils, 'is_admin', side_effect=lambda u: u == 'alice')` raises `TypeError` once `is_admin` is called with two arguments. Change to `side_effect=lambda u, display_name=None: u == 'web:alice'`.
- `tests/test_admin_panel.py::TestUsersListPage::test_lists_users_from_workspaces_and_schedules` (103-118) — should survive the extra template column since it only asserts the substrings `'alice'`/`'bob'`, but re-run it after the `users.html` edit; it is the only test covering that template.
- **Verified NOT affected — do not touch:** all of `tests/test_scheduler.py` (24 tests). `ScheduleManager` treats `user_id` as an opaque TEXT column throughout (`scheduler.py:142`, `:154`, `:176`, `:193`, `:214`); the tests use bare strings like `'user1'`/`'alice'` and remain valid.
- **Verified NOT affected — do not touch:** all of `tests/test_workspace_store.py` (14 tests). `workspace_store` is likewise identity-agnostic. Note specifically that `test_enable_sandboxed_creates_dir_and_registers` (line 73, `assertIn('alice', path)`) still holds for `'web:alice'` because `_safe_user_dir` maps it to `web_alice`.
- **Verified NOT affected, but only because of a deliberate API choice:** `tests/test_agent_tools.py::TestFormatPromptAndSource` (6 tests, lines 66-90) calls `format_prompt(prompt, source, user_id)` and `get_source_info(source, user_id)` positionally with three/two args. `display_name` must therefore be added as a *trailing keyword parameter with a default*. If you make it positional or required, all six break.

### Manual verification

- `python migrate_identity.py --dry-run` against a **copy** of the real `fritz.db` + `chroma_store/`. Expect it to report: `checkpoints`/`writes` legacy key `divora`, Chroma namespace `divora` with 11 documents, `schedules` empty, `workspaces` table absent (handled, not crashed), `store` table present but empty.
- Run the dry-run with Ollama **stopped**. It must succeed — confirmed in this session that `chromadb.PersistentClient(path='chroma_store').get_collection('langchain_store').get(include=['metadatas','documents','embeddings'])` works with no embedding function and returns 1024-dim vectors.
- After `--apply`, re-run the read-only probe and confirm `namespace == 'discord:<snowflake>'` for all 11 documents and that the embedding vector for a sampled document is byte-identical to the pre-migration value (proves nothing was re-embedded).
- On Windows specifically: DM the bot with an image attached and confirm a file appears in `temp_images/` named `discord_<snowflake>_<id>_<name>` with **no colon**. A colon here creates an NTFS alternate data stream or an `OSError` — this is the most likely regression in the whole item.
- Rename the Discord account, then send a message: Fritz must still recall prior memories, `/schedule list` must still show the schedules, `/workspace status` must still show the workspace, and admin commands must still work once `ROOT_USER=discord:<snowflake>`.
- Set `ADMIN_LEGACY_NAME_MATCH=false` with `ROOT_USER` still a display name and confirm admin commands are refused — proves the flag is actually load-bearing and not decorative.
- Log into `/chat` as the same name as the Discord account and confirm the conversations are now **separate** (this is the intended, documented break); then set `IDENTITY_LINKS=web:<name>=discord:<snowflake>`, restart, and confirm they merge again.
- Set `THREADS_PER_CHANNEL=true`, message the bot in a guild channel and in a DM, and confirm the two conversations do not see each other's context; then run `/forget conversation` and confirm both threads are cleared (this exercises the new `LIKE '<id>#%'` predicate).
- `ruff check .` and `pytest tests/ --cov=. --cov-fail-under=60` must both pass — the CI gate at `.github/workflows/ci.yml` runs exactly these.

## Risks

- **Colon in a Windows path.** `main_discord.py:161` and `:170` build `temp_images/{author}_...` and `temp_audio/{author}_...` from the same variable used as the identity. If `author` is replaced in place with `discord:123`, `open()` on Windows creates an NTFS alternate data stream (silently, for writes) or raises `OSError`. Detect: the manual image-attachment check above; also `bot_commands.py:295` (`filename=f"misterfritz_export_{user_id}.json"`, which Discord will reject) and `admin_panel.py:711-712` (already sanitises, but with its own regex — route it through `safe_user_token`).
- **Owner locked out of admin on upgrade.** If `ADMIN_LEGACY_NAME_MATCH` were defaulted to false, `ROOT_USER=divora` stops matching `discord:<snowflake>` and every admin command refuses. Mitigated by defaulting the flag to true plus the `validate_config` warning. Detect: `/workspace set` returns 'You do not have permission' immediately after upgrade. Recovery is editing `.env` and restarting — non-destructive.
- **The compatibility flag is itself the vulnerability.** While `ADMIN_LEGACY_NAME_MATCH=true`, anyone who registers the owner's released Discord username, or simply types it at `/chat/login`, inherits admin — including `POST /chat/upload/document` into shared RAG. This is the exact hole the item exists to close, and it stays open by default until the owner flips the flag. Detect: `grep ADMIN_LEGACY_NAME_MATCH .env`. Consider a hard deprecation (default false) in the following release.
- **Migration run at the wrong time.** The rewrite must happen while the bot is stopped. Running it live means the bot writes new rows under the canonical id while the script is rewriting old ones, and SQLite WAL + APScheduler jobs holding `SCHEDULE_DB` open makes the outcome non-deterministic. Detect: row counts in the mapping JSON not matching a post-run `SELECT DISTINCT`. Mitigation: the script refuses to `--apply` if `fritz.db-wal` is non-empty, with an override flag.
- **Wrong `--map` entry silently orphans data.** A typo in the snowflake moves 11 Chroma documents to a namespace nobody will ever query. Nothing errors. Detect: after migration Fritz behaves like it has amnesia. Mitigation: `--apply` writes the mapping JSON first and `--reverse` replays it; the dry-run prints the resolved mapping for eyeball confirmation before anything is written.
- **`LIKE '<id>#%'` prefix collision.** `forget_conversation('discord:1')` must not match `discord:10`. It does not — the `#` separator makes the pattern `discord:1#%`, which `discord:10` cannot match — but this is exactly the kind of thing that regresses if someone later changes the separator. Covered by `TestForgetConversationPerChannel`; keep that test.
- **Chroma `where={'namespace': ...}` on a colon-bearing value.** Verified working in this session (equality filter on an arbitrary metadata string), but only against chromadb 1.3.7 as installed. Detect: `search_memories` returns `{}` for every user after the cut-over. Settle it by re-running the read-only probe after `--apply` before starting the bot.
- **Alias write on the hot path.** `identity_store.record()` runs a SQLite upsert per turn. The in-process `_NAME_CACHE` makes the steady state a dict lookup, but a cold process pays one write per user per restart. Negligible against an Ollama call; flagged only because it sits inside `ask_stuff`. If it ever shows up, move it behind the same daemon thread `extract_memories_background` uses.
- **Sequencing against other planned items.** `history-window` and `web-auth` both touch code this item rewrites — `history-window` reads the same checkpoint `thread_id` this item redefines, and `web-auth` replaces the self-asserted cookie name that becomes `web:<name>` here. Landing either after this item is fine; landing them concurrently will conflict in `admin_panel.py` and `privacy.py`.

## Rollback
"Steps 1-3 are additive and independently revertable with no data implications. The risky boundary is Step 4 (the cut-over) plus Step 5 (the migration), because Step 5 mutates data in place.\n\n**Code rollback:** `git revert` the Step 4-7 commits. The code then writes and reads legacy-shaped keys again.\n\n**Data rollback:** the migration is the only irreversible-looking part, so it is built to be reversible. `--apply` writes `identity_migration_<timestamp>.json` containing the exact `{old: new}` mapping *before* mutating anything; `python migrate_identity.py --reverse identity_migration_<ts>.json --apply` swaps the direction and replays it against all five stores. Because every rewrite is a straight `UPDATE ... WHERE col = ?` (plus one Chroma `update(ids, metadatas)` and one profile-doc `add`/`delete`), the reverse is exact. Belt-and-braces: the runbook (Step 9) has the operator stop the bot and copy `fritz.db` and `chroma_store/` aside first — that is the real rollback, and it takes ten seconds.\n\n**Feature flags** cover the two behavioural changes that do not need a code revert at all: `THREADS_PER_CHANNEL=false` (the default) keeps one thread per identity, so per-channel branching can be turned off without redeploying; `IDENTITY_LINKS` restores the web↔Discord shared thread that `README.md:318` promises; `ADMIN_LEGACY_NAME_MATCH=true` (the default) keeps the pre-existing `.env` working. A flag is warranted here specifically because Step 7 branches live conversations — that is the one change users would feel immediately and would want backed out without a deploy."

## Open questions for you to decide

- **Separator character.** The brief specifies `:` (`discord:<id>`) and I have planned for it, but `:` is illegal in Windows filenames and forces `safe_user_token` discipline at four call sites (`main_discord.py:161`, `:170`, `bot_commands.py:295`, `admin_panel.py:711`). Using `-` or `.` — `discord-123456` — with `partition(sep)` and a closed platform allowlist would be equally unambiguous and would remove that whole class of bug on the owner's actual OS. Owner's call; I would take the safer separator, but `:` is the conventional one and reads better in logs.
- **Should the compatibility flag default to true or false?** Planned as `ADMIN_LEGACY_NAME_MATCH=true` so the upgrade does not lock the owner out. That ships the impersonation hole by default for one release. `false` is the secure default and the lockout is a ten-second `.env` edit. This is a genuine judgment call about which failure the owner would rather have on upgrade day.
- **Scheduled-task threads.** I have `scheduler._run_task` use `channel_key = str(channel_id)`, so a cron reminder lands in the same thread as human messages in that channel — the intuitive behaviour. The alternative is a dedicated `#sched` thread so an automated 9am prompt never pollutes conversational context. If the owner runs chatty recurring schedules, the second option is better.
- **Should `web:<name>` exist at all, or should the web chat be forced to link to a real platform identity?** Right now the web id is self-asserted and therefore not an identity in any meaningful sense — `web:alice` is a claim, not a fact. Namespacing it makes the *storage* correct but does not make it *authentic*. If `web-auth` is going to land anyway, it may be cleaner to have the web surface require an `IDENTITY_LINKS` entry (or a future login) rather than mint standalone `web:*` identities that will need re-migrating later.
- **Does the cross-platform shared thread need to survive at all?** `README.md:318` sells it as a feature, but it is really an artefact of the collision (`discord:divora` and `web:divora` both stripped to `divora`). `IDENTITY_LINKS` preserves it as an explicit opt-in. If the owner does not actually use the web UI as a Discord continuation, drop `IDENTITY_LINKS` and `resolve_identity` entirely and delete ~25 lines.
- **Backfilling display names for pre-migration users.** `user_aliases` starts empty; a user's name only appears after their first post-migration message. The scheduler and admin panel show the bare id until then. Acceptable for a one-owner deployment; if not, `migrate_identity.py --apply` could seed `user_aliases` from the legacy keys it is rewriting (the old key *was* the display name), which is a five-line addition. I would do it — flagging it because it slightly widens the migration's blast radius.
- **Unverifiable statically:** whether `collection.update(ids=..., metadatas=...)` persists across a `PersistentClient` reopen in this exact chromadb build. I confirmed the read path and the API signature in this session against chromadb 1.3.7, but not the write-then-reopen round trip. The experiment that settles it: copy `chroma_store/` to a scratch dir, run the migration's `migrate_chroma` against it with `--apply`, exit the process, reopen with a fresh `PersistentClient`, and assert `get(where={'namespace': 'discord:<id>'})` returns 11 documents with unchanged embeddings. Do this before touching the real store.
