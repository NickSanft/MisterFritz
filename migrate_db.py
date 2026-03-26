"""
One-time migration: merge schedules.db + chat_history.db → fritz.db

Run this script once before (or instead of) starting the bot for the first time
after upgrading to the unified-database configuration.

    python migrate_db.py

The old database files are NOT deleted — remove them manually once you have
confirmed that fritz.db contains all the expected data.

Exit codes:
    0  — migration completed (or nothing to migrate)
    1  — unexpected error
"""
import sqlite3
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LANGGRAPH_TABLES = ("checkpoints", "writes")
_CHAT_TABLES = ("store",) + _LANGGRAPH_TABLES


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _source_table_exists(source_db: str, table: str) -> bool:
    try:
        with sqlite3.connect(source_db) as c:
            return _table_exists(c, table)
    except sqlite3.Error:
        return False


def _migrate_table(
    conn: sqlite3.Connection,
    alias: str,
    table: str,
) -> int:
    """Copy rows from alias.table → main.table using INSERT OR IGNORE.

    Returns the number of rows copied.
    """
    if not _table_exists(conn, table):
        print(f"  [skip] {table} — table not present in target; no schema to copy into")
        return 0

    try:
        before = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        conn.execute(f"INSERT OR IGNORE INTO {table} SELECT * FROM {alias}.{table}")
        after = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        copied = after - before
        print(f"  {table}: {copied} row(s) copied ({before} → {after})")
        return copied
    except sqlite3.OperationalError as e:
        print(f"  [warn] {table}: {e}")
        return 0


# ---------------------------------------------------------------------------
# Ensure target tables exist
# ---------------------------------------------------------------------------

def _ensure_target_schema(conn: sqlite3.Connection) -> None:
    """Create all application tables in fritz.db if they don't already exist."""
    conn.execute("PRAGMA journal_mode=WAL")

    # store (SQLiteStore)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS store (
            namespace TEXT,
            key TEXT,
            value TEXT,
            PRIMARY KEY (namespace, key)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_store_namespace ON store(namespace)"
    )

    # schedules (ScheduleManager)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schedules (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            channel_id INTEGER NOT NULL,
            guild_id INTEGER,
            prompt TEXT NOT NULL,
            schedule_expr TEXT NOT NULL,
            description TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_schedules_user_id ON schedules(user_id)"
    )

    # LangGraph checkpoint tables (SqliteSaver creates these on first use, but
    # we pre-create them so the INSERT OR IGNORE in the migration can proceed
    # even if the bot hasn't been run against fritz.db yet).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            checkpoint_id TEXT NOT NULL,
            parent_checkpoint_id TEXT,
            type TEXT,
            checkpoint BLOB,
            metadata BLOB,
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS writes (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            checkpoint_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            idx INTEGER NOT NULL,
            channel TEXT NOT NULL,
            type TEXT,
            blob BLOB NOT NULL,
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
        )
    """)

    conn.commit()


# ---------------------------------------------------------------------------
# Per-source migration
# ---------------------------------------------------------------------------

def _migrate_schedules(conn: sqlite3.Connection, source: str) -> int:
    print(f"\nMigrating schedules from '{source}' ...")
    conn.execute(f"ATTACH DATABASE '{source}' AS old_sched")
    try:
        if not _source_table_exists(source, "schedules"):
            print("  [skip] schedules table not found in source")
            return 0
        total = _migrate_table(conn, "old_sched", "schedules")
        conn.commit()
        return total
    finally:
        conn.execute("DETACH DATABASE old_sched")


def _migrate_chat(conn: sqlite3.Connection, source: str) -> int:
    print(f"\nMigrating chat history from '{source}' ...")
    conn.execute(f"ATTACH DATABASE '{source}' AS old_chat")
    try:
        total = 0
        for table in _CHAT_TABLES:
            if not _source_table_exists(source, table):
                print(f"  [skip] {table} — not found in source")
                continue
            total += _migrate_table(conn, "old_chat", table)
        conn.commit()
        return total
    finally:
        conn.execute("DETACH DATABASE old_chat")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    import os

    # Allow overrides via env so this script respects the same config as the app.
    from fritz_utils import DB_NAME, CHAT_DB_NAME, SCHEDULE_DB

    target = DB_NAME
    sched_source = SCHEDULE_DB
    chat_source = CHAT_DB_NAME

    print(f"Target database : {target}")
    print(f"Schedules source: {sched_source}")
    print(f"Chat source     : {chat_source}")

    if sched_source == target and chat_source == target:
        print(
            "\nAll sources already point to the target — nothing to migrate.\n"
            "If you intended to migrate from the old defaults, set:\n"
            "  SCHEDULE_DB=schedules.db CHAT_DB_NAME=chat_history.db"
        )
        return 0

    try:
        with sqlite3.connect(target) as conn:
            _ensure_target_schema(conn)

            total = 0

            if sched_source != target and os.path.exists(sched_source):
                total += _migrate_schedules(conn, sched_source)
            elif sched_source != target:
                print(f"\n[info] '{sched_source}' not found — skipping schedules migration")

            if chat_source != target and os.path.exists(chat_source):
                total += _migrate_chat(conn, chat_source)
            elif chat_source != target:
                print(f"\n[info] '{chat_source}' not found — skipping chat migration")

        print(f"\nDone. {total} total row(s) copied into '{target}'.")
        print("Old files have NOT been deleted. Remove them once you have verified the migration.")
        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
