"""
One-time migration: merge schedules.db + chat_history.db → fritz.db

Run this script once before (or instead of) starting the bot for the first time
after upgrading to the unified-database configuration.

    python migrate_db.py

The old database files are NOT deleted — remove them manually once you have
confirmed that fritz.db contains all the expected data.

Exit codes:
    0  — migration completed (or nothing to migrate)
    1  — unexpected error, or a table that did not copy cleanly

A non-zero exit means DO NOT DELETE the old files yet: either something raised,
or at least one table reported rows that are in the source and not in the
target. The closing advice to remove the sources is printed only on 0.
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


def _columns(conn: sqlite3.Connection, table: str, alias: str = "main") -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA {alias}.table_info({table})")]


def _migrate_table(
    conn: sqlite3.Connection,
    alias: str,
    table: str,
) -> tuple[int, bool]:
    """Copy rows from alias.table → main.table using INSERT OR IGNORE.

    Returns (rows copied, warned). `warned` is what stops main() reporting
    success: a failed copy used to print one [warn] line, be counted as zero
    rows, and leave the closing message still advising removal of the sources.
    """
    if not _table_exists(conn, table):
        print(f"  [skip] {table} — table not present in target; no schema to copy into")
        return 0, False

    # Named columns rather than SELECT *. The positional copy silently
    # misfiled everything after the first difference whenever the source and
    # target column sets disagreed, which is exactly what happens across
    # checkpointer versions.
    target_cols = _columns(conn, table)
    source_cols = set(_columns(conn, table, alias))
    shared = [c for c in target_cols if c in source_cols]
    if not shared:
        print(f"  [warn] {table}: source and target share no columns; not copied")
        return 0, True
    dropped = sorted(c for c in source_cols if c not in target_cols)
    missing = [c for c in target_cols if c not in source_cols]
    if dropped:
        print(f"  [warn] {table}: source columns absent from target, NOT copied: "
              f"{', '.join(dropped)}")
    if missing:
        print(f"  [note] {table}: target columns absent from source, left at default: "
              f"{', '.join(missing)}")

    names = ", ".join(shared)
    try:
        before = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        conn.execute(
            f"INSERT OR IGNORE INTO {table} ({names}) "
            f"SELECT {names} FROM {alias}.{table}"
        )
        after = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        copied = after - before
        print(f"  {table}: {copied} row(s) copied ({before} → {after})")
    except sqlite3.Error as e:
        print(f"  [warn] {table}: {e}")
        return 0, True

    return copied, bool(dropped) or _rows_left_behind(conn, alias, table)


def _rows_left_behind(conn: sqlite3.Connection, alias: str, table: str) -> bool:
    """Warn about source rows that are still not in the target.

    OR IGNORE is what makes this necessary. It is there so the migration can
    be re-run safely - a row already present is skipped, which is correct and
    is why a second run copies nothing and is still a success. But it treats a
    rejected row exactly the same way: a source that predates a NOT NULL
    column loses every row, silently, with no exception to catch and a count
    of zero that is indistinguishable from "already migrated".

    So this asks the only question that matters before an operator deletes the
    source: is every row in it now also in the target? Keyed on the primary
    key, because that is what OR IGNORE itself matches on.
    """
    key = [row[1] for row in conn.execute(f"PRAGMA main.table_info({table})") if row[5]]
    source_cols = set(_columns(conn, table, alias))
    if not key or any(column not in source_cols for column in key):
        # No shared key to compare on; the counts above are all there is.
        return False
    match = " AND ".join(f"target.{column} IS source.{column}" for column in key)
    left = conn.execute(
        f"SELECT COUNT(*) FROM {alias}.{table} AS source WHERE NOT EXISTS "
        f"(SELECT 1 FROM main.{table} AS target WHERE {match})"
    ).fetchone()[0]
    if left:
        print(f"  [warn] {table}: {left} source row(s) did NOT arrive - rejected by the "
              "target schema, not merely already present")
    return bool(left)


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

    # The LangGraph tables are NOT written out here. They used to be, and the
    # copy drifted: this file created `writes.blob BLOB NOT NULL` where the
    # checkpointer reads and writes a nullable `value`. CREATE TABLE IF NOT
    # EXISTS no-ops on an existing table, so SqliteSaver.setup() could not
    # repair it afterwards - once this script had run against a fresh
    # fritz.db, every put_writes failed with "table writes has no column named
    # value", the bot could not finish a single turn, and nothing pointed back
    # at the migration.
    #
    # Asking the checkpointer for its own schema cannot drift, because it is
    # the same call the bot makes on first use.
    _ensure_langgraph_schema(conn)
    conn.commit()


def _ensure_langgraph_schema(conn: sqlite3.Connection) -> bool:
    """Have the installed checkpointer create its own tables.

    Returns False when it is not installed: the LangGraph tables are then left
    for the bot's first run, and the copy skips them rather than inventing a
    schema to copy into.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as e:
        print(f"  [skip] LangGraph tables - checkpointer not installed ({e}).")
        print("         Conversation history will NOT be copied. Install the")
        print("         dependencies and re-run to include it.")
        return False
    SqliteSaver(conn).setup()
    conn.commit()
    return True


# ---------------------------------------------------------------------------
# Per-source migration
# ---------------------------------------------------------------------------

def _migrate_schedules(conn: sqlite3.Connection, source: str) -> tuple[int, bool]:
    print(f"\nMigrating schedules from '{source}' ...")
    conn.execute(f"ATTACH DATABASE '{source}' AS old_sched")
    try:
        if not _source_table_exists(source, "schedules"):
            print("  [skip] schedules table not found in source")
            return 0, False
        total, warned = _migrate_table(conn, "old_sched", "schedules")
        conn.commit()
        return total, warned
    finally:
        conn.execute("DETACH DATABASE old_sched")


def _migrate_chat(conn: sqlite3.Connection, source: str) -> tuple[int, bool]:
    print(f"\nMigrating chat history from '{source}' ...")
    conn.execute(f"ATTACH DATABASE '{source}' AS old_chat")
    try:
        total = 0
        warned = False
        for table in _CHAT_TABLES:
            if not _source_table_exists(source, table):
                print(f"  [skip] {table} — not found in source")
                continue
            copied, table_warned = _migrate_table(conn, "old_chat", table)
            total += copied
            warned = warned or table_warned
        conn.commit()
        return total, warned
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
            warned = False

            if sched_source != target and os.path.exists(sched_source):
                copied, table_warned = _migrate_schedules(conn, sched_source)
                total += copied
                warned = warned or table_warned
            elif sched_source != target:
                print(f"\n[info] '{sched_source}' not found — skipping schedules migration")

            if chat_source != target and os.path.exists(chat_source):
                copied, table_warned = _migrate_chat(conn, chat_source)
                total += copied
                warned = warned or table_warned
            elif chat_source != target:
                print(f"\n[info] '{chat_source}' not found — skipping chat migration")

        if warned:
            # Deliberately not "Done", and deliberately silent about removing
            # anything: deleting the sources is the one irreversible step this
            # script leads to, and a [warn] line scrolling past above a
            # success message is how data gets deleted that was never copied.
            print(f"\nINCOMPLETE. {total} row(s) copied into '{target}', but at least "
                  "one table reported a problem above.", file=sys.stderr)
            print("KEEP the old files and resolve the warnings before deleting anything.",
                  file=sys.stderr)
            return 1

        print(f"\nDone. {total} total row(s) copied into '{target}'.")
        print("Old files have NOT been deleted. Remove them once you have verified the migration.")
        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
