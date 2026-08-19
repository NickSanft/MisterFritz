"""
One-time migration: rewrite legacy display-name keys to canonical identities.

Before this change every store was keyed off a mutable, platform-ambiguous
display string ("divora"). Now everything is keyed off
`<platform>-<immutable id>` ("discord-123456789"). This script rewrites the
existing rows so the upgrade does not look like total data loss.

Five stores are touched:

    schedules.user_id          (fritz.db)
    workspaces.user_id         (fritz.db — may not exist yet; that is fine)
    user_aliases.user_id       (fritz.db — new in this release)
    checkpoints/writes.thread_id  (fritz.db — LangGraph SqliteSaver)
    store.namespace            (fritz.db — currently unused, migrated anyway)
    Chroma metadata['namespace']  (chroma_store/ — memories and profiles)

DRY RUN IS THE DEFAULT. Nothing is written without --apply.

    # 1. See what is there and what it would become.
    python migrate_identity.py --dry-run

    # 2. Map each legacy key to its canonical identity, then apply.
    python migrate_identity.py --map divora=discord-123456789 --apply

    # 3. If it went wrong, undo it exactly.
    python migrate_identity.py --reverse identity_migration_<ts>.json

STOP THE BOT FIRST. SQLite is not the problem — a live checkpoint write during
the rewrite is.

Chroma is opened directly with chromadb.PersistentClient so Ollama does NOT
need to be running: metadata is rewritten in place with collection.update(),
which leaves the embedding vectors untouched. Nothing is re-embedded.

Exit codes:
    0  — completed (or nothing to do)
    1  — unexpected error
    2  — bad arguments, or unmapped keys found during --apply
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time

from fritz_utils import is_canonical_user_id

# Imported lazily inside main() so --help works without a configured .env.
CHROMA_COLLECTION = "langchain_store"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


# (table, key column). Order is cosmetic — each is rewritten independently.
_SQLITE_TARGETS = (
    ("schedules", "user_id"),
    ("workspaces", "user_id"),
    ("user_aliases", "user_id"),
    ("store", "namespace"),
)
# thread_id needs the channel-suffix treatment, so it is handled separately.
_THREAD_TARGETS = (
    ("checkpoints", "thread_id"),
    ("writes", "thread_id"),
)


def _distinct_keys(conn: sqlite3.Connection, table: str, column: str) -> dict[str, int]:
    """{key: row_count} for one table, or {} if the table does not exist.

    The live fritz.db has never had a `workspaces` table — _init_db creates it
    lazily on first /workspace use — so a missing table is normal, not an error.
    """
    if not _table_exists(conn, table):
        return {}
    rows = conn.execute(
        f"SELECT {column}, COUNT(*) FROM {table} GROUP BY {column}"  # noqa: S608 — names are module constants
    ).fetchall()
    return {r[0]: r[1] for r in rows if r[0]}


def _thread_identity(thread_id: str) -> str:
    """The identity half of a thread id, dropping any `#channel` suffix."""
    return thread_id.split("#", 1)[0]


def survey(db_path: str) -> dict:
    """Every distinct key in every SQLite store, without mutating anything."""
    found: dict[str, dict[str, int]] = {}
    with sqlite3.connect(db_path) as conn:
        for table, column in _SQLITE_TARGETS:
            keys = _distinct_keys(conn, table, column)
            if keys or _table_exists(conn, table):
                found[table] = keys
        for table, column in _THREAD_TARGETS:
            raw = _distinct_keys(conn, table, column)
            # Collapse `discord-1#999` down to `discord-1` so the operator maps
            # identities, not one entry per channel.
            collapsed: dict[str, int] = {}
            for thread_id, count in raw.items():
                collapsed[_thread_identity(thread_id)] = (
                    collapsed.get(_thread_identity(thread_id), 0) + count
                )
            if collapsed or _table_exists(conn, table):
                found[table] = collapsed
    return found


def survey_chroma(chroma_path: str) -> dict[str, int]:
    """{namespace: doc_count} from the Chroma collection, or {} if absent."""
    try:
        import chromadb
    except ImportError:
        print("  [skip] chromadb not installed")
        return {}
    if not os.path.isdir(chroma_path):
        print(f"  [skip] {chroma_path} does not exist")
        return {}
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(CHROMA_COLLECTION)
        data = collection.get(include=["metadatas"])
    except Exception as e:
        print(f"  [skip] could not read Chroma at {chroma_path}: {e}")
        return {}
    counts: dict[str, int] = {}
    for meta in data.get("metadatas") or []:
        ns = (meta or {}).get("namespace")
        if ns:
            counts[ns] = counts.get(ns, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Rewrite
# ---------------------------------------------------------------------------

def _rewrite_column(conn: sqlite3.Connection, table: str, column: str,
                    mapping: dict[str, str]) -> int:
    """UPDATE table SET column = new WHERE column = old, for each pair.

    Idempotent: a second run matches nothing because the old value is gone.
    """
    if not _table_exists(conn, table):
        return 0
    changed = 0
    for old, new in mapping.items():
        if old == new:
            continue
        cur = conn.execute(
            f"UPDATE {table} SET {column} = ? WHERE {column} = ?",  # noqa: S608
            (new, old),
        )
        changed += cur.rowcount
    return changed


def _rewrite_threads(conn: sqlite3.Connection, table: str, column: str,
                     mapping: dict[str, str]) -> int:
    """Rewrite thread ids, preserving any `#channel` suffix.

    `divora` → `discord-1` and `divora#999` → `discord-1#999`, in one pass, and
    without a LIKE that could catch a sibling: the rows are read first and
    matched in Python on the exact identity half.
    """
    if not _table_exists(conn, table):
        return 0
    rows = conn.execute(f"SELECT DISTINCT {column} FROM {table}").fetchall()  # noqa: S608
    changed = 0
    for (thread_id,) in rows:
        if not thread_id:
            continue
        identity = _thread_identity(thread_id)
        new_identity = mapping.get(identity)
        if not new_identity or new_identity == identity:
            continue
        suffix = thread_id[len(identity):]        # "" or "#999"
        cur = conn.execute(
            f"UPDATE {table} SET {column} = ? WHERE {column} = ?",  # noqa: S608
            (new_identity + suffix, thread_id),
        )
        changed += cur.rowcount
    return changed


def rewrite_sqlite(db_path: str, mapping: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    with sqlite3.connect(db_path) as conn:
        for table, column in _SQLITE_TARGETS:
            counts[table] = _rewrite_column(conn, table, column, mapping)
        for table, column in _THREAD_TARGETS:
            counts[table] = _rewrite_threads(conn, table, column, mapping)
        conn.commit()
    return counts


def rewrite_chroma(chroma_path: str, mapping: dict[str, str]) -> int:
    """Rewrite metadata['namespace'] in place.

    collection.update() with metadatas only does NOT re-embed — the vectors
    stay byte-identical, which is why this runs with Ollama stopped.

    The profile document is the one identity-derived *id* (`profile_<user_id>`);
    it is re-added under the new id carrying its existing embedding, then the
    old id is deleted.
    """
    try:
        import chromadb
    except ImportError:
        print("  [skip] chromadb not installed")
        return 0
    if not os.path.isdir(chroma_path):
        print(f"  [skip] {chroma_path} does not exist")
        return 0
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(CHROMA_COLLECTION)
        data = collection.get(include=["metadatas", "documents", "embeddings"])
    except Exception as e:
        print(f"  [skip] could not read Chroma at {chroma_path}: {e}")
        return 0

    ids = data.get("ids") or []
    metadatas = data.get("metadatas") or []
    documents = data.get("documents") or []
    embeddings = data.get("embeddings")
    if embeddings is None:
        embeddings = [None] * len(ids)

    update_ids, update_metas = [], []
    renames = []                      # (old_id, new_id, meta, document, embedding)
    for i, doc_id in enumerate(ids):
        meta = dict(metadatas[i] or {})
        old_ns = meta.get("namespace")
        new_ns = mapping.get(old_ns) if old_ns else None
        if not new_ns or new_ns == old_ns:
            continue
        meta["namespace"] = new_ns
        new_id = doc_id
        if doc_id == f"profile_{old_ns}":
            new_id = f"profile_{new_ns}"
        if new_id == doc_id:
            update_ids.append(doc_id)
            update_metas.append(meta)
        else:
            renames.append((doc_id, new_id, meta, documents[i], embeddings[i]))

    if update_ids:
        collection.update(ids=update_ids, metadatas=update_metas)
    for old_id, new_id, meta, document, embedding in renames:
        # add() rather than upsert() so an existing new_id is a loud failure
        # rather than a silent overwrite of somebody's profile.
        kwargs = {"ids": [new_id], "metadatas": [meta], "documents": [document]}
        if embedding is not None:
            kwargs["embeddings"] = [embedding]
        collection.add(**kwargs)
        collection.delete(ids=[old_id])
    return len(update_ids) + len(renames)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_map(pairs: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--map expects old=new, got {pair!r}")
        old, _, new = pair.partition("=")
        old, new = old.strip(), new.strip()
        if not old or not new:
            raise ValueError(f"--map expects old=new, got {pair!r}")
        mapping[old] = new
    return mapping


def _survey_arrow(key: str, mapping: dict) -> tuple[str, bool]:
    """Render one survey line's arrow. Returns (arrow, needs_a_mapping).

    A key that is ALREADY canonical needs no mapping — it is where the
    migration is trying to get to. Without this, any database the bot has been
    started against post-cutover contains canonical keys alongside legacy ones,
    every canonical key is reported as unmapped, and --apply refuses to run.
    The operator's only escape was to hand-write identity entries
    (discord-285…=discord-285…), which is exactly what used to break --reverse:
    inverting {divora: X, X: X} collapses to {X: X} and loses the real entry.
    """
    target = mapping.get(key)
    if target:
        return f"-> {target}", False
    if is_canonical_user_id(key):
        return "-- already canonical, left alone", False
    return "-> ?  (NO --map ENTRY)", True


def _live_wal_files(db_path: str, chroma_path: str,
                    skip_chroma: bool) -> list[tuple[str, int]]:
    """Return [(path, size)] for every non-empty write-ahead log we can see.

    A non-empty -wal means something has the database open with uncommitted
    pages — in practice, the bot. This is the guard against running the
    migration at the wrong time: SQLite will happily let two writers interleave
    here, and a checkpoint written mid-rewrite lands under the OLD key with no
    error anywhere.
    """
    candidates = [db_path + "-wal"]
    if not skip_chroma:
        candidates.append(os.path.join(chroma_path, "chroma.sqlite3-wal"))
    busy = []
    for path in candidates:
        try:
            size = os.path.getsize(path)
        except OSError:
            continue          # absent is the normal, healthy case
        if size > 0:
            busy.append((path, size))
    return busy


def _print_survey(sqlite_found: dict, chroma_found: dict, mapping: dict) -> list[str]:
    """Print what is there and return the keys that genuinely need a mapping."""
    unmapped: set[str] = set()
    print("\nSQLite stores:")
    for table, keys in sqlite_found.items():
        if not keys:
            print(f"  {table}: (empty)")
            continue
        print(f"  {table}:")
        for key, count in sorted(keys.items()):
            arrow, needs_mapping = _survey_arrow(key, mapping)
            if needs_mapping:
                unmapped.add(key)
            print(f"    {key!r} ({count} rows) {arrow}")
    print("\nChroma namespaces:")
    if not chroma_found:
        print("  (none)")
    for ns, count in sorted(chroma_found.items()):
        arrow, needs_mapping = _survey_arrow(ns, mapping)
        if needs_mapping:
            unmapped.add(ns)
        print(f"  {ns!r} ({count} docs) {arrow}")
    return sorted(unmapped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite legacy user keys to canonical identities.",
        epilog="Dry run is the default; nothing is written without --apply.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="report only (the default)")
    parser.add_argument("--apply", action="store_true",
                        help="actually rewrite the stores")
    parser.add_argument("--map", action="append", metavar="OLD=NEW", default=[],
                        help="map one legacy key to a canonical id (repeatable)")
    parser.add_argument("--map-file", metavar="PATH",
                        help="JSON object of {old: new} mappings")
    parser.add_argument("--skip-chroma", action="store_true",
                        help="leave the vector store alone")
    parser.add_argument("--reverse", metavar="MAPPING.JSON",
                        help="undo a previous --apply using its mapping file")
    parser.add_argument("--ignore-wal", action="store_true",
                        help="proceed even if a non-empty -wal suggests the bot "
                             "is still running (you have verified it is not)")
    parser.add_argument("--db", help="override the fritz.db path")
    parser.add_argument("--chroma", help="override the chroma_store path")
    args = parser.parse_args(argv)

    from fritz_utils import SCHEDULE_DB
    db_path = args.db or SCHEDULE_DB
    chroma_path = args.chroma or os.environ.get("CHROMA_PATH", "chroma_store")

    try:
        mapping = _parse_map(args.map)
    except ValueError as e:
        print(f"error: {e}")
        return 2
    if args.map_file:
        with open(args.map_file, encoding="utf-8") as f:
            mapping.update(json.load(f))

    if args.reverse:
        with open(args.reverse, encoding="utf-8") as f:
            saved = json.load(f)
        forward = saved.get("mapping", {})
        if not forward:
            print("error: mapping file has no 'mapping' object")
            return 2
        # Invert: the file records old→new, so undoing means new→old.
        #
        # Refuse rather than invert a non-injective mapping. A dict inversion
        # silently keeps only the last entry for a duplicated target, so
        # {"divora": X, X: X} would invert to {X: X} — the reverse would report
        # success and restore nothing, which is the worst possible outcome for
        # a command whose entire job is undoing a migration. Identity entries
        # are no longer needed (see _survey_arrow), so a collision now means a
        # genuinely ambiguous mapping the operator has to resolve.
        collisions: dict[str, list[str]] = {}
        for old, new in forward.items():
            collisions.setdefault(new, []).append(old)
        ambiguous = {new: olds for new, olds in collisions.items() if len(olds) > 1}
        if ambiguous:
            print("error: mapping is not reversible — these targets have more "
                  "than one source, so undoing them is ambiguous:")
            for new, olds in sorted(ambiguous.items()):
                print(f"  {new!r} <- {', '.join(repr(o) for o in sorted(olds))}")
            print("Edit the mapping file to keep exactly one source per target.")
            return 2
        mapping = {new: old for old, new in forward.items() if new != old}
        if not mapping:
            print("Nothing to reverse: every entry maps a key to itself.")
            return 0
        print(f"Reversing {len(mapping)} mapping(s) from {args.reverse}")

    if not os.path.exists(db_path):
        print(f"error: {db_path} does not exist")
        return 2

    print(f"Database: {db_path}")
    print(f"Chroma:   {chroma_path}" + ("  (skipped)" if args.skip_chroma else ""))

    sqlite_found = survey(db_path)
    chroma_found = {} if args.skip_chroma else survey_chroma(chroma_path)
    unmapped = _print_survey(sqlite_found, chroma_found, mapping)

    apply_changes = args.apply and not args.dry_run
    if not apply_changes:
        print("\nDRY RUN - nothing was written. Re-run with --apply to proceed.")
        if unmapped:
            print(f"Unmapped keys: {', '.join(repr(k) for k in unmapped)}")
        return 0

    if unmapped and not args.reverse:
        print(f"\nerror: no --map entry for: {', '.join(repr(k) for k in unmapped)}")
        print("Every discovered key must be mapped, or --apply would leave the")
        print("stores half-migrated. Add the missing entries and retry.")
        return 2

    if not args.ignore_wal:
        busy = _live_wal_files(db_path, chroma_path, args.skip_chroma)
        if busy:
            print()
            print("error: the bot appears to be running — these write-ahead "
                  "logs are not empty:")
            for path, size in busy:
                print(f"  {path}  ({size} bytes)")
            print("Rewriting keys underneath a live SqliteSaver loses whatever "
                  "it checkpoints mid-run, and the damage is silent. Stop the "
                  "bot and retry, or pass --ignore-wal if you have verified it "
                  "is stopped (a -wal can linger after an unclean shutdown).")
            return 2

    # Write the mapping BEFORE mutating anything, so --reverse is always
    # possible even if the run dies partway through.
    if not args.reverse:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        record_path = f"identity_migration_{stamp}.json"
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump({"db": db_path, "chroma": chroma_path,
                       "mapping": mapping}, f, indent=2)
        print(f"\nMapping written to {record_path} (keep it - --reverse needs it)")

    print("\nApplying...")
    counts = rewrite_sqlite(db_path, mapping)
    for table, n in counts.items():
        print(f"  {table}: {n} row(s) rewritten")
    if not args.skip_chroma:
        n = rewrite_chroma(chroma_path, mapping)
        print(f"  chroma: {n} document(s) rewritten")

    print("\nDone. Set ROOT_USER / ADMIN_USERS to the canonical ids in .env "
          "before restarting the bot, or you will lose admin commands.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:                       # pragma: no cover
        print(f"error: {e}")
        sys.exit(1)
