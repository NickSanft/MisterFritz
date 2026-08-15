"""
Persistent scheduled task manager.

Schedules are stored in SQLite so they survive restarts. When the bot starts,
all persisted schedules are re-registered with APScheduler.

Supported schedule expressions:
  Interval:  '30m', '2h', '1d'
  Cron:      '0 9 * * *'  (minute hour day month day_of_week)
"""
import asyncio
import logging
import re
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import discord
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from fritz_utils import MAX_SCHEDULES_PER_USER, SCHEDULE_DB, SCHEDULE_MIN_DELAY_MIN, MessageSource
import identity_store

logger = logging.getLogger(__name__)

_INTERVAL_RE = re.compile(r'^(\d+)([mhd])$')


class ScheduleManager:
    """Manages persistent scheduled tasks that invoke Fritz and post results to Discord."""

    def __init__(self, bot: discord.Client):
        self.bot = bot
        self.scheduler = AsyncIOScheduler()
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(SCHEDULE_DB) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
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
            conn.commit()

    def _parse_trigger(self, schedule_expr: str):
        """Parse a schedule expression into an APScheduler trigger.

        Raises ValueError if the expression is not recognised.
        """
        m = _INTERVAL_RE.match(schedule_expr.strip())
        if m:
            value, unit = int(m.group(1)), m.group(2)
            if unit == 'm':
                return IntervalTrigger(minutes=value)
            elif unit == 'h':
                return IntervalTrigger(hours=value)
            else:
                return IntervalTrigger(days=value)

        parts = schedule_expr.strip().split()
        if len(parts) == 5:
            minute, hour, day, month, day_of_week = parts
            return CronTrigger(
                minute=minute, hour=hour, day=day,
                month=month, day_of_week=day_of_week,
            )

        raise ValueError(
            f"Invalid schedule '{schedule_expr}'. "
            "Use an interval like '30m', '2h', '1d', "
            "or a 5-part cron expression like '0 9 * * *'."
        )

    async def _run_task(self, schedule_id: str, user_id: str, channel_id: int, prompt: str):
        """Execute a scheduled task and deliver the result to the Discord channel."""
        from mister_fritz import ask_stuff

        logger.info("Running scheduled task %s for %s", schedule_id, user_id)
        channel = self.bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self.bot.fetch_channel(channel_id)
            except discord.NotFound:
                logger.warning("Channel %d not found for schedule %s — skipping", channel_id, schedule_id)
                return
            except discord.Forbidden:
                logger.warning("No permission to access channel %d for schedule %s — skipping", channel_id, schedule_id)
                return

        status_msg = await channel.send("⏰ *Running scheduled task...*")

        loop = asyncio.get_running_loop()
        try:
            # A cron job has no live user object, which is exactly why the
            # alias table exists: without it Fritz would greet the owner as
            # "discord-123456789" every morning.
            display = identity_store.display_name(user_id)
            response_data = await loop.run_in_executor(
                None,
                lambda: ask_stuff(
                    prompt, MessageSource.LOCAL, user_id,
                    display_name=display, channel_key=str(channel_id),
                ),
            )
            text = response_data.get("text") or "No response generated."
        except Exception as e:
            logger.exception("Scheduled task %s failed", schedule_id)
            text = f"❌ Scheduled task failed: {e}"

        chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)]
        if chunks:
            await status_msg.edit(content=chunks[0])
            for chunk in chunks[1:]:
                await channel.send(chunk)

    def add_schedule(
        self,
        user_id: str,
        channel_id: int,
        guild_id: Optional[int],
        prompt: str,
        schedule_expr: str,
        description: str = "",
    ) -> str:
        """Register a new schedule. Returns the short schedule ID.

        Raises ValueError on bad schedule_expr or when the user is already at
        MAX_SCHEDULES_PER_USER.
        """
        trigger = self._parse_trigger(schedule_expr)  # raises ValueError on invalid

        with sqlite3.connect(SCHEDULE_DB) as conn:
            (existing,) = conn.execute(
                "SELECT COUNT(*) FROM schedules WHERE user_id = ?", (user_id,),
            ).fetchone()
            if existing >= MAX_SCHEDULES_PER_USER:
                raise ValueError(
                    f"You already have {existing} schedules (max {MAX_SCHEDULES_PER_USER}). "
                    "Remove one with /schedule remove before adding another."
                )

            schedule_id = str(uuid.uuid4())[:8]
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO schedules VALUES (?,?,?,?,?,?,?,?)",
                (schedule_id, user_id, channel_id, guild_id, prompt,
                 schedule_expr, description or None, now),
            )
            conn.commit()

        self.scheduler.add_job(
            self._run_task,
            trigger=trigger,
            args=[schedule_id, user_id, channel_id, prompt],
            id=schedule_id,
        )
        logger.info("Added schedule %s for %s: %s", schedule_id, user_id, schedule_expr)
        return schedule_id

    def remove_schedule(self, schedule_id: str, user_id: str) -> bool:
        """Remove a schedule by ID. Returns False if not found."""
        with sqlite3.connect(SCHEDULE_DB) as conn:
            row = conn.execute(
                "SELECT user_id FROM schedules WHERE id = ?", (schedule_id,)
            ).fetchone()
            if not row:
                return False
            if row[0] != user_id:
                raise PermissionError("You can only remove your own schedules.")
            conn.execute("DELETE FROM schedules WHERE id = ?", (schedule_id,))
            conn.commit()

        try:
            self.scheduler.remove_job(schedule_id)
        except Exception:
            pass
        logger.info("Removed schedule %s", schedule_id)
        return True

    def list_schedules(self, user_id: str) -> list[dict]:
        """Return all schedules belonging to user_id."""
        with sqlite3.connect(SCHEDULE_DB) as conn:
            rows = conn.execute(
                "SELECT id, prompt, schedule_expr, created_at, description "
                "FROM schedules WHERE user_id = ? ORDER BY created_at",
                (user_id,),
            ).fetchall()
        return [
            {
                "id": r[0],
                "prompt": r[1],
                "schedule": r[2],
                "created": r[3],
                "description": r[4] or "",
            }
            for r in rows
        ]

    def remove_all_for_user(self, user_id: str) -> int:
        """Delete every schedule belonging to user_id. Returns the count removed.

        Used by /forget schedules. Also detaches the live APScheduler jobs.
        """
        with sqlite3.connect(SCHEDULE_DB) as conn:
            rows = conn.execute(
                "SELECT id FROM schedules WHERE user_id = ?", (user_id,),
            ).fetchall()
            ids = [r[0] for r in rows]
            if ids:
                placeholders = ",".join(["?"] * len(ids))
                conn.execute(
                    f"DELETE FROM schedules WHERE id IN ({placeholders})", ids,
                )
                conn.commit()

        for sid in ids:
            try:
                self.scheduler.remove_job(sid)
            except Exception as e:
                logger.debug("remove_job(%s) raised on bulk removal: %s", sid, e)
        return len(ids)

    def list_all_schedules(self) -> list[dict]:
        """Return every schedule across all users. Admin / observability use."""
        with sqlite3.connect(SCHEDULE_DB) as conn:
            rows = conn.execute(
                "SELECT id, user_id, prompt, schedule_expr, created_at, description "
                "FROM schedules ORDER BY created_at"
            ).fetchall()
        return [
            {
                "id": r[0],
                "user_id": r[1],
                "prompt": r[2],
                "schedule": r[3],
                "created": r[4],
                "description": r[5] or "",
            }
            for r in rows
        ]

    def schedule_once(
        self,
        channel_id: int,
        user_id: str,
        delay_minutes: int,
        prompt: str,
    ) -> str:
        """Register a one-shot job that fires once after delay_minutes. Not persisted to DB."""
        if delay_minutes < SCHEDULE_MIN_DELAY_MIN:
            raise ValueError(f"Delay must be at least {SCHEDULE_MIN_DELAY_MIN} minute(s).")
        schedule_id = str(uuid.uuid4())[:8]
        run_at = datetime.now(timezone.utc) + timedelta(minutes=delay_minutes)
        self.scheduler.add_job(
            self._run_task,
            trigger=DateTrigger(run_date=run_at),
            args=[schedule_id, user_id, channel_id, prompt],
            id=schedule_id,
        )
        logger.info("One-shot task %s for %s in %dm", schedule_id, user_id, delay_minutes)
        return schedule_id

    def start(self):
        """Start the scheduler and restore all persisted schedules from SQLite."""
        with sqlite3.connect(SCHEDULE_DB) as conn:
            rows = conn.execute(
                "SELECT id, user_id, channel_id, prompt, schedule_expr FROM schedules"
            ).fetchall()

        restored = 0
        for schedule_id, user_id, channel_id, prompt, schedule_expr in rows:
            try:
                trigger = self._parse_trigger(schedule_expr)
                self.scheduler.add_job(
                    self._run_task,
                    trigger=trigger,
                    args=[schedule_id, user_id, channel_id, prompt],
                    id=schedule_id,
                    replace_existing=True,
                )
                restored += 1
            except Exception as e:
                logger.warning("Could not restore schedule %s: %s", schedule_id, e)

        # Internal daily job to keep the WAL file from growing unbounded.
        # Runs at 03:00 UTC. The job ID prefix `_internal_` is excluded from
        # any future user-facing schedule listings if we add filtering there.
        try:
            self.scheduler.add_job(
                self._wal_checkpoint,
                trigger=CronTrigger(hour=3, minute=0),
                id="_internal_wal_checkpoint",
                replace_existing=True,
            )
        except Exception as e:
            logger.warning("Could not register WAL checkpoint job: %s", e)

        self.scheduler.start()
        logger.info("Scheduler started, %d job(s) restored", restored)

    def _wal_checkpoint(self) -> None:
        """Truncate the SQLite WAL file. Called daily by an internal APScheduler job.

        Under WAL mode + heavy writes, the WAL file can grow into hundreds of
        MB before SQLite checkpoints on its own. `wal_checkpoint(TRUNCATE)`
        forces a checkpoint and shrinks the WAL back to zero bytes.
        """
        try:
            with sqlite3.connect(SCHEDULE_DB) as conn:
                cur = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                result = cur.fetchone()
            logger.info("WAL checkpoint on %s complete: %s", SCHEDULE_DB, result)
        except Exception as e:
            logger.warning("WAL checkpoint failed for %s: %s", SCHEDULE_DB, e)

    def stop(self):
        self.scheduler.shutdown(wait=False)
