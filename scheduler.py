"""
Persistent scheduled task manager.

Schedules are stored in SQLite so they survive restarts. When the bot starts,
all persisted schedules are re-registered with APScheduler.

Supported schedule expressions:
  Interval:  '30m', '2h', '1d'
  Cron:      '0 9 * * *'  (minute hour day month day_of_week)

A cron's hours are in the zone the schedule was made in (plan 14 PR 14,
plan 13 Q1): SCHEDULE_TIMEZONE at the time, kept in its row, so a reminder
keeps meaning 9am there across a daylight-saving change or a move of the
host. One made before the column existed has none, and keeps the host's
zone, as it always had; one whose zone this host can't load keeps the
host's too, and says so. On the nights the clocks change, APScheduler fires
a cron in the hour they go back twice, and one in the hour they skip an hour
late.

Where a reminder goes (plan 14 PR 14): through each DeliverySink that can
take it, the Discord channel first (today's path, so nothing changes with no
desk connected), then every desktop app connected as the schedule's owner
(companion_api, a `reminder` frame), including one that connected while the
turn ran. A reminder with nowhere to go when it fires is not run.
"""
import asyncio
import functools
import logging
import re
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol

import discord
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

import fritz_utils
from fritz_utils import MAX_SCHEDULES_PER_USER, SCHEDULE_DB, SCHEDULE_MIN_DELAY_MIN, MessageSource, resolve_identity
import identity_store

logger = logging.getLogger(__name__)

_INTERVAL_RE = re.compile(r'^(\d+)([mhd])$')

# Discord refuses a message over this many characters.
_DISCORD_MAX_CHARS = 2000


def is_cron(schedule_expr: str) -> bool:
    return len(schedule_expr.strip().split()) == 5


@functools.lru_cache(maxsize=64)
def zone_loads(zone: str) -> bool:
    """Whether this host knows `zone`."""
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(zone)
        return True
    except Exception:
        return False


def zone_of(schedule: dict) -> str | None:
    """The zone a cron's hours are in, as people read it: its own, or the
    host's (one made before zones were kept, or one whose zone this host
    doesn't know, which start() runs in the host's); None for an interval,
    which has none."""
    if not is_cron(schedule["schedule"]):
        return None
    zone = schedule.get("timezone")
    return zone if zone and zone_loads(zone) else "the host's zone"


def when(schedule: dict) -> str:
    """A schedule's "every …" as people read it: a cron with its zone (its
    hours mean nothing without one), an interval as it is."""
    every = f"every `{schedule['schedule']}`"
    zone = zone_of(schedule)
    return every if zone is None else f"{every} ({zone})"


class DeliverySink(Protocol):
    """Somewhere a reminder can go (plan 14 PR 14). `ready` says whether it
    can take this one now (and gets ready for it: Discord's placeholder);
    `deliver` hands it the answer, and says whether it went."""

    name: str

    async def ready(self, identity: str, schedule_id: str) -> bool: ...

    async def deliver(self, identity: str, text: str, *, schedule_id: str) -> bool: ...


class DiscordChannelSink:
    """The schedule's Discord channel, as before: a placeholder while the turn
    runs, then the answer in it, in chunks Discord takes. A channel the cache
    has lost is fetched; one gone, or no longer readable, can't take it."""

    name = "discord"

    def __init__(self, bot, channel_id: int):
        self.bot, self.channel_id = bot, channel_id
        self.channel = None
        self.status = None

    async def ready(self, identity: str, schedule_id: str) -> bool:
        if self.bot is None:
            return False
        channel = self.bot.get_channel(self.channel_id)
        if channel is None:
            try:
                channel = await self.bot.fetch_channel(self.channel_id)
            except discord.NotFound:
                logger.warning("Channel %d not found for schedule %s", self.channel_id, schedule_id)
                return False
            except discord.Forbidden:
                logger.warning("No permission to access channel %d for schedule %s", self.channel_id, schedule_id)
                return False
        self.channel = channel
        self.status = await channel.send("⏰ *Running scheduled task...*")
        return True

    async def deliver(self, identity: str, text: str, *, schedule_id: str) -> bool:
        chunks = [text[i:i + _DISCORD_MAX_CHARS] for i in range(0, len(text), _DISCORD_MAX_CHARS)]
        if not chunks:
            return False
        await self.status.edit(content=chunks[0])
        for chunk in chunks[1:]:
            await self.channel.send(chunk)
        return True


class CompanionSink:
    """Every desktop app connected as the schedule's owner (companion_api):
    a `reminder` frame to each. None connected when it fires, it can't take
    it; but it is still asked to deliver (`late`), for a desk that connected
    while the turn ran, and finds the desks connected then."""

    name = "desk"
    late = True

    async def ready(self, identity: str, schedule_id: str) -> bool:
        import companion_api
        return companion_api.desks_of(identity) > 0

    async def deliver(self, identity: str, text: str, *, schedule_id: str) -> bool:
        import companion_api
        return companion_api.deliver_reminder(identity, text, schedule_id) > 0


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
                    created_at TEXT NOT NULL,
                    timezone TEXT
                )
            """)
            # A table made before the zone was kept (plan 14 PR 14): its rows
            # have none, and keep the host's zone.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(schedules)")}
            if "timezone" not in columns:
                conn.execute("ALTER TABLE schedules ADD COLUMN timezone TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_schedules_user_id ON schedules(user_id)"
            )
            conn.commit()

    def _parse_trigger(self, schedule_expr: str, zone: str | None = None):
        """Parse a schedule expression into an APScheduler trigger, a cron's
        hours in `zone` (None: the host's).

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
                timezone=zone,
            )

        raise ValueError(
            f"Invalid schedule '{schedule_expr}'. "
            "Use an interval like '30m', '2h', '1d', "
            "or a 5-part cron expression like '0 9 * * *'."
        )

    def _sinks(self, channel_id: int) -> list[DeliverySink]:
        """Where a reminder may go, in order: its Discord channel first, so a
        reminder with no desk connected goes as it always did."""
        return [DiscordChannelSink(self.bot, channel_id), CompanionSink()]

    async def _run_task(self, schedule_id: str, user_id: str, channel_id: int, prompt: str):
        """Execute a scheduled task and deliver the result through every sink
        that can take it: the Discord channel, then the owner's desks."""
        from mister_fritz import ask_stuff

        logger.info("Running scheduled task %s for %s", schedule_id, user_id)
        identity = resolve_identity(user_id)
        sinks, ready = self._sinks(channel_id), []
        for sink in sinks:
            try:
                if await sink.ready(identity, schedule_id):
                    ready.append(sink)
            except Exception:
                logger.exception("Schedule %s: the %s sink failed to get ready", schedule_id, sink.name)
        if not ready:
            logger.warning("Schedule %s has nowhere to go (channel %d unreachable, no desk of its owner "
                           "connected) — skipping", schedule_id, channel_id)
            return

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
                    # Queue behind a live conversation on this thread rather
                    # than being turned away: the text that comes back is
                    # posted to the channel as the reminder, so a refusal
                    # would be delivered as the reminder and the real one
                    # lost. A schedule fires once per period, so this cannot
                    # pile up.
                    may_wait=True,
                ),
            )
            text = response_data.get("text") or "No response generated."
        except Exception as e:
            logger.exception("Scheduled task %s failed", schedule_id)
            text = f"❌ Scheduled task failed: {e}"

        delivered = []
        # Those that were ready, and those that look again as they deliver.
        for sink in [s for s in sinks if s in ready or getattr(s, "late", False)]:
            try:
                if await sink.deliver(identity, text, schedule_id=schedule_id):
                    delivered.append(sink.name)
            except Exception:
                logger.exception("Schedule %s: the %s sink failed to deliver", schedule_id, sink.name)
        logger.info("Schedule %s delivered to %s", schedule_id, ", ".join(delivered) or "nowhere")

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
        zone = fritz_utils.SCHEDULE_TIMEZONE
        trigger = self._parse_trigger(schedule_expr, zone)  # raises ValueError on invalid

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
                "INSERT INTO schedules (id, user_id, channel_id, guild_id, prompt, schedule_expr, "
                "description, created_at, timezone) VALUES (?,?,?,?,?,?,?,?,?)",
                (schedule_id, user_id, channel_id, guild_id, prompt,
                 schedule_expr, description or None, now, zone),
            )
            conn.commit()

        self.scheduler.add_job(
            self._run_task,
            trigger=trigger,
            args=[schedule_id, user_id, channel_id, prompt],
            id=schedule_id,
        )
        logger.info("Added schedule %s for %s: %s (%s)", schedule_id, user_id, schedule_expr, zone)
        return schedule_id

    def remove_schedule(self, schedule_id: str, user_id: str) -> bool:
        """Remove a schedule by ID. Returns False if not found."""
        with sqlite3.connect(SCHEDULE_DB) as conn:
            row = conn.execute(
                "SELECT user_id FROM schedules WHERE id = ?", (schedule_id,)
            ).fetchone()
            if not row:
                # Not persisted, so it is either a one-shot from
                # schedule_once or nothing at all. Ask the job store, which
                # is a one-shot's only record.
                return self._remove_one_shot(schedule_id, user_id)
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

    def _jobs_for(self, user_id: str) -> list:
        """Live jobs whose owner is user_id.

        Every job this class registers carries
        `[schedule_id, user_id, channel_id, prompt]` as its args, so the owner
        is args[1] whether the job was persisted or not. Callers decide which
        kind they are looking at: a persisted one has a row in the schedules
        table, a one-shot never does.
        """
        owned = []
        for job in self.scheduler.get_jobs():
            args = getattr(job, "args", None) or ()
            if len(args) >= 2 and args[1] == user_id:
                owned.append(job)
        return owned

    def _remove_one_shot(self, schedule_id: str, user_id: str) -> bool:
        """Cancel a job that exists only in the scheduler.

        schedule_once does not persist (there is nothing to restore - a
        one-shot that missed its moment should not fire on the next boot), so
        a one-shot has no row in the schedules table. That made every id
        schedule_message ever handed out uncancellable: remove_schedule
        returned False, so Fritz confirmed a reminder and then denied it
        existed, and /forget schedules left it running.
        """
        job = self.scheduler.get_job(schedule_id)
        if job is None:
            return False
        args = getattr(job, "args", None) or ()
        if len(args) < 2:
            # A job this class did not book, or one whose args no longer carry
            # an owner. "Not yours" would be a claim about an owner there is
            # no evidence for, so report it the way an absent id is reported.
            logger.warning("Job %s has no identifiable owner; not removing", schedule_id)
            return False
        if args[1] != user_id:
            # Same answer the persisted path gives, deliberately: a one-shot
            # should not be the one place that reports somebody else's
            # schedule as simply absent.
            raise PermissionError("You can only remove your own schedules.")
        self.scheduler.remove_job(schedule_id)
        logger.info("Removed one-shot %s", schedule_id)
        return True

    def list_schedules(self, user_id: str) -> list[dict]:
        """Return all schedules belonging to user_id."""
        with sqlite3.connect(SCHEDULE_DB) as conn:
            rows = conn.execute(
                "SELECT id, prompt, schedule_expr, created_at, description, timezone "
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
                "timezone": r[5],
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

        # The persisted jobs are detached by now, so anything still in the
        # store under this user is a one-shot - which has no row to delete and
        # so used to survive /forget schedules entirely, still holding the
        # prompt it was booked with.
        one_shots = 0
        for job in self._jobs_for(user_id):
            try:
                self.scheduler.remove_job(job.id)
                one_shots += 1
            except Exception as e:
                logger.debug("remove_job(%s) raised on one-shot sweep: %s", job.id, e)
        return len(ids) + one_shots

    def list_all_schedules(self) -> list[dict]:
        """Return every schedule across all users. Admin / observability use."""
        with sqlite3.connect(SCHEDULE_DB) as conn:
            rows = conn.execute(
                "SELECT id, user_id, prompt, schedule_expr, created_at, description, timezone "
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
                "timezone": r[6],
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
                "SELECT id, user_id, channel_id, prompt, schedule_expr, timezone FROM schedules"
            ).fetchall()

        restored = 0
        for schedule_id, user_id, channel_id, prompt, schedule_expr, zone in rows:
            try:
                # A cron made in a zone this host doesn't know (made on
                # another host, say) fires in the host's, rather than never
                # again. An interval has no zone to miss; a bad expression is
                # another matter, and goes to the warning below.
                unknown = zone is not None and is_cron(schedule_expr) and not zone_loads(zone)
                trigger = self._parse_trigger(schedule_expr, None if unknown else zone)
                self.scheduler.add_job(
                    self._run_task,
                    trigger=trigger,
                    args=[schedule_id, user_id, channel_id, prompt],
                    id=schedule_id,
                    replace_existing=True,
                )
                restored += 1
                if unknown:
                    logger.error("Schedule %s was made in %r, which this host doesn't know: it fires in the "
                                 "host's zone until it is made again", schedule_id, zone)
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
