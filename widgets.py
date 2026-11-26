"""The desk's widgets (plan 14 PR 14): what the brain feeds a desk that
subscribes, each made by plain Python, never the model.

- clock: the brain's time and zone (SCHEDULE_TIMEZONE), every CLOCK_EVERY_S.
  The desk ticks the seconds itself.
- weather: Open-Meteo's now and today at WEATHER_LATITUDE and
  WEATHER_LONGITUDE (no key): one fetch every WEATHER_EVERY_S however many
  desks ask, or every RETRY_S while fetches fail, by one fetch they share (a
  desk going doesn't cut it off), offered again just after it is due.
  Nothing but the place, the units and what to report is sent. A failed
  fetch is logged once until one works again. Sent with the time it has left
  of its TTL_S, so a desk can tell an old one.
- desk: the last exchange at a desk of the identity since the brain started
  (record_exchange), at subscribe and after each turn that ends whole (not
  cut short, something said); gone with the
  identity's conversation (forget, from privacy.forget_conversation), and a
  turn that was under way when it was forgotten leaves no exchange behind.
  Each change is told to every watcher (companion_api, which sends it on).

companion_api runs the feed (next_in says when to offer a widget again);
this module only says what each widget holds (widget).
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx

import fritz_utils
from companion_protocol import WIDGET_PAYLOADS, ClockPayload, DeskExchange, DeskPayload, WeatherPayload

logger = logging.getLogger(__name__)

# Read at call time, so tests can shorten them.
CLOCK_EVERY_S = 60.0
WEATHER_EVERY_S = 900.0
RETRY_S = 120.0
# How long past a fetch's due time the feed asks: a timer may wake a clock
# tick early (Windows' monotonic clock is 15.6 ms coarse), and one that woke
# just before the fetch was due would find it not due, send the old weather
# again and wake once more.
SLACK_S = 1.0
WEATHER_TIMEOUT_S = 10.0
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"

# How long each widget's news holds, in seconds (0: until the next).
TTL_S = {"clock": 120, "weather": 1800, "desk": 0}
# Every widget the brain feeds (the protocol's list), those fed on a
# cadence, and the one fed on events.
WIDGETS = frozenset(WIDGET_PAYLOADS)
TIMED = ("clock", "weather")

ASKED_MAX_CHARS = 200
ANSWERED_MAX_CHARS = 400

# The WMO weather codes Open-Meteo reports, in words.
WMO = {
    0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Freezing fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    56: "Freezing drizzle", 57: "Heavy freezing drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    66: "Freezing rain", 67: "Heavy freezing rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Light showers", 81: "Showers", 82: "Heavy showers",
    85: "Snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}


def next_in(name: str) -> float:
    """How long until a timed widget is offered again: the clock every
    minute, the weather every quarter hour, or sooner while fetches fail."""
    return CLOCK_EVERY_S if name == "clock" else WEATHER.next_in()


# ── clock ────────────────────────────────────────────────────────────────────

def clock_payload(now: datetime | None = None) -> ClockPayload:
    zone = fritz_utils.SCHEDULE_TIMEZONE
    at = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo(zone))
    return ClockPayload(now=at.isoformat(timespec="seconds"), zone=zone)


# ── weather ──────────────────────────────────────────────────────────────────

def weather_params(place: tuple[float, float]) -> dict:
    """What is asked of Open-Meteo: the place, the units, and what to report."""
    fahrenheit = fritz_utils.WEATHER_UNIT == "fahrenheit"
    return {
        "latitude": f"{place[0]:.2f}",
        "longitude": f"{place[1]:.2f}",
        "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,is_day",
        "daily": "temperature_2m_max,temperature_2m_min",
        "temperature_unit": "fahrenheit" if fahrenheit else "celsius",
        "wind_speed_unit": "mph" if fahrenheit else "kmh",
        "timezone": "auto",
        "forecast_days": "1",
    }


def parse_weather(data: dict) -> WeatherPayload:
    """Open-Meteo's answer as the widget's payload. ValueError (or KeyError,
    TypeError) for one missing what the widget needs."""
    current, daily = data["current"], data["daily"]
    offset = timedelta(seconds=int(data.get("utc_offset_seconds", 0)))
    observed = datetime.fromisoformat(current["time"]).replace(tzinfo=timezone(offset))
    fahrenheit = fritz_utils.WEATHER_UNIT == "fahrenheit"
    code = int(current["weather_code"])
    return WeatherPayload(
        temperature=float(current["temperature_2m"]),
        feels_like=float(current["apparent_temperature"]),
        high=float(daily["temperature_2m_max"][0]),
        low=float(daily["temperature_2m_min"][0]),
        unit="F" if fahrenheit else "C",
        code=code,
        summary=WMO.get(code, "Unsettled"),
        wind=float(current["wind_speed_10m"]),
        wind_unit="mph" if fahrenheit else "km/h",
        is_day=bool(current["is_day"]),
        observed_at=observed.isoformat(timespec="minutes"),
    )


async def fetch_weather(place: tuple[float, float]) -> dict:
    """One request to Open-Meteo, its JSON. Raises on any failure."""
    async with httpx.AsyncClient(timeout=WEATHER_TIMEOUT_S, headers={"User-Agent": "MisterFritz desk weather"}) as client:
        r = await client.get(OPEN_METEO, params=weather_params(place))
        r.raise_for_status()
        return r.json()


class _Weather:
    """The one weather all desks share: fetched at most every
    WEATHER_EVERY_S, by one fetch task its askers wait on (one of them
    cancelled leaves it running for the rest), a failure tried again after
    RETRY_S. The last that came is given while it holds (TTL_S)."""

    def __init__(self) -> None:
        self.payload: WeatherPayload | None = None
        self.fetched = float("-inf")
        self.failed = float("-inf")
        self.failing = False
        self.lock: asyncio.Lock | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.task: asyncio.Task | None = None

    def next_in(self) -> float:
        """How long a desk waits before asking again: until the next fetch is
        due (RETRY_S after a failure, WEATHER_EVERY_S after a success), and
        SLACK_S past it."""
        if self.failing:
            due = self.failed + RETRY_S
        elif self.payload is not None:
            due = self.fetched + WEATHER_EVERY_S
        else:
            return WEATHER_EVERY_S      # no place, or not asked yet
        return max(0.0, due - time.monotonic()) + SLACK_S

    def ttl_left(self) -> int:
        """The seconds the last weather has left to hold, at least one (and
        one for weather never fetched)."""
        left = TTL_S["weather"] - (time.monotonic() - self.fetched)
        return int(left) if left > 1 else 1

    def _held(self) -> WeatherPayload | None:
        fresh = time.monotonic() - self.fetched < TTL_S["weather"]
        return self.payload if self.payload is not None and fresh else None

    async def get(self) -> WeatherPayload | None:
        try:
            place = fritz_utils.weather_place()
        except ValueError:
            return None     # validate_config refused it at startup
        if place is None:
            return None
        loop = asyncio.get_running_loop()
        if self.loop is not loop:
            self.lock, self.loop, self.task = asyncio.Lock(), loop, None
        async with self.lock:
            now = time.monotonic()
            if self.payload is not None and now - self.fetched < WEATHER_EVERY_S:
                return self.payload
            if now - self.failed < RETRY_S:
                return self._held()
            if self.task is None or self.task.done():
                self.task = loop.create_task(self._refresh(place))
            fetched = await asyncio.shield(self.task)
            return self.payload if fetched else self._held()

    async def _refresh(self, place: tuple[float, float]) -> bool:
        """One fetch; whether it worked."""
        try:
            payload = parse_weather(await fetch_weather(place))
        except Exception as e:
            self.failed = time.monotonic()
            if not self.failing:
                logger.warning("weather: couldn't get it from Open-Meteo (%s); trying again in %ds",
                               type(e).__name__, int(RETRY_S))
            self.failing = True
            return False
        self.payload, self.fetched = payload, time.monotonic()
        if self.failing:
            logger.info("weather: Open-Meteo answers again")
        self.failing = False
        return True


WEATHER = _Weather()


# ── desk ─────────────────────────────────────────────────────────────────────

_LAST: dict[str, DeskExchange] = {}
# How many times each identity's desk has been forgotten: a turn notes it as
# it begins, and records nothing if it has moved by the time it ends.
_FORGOTTEN: dict[str, int] = {}
# Held over a record's check and its store, and over a forget, which come
# from different threads (the companion's loop; a forget's caller): either
# the forget comes first and the record is dropped, or the record does and
# the forget takes it away. And over telling the watchers, so a desk hears
# them in that order too.
_DESK_LOCK = threading.Lock()
# Told the identity whenever its desk exchange changes.
_WATCHERS: list = []


def watch(callback) -> None:
    """`callback(identity)` whenever an identity's desk exchange changes."""
    _WATCHERS.append(callback)


def _changed(identity: str) -> None:
    for callback in list(_WATCHERS):
        try:
            callback(identity)
        except Exception:
            logger.exception("widgets: a desk watcher failed")


def _cut(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def generation(identity: str) -> int:
    """The identity's forgettings so far, for record_exchange's `since`."""
    return _FORGOTTEN.get(identity, 0)


def record_exchange(identity: str, asked: str, answered: str, mood: str | None,
                    since: int | None = None) -> DeskExchange | None:
    """The identity's last desk exchange is now this one; or, when it has
    been forgotten since `since` (its generation when the turn began),
    nothing: that conversation is gone, and so is this."""
    exchange = DeskExchange(asked=_cut(asked, ASKED_MAX_CHARS), answered=_cut(answered, ANSWERED_MAX_CHARS),
                            mood=mood, at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    with _DESK_LOCK:
        if since is not None and since != generation(identity):
            return None
        _LAST[identity] = exchange
        _changed(identity)
    return exchange


def desk_payload(identity: str) -> DeskPayload:
    return DeskPayload(last=_LAST.get(identity))


def forget(identity: str) -> None:
    """The identity's desk exchange gone (its conversation forgotten), and
    any turn under way for it leaves none behind."""
    with _DESK_LOCK:
        _FORGOTTEN[identity] = generation(identity) + 1
        if _LAST.pop(identity, None) is not None:
            _changed(identity)


async def widget(name: str, identity: str):
    """Widget `name` for a desk of `identity` now, as (payload, ttl_s), or
    None when there is none to give (no weather to be had)."""
    if name == "clock":
        return clock_payload(), TTL_S["clock"]
    if name == "weather":
        weather = await WEATHER.get()
        return None if weather is None else (weather, WEATHER.ttl_left())
    if name == "desk":
        return desk_payload(identity), TTL_S["desk"]
    return None
