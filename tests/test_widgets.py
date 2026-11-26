"""The desk's widgets (plan 14 PR 14): what each holds, made by plain Python.

The clock in the brain's zone; the weather from Open-Meteo (its answer as a
fixture: these tests never reach the network), shared by every desk, one
fetch a quarter hour, a failure logged once and tried again later; his desk,
the identity's last exchange, cut short, told to whoever watches, and
forgotten with the conversation.
"""
import asyncio
import json
import logging
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import fritz_utils
import widgets
from companion_protocol import WIDGET_PAYLOADS, ClockPayload, DeskPayload, WeatherPayload

# What Open-Meteo answers for weather_params(), trimmed to what is read.
OPEN_METEO_ANSWER = {
    "latitude": 35.85, "longitude": -86.39, "utc_offset_seconds": -21600, "timezone": "America/Chicago",
    "current": {"time": "2026-11-26T15:15", "interval": 900, "temperature_2m": 51.3,
                "apparent_temperature": 47.9, "weather_code": 2, "wind_speed_10m": 7.4, "is_day": 1},
    "daily": {"time": ["2026-11-26"], "temperature_2m_max": [58.1], "temperature_2m_min": [39.2]},
}


def run(coro):
    return asyncio.run(coro)


class TestTheClock(unittest.TestCase):
    def test_the_brains_time_in_its_zone(self):
        at = datetime(2026, 11, 27, 3, 15, 30, tzinfo=timezone.utc)
        with patch.object(fritz_utils, "SCHEDULE_TIMEZONE", "America/Chicago"):
            p = widgets.clock_payload(at)
        self.assertEqual(p, ClockPayload(now="2026-11-26T21:15:30-06:00", zone="America/Chicago"))
        with patch.object(fritz_utils, "SCHEDULE_TIMEZONE", "UTC"):
            self.assertEqual(widgets.clock_payload(at).now, "2026-11-27T03:15:30+00:00")

    def test_by_default_now(self):
        before = datetime.now(timezone.utc)
        p = widgets.clock_payload()
        self.assertGreaterEqual(datetime.fromisoformat(p.now), before.replace(microsecond=0))


class TestTheWeather(unittest.TestCase):
    def setUp(self):
        self.cache = widgets._Weather()
        for name, value in (("WEATHER_LATITUDE", "35.85"), ("WEATHER_LONGITUDE", "-86.39"),
                            ("WEATHER_UNIT", "fahrenheit")):
            p = patch.object(fritz_utils, name, value)
            p.start()
            self.addCleanup(p.stop)

    def test_what_is_asked_of_open_meteo(self):
        self.assertEqual(widgets.weather_params((35.8456, -86.3903)), {
            "latitude": "35.85", "longitude": "-86.39",
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,is_day",
            "daily": "temperature_2m_max,temperature_2m_min",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "timezone": "auto", "forecast_days": "1",
        })
        with patch.object(fritz_utils, "WEATHER_UNIT", "celsius"):
            params = widgets.weather_params((51.5, -0.12))
        self.assertEqual((params["temperature_unit"], params["wind_speed_unit"]), ("celsius", "kmh"))

    def test_its_answer_as_the_payload(self):
        self.assertEqual(widgets.parse_weather(OPEN_METEO_ANSWER), WeatherPayload(
            temperature=51.3, feels_like=47.9, high=58.1, low=39.2, unit="F", code=2, summary="Partly cloudy",
            wind=7.4, wind_unit="mph", is_day=True, observed_at="2026-11-26T15:15-06:00"))
        with patch.object(fritz_utils, "WEATHER_UNIT", "celsius"):
            p = widgets.parse_weather(OPEN_METEO_ANSWER)
        self.assertEqual((p.unit, p.wind_unit), ("C", "km/h"))

    def test_every_code_has_words(self):
        for code in (0, 1, 2, 3, 45, 48, 51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 71, 73, 75, 77, 80, 81, 82,
                     85, 86, 95, 96, 99):
            self.assertIn(code, widgets.WMO)
        answer = json.loads(json.dumps(OPEN_METEO_ANSWER))
        answer["current"]["weather_code"] = 42
        self.assertEqual(widgets.parse_weather(answer).summary, "Unsettled")

    def test_an_answer_missing_what_it_needs_is_refused(self):
        for drop in ("current", "daily"):
            answer = {k: v for k, v in OPEN_METEO_ANSWER.items() if k != drop}
            with self.subTest(drop=drop), self.assertRaises(KeyError):
                widgets.parse_weather(answer)

    def test_fetched_once_a_quarter_hour_whoever_asks(self):
        fetch = AsyncMock(return_value=OPEN_METEO_ANSWER)
        with patch.object(widgets, "fetch_weather", fetch):
            async def go():
                return [await self.cache.get() for _ in range(3)]
            got = run(go())
        self.assertEqual(fetch.await_count, 1)
        self.assertEqual(fetch.await_args.args, ((35.85, -86.39),))
        self.assertTrue(all(g == got[0] for g in got))
        self.assertEqual(got[0].temperature, 51.3)

    def test_and_again_once_it_is_old(self):
        fetch = AsyncMock(return_value=OPEN_METEO_ANSWER)
        with patch.object(widgets, "fetch_weather", fetch), patch.object(widgets, "WEATHER_EVERY_S", 0.0):
            run(self.cache.get())
            run(self.cache.get())
        self.assertEqual(fetch.await_count, 2)

    def test_a_failure_is_logged_once_and_tried_again_later(self):
        fetch = AsyncMock(side_effect=OSError("no route"))
        with patch.object(widgets, "fetch_weather", fetch), self.assertLogs(widgets.logger, logging.WARNING) as logs:
            self.assertIsNone(run(self.cache.get()))
            self.assertIsNone(run(self.cache.get()))   # within RETRY_S: not asked again
            with patch.object(widgets, "RETRY_S", 0.0):
                self.assertIsNone(run(self.cache.get()))
        self.assertEqual(fetch.await_count, 2)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("OSError", logs.output[0])
        self.assertNotIn("no route", logs.output[0])
        fetch = AsyncMock(return_value=OPEN_METEO_ANSWER)
        with patch.object(widgets, "fetch_weather", fetch), patch.object(widgets, "RETRY_S", 0.0), \
                self.assertLogs(widgets.logger, logging.INFO) as logs:
            self.assertIsNotNone(run(self.cache.get()))
        self.assertIn("answers again", logs.output[0])

    def test_after_a_recovery_the_next_failure_is_logged_again(self):
        fail = AsyncMock(side_effect=OSError())
        good = AsyncMock(return_value=OPEN_METEO_ANSWER)
        with patch.object(widgets, "RETRY_S", 0.0), patch.object(widgets, "WEATHER_EVERY_S", 0.0):
            with patch.object(widgets, "fetch_weather", fail), self.assertLogs(widgets.logger, logging.WARNING):
                run(self.cache.get())
            with patch.object(widgets, "fetch_weather", good), self.assertLogs(widgets.logger, logging.INFO) as logs:
                run(self.cache.get())
                run(self.cache.get())
            self.assertEqual(len(logs.output), 1, "a recovery said once")
            with patch.object(widgets, "fetch_weather", fail), self.assertLogs(widgets.logger, logging.WARNING) as logs:
                run(self.cache.get())
            self.assertEqual(len(logs.output), 1, "a new run of failures said again")

    def test_a_failure_once_the_last_is_stale_gives_none(self):
        good = AsyncMock(return_value=OPEN_METEO_ANSWER)
        with patch.object(widgets, "fetch_weather", good):
            run(self.cache.get())
        with patch.object(widgets, "fetch_weather", AsyncMock(side_effect=OSError())), \
                patch.object(widgets, "WEATHER_EVERY_S", 0.0), patch.object(widgets, "TTL_S", {**widgets.TTL_S, "weather": 0}), \
                self.assertLogs(widgets.logger, logging.WARNING):
            self.assertIsNone(run(self.cache.get()))

    def test_a_failure_keeps_the_last_while_it_holds(self):
        good = AsyncMock(return_value=OPEN_METEO_ANSWER)
        with patch.object(widgets, "fetch_weather", good):
            first = run(self.cache.get())
        with patch.object(widgets, "fetch_weather", AsyncMock(side_effect=OSError())), \
                patch.object(widgets, "WEATHER_EVERY_S", 0.0), self.assertLogs(widgets.logger, logging.WARNING):
            self.assertEqual(run(self.cache.get()), first)
            with patch.object(widgets, "TTL_S", {**widgets.TTL_S, "weather": 0}):
                self.assertIsNone(run(self.cache.get()), "not once it is stale")

    def test_no_place_no_weather_and_no_fetch(self):
        fetch = AsyncMock(return_value=OPEN_METEO_ANSWER)
        with patch.object(widgets, "fetch_weather", fetch), patch.object(fritz_utils, "WEATHER_LATITUDE", ""), \
                patch.object(fritz_utils, "WEATHER_LONGITUDE", ""):
            self.assertIsNone(run(self.cache.get()))
        fetch.assert_not_awaited()

    def test_one_fetch_for_every_asker_and_a_cancelled_one_leaves_it_running(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def slow(place):
            started.set()
            await release.wait()
            return OPEN_METEO_ANSWER

        fetch = AsyncMock(side_effect=slow)

        async def go():
            first = asyncio.create_task(self.cache.get())
            await started.wait()
            second = asyncio.create_task(self.cache.get())
            await asyncio.sleep(0)
            first.cancel()              # its desk went, mid-fetch
            # The second takes the lock while the fetch is still under way,
            # and must wait on that one rather than start its own.
            for _ in range(5):
                await asyncio.sleep(0)
            self.assertFalse(self.cache.task.done())
            release.set()
            return await second, first

        with patch.object(widgets, "fetch_weather", fetch):
            got, first = run(go())
        self.assertTrue(first.cancelled())
        self.assertEqual(got.temperature, 51.3)
        self.assertEqual(fetch.await_count, 1)

    def test_the_time_it_has_left(self):
        clock = [1000.0]
        with patch.object(widgets.time, "monotonic", lambda: clock[0]), \
                patch.object(widgets, "fetch_weather", AsyncMock(return_value=OPEN_METEO_ANSWER)):
            run(self.cache.get())
            self.assertEqual(self.cache.ttl_left(), 1800)
            clock[0] += 600
            self.assertEqual(self.cache.ttl_left(), 1200)
            clock[0] += 5000
            self.assertEqual(self.cache.ttl_left(), 1, "never none: it is sent only while it holds")

    def test_asked_again_just_after_it_is_due(self):
        """Until the next fetch is due, and SLACK_S past it: a timer that wakes
        a clock tick early must not find it not due yet and wait a whole
        cadence more."""
        clock = [5000.0]
        self.assertEqual(self.cache.next_in(), widgets.WEATHER_EVERY_S, "not asked yet")
        with patch.object(widgets.time, "monotonic", lambda: clock[0]):
            with patch.object(widgets, "fetch_weather", AsyncMock(side_effect=OSError())), \
                    self.assertLogs(widgets.logger, logging.WARNING):
                run(self.cache.get())
            self.assertEqual(self.cache.next_in(), widgets.RETRY_S + widgets.SLACK_S)
            clock[0] += 100
            self.assertEqual(self.cache.next_in(), widgets.RETRY_S - 100 + widgets.SLACK_S)
            clock[0] += widgets.RETRY_S
            self.assertEqual(self.cache.next_in(), widgets.SLACK_S, "overdue: at once, near enough")
            with patch.object(widgets, "fetch_weather", AsyncMock(return_value=OPEN_METEO_ANSWER)), \
                    self.assertLogs(widgets.logger, logging.INFO):
                run(self.cache.get())
            self.assertEqual(self.cache.next_in(), widgets.WEATHER_EVERY_S + widgets.SLACK_S)
            clock[0] += 300
            self.assertEqual(self.cache.next_in(), widgets.WEATHER_EVERY_S - 300 + widgets.SLACK_S,
                             "counted from the fetch, not from the asking")
            clock[0] -= 300
            # Waking that much later, it is due: fetched again.
            clock[0] += self.cache.next_in() - 0.5
            fetch = AsyncMock(return_value=OPEN_METEO_ANSWER)
            with patch.object(widgets, "fetch_weather", fetch):
                run(self.cache.get())
            fetch.assert_awaited_once()

    def test_one_lock_a_loop(self):
        fetch = AsyncMock(return_value=OPEN_METEO_ANSWER)
        with patch.object(widgets, "fetch_weather", fetch), patch.object(widgets, "WEATHER_EVERY_S", 0.0):
            run(self.cache.get())
            first = self.cache.lock
            run(self.cache.get())
        self.assertIsNot(self.cache.lock, first)


class TestHisDesk(unittest.TestCase):
    def setUp(self):
        for d in (widgets._LAST, widgets._FORGOTTEN):
            p = patch.dict(d, clear=True)
            p.start()
            self.addCleanup(p.stop)
        self.told = []
        widgets.watch(self.told.append)
        self.addCleanup(widgets._WATCHERS.remove, self.told.append)

    def test_none_yet(self):
        self.assertEqual(widgets.desk_payload("desktop-x"), DeskPayload(last=None))

    def test_the_last_exchange_cut_short_with_its_mood(self):
        widgets.record_exchange("desktop-x", "what time  is\nit", "A quarter past nine, sir.", "dry")
        last = widgets.desk_payload("desktop-x").last
        self.assertEqual((last.asked, last.answered, last.mood), ("what time is it", "A quarter past nine, sir.", "dry"))
        self.assertLess(abs((datetime.now(timezone.utc) - datetime.fromisoformat(last.at)).total_seconds()), 5)
        widgets.record_exchange("desktop-x", "q" * 500, "a" * 900, None)
        last = widgets.desk_payload("desktop-x").last
        self.assertEqual((len(last.asked), len(last.answered)), (widgets.ASKED_MAX_CHARS, widgets.ANSWERED_MAX_CHARS))
        self.assertTrue(last.answered.endswith("…"))
        self.assertIsNone(last.mood)
        self.assertEqual(self.told, ["desktop-x", "desktop-x"])

    def test_each_identity_its_own(self):
        widgets.record_exchange("desktop-a", "a?", "A.", None)
        self.assertIsNone(widgets.desk_payload("desktop-b").last)

    def test_forgotten_with_the_conversation(self):
        widgets.record_exchange("desktop-x", "q", "a", None)
        widgets.forget("desktop-x")
        self.assertIsNone(widgets.desk_payload("desktop-x").last)
        widgets.forget("desktop-x")    # nothing to forget: nobody told
        self.assertEqual(self.told, ["desktop-x", "desktop-x"])

    def test_a_turn_under_way_when_it_was_forgotten_leaves_nothing(self):
        since = widgets.generation("desktop-x")
        widgets.forget("desktop-x")
        self.assertIsNone(widgets.record_exchange("desktop-x", "q", "a", None, since=since))
        self.assertIsNone(widgets.desk_payload("desktop-x").last)
        self.assertEqual(self.told, [], "nobody told of it")
        # A turn begun after the forget is recorded.
        self.assertIsNotNone(widgets.record_exchange("desktop-x", "q", "a", None,
                                                     since=widgets.generation("desktop-x")))
        # Each identity counts its own.
        self.assertIsNotNone(widgets.record_exchange("desktop-y", "q", "a", None, since=0))

    def test_a_record_and_a_forget_each_hold_the_lock(self):
        """They come from different threads: the check and the store, and the
        count and the pop, are each one step to the other, and the watchers
        are told under it too, so a desk hears them in their order."""
        held, told = [], []
        real = widgets.generation

        def noting(identity):
            held.append(widgets._DESK_LOCK.locked())
            return real(identity)

        def watcher(identity):
            told.append(widgets._DESK_LOCK.locked())

        widgets.watch(watcher)
        self.addCleanup(widgets._WATCHERS.remove, watcher)
        with patch.object(widgets, "generation", noting):
            widgets.record_exchange("desktop-x", "q", "a", None, since=0)
            widgets.forget("desktop-x")
        self.assertEqual(held, [True, True])
        self.assertEqual(told, [True, True])

    def test_a_broken_watcher_breaks_nothing(self):
        def boom(identity):
            raise RuntimeError("watcher exploded")
        widgets.watch(boom)
        self.addCleanup(widgets._WATCHERS.remove, boom)
        with self.assertLogs(widgets.logger, logging.ERROR):
            widgets.record_exchange("desktop-x", "q", "a", None)
        self.assertEqual(widgets.desk_payload("desktop-x").last.asked, "q")


class TestWhatEachHolds(unittest.TestCase):
    def test_each_widget_by_name_is_its_payload_model(self):
        weather = widgets.parse_weather(OPEN_METEO_ANSWER)
        with patch.object(widgets.WEATHER, "get", AsyncMock(return_value=weather)), \
                patch.object(widgets.WEATHER, "ttl_left", lambda: 1234):
            for name in widgets.WIDGETS:
                payload, ttl = run(widgets.widget(name, "desktop-x"))
                self.assertIsInstance(payload, WIDGET_PAYLOADS[name], name)
                self.assertEqual(ttl, 1234 if name == "weather" else widgets.TTL_S[name])
        with patch.object(widgets.WEATHER, "get", AsyncMock(return_value=None)):
            self.assertIsNone(run(widgets.widget("weather", "desktop-x")))
        self.assertIsNone(run(widgets.widget("agenda", "desktop-x")))

    def test_the_brain_feeds_what_the_protocol_lists(self):
        self.assertEqual(widgets.WIDGETS, set(WIDGET_PAYLOADS))
        self.assertEqual(widgets.WIDGETS, {"clock", "weather", "desk"})
        self.assertEqual(set(widgets.TTL_S), widgets.WIDGETS)
        self.assertLessEqual(set(widgets.TIMED), widgets.WIDGETS)

    def test_when_each_is_offered_again(self):
        with patch.object(widgets, "WEATHER", widgets._Weather()):
            self.assertEqual((widgets.next_in("clock"), widgets.next_in("weather")), (60.0, 900.0))
            with patch.object(widgets.WEATHER, "next_in", lambda: 7.5):
                self.assertEqual(widgets.next_in("weather"), 7.5)


if __name__ == "__main__":
    unittest.main()
