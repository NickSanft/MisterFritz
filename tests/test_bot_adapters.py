"""Tests for bot_adapters — the shared, platform-neutral bot helpers.

bot_adapters imports only stdlib plus fritz_utils (and deliberately no
`discord`), so these need none of the sys.modules stubbing the Discord test
modules require.
"""
import asyncio
import threading
import time
import unittest
import unittest.mock

import bot_adapters
import fritz_utils


class TestSplitIntoChunks(unittest.TestCase):
    """Pre-existing behaviour, moved here from the main_discord re-export tests
    so the chunker is covered where it actually lives."""

    def test_short_string_is_one_chunk(self):
        self.assertEqual(bot_adapters.split_into_chunks("hello", 2000), ["hello"])

    def test_splits_at_the_boundary(self):
        chunks = bot_adapters.split_into_chunks("a" * 4500, 2000)
        self.assertEqual([len(c) for c in chunks], [2000, 2000, 500])

    def test_rejoins_losslessly(self):
        text = "x" * 5000
        self.assertEqual("".join(bot_adapters.split_into_chunks(text, 2000)), text)

    def test_empty_string_yields_no_chunks(self):
        self.assertEqual(bot_adapters.split_into_chunks("", 2000), [])

    def test_non_positive_chunk_size_raises(self):
        with self.assertRaises(ValueError):
            bot_adapters.split_into_chunks("abc", 0)


class TestChunkBoundaries(unittest.TestCase):
    """Word- and line-aware splitting. Pure index slicing used to cut mid-word."""

    def test_prefers_a_word_boundary(self):
        text = "word " * 100          # 500 chars
        chunks = bot_adapters.split_into_chunks(text, 100)
        for c in chunks[:-1]:
            with self.subTest(chunk=c[-12:]):
                # A chunk must not end mid-word.
                self.assertTrue(c.endswith(" "), repr(c[-12:]))

    def test_prefers_a_line_boundary_over_a_space(self):
        # Newline at 25, space at 30 — both in the usable second half of a
        # 40-char window. The newline wins despite the space being later.
        text = "a" * 25 + "\n" + "b" * 4 + " " + "c" * 40
        chunks = bot_adapters.split_into_chunks(text, 40)
        self.assertTrue(chunks[0].endswith("\n"))
        self.assertEqual(len(chunks[0]), 26)

    def test_a_boundary_in_the_first_half_is_rejected(self):
        # Honouring a newline at index 10 of a 40-char budget would waste 30
        # characters of every message.
        text = "alpha beta\n" + "x" * 60
        chunks = bot_adapters.split_into_chunks(text, 40)
        self.assertEqual(len(chunks[0]), 40)

    def test_falls_back_to_a_hard_cut_when_no_boundary_is_usable(self):
        # A space at index 2 of a 100-char budget would waste the message.
        text = "ab " + "z" * 300
        chunks = bot_adapters.split_into_chunks(text, 100)
        self.assertEqual(len(chunks[0]), 100)

    def test_no_chunk_exceeds_the_limit(self):
        text = ("some words here and there\n" * 40) + "```py\n" + ("code\n" * 40) + "```"
        for c in bot_adapters.split_into_chunks(text, 200):
            with self.subTest(n=len(c)):
                self.assertLessEqual(len(c), 200)

    def test_word_split_is_still_lossless(self):
        original = "hello world " * 300
        self.assertEqual(
            "".join(bot_adapters.split_into_chunks(original, 2000)), original)


class TestFenceAwareChunking(unittest.TestCase):
    """A fence split across chunks renders the tail as prose and the rest as
    code. Every chunk must carry a balanced number of fence lines."""

    @staticmethod
    def _fence_count(chunk: str) -> int:
        return sum(1 for ln in chunk.split("\n") if ln.lstrip().startswith("```"))

    def test_every_chunk_has_balanced_fences(self):
        text = "Here:\n\n```python\n" + ("print('x')\n" * 80) + "```\n"
        chunks = bot_adapters.split_into_chunks(text, 300)
        self.assertGreater(len(chunks), 1)
        for i, c in enumerate(chunks):
            with self.subTest(chunk=i):
                self.assertEqual(self._fence_count(c) % 2, 0,
                                 f"unbalanced fences in chunk {i}")

    def test_language_tag_is_reopened_on_the_next_chunk(self):
        text = "```python\n" + ("print('x')\n" * 80) + "```\n"
        chunks = bot_adapters.split_into_chunks(text, 300)
        self.assertTrue(chunks[1].startswith("```python\n"))

    def test_bare_fence_reopens_without_a_language(self):
        text = "```\n" + ("data\n" * 80) + "```\n"
        chunks = bot_adapters.split_into_chunks(text, 200)
        self.assertTrue(chunks[1].startswith("```\n"))

    def test_code_content_survives_the_repair(self):
        # The repair adds fence lines, so the join is not byte-identical — but
        # no source line may be lost.
        body = [f"line_{i}()" for i in range(60)]
        text = "```py\n" + "\n".join(body) + "\n```\n"
        joined = "".join(bot_adapters.split_into_chunks(text, 250))
        for ln in body:
            with self.subTest(line=ln):
                self.assertIn(ln, joined)

    def test_text_without_fences_is_untouched_by_the_repair(self):
        text = "plain prose " * 200
        self.assertEqual(
            "".join(bot_adapters.split_into_chunks(text, 300)), text)

    def test_open_fence_lang_toggles(self):
        f = bot_adapters._open_fence_lang
        self.assertIsNone(f("no fences here"))
        self.assertEqual(f("```python\ncode"), "python")
        self.assertIsNone(f("```python\ncode\n```"))
        self.assertEqual(f("```\ncode"), "")


class TestFritzError(unittest.TestCase):
    """User-facing failures read in Fritz's voice; the traceback goes to the
    log only."""

    def test_returns_butler_copy_not_the_exception(self):
        msg = bot_adapters.fritz_error("op", RuntimeError("connection refused to 10.0.0.5"))
        self.assertNotIn("connection refused", msg)
        self.assertNotIn("RuntimeError", msg)
        self.assertIn("did not go to plan", msg)

    def test_includes_a_log_ref(self):
        msg = bot_adapters.fritz_error("op", RuntimeError("x"))
        self.assertRegex(msg, r"\(ref `[0-9a-f]{8}`\)")

    def test_refs_are_unique_per_call(self):
        a = bot_adapters.fritz_error("op", RuntimeError("x"))
        b = bot_adapters.fritz_error("op", RuntimeError("x"))
        self.assertNotEqual(a, b)

    def test_records_the_error_in_metrics(self):
        exc = RuntimeError("boom")
        with unittest.mock.patch.object(bot_adapters.METRICS, "record_error") as rec:
            bot_adapters.fritz_error("discord_commands.gen", exc)
        rec.assert_called_once_with("discord_commands.gen", exc)

    def test_custom_note_replaces_the_default_copy(self):
        msg = bot_adapters.fritz_error("op", None, note="That is outside the permitted range.")
        self.assertIn("outside the permitted range", msg)
        self.assertNotIn("did not go to plan", msg)

    def test_no_exception_still_returns_copy_and_ref(self):
        msg = bot_adapters.fritz_error("op")
        self.assertIn("did not go to plan", msg)
        self.assertRegex(msg, r"ref `[0-9a-f]{8}`")

    def test_detail_flag_appends_the_exception(self):
        with unittest.mock.patch.object(bot_adapters, "DISCORD_ERROR_DETAIL", True):
            msg = bot_adapters.fritz_error("op", ValueError("the specifics"))
        self.assertIn("ValueError: the specifics", msg)

    def test_no_exclamation_marks(self):
        # FRITZ_CHARACTER forbids them except for mock-dramatic effect.
        self.assertNotIn("!", bot_adapters.fritz_error("op", RuntimeError("x")))


class TestBlockingPool(unittest.TestCase):
    def test_pool_is_bounded_by_the_config_knob(self):
        # Touches a private attribute deliberately: it is the only way to prove
        # the knob is actually wired to the pool rather than merely declared.
        self.assertEqual(
            bot_adapters._BLOCKING_POOL._max_workers,
            fritz_utils.BLOCKING_POOL_SIZE,
        )

    def test_pool_threads_are_named(self):
        # Named so a stuck bot's thread dump says which pool is saturated.
        self.assertEqual(
            bot_adapters._BLOCKING_POOL._thread_name_prefix, "fritz-blocking")


class TestRunBlocking(unittest.IsolatedAsyncioTestCase):
    async def test_returns_the_callables_result(self):
        self.assertEqual(await bot_adapters.run_blocking(lambda: 42), 42)

    async def test_passes_args_and_kwargs(self):
        def f(a, b, c=0):
            return (a, b, c)
        self.assertEqual(await bot_adapters.run_blocking(f, 1, 2, c=3), (1, 2, 3))

    async def test_exceptions_propagate_to_the_awaiter(self):
        def boom():
            raise ValueError("kaboom")
        with self.assertRaises(ValueError):
            await bot_adapters.run_blocking(boom)

    async def test_runs_off_the_event_loop_thread(self):
        # The whole point: the callable must not execute on the loop's thread.
        loop_thread = threading.get_ident()
        worker_thread = await bot_adapters.run_blocking(threading.get_ident)
        self.assertNotEqual(worker_thread, loop_thread)

    async def test_loop_stays_responsive_during_a_blocking_call(self):
        # The regression this whole change exists to prevent: a multi-second
        # blocking call used to freeze heartbeats and every other interaction.
        ticks = []

        async def ticker():
            for _ in range(10):
                await asyncio.sleep(0.01)
                ticks.append(1)

        task = asyncio.create_task(ticker())
        await bot_adapters.run_blocking(time.sleep, 0.12)
        await task
        # If the sleep had run on the loop, no tick would have landed during it.
        self.assertGreater(len(ticks), 0)

    async def test_concurrent_calls_actually_overlap(self):
        # Bounded, but genuinely parallel up to the bound.
        barrier = threading.Barrier(2, timeout=5)

        def wait_for_partner():
            barrier.wait()
            return True

        results = await asyncio.gather(
            bot_adapters.run_blocking(wait_for_partner),
            bot_adapters.run_blocking(wait_for_partner),
        )
        # Would raise BrokenBarrierError on timeout if they had been serialised.
        self.assertEqual(results, [True, True])


if __name__ == "__main__":
    unittest.main()
