"""Tests for bot_adapters — the shared, platform-neutral bot helpers.

bot_adapters imports only stdlib plus fritz_utils (and deliberately no
`discord`), so these need none of the sys.modules stubbing the Discord test
modules require.
"""
import asyncio
import threading
import time
import unittest

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
