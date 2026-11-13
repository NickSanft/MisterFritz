"""Recovering when Ollama's runner crashes loading a model (ollama_recovery).

No Ollama is needed: the crash is Ollama's own ResponseError, raised by fakes
standing in for the client, and the model list is a fake `ps()`.
"""
import asyncio
import types
import unittest
from unittest.mock import MagicMock, patch

import ollama
from langchain_core.messages import HumanMessage

import ollama_recovery
from ollama_recovery import (
    RecoveringChatOllama,
    call_with_recovery,
    is_runner_crash,
    make_room,
)

CRASH_TEXT = (
    "llama-server process has terminated: exit status 0xc0000409: The system "
    "detected an overrun of a stack-based buffer in this application. CUDA error"
)


def crash() -> ollama.ResponseError:
    """What Ollama 0.40.16 returned for the crash, as the client raises it."""
    return ollama.ResponseError('{"error":"%s"}' % CRASH_TEXT, 500)


def ps(*names):
    return types.SimpleNamespace(models=[types.SimpleNamespace(model=n) for n in names])


class FakeOllama:
    """`ps()` lists what is loaded; `generate(keep_alive=0)` unloads it."""

    def __init__(self, *loaded):
        self.loaded = list(loaded)
        self.unloaded = []

    def ps(self):
        return ps(*self.loaded)

    def generate(self, model, keep_alive):
        assert keep_alive == 0
        self.unloaded.append(model)
        self.loaded.remove(model)


class TheCrashIsRecognised(unittest.TestCase):
    def test_the_runner_crash_is_one(self):
        self.assertTrue(is_runner_crash(crash()))

    def test_other_errors_are_not(self):
        self.assertFalse(is_runner_crash(ollama.ResponseError("model 'x' not found", 404)))
        self.assertFalse(is_runner_crash(ollama.ResponseError(CRASH_TEXT, 400)), "not a 500")
        self.assertFalse(is_runner_crash(ollama.ResponseError("out of memory", 500)))
        self.assertFalse(is_runner_crash(RuntimeError(CRASH_TEXT)), "not Ollama's error")

    def test_a_model_that_cannot_load_is_not(self):
        # It fails every time: unloading the others and asking again only
        # makes them load again, for a second failure.
        self.assertFalse(is_runner_crash(ollama.ResponseError(
            "llama-server process has terminated: exit status 1: "
            "error loading model: unknown model architecture: 'x'", 500)))


class TheRecoveryClient(unittest.TestCase):
    def test_its_requests_have_a_timeout(self):
        # The brain's other direct calls have none; recovery must not hold a
        # turn longer than it saves.
        with patch.object(ollama_recovery.ollama, "Client") as client:
            ollama_recovery._client()
        self.assertEqual(client.call_args.kwargs["timeout"], ollama_recovery.REQUEST_TIMEOUT_S)
        self.assertLessEqual(ollama_recovery.REQUEST_TIMEOUT_S, 30)


class MakingRoom(unittest.TestCase):
    def test_every_other_model_is_unloaded(self):
        fake = FakeOllama("mxbai-embed-large:latest", "gpt-oss:latest", "llava:latest")
        self.assertEqual(make_room("gpt-oss", fake, wait_s=1),
                         ["mxbai-embed-large:latest", "llava:latest"])
        self.assertEqual(fake.loaded, ["gpt-oss:latest"])

    def test_the_tag_is_the_same_model(self):
        for asked, loaded in (("gpt-oss", "gpt-oss:latest"), ("gpt-oss:latest", "gpt-oss"),
                              ("gpt-oss:20b", "gpt-oss:20b")):
            fake = FakeOllama(loaded, "llava:latest")
            make_room(asked, fake, wait_s=1)
            self.assertEqual(fake.loaded, [loaded], asked)

    def test_another_tag_is_another_model(self):
        fake = FakeOllama("gpt-oss:20b")
        make_room("gpt-oss", fake, wait_s=1)
        self.assertEqual(fake.unloaded, ["gpt-oss:20b"])

    def test_it_waits_for_them_to_go(self):
        # Ollama unloads in the background: ps() still lists the model a while.
        fake = FakeOllama("llava:latest", "gpt-oss:latest")
        fake.generate = lambda model, keep_alive: None  # asked, not yet gone
        polls = []

        def ps_():
            polls.append(1)
            if len(polls) >= 3:
                fake.loaded = ["gpt-oss:latest"]
            return ps(*fake.loaded)

        fake.ps = ps_
        with patch.object(ollama_recovery.time, "sleep"):
            make_room("gpt-oss", fake, wait_s=5)
        self.assertGreaterEqual(len(polls), 3)

    def test_an_unload_that_fails_is_not_fatal(self):
        fake = FakeOllama("llava:latest", "mxbai-embed-large:latest")
        fake.generate = MagicMock(side_effect=[ConnectionError("gone"), None])
        with self.assertLogs("ollama_recovery", "WARNING"):
            make_room("gpt-oss", fake, wait_s=0)
        self.assertEqual(fake.generate.call_count, 2, "the second is still unloaded")


class CallWithRecovery(unittest.TestCase):
    def setUp(self):
        self.fake = FakeOllama("llava:latest", "gpt-oss:latest")
        p = patch.object(ollama_recovery, "_client", return_value=self.fake)
        p.start()
        self.addCleanup(p.stop)

    def test_a_crash_is_asked_again_alone(self):
        calls = []

        def chat(**kw):
            calls.append(list(self.fake.loaded))
            if len(calls) == 1:
                raise crash()
            return "reply"

        with self.assertLogs("ollama_recovery", "WARNING") as logs:
            self.assertEqual(call_with_recovery("gpt-oss", chat, model="gpt-oss"), "reply")
        self.assertEqual(calls, [["llava:latest", "gpt-oss:latest"], ["gpt-oss:latest"]])
        self.assertIn("crashed as it loaded", logs.output[0])
        self.assertIn("llava:latest", logs.output[0])

    def test_once_only(self):
        chat = MagicMock(side_effect=[crash(), crash(), "never"])
        with self.assertLogs("ollama_recovery", "WARNING"), self.assertRaises(ollama.ResponseError):
            call_with_recovery("gpt-oss", chat)
        self.assertEqual(chat.call_count, 2)

    def test_other_errors_are_not_retried(self):
        chat = MagicMock(side_effect=ollama.ResponseError("model 'x' not found", 404))
        with self.assertRaises(ollama.ResponseError):
            call_with_recovery("x", chat)
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(self.fake.unloaded, [])

    def test_the_crash_is_counted(self):
        with patch.object(ollama_recovery.METRICS, "increment") as inc, \
                self.assertLogs("ollama_recovery", "WARNING"):
            call_with_recovery("gpt-oss", MagicMock(side_effect=[crash(), "ok"]))
        inc.assert_called_once_with("ollama_runner_crash")

    def test_a_retry_is_made_even_if_ollama_cannot_be_asked(self):
        self.fake.ps = MagicMock(side_effect=ConnectionError("down"))
        chat = MagicMock(side_effect=[crash(), "ok"])
        with self.assertLogs("ollama_recovery", "WARNING"):
            self.assertEqual(call_with_recovery("gpt-oss", chat), "ok")


def _chunk(text, done=False):
    return {"model": "gpt-oss", "created_at": "2026-11-12T00:00:00Z",
            "message": {"role": "assistant", "content": text}, "done": done,
            **({"done_reason": "stop"} if done else {})}


class TheChatModelRecovers(unittest.TestCase):
    """RecoveringChatOllama, over a fake client: the real request path of
    ChatOllama (_create_chat_stream > self._client.chat) is what is wrapped."""

    def setUp(self):
        self.fake = FakeOllama("mxbai-embed-large:latest", "gpt-oss:latest")
        p = patch.object(ollama_recovery, "_client", return_value=self.fake)
        p.start()
        self.addCleanup(p.stop)
        self.llm = RecoveringChatOllama(model="gpt-oss")
        self.answers = []
        self.llm._client = types.SimpleNamespace(chat=self._chat)

    def _chat(self, **params):
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return iter(answer)

    def test_a_crash_before_the_first_word_is_asked_again(self):
        self.answers = [crash(), [_chunk("Good "), _chunk("evening.", done=True)]]
        with self.assertLogs("ollama_recovery", "WARNING"):
            reply = self.llm.invoke([HumanMessage("hello")])
        self.assertEqual(reply.content, "Good evening.")
        self.assertEqual(self.fake.unloaded, ["mxbai-embed-large:latest"])

    def test_streaming_too(self):
        self.answers = [crash(), [_chunk("Good "), _chunk("evening.", done=True)]]
        with self.assertLogs("ollama_recovery", "WARNING"):
            text = "".join(c.content for c in self.llm.stream([HumanMessage("hello")]))
        self.assertEqual(text, "Good evening.")

    def test_a_crash_after_words_went_out_is_not_retried(self):
        def broken():
            yield _chunk("Good ")
            raise crash()

        self.answers = [broken(), [_chunk("never", done=True)]]
        with self.assertRaises(ollama.ResponseError):
            list(self.llm.stream([HumanMessage("hello")]))
        self.assertEqual(len(self.answers), 1, "no second request")
        self.assertEqual(self.fake.unloaded, [])

    def test_two_crashes_are_the_callers(self):
        self.answers = [crash(), crash(), [_chunk("never", done=True)]]
        with self.assertLogs("ollama_recovery", "WARNING"), self.assertRaises(ollama.ResponseError):
            self.llm.invoke([HumanMessage("hello")])

    def test_other_errors_pass_through(self):
        self.answers = [ollama.ResponseError("model 'gpt-oss' not found", 404)]
        with self.assertRaises(ollama.ResponseError):
            self.llm.invoke([HumanMessage("hello")])
        self.assertEqual(self.fake.unloaded, [])

    def test_async_too(self):
        answers = [crash(), [_chunk("Good "), _chunk("evening.", done=True)]]

        async def achat(**params):
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer

            async def gen():
                for c in answer:
                    yield c
            return gen()

        self.llm._async_client = types.SimpleNamespace(chat=achat)
        with self.assertLogs("ollama_recovery", "WARNING"):
            reply = asyncio.run(self.llm.ainvoke([HumanMessage("hello")]))
        self.assertEqual(reply.content, "Good evening.")
        self.assertEqual(self.fake.unloaded, ["mxbai-embed-large:latest"])


class TheBrainUsesIt(unittest.TestCase):
    """Every chat model the brain makes, and its direct Ollama calls."""

    def test_the_conversation_models(self):
        import mister_fritz
        self.assertIsInstance(mister_fritz.ollama_instance, RecoveringChatOllama)
        self.assertIsInstance(mister_fritz.fast_ollama_instance, RecoveringChatOllama)

    def test_the_document_models(self):
        # document_engine is a stub under test (conftest), so read its source.
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / "document_engine.py").read_text(encoding="utf-8")
        self.assertIn("thinking_llm = RecoveringChatOllama(", source)
        self.assertIn("fast_llm = RecoveringChatOllama(", source)
        self.assertNotIn("= ChatOllama(", source)

    def test_memory_extraction(self):
        import agent_tools
        reply = types.SimpleNamespace(message=types.SimpleNamespace(content='{"facts": []}'))
        with patch.object(agent_tools, "_ollama_client") as client, \
                patch.object(ollama_recovery, "_client", return_value=FakeOllama()), \
                self.assertLogs("ollama_recovery", "WARNING"):
            client.chat.side_effect = [crash(), reply]
            agent_tools._extract_and_store_memories(
                "web-owner", "I live in a small flat with a cold kitchen, as it happens.",
                "Then your dough will want a longer rise; give it the warmest shelf you have.")
        self.assertEqual(client.chat.call_count, 2)

    def test_image_analysis(self):
        import inspect

        import agent_tools
        source = inspect.getsource(agent_tools)
        self.assertIn("call_with_recovery(\n                VISION_MODEL,\n                _ollama_client.chat,",
                      source.replace("\r\n", "\n"))


if __name__ == "__main__":
    unittest.main()
