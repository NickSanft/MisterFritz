"""
companion_protocol: the wire format between the brain and the desktop app
(plans/14-desktop-companion-build.md § 2).

The models are the source of truth for both ends, so these tests hold them to
the plan's examples, check the bounds that keep a client's input from being
echoed back at any size, and make sure no model is left out of the unions a
parser or the TypeScript generator walks.
"""
import inspect
import json
import typing
import unittest

from pydantic import BaseModel, ValidationError

import companion_protocol as cp
from companion_protocol import BadFrame, parse_client_frame, parse_server_frame

UUID = "7f3c2a10-4b5e-4f6a-9c8d-0e1f2a3b4c5d"


def union_members(union) -> set[type]:
    """The models in an Annotated[Union[...], Field(discriminator=...)]."""
    inner = typing.get_args(union)[0]
    return set(typing.get_args(inner))


class TestTheClientFrames(unittest.TestCase):
    def test_the_plans_hello(self):
        frame = parse_client_frame(
            '{"type":"hello","protocol":1,"token":"abc","client":{"id":"nick-pc",'
            '"name":"Nick\'s PC","caps":["text","audio-in","audio-out","vrm"]}}')
        self.assertIsInstance(frame, cp.Hello)
        self.assertEqual((frame.protocol, frame.token, frame.client.id, frame.client.name),
                         (1, "abc", "nick-pc", "Nick's PC"))
        self.assertEqual(frame.client.caps, ["text", "audio-in", "audio-out", "vrm"])

    def test_a_hello_without_a_protocol_is_version_one(self):
        frame = parse_client_frame(json.dumps({"type": "hello", "token": "t",
                                               "client": {"id": "pc", "name": "PC", "caps": []}}))
        self.assertEqual(frame.protocol, cp.PROTOCOL_VERSION)
        self.assertEqual(cp.PROTOCOL_VERSION, 1)

    def test_every_client_frame(self):
        for raw, kind in (
            ({"type": "say", "turn_id": UUID, "text": "good evening"}, cp.Say),
            ({"type": "audio_end", "turn_id": UUID}, cp.AudioEnd),
            ({"type": "cancel", "turn_id": UUID}, cp.Cancel),
            ({"type": "subscribe", "widgets": ["clock", "weather"]}, cp.Subscribe),
            ({"type": "ping", "t": 1731370000123}, cp.Ping),
            ({"type": "pong", "t": 1731370000123}, cp.Pong),
        ):
            with self.subTest(type=raw["type"]):
                frame = parse_client_frame(json.dumps(raw))
                self.assertIsInstance(frame, kind)
                self.assertEqual(frame.model_dump(exclude_defaults=False) | raw, frame.model_dump())

    def test_audio_end_defaults(self):
        frame = parse_client_frame(json.dumps({"type": "audio_end", "turn_id": "t"}))
        self.assertEqual((frame.sample_rate, frame.channels, frame.format), (16000, 1, "pcm16"))

    def test_a_client_id_canonical_user_id_keeps_intact(self):
        for good in ("nick-pc", "pc", "0", "a" * 32, "laptop-2"):
            with self.subTest(good=good):
                self.assertEqual(cp.ClientInfo(id=good, name="x", caps=[]).id, good)
        for bad in ("", "Nick-PC", "-pc", "a" * 33, "nick pc", "nick_pc", "pc\n", "pc!"):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                cp.ClientInfo(id=bad, name="x", caps=[])

    def test_the_client_name_is_bounded(self):
        cp.ClientInfo(id="pc", name="n" * 64, caps=[])
        with self.assertRaises(ValidationError):
            cp.ClientInfo(id="pc", name="n" * 65, caps=[])

    def test_a_cap_must_be_one_the_protocol_names(self):
        with self.assertRaises(ValidationError):
            cp.ClientInfo(id="pc", name="x", caps=["telepathy"])

    def test_a_turn_id_has_room_for_a_uuid_and_no_more_than_64(self):
        for good in (UUID, "t", "a" * 64, "turn_1"):
            with self.subTest(good=good):
                self.assertEqual(cp.Cancel(turn_id=good).turn_id, good)
        for bad in ("", "a" * 65, "turn 1", "t\n", "<script>", "t/1"):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                cp.Cancel(turn_id=bad)

    def test_what_is_said_is_stripped_then_measured(self):
        self.assertEqual(cp.Say(turn_id="t", text="  hello \n").text, "hello")
        self.assertEqual(len(cp.Say(turn_id="t", text="x" * 8000).text), 8000)
        self.assertEqual(len(cp.Say(turn_id="t", text=" " + "x" * 8000 + " ").text), 8000)
        for bad in ("", "   ", "\n\t", "x" * 8001):
            with self.subTest(length=len(bad)), self.assertRaises(ValidationError):
                cp.Say(turn_id="t", text=bad)


class TestBadFrames(unittest.TestCase):
    def test_what_is_not_a_client_frame(self):
        for raw in ("", "not json", "[]", "null", "42", '{"type":"nope"}', '{"no":"type"}',
                    '{"type":"welcome","protocol":1,"identity":"a","display_name":"a","server_caps":[],"thread_id":"a"}',
                    '{"type":"say","turn_id":"t"}', b"\xff\xfe"):
            with self.subTest(raw=raw), self.assertRaises(BadFrame):
                parse_client_frame(raw)

    def test_the_reason_is_brief_and_does_not_echo_what_was_said(self):
        secret = "my bank password is hunter2 " * 400
        with self.assertRaises(BadFrame) as caught:
            parse_client_frame(json.dumps({"type": "say", "turn_id": "t", "text": secret}))
        reason = str(caught.exception)
        self.assertNotIn("hunter2", reason)
        self.assertLessEqual(len(reason), 300)
        self.assertIn("text", reason)

    def test_a_broken_say_still_names_its_turn(self):
        """So the server can answer the turn the client is waiting on."""
        with self.assertRaises(BadFrame) as caught:
            parse_client_frame(json.dumps({"type": "say", "turn_id": UUID, "text": ""}))
        self.assertEqual(caught.exception.turn_id, UUID)

    def test_but_only_a_well_formed_turn_id(self):
        for raw in (json.dumps({"type": "say", "turn_id": "a" * 65, "text": ""}),
                    json.dumps({"type": "say", "turn_id": 123, "text": ""}),
                    json.dumps({"type": "say", "turn_id": ["t"], "text": ""}),
                    json.dumps({"type": "nope"}), "[1]", "not json", "[" * 100000):
            with self.subTest(raw=raw[:40]), self.assertRaises(BadFrame) as caught:
                parse_client_frame(raw)
            self.assertIsNone(caught.exception.turn_id)

    def test_it_is_a_value_error(self):
        self.assertTrue(issubclass(BadFrame, ValueError))


class TestTheServerFrames(unittest.TestCase):
    def test_the_plans_example_turn(self):
        """§ 2.3, line by line (the speech frame is Phase 2's)."""
        for raw in (
            '{"type":"welcome","protocol":1,"identity":"discord-1234","display_name":"Nick",'
            '"server_caps":["text","audio-in","audio-out"],"thread_id":"discord-1234#desktop"}',
            '{"type":"state","turn_id":"7f3","state":"thinking"}',
            '{"type":"token","turn_id":"7f3","delta":"Good evening, sir.","restart":true}',
            '{"type":"token","turn_id":"7f3","delta":" The hour is late"}',
            '{"type":"state","turn_id":"7f3","state":"speaking"}',
            '{"type":"speech","turn_id":"7f3","seq":0,"mime":"audio/wav","b64":"UklG","final":false}',
            '{"type":"done","turn_id":"7f3","text":"Good evening, sir. The hour is late",'
            '"images":[],"interrupted":false}',
            '{"type":"state","turn_id":"7f3","state":"idle"}',
        ):
            with self.subTest(raw=raw[:30]):
                frame = parse_server_frame(raw)
                self.assertEqual(frame.model_dump(), json.loads(raw) | frame.model_dump())

    def test_each_frame_says_its_own_type(self):
        """The server builds frames without naming the type; the wire must
        still carry it, or the client's discriminator has nothing to read."""
        for frame in (cp.Token(turn_id="t", delta="x"), cp.Done(turn_id="t", text="x"),
                      cp.Error(code="busy", text="x"), cp.Ping(t=1), cp.Pong(t=1),
                      cp.Welcome(protocol=1, identity="i", display_name="d", server_caps=["text"],
                                 thread_id="i#desktop")):
            with self.subTest(frame=type(frame).__name__):
                wire = json.loads(frame.model_dump_json())
                self.assertEqual(wire["type"], frame.type)
                self.assertEqual(parse_server_frame(frame.model_dump_json()), frame)

    def test_done_and_error_defaults(self):
        done = json.loads(cp.Done(turn_id="t", text="x").model_dump_json())
        self.assertEqual((done["images"], done["interrupted"]), ([], False))
        error = json.loads(cp.Error(code="bad_frame", text="x").model_dump_json())
        self.assertIsNone(error["turn_id"])

    def test_two_dones_do_not_share_their_images(self):
        first, second = cp.Done(turn_id="a", text=""), cp.Done(turn_id="b", text="")
        first.images.append("data:image/png;base64,AA==")
        self.assertEqual(second.images, [])

    def test_an_error_code_must_be_one_the_protocol_names(self):
        with self.assertRaises(ValidationError):
            cp.Error(code="oops", text="x")


class TestTheUnions(unittest.TestCase):
    """The parsers and the TypeScript generator walk ClientFrame and
    ServerFrame. A model missing from both would exist in Python only."""

    FRAMES = {obj for _name, obj in inspect.getmembers(cp, inspect.isclass)
              if issubclass(obj, BaseModel) and obj.__module__ == cp.__name__ and "type" in obj.model_fields}

    def test_every_frame_model_is_in_a_union(self):
        listed = union_members(cp.ClientFrame) | union_members(cp.ServerFrame)
        self.assertEqual(sorted(m.__name__ for m in self.FRAMES - listed), [])
        self.assertEqual(len(self.FRAMES), 18)

    def test_ping_and_pong_go_both_ways(self):
        """The server pings and the client mirrors it (§ 2.2)."""
        for union in (cp.ClientFrame, cp.ServerFrame):
            self.assertLessEqual({cp.Ping, cp.Pong}, union_members(union))

    def test_the_directions_are_otherwise_apart(self):
        both = union_members(cp.ClientFrame) & union_members(cp.ServerFrame)
        self.assertEqual(both, {cp.Ping, cp.Pong})

    def test_each_type_is_its_own_word(self):
        words = [m.model_fields["type"].default for m in self.FRAMES]
        self.assertEqual(len(words), len(set(words)))
        for model in self.FRAMES:
            with self.subTest(model=model.__name__):
                self.assertEqual(typing.get_args(model.model_fields["type"].annotation),
                                 (model.model_fields["type"].default,))


if __name__ == "__main__":
    unittest.main()
