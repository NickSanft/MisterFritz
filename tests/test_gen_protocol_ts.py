"""
scripts/gen_protocol_ts.py: the desktop app's TypeScript for the companion
protocol (plan 14, PR 3), generated from companion_protocol.py.

The committed protocol.ts and protocol.fixtures.json must be what the
generator writes now, so a change to the protocol that isn't regenerated fails
here, in CI's Python job, without the desktop's tools. The fixtures' valid
frames must parse with the models, and their invalid ones must each be
refused by something: the desktop's vitest (src/shared/protocol.test.ts)
holds isServerFrame to both lists.
"""
import importlib.util
import json
import shutil
import tempfile
import typing
import unittest
from unittest import mock
from pathlib import Path

import companion_api
import companion_protocol as cp

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("gen_protocol_ts", ROOT / "scripts" / "gen_protocol_ts.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def committed(name: str) -> str:
    # git may check the files out with CRLF; the generator writes LF.
    return (gen.OUT / name).read_text(encoding="utf-8").replace("\r\n", "\n")


class TestTheCommittedFilesAreCurrent(unittest.TestCase):
    def test_protocol_ts_is_what_the_generator_writes(self):
        self.assertEqual(committed(gen.TS_FILE), gen.render_ts(),
                         "protocol.ts is out of date: run python scripts/gen_protocol_ts.py")

    def test_the_fixtures_are_what_the_generator_writes(self):
        self.assertEqual(committed(gen.FIXTURES_FILE), gen.render_fixtures(),
                         "protocol.fixtures.json is out of date: run python scripts/gen_protocol_ts.py")

    def test_check_passes_on_the_committed_files(self):
        self.assertEqual(gen.main(["--check"]), 0)


class TestCheck(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="gen-protocol-ts-"))
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.assertEqual(gen.main(["--out", str(self.dir)]), 0)

    def test_a_fresh_copy_passes(self):
        self.assertEqual(gen.main(["--check", "--out", str(self.dir)]), 0)

    def test_crlf_line_endings_pass(self):
        path = self.dir / gen.TS_FILE
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        self.assertEqual(gen.main(["--check", "--out", str(self.dir)]), 0)

    def test_a_tampered_file_fails_and_is_named(self):
        for name in (gen.TS_FILE, gen.FIXTURES_FILE):
            with self.subTest(name):
                path = self.dir / name
                original = path.read_text(encoding="utf-8")
                path.write_text(original.replace("turn_id", "turnId", 1), encoding="utf-8")
                with mock.patch("sys.stderr") as err:
                    self.assertEqual(gen.main(["--check", "--out", str(self.dir)]), 1)
                said = "".join(call.args[0] for call in err.write.call_args_list)
                self.assertIn(name, said)
                path.write_text(original, encoding="utf-8")

    def test_a_missing_file_fails(self):
        (self.dir / gen.FIXTURES_FILE).unlink()
        with mock.patch("sys.stderr"):
            self.assertEqual(gen.main(["--check", "--out", str(self.dir)]), 1)


class TestWhatIsGenerated(unittest.TestCase):
    ts = gen.render_ts()
    fixtures = json.loads(gen.render_fixtures())

    def test_every_frame_has_an_interface(self):
        for model in [*gen.CLIENT, *gen.SERVER, cp.ClientInfo]:
            self.assertIn(f"export interface {gen.ts_name(model)} {{", self.ts)
        # Error would shadow the global.
        self.assertIn("export interface ErrorFrame {", self.ts)
        self.assertNotIn("export interface Error {", self.ts)

    def test_the_unions_name_every_frame(self):
        server = " | ".join(gen.ts_name(m) for m in typing.get_args(typing.get_args(cp.ServerFrame)[0]))
        client = " | ".join(gen.ts_name(m) for m in typing.get_args(typing.get_args(cp.ClientFrame)[0]))
        self.assertIn(f"export type ServerFrame = {server};", self.ts)
        self.assertIn(f"export type ClientFrame = {client};", self.ts)
        for model in gen.SERVER:
            self.assertIn(f'["{model.model_fields["type"].default}", shape(', self.ts)

    def test_the_constants_are_the_protocols(self):
        self.assertIn(f"export const PROTOCOL_VERSION = {cp.PROTOCOL_VERSION};", self.ts)
        self.assertIn("export const TURN_ID_PATTERN = /^[A-Za-z0-9_-]{1,64}$/;", self.ts)
        self.assertIn("export const CLIENT_ID_PATTERN = /^[a-z0-9][a-z0-9-]{0,31}$/;", self.ts)
        self.assertIn("export const SAY_MAX_CHARS = 8000;", self.ts)
        self.assertIn("export const CLIENT_NAME_MAX_CHARS = 64;", self.ts)
        for key, const in gen.CLOSE_NAMES.items():
            self.assertIn(f"  {key}: {getattr(cp, const)},", self.ts)

    def test_server_fields_are_required_and_client_defaults_optional(self):
        # The server sends model_dump_json: Token.restart is always there.
        self.assertIn("  restart: boolean;", self.ts)
        self.assertIn("  turn_id: string | null;", self.ts)
        # A client may leave a defaulted field out, but never `type`.
        self.assertIn("  protocol?: number;", self.ts)
        self.assertIn("  sample_rate?: number;", self.ts)
        self.assertIn('  type: "hello";', self.ts)

    def test_an_unknown_kind_of_field_stops_the_generator(self):
        with self.assertRaises(SystemExit):
            gen.ts_type(float)
        with self.assertRaises(SystemExit):
            gen.ts_check(set[str])
        with self.assertRaises(SystemExit):
            gen.ts_type(int | str)

    def test_every_valid_server_frame_parses(self):
        valid = self.fixtures["server"]["valid"]
        types = {entry["frame"]["type"] for entry in valid}
        self.assertEqual(types, {m.model_fields["type"].default for m in gen.SERVER})
        for entry in valid:
            with self.subTest(entry["why"]):
                cp.parse_server_frame(json.dumps(entry["frame"]))

    def test_every_value_of_each_literal_is_a_valid_frame(self):
        valid = [entry["frame"] for entry in self.fixtures["server"]["valid"]]
        for state in typing.get_args(cp.TurnState):
            self.assertIn({**gen.valid_frame(cp.State), "state": state}, valid)
        for code in typing.get_args(cp.ErrorCode):
            self.assertIn({**gen.valid_frame(cp.Error), "code": code}, valid)
        for cap in typing.get_args(cp.Cap):
            self.assertIn({**gen.valid_frame(cp.Welcome), "server_caps": [cap]}, valid)

    def test_every_valid_client_frame_parses(self):
        valid = self.fixtures["client"]["valid"]
        self.assertEqual(set(valid), {m.model_fields["type"].default for m in gen.CLIENT})
        for kind, frame in valid.items():
            with self.subTest(kind):
                self.assertEqual(cp.parse_client_frame(json.dumps(frame)).type, kind)

    def test_the_invalid_frames_break_each_field(self):
        invalid = self.fixtures["server"]["invalid"]
        whys = [entry["why"] for entry in invalid]
        self.assertEqual(len(whys), len(set(whys)), "each says what it breaks")
        for model in gen.SERVER:
            kind = model.model_fields["type"].default
            for name in model.model_fields:
                if name != "type":
                    self.assertIn(f"{kind} without {name}", whys)
        self.assertIn("a type Object.prototype has", whys)


class TestTheCloseCodesMoved(unittest.TestCase):
    def test_companion_api_still_has_them(self):
        for const in gen.CLOSE_NAMES.values():
            self.assertEqual(getattr(companion_api, const), getattr(cp, const))
        self.assertEqual((cp.CLOSE_GONE_QUIET, cp.CLOSE_BAD_FRAME, cp.CLOSE_BAD_TOKEN, cp.CLOSE_NO_HELLO,
                          cp.CLOSE_FULL, cp.CLOSE_UNSUPPORTED), (4000, 4400, 4401, 4408, 4409, 4426))


if __name__ == "__main__":
    unittest.main()
