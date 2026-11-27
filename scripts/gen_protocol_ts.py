"""Write the desktop app's TypeScript for the companion protocol.

companion_protocol.py is the protocol's source of truth (plan 14 build § 2).
This walks its frame unions and writes, into desktop/src/shared/:

- protocol.ts: an interface per frame, the ClientFrame and ServerFrame unions,
  the protocol's constants and close codes, and isServerFrame(x), a
  structural check of every frame the server sends; and for the widgets
  (PR 14) an interface per payload (WIDGET_PAYLOADS), WidgetPayloads (each
  by its widget's name), WIDGET_NAMES, WIDGETS_MAX and isWidgetPayload(name,
  x), the same check of a payload;
- protocol.fixtures.json: frames for the desktop's tests. The valid ones are
  built from the models and parsed back through companion_protocol; the
  invalid ones break one thing each. And each widget's payload, valid and
  broken one thing at a time (its WIDGET_PAYLOADS model, PR 14).

    python scripts/gen_protocol_ts.py            write both files
    python scripts/gen_protocol_ts.py --check    exit 1 if either is out of date
    python scripts/gen_protocol_ts.py --out DIR  somewhere else

tests/test_gen_protocol_ts.py fails while the committed files are out of date,
so CI's Python job holds the two ends together without the desktop's tools.

How Python's types are read:
- str, int, float, bool, dict, list[X], X | None and Literal map to string,
  number (an integer for int, any finite one for float), boolean,
  Record<string, unknown>, X[], X | null and a union of literals.
  Anything else stops the script, so a new kind of field is decided here.
- The module's Literal aliases (Cap, TurnState, ErrorCode, MoodName) become
  type aliases.
- A server frame's fields are all required: the server sends model_dump_json,
  defaults included. A client frame's defaulted fields are optional, but for
  `type`.
- Error is emitted as ErrorFrame, so it doesn't shadow the global Error, and
  Mood as MoodFrame, so it isn't taken for settings.ts's Mood (the colours).
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import types
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import companion_protocol as cp  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from pydantic.fields import FieldInfo  # noqa: E402

OUT = ROOT / "desktop" / "src" / "shared"
TS_FILE = "protocol.ts"
FIXTURES_FILE = "protocol.fixtures.json"

# The module's own Literal aliases, emitted as named types.
ALIASES = {"Cap": cp.Cap, "TurnState": cp.TurnState, "ErrorCode": cp.ErrorCode,
           "MoodName": cp.MoodName}
# A TypeScript name for a model whose own would shadow a global (Error) or be
# taken for another module's type (settings.ts's Mood, the colours).
RENAMED = {"Error": "ErrorFrame", "Mood": "MoodFrame"}

CLOSE_NAMES = {
    "goneQuiet": "CLOSE_GONE_QUIET",
    "badFrame": "CLOSE_BAD_FRAME",
    "badToken": "CLOSE_BAD_TOKEN",
    "noHello": "CLOSE_NO_HELLO",
    "full": "CLOSE_FULL",
    "unsupported": "CLOSE_UNSUPPORTED",
}

# What the samples say, by field (and by model, where one field name means
# two things). Each sample is checked by the models before it is written.
SAMPLE_TEXT = {
    "turn_id": "7f3c2a10-4b5e-4f6a-9c8d-0e1f2a3b4c5d",
    "id": "nick-pc",
    "token": "a-companion-token",
    "text": "Good evening, sir.",
    "delta": "Good evening",
    "identity": "desktop-nick-pc",
    "display_name": "Nick",
    "thread_id": "desktop-nick-pc#desktop",
    "b64": "UklGRiQAAABXQVZF",
}
SAMPLE_BY_MODEL = {
    ("Hello", "protocol"): cp.PROTOCOL_VERSION,
    ("Welcome", "protocol"): cp.PROTOCOL_VERSION,
    ("AudioEnd", "sample_rate"): 16000,
    ("Done", "interrupted"): False,
    ("ClientInfo", "name"): "Nick's PC",
    ("Widget", "name"): "weather",
    ("Subscribe", "widgets"): ["weather"],
    ("Done", "images"): ["data:image/png;base64,iVBORw0KGgo="],
    ("Widget", "payload"): {"temp_c": 11},
    ("Ping", "t"): 1763112000000,
    ("Pong", "t"): 1763112000000,
    ("Widget", "ttl_s"): 600,
    ("Reminder", "schedule_id"): "3f9c2a1b",
    ("ClockPayload", "now"): "2026-11-26T21:15:00-06:00",
    ("ClockPayload", "zone"): "America/Chicago",
    ("WeatherPayload", "code"): 2,
    ("WeatherPayload", "summary"): "Partly cloudy",
    ("WeatherPayload", "observed_at"): "2026-11-26T21:15-06:00",
    ("DeskExchange", "asked"): "what time is it",
    ("DeskExchange", "answered"): "A quarter past nine, sir.",
    ("DeskExchange", "at"): "2026-11-27T03:15:00+00:00",
}


def union_members(union) -> list[type[BaseModel]]:
    """The models of an Annotated[Union[...], Field(discriminator=...)], in order."""
    return list(typing.get_args(typing.get_args(union)[0]))


CLIENT = union_members(cp.ClientFrame)
PAYLOADS = cp.WIDGET_PAYLOADS
SERVER = union_members(cp.ServerFrame)
BOTH = [m for m in CLIENT if m in SERVER]


def ts_name(model: type[BaseModel]) -> str:
    return RENAMED.get(model.__name__, model.__name__)


def alias_of(annotation) -> str | None:
    for name, alias in ALIASES.items():
        if annotation is alias:
            return name
    return None


def is_optional(annotation) -> tuple[bool, object]:
    """Whether `annotation` is `X | None`, and X."""
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        args = typing.get_args(annotation)
        rest = [a for a in args if a is not type(None)]
        if len(rest) == 1 and len(args) == 2:
            return True, rest[0]
        raise SystemExit(f"gen_protocol_ts: can't map the union {annotation!r}")
    return False, annotation


def is_model(annotation) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def ts_type(annotation) -> str:
    optional, inner = is_optional(annotation)
    if optional:
        return f"{ts_type(inner)} | null"
    if (name := alias_of(annotation)) is not None:
        return name
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        return " | ".join(json.dumps(v) for v in typing.get_args(annotation))
    if origin is list:
        (item,) = typing.get_args(annotation)
        t = ts_type(item)
        return f"({t})[]" if "|" in t else f"{t}[]"
    if annotation is str:
        return "string"
    if annotation is bool:
        return "boolean"
    if annotation is int or annotation is float:
        return "number"
    if annotation is dict:
        return "Record<string, unknown>"
    if is_model(annotation):
        return ts_name(annotation)
    raise SystemExit(f"gen_protocol_ts: can't map {annotation!r}")


def ts_check(annotation) -> str:
    """The generated check for one field, built from protocol.ts's helpers."""
    optional, inner = is_optional(annotation)
    if optional:
        return f"orNull({ts_check(inner)})"
    if (name := alias_of(annotation)) is not None:
        return f"oneOf({values_const(name)})"
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        return f"oneOf([{', '.join(json.dumps(v) for v in typing.get_args(annotation))}])"
    if origin is list:
        (item,) = typing.get_args(annotation)
        return f"listOf({ts_check(item)})"
    if annotation is str:
        return "str"
    if annotation is bool:
        return "bool"
    if annotation is int:
        return "int"
    if annotation is float:
        return "num"
    if annotation is dict:
        return "isObj"
    if is_model(annotation):
        return shape_expr(annotation)
    raise SystemExit(f"gen_protocol_ts: can't check {annotation!r}")


def values_const(alias: str) -> str:
    """CAP_VALUES for Cap, TURN_STATE_VALUES for TurnState."""
    snake = "".join(f"_{c}" if c.isupper() and i else c for i, c in enumerate(alias)).upper()
    return f"{snake}_VALUES"


def shape_expr(model: type[BaseModel]) -> str:
    fields = ", ".join(f"{name}: {ts_check(f.annotation)}" for name, f in model.model_fields.items())
    return f"shape({{ {fields} }})"


def doc_comment(model: type[BaseModel], indent: str = "") -> list[str]:
    doc = model.__dict__.get("__doc__")
    if not doc:
        return []
    lines = inspect.cleandoc(doc).replace("*/", "*\\/").splitlines()
    return [f"{indent}/**", *(f"{indent} *{' ' + line if line else ''}" for line in lines), f"{indent} */"]


def optional_on_the_wire(model: type[BaseModel], name: str, field: FieldInfo) -> bool:
    """A client frame's defaulted field may be left out; `type` never is, and
    nothing a server sends is."""
    if name == "type" or model in SERVER:
        return False
    return not field.is_required()


def interface(model: type[BaseModel]) -> list[str]:
    out = doc_comment(model)
    out.append(f"export interface {ts_name(model)} {{")
    for name, field in model.model_fields.items():
        mark = "?" if optional_on_the_wire(model, name, field) else ""
        out.append(f"  {name}{mark}: {ts_type(field.annotation)};")
    out.append("}")
    return out


def nested_models(model: type[BaseModel]) -> list[type[BaseModel]]:
    """The models `model`'s fields hold, before it, in field order."""
    found: list[type[BaseModel]] = []

    def visit(annotation) -> None:
        _, inner = is_optional(annotation)
        if typing.get_origin(inner) is list:
            visit(typing.get_args(inner)[0])
        elif is_model(inner) and inner not in found:
            for deeper in nested_models(inner):
                if deeper not in found:
                    found.append(deeper)
            found.append(inner)

    for field in model.model_fields.values():
        visit(field.annotation)
    return found


def in_order(models: list[type[BaseModel]], done: set) -> list[type[BaseModel]]:
    out = []
    for m in models:
        for dep in [*nested_models(m), m]:
            if dep not in done:
                done.add(dep)
                out.append(dep)
    return out


def string_constraint(annotated, attr: str):
    """An attribute of the StringConstraints in an Annotated alias."""
    for meta in typing.get_args(annotated)[1:]:
        if getattr(meta, attr, None) is not None:
            return getattr(meta, attr)
    raise SystemExit(f"gen_protocol_ts: {annotated!r} has no {attr}")


def field_max_length(model: type[BaseModel], name: str) -> int:
    for meta in model.model_fields[name].metadata:
        if getattr(meta, "max_length", None) is not None:
            return meta.max_length
    raise SystemExit(f"gen_protocol_ts: {model.__name__}.{name} has no max_length")


def regex(pattern: str) -> str:
    return "/" + pattern.replace("/", "\\/") + "/"


def render_ts() -> str:
    for shared in BOTH:
        for name, field in shared.model_fields.items():
            if name != "type" and not field.is_required():
                raise SystemExit(f"gen_protocol_ts: {shared.__name__}.{name} has a default, and the frame "
                                 "goes both ways: decide whether it is optional")
    lines = [
        "// Generated by scripts/gen_protocol_ts.py from companion_protocol.py: do not",
        "// edit. Run `python scripts/gen_protocol_ts.py` after changing the protocol;",
        "// tests/test_gen_protocol_ts.py fails while this file is out of date.",
        "",
        "/** The version hello asks for and welcome answers with. */",
        f"export const PROTOCOL_VERSION = {cp.PROTOCOL_VERSION};",
        "/** A turn's id: what `say` names, and every frame of the turn echoes. */",
        f"export const TURN_ID_PATTERN = {regex(string_constraint(cp.TurnId, 'pattern'))};",
        "/** A client's id: its identity on the brain is `desktop-<id>`. */",
        f"export const CLIENT_ID_PATTERN = {regex(string_constraint(cp.ClientId, 'pattern'))};",
        "/** The longest `say`, in characters (code points), once trimmed. */",
        f"export const SAY_MAX_CHARS = {field_max_length(cp.Say, 'text')};",
        "/** The longest client name, in characters (code points). */",
        f"export const CLIENT_NAME_MAX_CHARS = {field_max_length(cp.ClientInfo, 'name')};",
        "/** The most widgets one subscribe may name. */",
        f"export const WIDGETS_MAX = {cp.WIDGETS_MAX};",
        "",
        "/** The codes the server closes a session with. */",
        "export const CLOSE = {",
        *(f"  {key}: {getattr(cp, const)}," for key, const in CLOSE_NAMES.items()),
        "} as const;",
        "",
    ]
    for name, alias in ALIASES.items():
        values = typing.get_args(alias)
        lines.append(f"export type {name} = {' | '.join(json.dumps(v) for v in values)};")
        lines.append(f"export const {values_const(name)}: readonly {name}[] = "
                     f"[{', '.join(json.dumps(v) for v in values)}];")
    done: set = set()
    for title, models in (("Client to server", [m for m in CLIENT if m not in BOTH]),
                          ("Server to client", [m for m in SERVER if m not in BOTH]),
                          ("Both ways: either side pings, and the other answers", BOTH)):
        lines += ["", f"// ── {title} " + "─" * max(0, 72 - len(title))]
        for model in in_order(models, done):
            lines += ["", *interface(model)]
    lines += ["", "// ── Widget payloads " + "─" * 57]
    for model in in_order(list(PAYLOADS.values()), done):
        lines += ["", *interface(model)]
    lines += [
        "",
        "/** Each widget's payload, by its name (a Widget frame's `name`). */",
        "export interface WidgetPayloads {",
        *(f"  {name}: {ts_name(model)};" for name, model in PAYLOADS.items()),
        "}",
        "/** Every widget the server may feed. */",
        "export const WIDGET_NAMES = [",
        *(f"  {json.dumps(name)}," for name in PAYLOADS),
        "] as const satisfies readonly (keyof WidgetPayloads)[];",
    ]
    lines += [
        "",
        f"export type ClientFrame = {' | '.join(ts_name(m) for m in CLIENT)};",
        f"export type ServerFrame = {' | '.join(ts_name(m) for m in SERVER)};",
        "",
        "/** Every `type` the server sends. */",
        "export const SERVER_FRAME_TYPES = [",
        *(f"  {json.dumps(m.model_fields['type'].default)}," for m in SERVER),
        '] as const satisfies readonly ServerFrame["type"][];',
        "",
        "// ── The check ────────────────────────────────────────────────────────────",
        "",
        "type Check = (v: unknown) => boolean;",
        "const isObj = (v: unknown): v is Record<string, unknown> =>",
        '  typeof v === "object" && v !== null && !Array.isArray(v);',
        'const str: Check = (v) => typeof v === "string";',
        "const int: Check = (v) => Number.isInteger(v);",
        'const bool: Check = (v) => typeof v === "boolean";',
        'const num: Check = (v) => typeof v === "number" && Number.isFinite(v);',
        "const oneOf = (values: readonly unknown[]): Check => (v) => values.includes(v);",
        "const listOf = (item: Check): Check => (v) => Array.isArray(v) && v.every((e) => item(e));",
        "const orNull = (inner: Check): Check => (v) => v === null || inner(v);",
        "const shape = (fields: Readonly<Record<string, Check>>): Check => (v) =>",
        "  isObj(v) && Object.entries(fields).every(([k, check]) => check(v[k]));",
        "",
        "/** Each server frame's fields, by its `type`. A Map, so no `type` reaches",
        " * Object.prototype (`constructor`, `__proto__`). */",
        "const SERVER_CHECKS: ReadonlyMap<string, Check> = new Map<string, Check>([",
        *(f"  [{json.dumps(m.model_fields['type'].default)}, {shape_expr(m)}]," for m in SERVER),
        "]);",
        "",
        "/** Whether `x` is a frame the server sends: an object whose `type` is one",
        " * of SERVER_FRAME_TYPES, with every field of that frame there, of its type.",
        " * Fields it doesn't know are allowed: a newer server may add some. */",
        "export function isServerFrame(x: unknown): x is ServerFrame {",
        '  if (!isObj(x) || typeof x.type !== "string") return false;',
        "  const check = SERVER_CHECKS.get(x.type);",
        "  return check !== undefined && check(x);",
        "}",
        "",
        "/** Each widget's payload's fields, by the widget's name. */",
        "const PAYLOAD_CHECKS: ReadonlyMap<string, Check> = new Map<string, Check>([",
        *(f"  [{json.dumps(name)}, {shape_expr(model)}]," for name, model in PAYLOADS.items()),
        "]);",
        "",
        "/** Whether `payload` is widget `name`'s, every field there, of its type.",
        " * Fields it doesn't know are allowed, as in a frame. */",
        "export function isWidgetPayload<K extends keyof WidgetPayloads>(name: K, payload: unknown): payload is WidgetPayloads[K] {",
        "  const check = PAYLOAD_CHECKS.get(name);",
        "  return check !== undefined && check(payload);",
        "}",
        "",
    ]
    return "\n".join(lines)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def sample(model: type[BaseModel], name: str, annotation):
    if (model.__name__, name) in SAMPLE_BY_MODEL:
        return SAMPLE_BY_MODEL[(model.__name__, name)]
    optional, inner = is_optional(annotation)
    if optional:
        return sample(model, name, inner)
    if typing.get_origin(inner) is typing.Literal:
        return typing.get_args(inner)[0]
    if typing.get_origin(inner) is list:
        return [sample(model, name, typing.get_args(inner)[0])]
    if inner is str:
        return SAMPLE_TEXT.get(name, name)
    if inner is bool:
        return True
    if inner is int:
        return 1
    if inner is float:
        return 12.5
    if inner is dict:
        return {"key": "value"}
    if is_model(inner):
        return valid_frame(inner)
    raise SystemExit(f"gen_protocol_ts: no sample for {model.__name__}.{name}")


def valid_frame(model: type[BaseModel]) -> dict:
    """A frame of `model` with every field given, as the models write it."""
    given = {name: sample(model, name, f.annotation) for name, f in model.model_fields.items() if name != "type"}
    return model.model_validate(given).model_dump(mode="json")


def wrong_values(annotation) -> list[tuple[str, object]]:
    """Values of the wrong JSON type for a field, each with why."""
    optional, inner = is_optional(annotation)
    out = [] if optional else [("null where null isn't allowed", None)]
    origin = typing.get_origin(inner)
    if origin is typing.Literal:
        out.append(("a value not in its Literal", "nonsense"))
        out.append(("a number for a Literal", 1))
    elif origin is list:
        out.append(("a string for a list", "x"))
        (item,) = typing.get_args(inner)
        out += [(f"a list holding {why}", [value]) for why, value in wrong_values(item)]
    elif inner is str:
        out.append(("a number for a string", 1))
    elif inner is int:
        out.append(("a string for an int", "1"))
        out.append(("a fraction for an int", 1.5))
    elif inner is float:
        out.append(("a string for a number", "12.5"))
    elif inner is bool:
        out.append(("a string for a bool", "true"))
        out.append(("a number for a bool", 1))
    elif inner is dict:
        out.append(("a list for an object", []))
    elif is_model(inner):
        out.append(("a string for an object", "x"))
    return out


def invalid_frames() -> list[dict]:
    out = []
    for model in SERVER:
        base = valid_frame(model)
        kind = base["type"]
        for name, field in model.model_fields.items():
            if name == "type":
                continue
            out.append({"why": f"{kind} without {name}",
                        "frame": {k: v for k, v in base.items() if k != name}})
            for why, value in wrong_values(field.annotation):
                out.append({"why": f"{kind}.{name}: {why}", "frame": {**base, name: value}})
    welcome = valid_frame(cp.Welcome)
    out += [
        {"why": "an unknown type", "frame": {**welcome, "type": "nonsense"}},
        {"why": "a type Object.prototype has", "frame": {**welcome, "type": "constructor"}},
        {"why": "a type that is __proto__", "frame": {**welcome, "type": "__proto__"}},
        {"why": "no type", "frame": {k: v for k, v in welcome.items() if k != "type"}},
        {"why": "a type that isn't a string", "frame": {**welcome, "type": 1}},
        {"why": "an array", "frame": [welcome]},
        {"why": "a string", "frame": "welcome"},
        {"why": "null", "frame": None},
        {"why": "a client's frame", "frame": valid_frame(cp.Say)},
    ]
    return out


def literal_values(annotation) -> tuple:
    """The values a Literal field (or a list of one) may take, else ()."""
    _, inner = is_optional(annotation)
    if typing.get_origin(inner) is list:
        inner = typing.get_args(inner)[0]
    return typing.get_args(inner) if typing.get_origin(inner) is typing.Literal else ()


def valid_server_frames() -> list[dict]:
    out = []
    for model in SERVER:
        frame = valid_frame(model)
        out.append({"why": frame["type"], "frame": frame})
        for name, field in model.model_fields.items():
            if is_optional(field.annotation)[0]:
                out.append({"why": f"{frame['type']} with {name} null", "frame": {**frame, name: None}})
            if name == "type":
                continue
            # Every value it may take, not only the sample's first.
            listed = typing.get_origin(is_optional(field.annotation)[1]) is list
            for value in literal_values(field.annotation)[1:]:
                out.append({"why": f"{frame['type']} with {name} {value}",
                            "frame": {**frame, name: [value] if listed else value}})
    welcome = valid_frame(cp.Welcome)
    out.append({"why": "welcome with a field this version doesn't know",
                "frame": {**welcome, "since_v2": True}})
    for entry in out:
        cp.parse_server_frame(json.dumps(entry["frame"]))  # each must parse
    return out


def payload_cases(model: type[BaseModel], base: dict, path: str = "") -> tuple[list, list]:
    """`base` (a valid `model`) changed one thing at a time, as (why,
    payload): the valid ways (a null, each Literal's other values) and the
    broken ones (a field missing, of a wrong type), down into the models its
    fields hold, by the field's path."""
    valid, invalid = [], []
    for field_name, field in model.model_fields.items():
        at = f"{path}{field_name}"
        optional, inner = is_optional(field.annotation)
        listed = typing.get_origin(inner) is list
        if optional:
            valid.append((f"with {at} null", {**base, field_name: None}))
        for value in literal_values(field.annotation)[1:]:
            valid.append((f"with {at} {value}", {**base, field_name: [value] if listed else value}))
        invalid.append((f"without {at}", {k: v for k, v in base.items() if k != field_name}))
        for why, value in wrong_values(field.annotation):
            invalid.append((f"{at}: {why}", {**base, field_name: value}))
        if is_model(inner) and isinstance(base.get(field_name), dict):
            deeper_valid, deeper_invalid = payload_cases(inner, base[field_name], f"{at}.")
            valid += [(why, {**base, field_name: p}) for why, p in deeper_valid]
            invalid += [(why, {**base, field_name: p}) for why, p in deeper_invalid]
    return valid, invalid


def payload_fixtures() -> dict:
    """Each widget's payload: valid (each checked by its model), and broken
    one thing at a time, by the widget's name."""
    valid, invalid = [], []
    for name, model in PAYLOADS.items():
        base = valid_frame(model)
        valid.append({"why": name, "name": name, "payload": base})
        good, bad = payload_cases(model, base)
        valid += [{"why": f"{name} {why}", "name": name, "payload": p} for why, p in good]
        invalid += [{"why": f"{name}: {why}", "name": name, "payload": p} for why, p in bad]
        invalid.append({"why": f"{name}: not an object", "name": name, "payload": [base]})
    for entry in valid:
        PAYLOADS[entry["name"]].model_validate(entry["payload"])  # each must hold
    return {"valid": valid, "invalid": invalid}


def render_fixtures() -> str:
    client = {}
    for model in CLIENT:
        frame = valid_frame(model)
        cp.parse_client_frame(json.dumps(frame))  # each must parse
        client[frame["type"]] = frame
    fixtures = {
        "about": ("Generated by scripts/gen_protocol_ts.py from companion_protocol.py: do not "
                  "edit. Frames for the desktop's tests: valid server frames (each parsed back "
                  "by companion_protocol) and invalid ones (each breaking one thing), a "
                  "valid frame of each type the client sends, and each widget's payload, "
                  "valid and broken one thing at a time."),
        "server": {"valid": valid_server_frames(), "invalid": invalid_frames()},
        "client": {"valid": client},
        "widgets": payload_fixtures(),
    }
    return json.dumps(fixtures, indent=2, ensure_ascii=False) + "\n"


def render() -> dict[str, str]:
    """Each file's text, by name."""
    return {TS_FILE: render_ts(), FIXTURES_FILE: render_fixtures()}


def stale(out: Path) -> list[str]:
    """The files in `out` that differ from a fresh render (or are missing).
    Line endings don't count: git may check them out as CRLF."""
    differ = []
    for name, text in render().items():
        path = out / name
        try:
            on_disk = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        except OSError:
            on_disk = None
        if on_disk != text:
            differ.append(name)
    return differ


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if a file is out of date")
    parser.add_argument("--out", type=Path, default=OUT, help=f"where to write (default {OUT})")
    args = parser.parse_args(argv)
    if args.check:
        differ = stale(args.out)
        for name in differ:
            print(f"{args.out / name} is out of date: run python scripts/gen_protocol_ts.py",
                  file=sys.stderr)
        return 1 if differ else 0
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in render().items():
        (args.out / name).write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {args.out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
