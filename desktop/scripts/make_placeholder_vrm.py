"""Write desktop/public/placeholder.vrm: a small CC0 VRM 1.0 stand-in for Fritz.

The real model (plan 14, P6) is the owner's VRoid export, kept out of git and
loaded from %APPDATA%. This one is built from primitives so the repository can
carry a model that CI and a fresh checkout load, with no licence question: it
is dedicated to the public domain (CC0 1.0). A butler in a charcoal suit, white
gloves, a burgundy tie and a silver moustache, with what the body drives:

- a VRM 1.0 humanoid in T-pose (translation-only bones, facing +Z);
- the five visemes (aa ih ou ee oh), blink / blinkLeft / blinkRight and the
  five moods as morph targets on the face;
- MToon materials with outlines, and spring bones on the tie and coat tails,
  so the idle loop exercises roughly what a VRoid export costs.

    python desktop/scripts/make_placeholder_vrm.py [--out PATH]

Deterministic: the same script writes the same bytes. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "public" / "placeholder.vrm"

FLOAT, USHORT = 5126, 5123
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963

Vec = tuple[float, float, float]


def f32(v: float) -> float:
    """v as float32 will store it, so accessor min/max match the data exactly."""
    return struct.unpack("<f", struct.pack("<f", v))[0]


def add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: Vec, s: float) -> Vec:
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec, b: Vec) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(a: Vec) -> Vec:
    n = math.sqrt(dot(a, a))
    return scale(a, 1.0 / n) if n else (0.0, 1.0, 0.0)


# --- the skeleton -----------------------------------------------------------

# (name, parent, translation from the parent). VRM 1.0: T-pose, no rotation or
# scale on any humanoid bone, the model facing +Z, so its left is +X.
BONES: list[tuple[str, str | None, Vec]] = [
    ("hips", None, (0.0, 0.95, 0.0)),
    ("spine", "hips", (0.0, 0.10, 0.0)),
    ("chest", "spine", (0.0, 0.12, 0.0)),
    ("upperChest", "chest", (0.0, 0.12, 0.0)),
    ("neck", "upperChest", (0.0, 0.13, 0.0)),
    ("head", "neck", (0.0, 0.08, 0.0)),
]
for side, sx in (("left", 1.0), ("right", -1.0)):
    BONES += [
        (f"{side}Shoulder", "upperChest", (0.03 * sx, 0.09, 0.0)),
        (f"{side}UpperArm", f"{side}Shoulder", (0.13 * sx, 0.0, 0.0)),
        (f"{side}LowerArm", f"{side}UpperArm", (0.26 * sx, 0.0, 0.0)),
        (f"{side}Hand", f"{side}LowerArm", (0.24 * sx, 0.0, 0.0)),
        (f"{side}UpperLeg", "hips", (0.085 * sx, -0.06, 0.0)),
        (f"{side}LowerLeg", f"{side}UpperLeg", (0.0, -0.40, 0.0)),
        (f"{side}Foot", f"{side}LowerLeg", (0.0, -0.40, 0.0)),
        (f"{side}Toes", f"{side}Foot", (0.0, -0.05, 0.11)),
    ]
# Spring chains (not humanoid): the tie, and a coat tail each side.
BONES += [
    ("tie1", "upperChest", (0.0, 0.105, 0.118)),
    ("tie2", "tie1", (0.0, -0.09, 0.004)),
    ("tie3", "tie2", (0.0, -0.09, 0.003)),
    ("tieTip", "tie3", (0.0, -0.08, 0.002)),
]
for side, sx in (("L", 1.0), ("R", -1.0)):
    BONES += [
        (f"tail{side}1", "hips", (0.075 * sx, 0.0, -0.118)),
        (f"tail{side}2", f"tail{side}1", (0.0, -0.14, -0.006)),
        (f"tail{side}Tip", f"tail{side}2", (0.0, -0.14, -0.004)),
    ]

HUMANOID = [name for name, _, _ in BONES if not name.startswith(("tie", "tail"))]
REQUIRED = {"hips", "spine", "head", "leftUpperLeg", "leftLowerLeg", "leftFoot", "rightUpperLeg",
            "rightLowerLeg", "rightFoot", "leftUpperArm", "leftLowerArm", "leftHand",
            "rightUpperArm", "rightLowerArm", "rightHand"}

INDEX = {name: i for i, (name, _, _) in enumerate(BONES)}


def world(name: str) -> Vec:
    _, parent, t = BONES[INDEX[name]]
    return add(world(parent), t) if parent else t


# --- geometry -----------------------------------------------------------------

class Part:
    """Positions, normals and triangles, all bound rigidly to one joint."""

    def __init__(self, joint: str, centre: Vec | None = None):
        self.joint = joint
        self.centre = centre  # for the winding check on closed convex parts
        self.pos: list[Vec] = []
        self.nrm: list[Vec] = []
        self.tri: list[tuple[int, int, int]] = []

    def vertex(self, p: Vec, n: Vec) -> int:
        self.pos.append(p)
        self.nrm.append(norm(n))
        return len(self.pos) - 1


def ellipsoid(joint: str, c: Vec, r: Vec, seg_u: int = 24, seg_v: int = 14,
              theta_max=lambda phi: math.pi, closed: bool = True) -> Part:
    """Rows run down from the top (theta), columns round from +X towards +Z
    (phi), so (a, b, d) and (a, d, c) wind counter-clockwise seen from outside.
    `theta_max(phi)` cuts the bottom off (the hair is an open shell)."""
    part = Part(joint, c if closed else None)
    grid = []
    for i in range(seg_v + 1):
        row = []
        for j in range(seg_u + 1):
            phi = 2 * math.pi * j / seg_u
            theta = theta_max(phi) * i / seg_v
            u = (math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi))
            p = add(c, (r[0] * u[0], r[1] * u[1], r[2] * u[2]))
            row.append(part.vertex(p, (u[0] / r[0], u[1] / r[1], u[2] / r[2])))
        grid.append(row)
    for i in range(seg_v):
        for j in range(seg_u):
            a, b, cc, d = grid[i][j], grid[i][j + 1], grid[i + 1][j], grid[i + 1][j + 1]
            if i > 0:  # the top row's first triangle is degenerate at the pole
                part.tri.append((a, b, d))
            if i < seg_v - 1 or not closed:
                part.tri.append((a, d, cc))
    return part


def cylinder(joint: str, p0: Vec, p1: Vec, ra: float, rb: float, hint: Vec = (1.0, 0.0, 0.0),
             seg: int = 16) -> Part:
    """An elliptical tube from p0 to p1 with flat caps. `ra` lies along `hint`
    (made perpendicular to the axis), `rb` across it: e1 x e2 = axis, which
    makes (a, b, d) / (a, d, c) outward."""
    axis = norm(sub(p1, p0))
    e1 = norm(sub(hint, scale(axis, dot(hint, axis))))
    e2 = cross(axis, e1)
    part = Part(joint, scale(add(p0, p1), 0.5))

    def ring(p: Vec, j: int) -> tuple[Vec, Vec]:
        phi = 2 * math.pi * j / seg
        off = add(scale(e1, ra * math.cos(phi)), scale(e2, rb * math.sin(phi)))
        n = add(scale(e1, math.cos(phi) / ra), scale(e2, math.sin(phi) / rb))
        return add(p, off), n

    rows = [[part.vertex(*ring(p, j)) for j in range(seg + 1)] for p in (p0, p1)]
    for j in range(seg):
        a, b, c, d = rows[0][j], rows[0][j + 1], rows[1][j], rows[1][j + 1]
        part.tri += [(a, b, d), (a, d, c)]
    for p, outward in ((p0, scale(axis, -1.0)), (p1, axis)):
        centre = part.vertex(p, outward)
        edge = [part.vertex(ring(p, j)[0], outward) for j in range(seg + 1)]
        for j in range(seg):
            if outward is axis:
                part.tri.append((centre, edge[j], edge[j + 1]))
            else:
                part.tri.append((centre, edge[j + 1], edge[j]))
    return part


# (normal, s, t) with s x t = normal, so the quad below winds outward.
_FACES: list[tuple[Vec, Vec, Vec]] = [
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((0, 1, 0), (0, 0, 1), (1, 0, 0)), ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
]


def box(joint: str, c: Vec, h: Vec) -> Part:
    part = Part(joint, c)
    for n, s, t in _FACES:
        def corner(a: float, b: float) -> Vec:
            off = add(add((n[0] * h[0], n[1] * h[1], n[2] * h[2]),
                          (s[0] * h[0] * a, s[1] * h[1] * a, s[2] * h[2] * a)),
                      (t[0] * h[0] * b, t[1] * h[1] * b, t[2] * h[2] * b))
            return add(c, off)
        q = [part.vertex(corner(a, b), n) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        part.tri += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    return part


def mirrored(make) -> list[Part]:
    """make(sx, side) for the model's left (+X) and right."""
    return [make(1.0, "left"), make(-1.0, "right")]


HEAD_C, HEAD_R = (0.0, 1.60, 0.0), (0.095, 0.115, 0.105)


def on_face(x: float, y: float, inset: float = 0.0) -> Vec:
    """A point on the front of the head ellipsoid at (x, y), pulled in by inset."""
    k = 1 - (x / HEAD_R[0]) ** 2 - ((y - HEAD_C[1]) / HEAD_R[1]) ** 2
    return (x, y, HEAD_C[2] + HEAD_R[2] * math.sqrt(max(k, 0.0)) - inset)


def body_parts() -> dict[str, list[Part]]:
    suit: list[Part] = [
        cylinder("upperChest", (0, 1.43, 0), (0, 1.27, 0), 0.165, 0.11),
        cylinder("chest", (0, 1.29, 0), (0, 1.15, 0), 0.15, 0.105),
        cylinder("spine", (0, 1.17, 0), (0, 1.03, 0), 0.14, 0.10),
        cylinder("hips", (0, 1.05, 0), (0, 0.84, 0), 0.155, 0.105),
    ]
    for sx, side in ((1.0, "left"), (-1.0, "right")):
        up, low, hand = world(f"{side}UpperArm"), world(f"{side}LowerArm"), world(f"{side}Hand")
        hip, knee, foot = world(f"{side}UpperLeg"), world(f"{side}LowerLeg"), world(f"{side}Foot")
        suit += [
            ellipsoid(f"{side}UpperArm", up, (0.06, 0.06, 0.06), 16, 10),
            cylinder(f"{side}UpperArm", up, low, 0.052, 0.052, (0, 1, 0)),
            ellipsoid(f"{side}LowerArm", low, (0.05, 0.05, 0.05), 16, 10),
            cylinder(f"{side}LowerArm", low, add(hand, (-0.01 * sx, 0, 0)), 0.046, 0.046, (0, 1, 0)),
            cylinder(f"{side}UpperLeg", add(hip, (0, 0.01, 0)), knee, 0.072, 0.072),
            ellipsoid(f"{side}LowerLeg", knee, (0.065, 0.065, 0.065), 16, 10),
            cylinder(f"{side}LowerLeg", knee, add(foot, (0, 0.01, 0)), 0.058, 0.058),
            box(f"{side}Foot", (foot[0], 0.045, 0.045), (0.05, 0.045, 0.12)),
        ]
        tail = "L" if sx > 0 else "R"
        t1, t2, tip = world(f"tail{tail}1"), world(f"tail{tail}2"), world(f"tail{tail}Tip")
        suit += [
            box(f"tail{tail}1", scale(add(t1, t2), 0.5), (0.06, 0.07, 0.006)),
            box(f"tail{tail}2", scale(add(t2, tip), 0.5), (0.06, 0.07, 0.006)),
        ]
    shirt: list[Part] = [box("upperChest", (0, 1.345, 0.1065), (0.045, 0.085, 0.0115))]
    for sx, side in ((1.0, "left"), (-1.0, "right")):
        shirt.append(ellipsoid(f"{side}Hand", add(world(f"{side}Hand"), (0.07 * sx, 0, 0)),
                               (0.075, 0.03, 0.045), 16, 10))
    skin: list[Part] = [
        ellipsoid("head", HEAD_C, HEAD_R, 32, 20),
        cylinder("neck", (0, 1.40, 0), (0, 1.52, 0), 0.045, 0.045),
        ellipsoid("head", add(on_face(0.0, 1.585, 0.012), (0, 0, 0)), (0.012, 0.02, 0.015), 12, 8),
    ]
    for sx in (1.0, -1.0):
        skin.append(ellipsoid("head", (0.095 * sx, 1.60, -0.005), (0.015, 0.028, 0.02), 12, 8))
    hair: list[Part] = [
        # An open shell: low at the back, high over the brow, grey at the temples.
        ellipsoid("head", (0.0, 1.612, -0.006), (0.102, 0.122, 0.112), 32, 14,
                  theta_max=lambda phi: math.pi * (0.475 - 0.145 * math.sin(phi)), closed=False),
        ellipsoid("head", on_face(0.0, 1.567, -0.004), (0.03, 0.007, 0.008), 16, 8),
    ]
    tie: list[Part] = [
        ellipsoid("upperChest", world("tie1"), (0.018, 0.016, 0.012), 12, 8),
        box("tie1", scale(add(world("tie1"), world("tie2")), 0.5), (0.022, 0.045, 0.004)),
        box("tie2", scale(add(world("tie2"), world("tie3")), 0.5), (0.026, 0.045, 0.004)),
        box("tie3", scale(add(world("tie3"), world("tieTip")), 0.5), (0.03, 0.04, 0.004)),
    ]
    return {"Suit": suit, "Shirt": shirt, "Skin": skin, "Hair": hair, "Tie": tie}


# --- the face and its expressions ----------------------------------------------

EYE_Y, EYE_X, BROW_Y, BROW_X, MOUTH_Y = 1.615, 0.035, 1.648, 0.037, 1.553


def face_features() -> tuple[Part, dict[str, tuple[int, int]], dict[str, Vec]]:
    """One part (so one primitive and one set of morph targets) holding both
    eyes, both brows and the mouth, with the vertex range and centre of each."""
    face = Part("head")
    ranges: dict[str, tuple[int, int]] = {}
    centres: dict[str, Vec] = {}

    def take(name: str, piece: Part, centre: Vec) -> None:
        base = len(face.pos)
        face.pos += piece.pos
        face.nrm += piece.nrm
        face.tri += [(a + base, b + base, c + base) for a, b, c in piece.tri]
        ranges[name] = (base, len(face.pos))
        centres[name] = centre

    for sx, side in ((1.0, "L"), (-1.0, "R")):
        eye = on_face(EYE_X * sx, EYE_Y, 0.002)
        take(f"eye{side}", ellipsoid("head", eye, (0.012, 0.016, 0.006), 16, 10), eye)
        brow = on_face(BROW_X * sx, BROW_Y, -0.001)
        take(f"brow{side}", box("head", brow, (0.017, 0.0035, 0.004)), brow)
    mouth = on_face(0.0, MOUTH_Y, 0.002)
    take("mouth", ellipsoid("head", mouth, (0.02, 0.005, 0.006), 16, 8), mouth)
    return face, ranges, centres


def morph_targets(face: Part, ranges, centres) -> list[tuple[str, list[Vec]]]:
    """(name, per-vertex position deltas) for each preset the body drives."""
    def shape(eyes: float = 1.0, eyes_only: str = "", brow_lift: float = 0.0, brow_tilt: float = 0.0,
              mouth: tuple[float, float, float] = (1.0, 1.0, 0.0)) -> list[Vec]:
        out: list[Vec] = [(0.0, 0.0, 0.0)] * len(face.pos)
        for name, (lo, hi) in ranges.items():
            cx, cy, _ = centres[name]
            for i in range(lo, hi):
                x, y, _ = face.pos[i]
                if name.startswith("eye") and (not eyes_only or name.endswith(eyes_only)):
                    out[i] = (0.0, (y - cy) * (eyes - 1.0), 0.0)
                elif name.startswith("brow"):
                    # Tilt > 0 raises the outer end and drops the inner one.
                    out[i] = (0.0, brow_lift + brow_tilt * (abs(x) - abs(cx)) / 0.017, 0.0)
                elif name == "mouth":
                    sx, sy, dy = mouth
                    out[i] = ((x - cx) * (sx - 1.0), (y - cy) * (sy - 1.0) + dy, 0.0)
        return out

    return [
        ("aa", shape(mouth=(1.05, 3.6, -0.004))),
        ("ih", shape(mouth=(1.25, 1.8, -0.001))),
        ("ou", shape(mouth=(0.6, 2.4, -0.002))),
        ("ee", shape(mouth=(1.4, 1.4, 0.0))),
        ("oh", shape(mouth=(0.8, 3.0, -0.003))),
        ("blink", shape(eyes=0.08)),
        ("blinkLeft", shape(eyes=0.08, eyes_only="L")),
        ("blinkRight", shape(eyes=0.08, eyes_only="R")),
        ("happy", shape(eyes=0.55, brow_lift=0.004, mouth=(1.35, 1.6, 0.001))),
        ("angry", shape(eyes=0.8, brow_lift=-0.003, brow_tilt=0.005, mouth=(0.8, 1.0, 0.0))),
        ("sad", shape(eyes=0.85, brow_lift=0.001, brow_tilt=-0.005, mouth=(0.75, 1.2, -0.002))),
        ("relaxed", shape(eyes=0.5, brow_lift=-0.002, mouth=(1.15, 1.0, 0.0))),
        ("surprised", shape(eyes=1.35, brow_lift=0.01, mouth=(0.7, 3.0, -0.004))),
    ]


# --- materials ----------------------------------------------------------------

def mtoon(name: str, base: list[float], shade: list[float], outline: float = 0.004) -> dict:
    return {
        "name": name,
        "pbrMetallicRoughness": {"baseColorFactor": base + [1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0},
        "extensions": {"VRMC_materials_mtoon": {
            "specVersion": "1.0",
            "transparentWithZWrite": False,
            "renderQueueOffsetNumber": 0,
            "shadeColorFactor": shade,
            "shadingShiftFactor": 0.0,
            "shadingToonyFactor": 0.9,
            "giEqualizationFactor": 0.9,
            "matcapFactor": [0.0, 0.0, 0.0],
            "parametricRimColorFactor": [0.0, 0.0, 0.0],
            "rimLightingMixFactor": 1.0,
            "parametricRimFresnelPowerFactor": 5.0,
            "parametricRimLiftFactor": 0.0,
            "outlineWidthMode": "worldCoordinates" if outline else "none",
            "outlineWidthFactor": outline,
            "outlineColorFactor": [0.02, 0.02, 0.025],
            "outlineLightingMixFactor": 1.0,
        }},
    }


MATERIALS = {  # linear colour factors
    "Suit": mtoon("Suit", [0.035, 0.035, 0.042], [0.012, 0.012, 0.018]),
    "Shirt": mtoon("Shirt", [0.82, 0.82, 0.8], [0.45, 0.46, 0.5]),
    "Skin": mtoon("Skin", [0.8, 0.56, 0.45], [0.55, 0.32, 0.28]),
    "Hair": mtoon("Hair", [0.5, 0.5, 0.53], [0.25, 0.25, 0.3]),
    "Tie": mtoon("Tie", [0.22, 0.015, 0.03], [0.09, 0.005, 0.015]),
    "Face": mtoon("Face", [0.015, 0.012, 0.012], [0.01, 0.008, 0.008], outline=0.0),
}


# --- glTF ----------------------------------------------------------------------

class Writer:
    def __init__(self) -> None:
        self.bin = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def _view(self, data: bytes, target: int | None) -> int:
        self.bin += b"\0" * (-len(self.bin) % 4)
        view = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target:
            view["target"] = target
        self.bin += data
        self.views.append(view)
        return len(self.views) - 1

    def floats(self, rows: list, kind: str, target: int | None = ARRAY_BUFFER, bounds: bool = False) -> int:
        rows = [tuple(f32(v) for v in row) for row in rows]
        flat = [v for row in rows for v in row]
        acc = {"bufferView": self._view(struct.pack(f"<{len(flat)}f", *flat), target),
               "componentType": FLOAT, "count": len(rows), "type": kind}
        if bounds:
            width = len(rows[0])
            acc["min"] = [min(r[k] for r in rows) for k in range(width)]
            acc["max"] = [max(r[k] for r in rows) for k in range(width)]
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def ushorts(self, values: list[int], kind: str, target: int) -> int:
        count = len(values) // (4 if kind == "VEC4" else 1)
        acc = {"bufferView": self._view(struct.pack(f"<{len(values)}H", *values), target),
               "componentType": USHORT, "count": count, "type": kind}
        self.accessors.append(acc)
        return len(self.accessors) - 1


def merged(parts: list[Part]) -> Part:
    out = Part(parts[0].joint)
    out.joints: list[int] = []  # type: ignore[attr-defined]
    for p in parts:
        base = len(out.pos)
        out.pos += p.pos
        out.nrm += p.nrm
        out.tri += [(a + base, b + base, c + base) for a, b, c in p.tri]
        out.joints += [INDEX[p.joint]] * len(p.pos)  # type: ignore[attr-defined]
    return out


def primitive(w: Writer, part: Part, material: int, targets=None) -> dict:
    if len(part.pos) >= 65536:
        raise ValueError("a primitive outgrew 16-bit indices")
    joints = getattr(part, "joints", [INDEX[part.joint]] * len(part.pos))
    prim = {
        "attributes": {
            "POSITION": w.floats(part.pos, "VEC3", bounds=True),
            "NORMAL": w.floats(part.nrm, "VEC3"),
            "JOINTS_0": w.ushorts([v for j in joints for v in (j, 0, 0, 0)], "VEC4", ARRAY_BUFFER),
            "WEIGHTS_0": w.floats([(1.0, 0.0, 0.0, 0.0)] * len(part.pos), "VEC4"),
        },
        "indices": w.ushorts([i for t in part.tri for i in t], "SCALAR", ELEMENT_ARRAY_BUFFER),
        "material": material,
        "mode": 4,
    }
    if targets:
        prim["targets"] = [{"POSITION": w.floats(deltas, "VEC3", bounds=True)} for _, deltas in targets]
    return prim


def check_winding(parts: list[Part]) -> None:
    """Every triangle of a closed convex part faces away from its centre, so
    back-face culling (MToon's default) never hides the outside."""
    for part in parts:
        if part.centre is None:
            continue
        for a, b, c in part.tri:
            pa, pb, pc = part.pos[a], part.pos[b], part.pos[c]
            n = cross(sub(pb, pa), sub(pc, pa))
            if dot(n, n) < 1e-18:
                continue  # degenerate at a pole
            mid = scale(add(add(pa, pb), pc), 1 / 3)
            if dot(n, sub(mid, part.centre)) <= 0:
                raise AssertionError(f"inward triangle on a part bound to {part.joint}")


def build() -> bytes:
    if not REQUIRED <= set(HUMANOID):
        raise AssertionError(f"missing humanoid bones: {REQUIRED - set(HUMANOID)}")
    w = Writer()
    groups = body_parts()
    face, ranges, centres = face_features()
    check_winding([p for parts in groups.values() for p in parts])
    targets = morph_targets(face, ranges, centres)
    names = list(MATERIALS)

    body = {"name": "Body", "primitives": [
        primitive(w, merged(parts), names.index(material)) for material, parts in groups.items()]}
    face_mesh = {"name": "Face", "primitives": [primitive(w, face, names.index("Face"), targets)],
                 "weights": [0.0] * len(targets), "extras": {"targetNames": [n for n, _ in targets]}}

    nodes: list[dict] = []
    for name, _, t in BONES:
        node: dict = {"name": name}
        if any(t):
            node["translation"] = [f32(v) for v in t]
        kids = [INDEX[n] for n, parent, _ in BONES if parent == name]
        if kids:
            node["children"] = kids
        nodes.append(node)
    body_node, face_node = len(nodes), len(nodes) + 1
    nodes += [{"name": "Body", "mesh": 0, "skin": 0}, {"name": "Face", "mesh": 1, "skin": 0}]

    inverse_bind = []
    for name, _, _ in BONES:
        x, y, z = world(name)
        inverse_bind.append((1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -x, -y, -z, 1))
    skin = {"joints": list(range(len(BONES))), "skeleton": INDEX["hips"],
            "inverseBindMatrices": w.floats(inverse_bind, "MAT4", target=None)}

    def bind(name: str) -> dict:
        k = [n for n, _ in targets].index(name)
        return {"morphTargetBinds": [{"node": face_node, "index": k, "weight": 1.0}],
                "isBinary": False, "overrideBlink": "none", "overrideLookAt": "none", "overrideMouth": "none"}

    def joints(*chain: str) -> list[dict]:
        return [{"node": INDEX[n], "hitRadius": 0.01, "stiffness": 1.0, "gravityPower": 0.4,
                 "gravityDir": [0.0, -1.0, 0.0], "dragForce": 0.4} for n in chain]

    unrange = {"inputMaxValue": 90.0, "outputScale": 1.0}
    vrm = {
        "specVersion": "1.0",
        # CC0 waives every right, so every permission is at its most
        # permissive: the VRM licence fields must not claim a restriction the
        # dedication can't carry (placeholder.test.ts checks this).
        "meta": {
            "name": "Fritz (placeholder)",
            "version": "1",
            "authors": ["Mister Fritz contributors"],
            "copyrightInformation": "Dedicated to the public domain under CC0 1.0.",
            "licenseUrl": "https://vrm.dev/licenses/1.0/",
            "avatarPermission": "everyone",
            "allowExcessivelyViolentUsage": True,
            "allowExcessivelySexualUsage": True,
            "commercialUsage": "corporation",
            "allowPoliticalOrReligiousUsage": True,
            "allowAntisocialOrHateUsage": True,
            "creditNotation": "unnecessary",
            "allowRedistribution": True,
            "modification": "allowModificationRedistribution",
            "otherLicenseUrl": "https://creativecommons.org/publicdomain/zero/1.0/",
        },
        "humanoid": {"humanBones": {name: {"node": INDEX[name]} for name in HUMANOID}},
        "lookAt": {"offsetFromHeadBone": [0.0, 0.11, 0.07], "type": "expression",
                   "rangeMapHorizontalInner": unrange, "rangeMapHorizontalOuter": unrange,
                   "rangeMapVerticalDown": unrange, "rangeMapVerticalUp": unrange},
        "expressions": {"preset": {name: bind(name) for name, _ in targets}},
    }
    spring = {
        "specVersion": "1.0",
        "colliders": [
            {"node": INDEX["chest"], "shape": {"sphere": {"offset": [0.0, 0.0, 0.0], "radius": 0.1}}},
            {"node": INDEX["leftUpperLeg"], "shape": {"capsule": {"offset": [0.0, 0.0, 0.0], "radius": 0.08, "tail": [0.0, -0.4, 0.0]}}},
            {"node": INDEX["rightUpperLeg"], "shape": {"capsule": {"offset": [0.0, 0.0, 0.0], "radius": 0.08, "tail": [0.0, -0.4, 0.0]}}},
        ],
        "colliderGroups": [{"name": "Torso", "colliders": [0]}, {"name": "Legs", "colliders": [1, 2]}],
        "springs": [
            {"name": "Tie", "joints": joints("tie1", "tie2", "tie3", "tieTip"), "colliderGroups": [0]},
            {"name": "CoatTailL", "joints": joints("tailL1", "tailL2", "tailLTip"), "colliderGroups": [1]},
            {"name": "CoatTailR", "joints": joints("tailR1", "tailR2", "tailRTip"), "colliderGroups": [1]},
        ],
    }
    gltf = {
        "asset": {"version": "2.0", "generator": "Mister Fritz desktop/scripts/make_placeholder_vrm.py",
                  "copyright": "CC0 1.0"},
        "extensionsUsed": ["VRMC_vrm", "VRMC_springBone", "VRMC_materials_mtoon"],
        "scene": 0,
        "scenes": [{"nodes": [INDEX["hips"], body_node, face_node]}],
        "nodes": nodes,
        "meshes": [body, face_mesh],
        "skins": [skin],
        "materials": list(MATERIALS.values()),
        "accessors": w.accessors,
        "bufferViews": w.views,
        "buffers": [{"byteLength": len(w.bin) + (-len(w.bin) % 4)}],
        "extensions": {"VRMC_vrm": vrm, "VRMC_springBone": spring},
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    data = bytes(w.bin) + b"\0" * (-len(w.bin) % 4)
    total = 12 + 8 + len(js) + 8 + len(data)
    return (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(data), 0x004E4942) + data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    data = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    print(f"wrote {args.out} ({len(data):,} bytes)")


if __name__ == "__main__":
    main()
