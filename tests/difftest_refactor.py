"""Differential harness: identical inputs through image.py before and after a split.

WHY THIS EXISTS. image.py sits at ~15% coverage, so a green suite proves very
little about a refactor that moved 5,800 lines between modules. This harness
does not assert on expected VALUES at all -- it captures the actual output of
every pure/deterministic entry point and lets you diff two checkouts. Anything
that changed shows up; anything that did not is proven unchanged.

It is the only evidence available for the ComfyUI workflow builders when the GPU
is busy: it cannot prove a graph RENDERS, but it does prove the emitted workflow
JSON is byte-identical to what the pre-split code emitted.

USAGE -- run the SAME copy of this file against two trees and diff:

    git worktree add --detach /tmp/pre <commit-before-the-refactor>
    cd <refactored tree> && PYTHONPATH=. venv/Scripts/python.exe -u         tests/difftest_refactor.py > /tmp/after.json
    cd /tmp/pre        && PYTHONPATH=. <same venv>/python.exe -u         /path/to/this/file    > /tmp/before.json
    diff /tmp/before.json /tmp/after.json      # empty == behaviour preserved
    git worktree remove --force /tmp/pre

Note you must invoke the same FILE in both trees (not the tree's own copy), or
you are diffing two different harnesses.

Everything here is deterministic and offline: seeded RNG, fixed seeds passed to
the graph builders, no LLM, no network, no GPU. Absolute paths that depend on
which checkout you are in are scrubbed by _scrub() so the diff shows BEHAVIOUR
differences only.

Result when it was written (image.py 4,696 -> 1,042 lines across 8 extractions):
byte-identical, 23 groups / 730 lines.
"""
import json
import os
import random
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
random.seed(20260810)   # build_edit_plan draws seeds; pin them so the diff is meaningful
import logging
logging.disable(logging.CRITICAL)

from PIL import Image

import image as M

OUT = {}


_REPO = os.path.dirname(os.path.abspath(__file__))


def _scrub(v):
    """Strip checkout- and tempdir-dependent absolute paths so the diff shows
    BEHAVIOUR differences only, not the fact that the two trees live in
    different directories."""
    if isinstance(v, str):
        low = v.replace("\\\\", "/")
        for marker in ("/scratchpad/pre/", "/f5-refactor/image/"):
            i = low.find(marker)
            if i != -1:
                low = "<TREE>/" + low[i + len(marker):]
        i = low.lower().find("difftest_")
        if i != -1:
            j = low.find("/", i)
            low = low[:i] + "<TMP>" + (low[j:] if j != -1 else "")
        return low
    if isinstance(v, dict):
        return {k: _scrub(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_scrub(x) for x in v]
    return v


def rec(name, fn):
    try:
        OUT[name] = _scrub(fn())
    except Exception as e:                      # record the failure, don't hide it
        OUT[name] = "EXC:%s:%s" % (type(e).__name__, e)


# ---------------------------------------------------------------- fixtures
_TMP = tempfile.mkdtemp(prefix="difftest_")
SRC = os.path.join(_TMP, "src.png")
Image.new("RGB", (1280, 720), "gray").save(SRC)
MASK = os.path.join(_TMP, "mask.png")
Image.new("L", (1280, 720), 0).save(MASK)

# ---------------------------------------------------------------- 1. router
CASES = [
    "upscale it", "make it 4k and sharper", "remove the background",
    "replace the background with a beach", "remove the hat", "убери шляпу",
    "remove the man on the left", "add sunglasses to him",
    "restore this old damaged photo", "fix the text on the sign", "исправь надписи",
    "make the light come from the left", "extend the canvas to the left",
    "make her jacket red", "redraw this in anime style",
    "convert it to black and white", "give her blue eyes",
    "make the sky more dramatic", "delete the woman in red", "blur the background",
    "сделай фон белым", "add a hat and remove the glasses",
    "make it look like a painting", "sharpen the eyes", "crop tighter",
    "turn the shirt into a leather jacket", "увеличь разрешение",
    "erase the logo", "put her on a mountain top", "fix the lighting on her face",
]
rec("router.classify", lambda: {c: M.classify_edit_intent(c) for c in CASES})

# ---------------------------------------------------------------- 2. sizing
SIZE_CASES = [
    "draw a cat", "draw a wide banner", "a tall portrait of a knight",
    "нарисуй кота", "a 512x512 icon", "make a 4k wallpaper, landscape",
]
rec("sizing.parse_generation_params",
    lambda: {c: list(M.parse_generation_params(c)) for c in SIZE_CASES})
rec("sizing.normalize_resolution",
    lambda: {f"{w}x{h}": list(M.normalize_resolution(w, h))
             for w, h in [(4000, 300), (300, 4000), (100, 100), (5000, 5000),
                          (1280, 720), (1023, 767)]})
rec("sizing.snap_to_8", lambda: [M._snap_to_8(v) for v in (1, 7, 9, 13, 100, 1023)])
rec("sizing.source_dims", lambda: list(M._source_dims(SRC)))

# ---------------------------------------------------------------- 3. graphs
# The ComfyUI workflow builders. Fixed seeds so the JSON is deterministic.
def _graph(fn_name, *a, **kw):
    fn = getattr(M, fn_name)
    return json.loads(json.dumps(fn(*a, **kw), sort_keys=True, default=str))


rec("graph.inpaint_crop", lambda: _graph(
    "_inpaint_crop_graph", "u.png", "m.png", "a red hat", "blurry", 1234, 1024, 1024))
rec("graph.brushnet_crop", lambda: _graph(
    "_brushnet_crop_graph", "u.png", "m.png", "a red hat", "blurry", 1234, 1024, 1024))
rec("graph.flux_fill_crop", lambda: _graph(
    "_flux_fill_crop_graph", "u.png", "m.png", "a red hat", 1234, 1024, 1024))
rec("graph.tile_detail", lambda: _graph("_tile_detail_graph", "u.png", 1234, 2.0))
rec("graph.oldmodel_bg", lambda: _graph("_oldmodel_bg_subgraph", "u.png", "a beach", 1234))
rec("graph.instantid_facelock", lambda: _graph(
    "_instantid_facelock_graph", "u.png", "f.png", "a portrait", 1234))
rec("graph.handfix_detect", lambda: _graph("_handfix_detect_graph", "u.png"))

# ---------------------------------------------------------------- 4. masks/geom
rec("mask.dilate_outward", lambda: M._dilate_mask_outward(MASK, 8) is not None)
rec("engines.edit_engine_workflow",
    lambda: {e: str(M._edit_engine_workflow(e)) for e in ("firered", "oldmodel", "qwen")})

# ---------------------------------------------------------------- 5. transfer
rec("transfer.roles", lambda: [
    [r.role for r in M.infer_reference_roles(instr, [M.ReferenceImage(SRC)])]
    for instr in ("put the jacket on him", "use this face",
                  "make it in this style", "copy this pose")])
rec("transfer.plan", lambda: json.loads(json.dumps(
    M.build_edit_plan(SRC, [M.ReferenceImage(SRC, role=M.ROLE_CLOTHING),
                            M.ReferenceImage(SRC, role=M.ROLE_FACE)], ""),
    sort_keys=True, default=str)))
rec("transfer.placement", lambda: {
    i: M._placement_region_from_instruction(i)
    for i in ("put it on her head", "add a tattoo on his left arm",
              "place the logo on the shirt", "nothing here")})

# ---------------------------------------------------------------- 6. ideogram
import ideogram as G
rec("ideogram.size", lambda: {f"{w}x{h}": list(G.ideogram_size(w, h))
                              for w, h in [(4000, 300), (300, 4000), (1280, 720), (100, 100)]})
rec("ideogram.caption", lambda: G.build_caption(
    "a park bench", [G.element("bench", G.bbox(10, 20, 300, 400))],
    aesthetics="calm", lighting="golden hour", photo="35mm",
    medium="photograph", palette=["#1b1b2f", "#e43f5a"]))
rec("ideogram.roundtrip", lambda: json.loads(json.dumps(
    G.caption_to_layout(G.layout_to_caption(G.normalize_layout({
        "background": "a park", "elements": [{"name": "bench",
        "bbox": [10, 20, 300, 400], "text": "STOP"}]}))),
    sort_keys=True, default=str)))
rec("ideogram.style_floor", lambda: [
    G.apply_style_floor(c, "") for c in
    ("a photo of a dog", "an oil painting of a dog", "")])
rec("ideogram.extract_json", lambda: [
    G._extract_json(s) for s in
    ('{"a":1}', 'noise {"a":1} tail', '{"a":1,}', 'no json here',
     '```json\n{"a":[1,2]}\n```')])

# ---------------------------------------------------------------- 7. delivery
rec("delivery.is_intermediate", lambda: {
    n: M.is_intermediate_artifact(n) for n in
    ("out.png", "_INTERMEDIATE_tile_1.png", "_INTERMEDIATE_refphoto_9.jpg",
     "upload_123.png", "final_result.png")})

print(json.dumps(OUT, ensure_ascii=False, indent=1, sort_keys=True))
