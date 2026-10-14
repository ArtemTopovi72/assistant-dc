import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sweep_stub  # the model's narrow reads, stubbed
import image_router as R

# Which kind of edit a phrase is: the model's read, bench/edit_intent_live.py.
import types
_EDIT_READ = {"make the background blurry": "background_replace",
              "change the background to a beach": "background_replace",
              "rotate the image 90 degrees": "transform", "flip it horizontally": "transform",
              "remove the cat and the dog and add a tree":
                  {"kind": "multi_op", "steps": ["remove the cat and the dog", "add a tree"]},
              "remove the cat and the dog": "object_remove", "add a tree": "object_insert"}.get
import image as I
from fakes import Over
seen = []
_IMG = Over(I, edit_image_with_firered=lambda ctx, path, instr, **k: seen.append(instr) or None,
            load_layout_for=lambda p: None)
for t, want in (("make the background blurry", "make the background blurry. Keep every"),
                ("change the background to a beach", "Replace the background with a beach")):
    seen.clear(); R.route_edit_request(_sweep_stub.reads_ctx(deps=types.SimpleNamespace(edit_read=_EDIT_READ, image=_IMG)), "x.png", t)
    assert seen and seen[0].startswith(want), (t, seen)
print("PASS a background modifier is not read as a new background")

steps = []
_route = lambda ctx, path, clause, **k: (steps.append(clause), ("x", None))[1]
R._orchestrate_multi_op(_sweep_stub.reads_ctx(deps=types.SimpleNamespace(edit_read=_EDIT_READ, route_edit_request=_route)), "x.png", "remove the cat and the dog and add a tree")
assert steps == ["remove the cat and the dog", "add a tree"], steps   # the model's steps, removals first
print("PASS multi-op runs the model's own steps, removals first")

assert [m.group(1) for m in R._CAPS_TEXT_RE.finditer("add the title BIG SALE 50% at the top")] == ["BIG SALE 50%"]
print("PASS unquoted capital lettering is read")

from PIL import Image
import tempfile
_p = os.path.join(tempfile.mkdtemp(), "w.png"); Image.new("RGB", (40, 20), "red").save(_p)
_OUT = __import__("pathlib").Path(tempfile.mkdtemp())
_TCTX = _sweep_stub.reads_ctx(deps=types.SimpleNamespace(edit_read=_EDIT_READ, OUTPUT_DIR=_OUT))
assert Image.open(R.transform_image(_p, "rotate the image 90 degrees", ctx=_TCTX)).size == (20, 40)
assert Image.open(R.transform_image(_p, "flip it horizontally", ctx=_TCTX)).size == (40, 20)
print("PASS rotate/flip/crop are exact pixel operations")
import tool_descriptions
assert "rotate the image 90 degrees" in tool_descriptions._REDRAW_IMAGE_DESC
print("PASS the agent is told rotation exists; 'the entire image' still counts")
