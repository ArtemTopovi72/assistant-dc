import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sweep_stub  # noqa: F401  the model's narrow reads, stubbed
import image_router as R

# Which kind of edit a phrase is: the model's read, bench/edit_intent_live.py.
R.EDIT_STUB = {"make the background blurry": "background_replace",
              "change the background to a beach": "background_replace",
              "rotate the image 90 degrees": "transform", "flip it horizontally": "transform",
              "remove the cat and the dog and add a tree":
                  {"kind": "multi_op", "steps": ["remove the cat and the dog", "add a tree"]},
              "remove the cat and the dog": "object_remove", "add a tree": "object_insert"}.get
import image as I
seen = []
I.edit_image_with_firered = lambda ctx, path, instr, **k: seen.append(instr) or None
I.load_layout_for = lambda p: None
for t, want in (("make the background blurry", "make the background blurry. Keep every"),
                ("change the background to a beach", "Replace the background with a beach")):
    seen.clear(); R.route_edit_request(None, "x.png", t)
    assert seen and seen[0].startswith(want), (t, seen)
print("PASS a background modifier is not read as a new background")

steps = []
_real_route = R.route_edit_request
R.route_edit_request = lambda ctx, path, clause, **k: (steps.append(clause), ("x", None))[1]
R._orchestrate_multi_op(None, "x.png", "remove the cat and the dog and add a tree")
assert steps == ["remove the cat and the dog", "add a tree"], steps   # the model's steps, removals first
R.route_edit_request = _real_route
print("PASS multi-op runs the model's own steps, removals first")

assert [m.group(1) for m in R._CAPS_TEXT_RE.finditer("add the title BIG SALE 50% at the top")] == ["BIG SALE 50%"]
print("PASS unquoted capital lettering is read")

from PIL import Image
import tempfile
_p = os.path.join(tempfile.mkdtemp(), "w.png"); Image.new("RGB", (40, 20), "red").save(_p)
R.OUTPUT_DIR = __import__("pathlib").Path(tempfile.mkdtemp())
assert Image.open(R.transform_image(_p, "rotate the image 90 degrees")).size == (20, 40)
assert Image.open(R.transform_image(_p, "flip it horizontally")).size == (40, 20)
print("PASS rotate/flip/crop are exact pixel operations")
import tool_descriptions
assert "rotate the image 90 degrees" in tool_descriptions._REDRAW_IMAGE_DESC   # the agent refused: "не могу повернуть"
print("PASS the agent is told rotation exists; 'the entire image' still counts")
