"""A new format is a new generate_image: redraw keeps the size, so it must refuse."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from types import SimpleNamespace
from PIL import Image
import tool_retrieval, tool_image_handlers as h
import image_router  # the read itself: bench/removal_intent_live.py
_DEPS = SimpleNamespace(edit_read=lambda t: {"reformats": "horizontal" in t})

img = os.path.join(tempfile.mkdtemp(), "a.png"); Image.new("RGB", (64, 96)).save(img)
ctx = SimpleNamespace(last_image_path=img, set_stage=lambda *a, **k: None, deps=_DEPS)
r = h._handle_redraw_image(ctx, {"image_path": img}, {"instructions": "horizontal landscape for YouTube"})
assert r.startswith("[TOOL ERROR]") and "generate_image" in r, r

print("ok")
from tool_args import GenerateImageArgs as _G
import image_sizing as _S, _sweep_stub
_R = _sweep_stub.reads_ctx()
# The format a description names is the model's read (bench/frame_read_eval.py); here, what is done with it.
assert _S.fix_image_params("a wide landscape banner", "", 1920, 3508, given=True, ctx=_R) > (0, 0)
_w, _h = _S.fix_image_params("a wide landscape banner", "", 1920, 3508, given=True, ctx=_R)
assert _w > _h, (_w, _h)
_w, _h = _S.fix_image_params("a single person standing, headshot", "", 1680, 944, given=True, ctx=_R)
assert _w > _h, "a scene's lean does not turn a size the agent chose"
assert _S.fix_image_params("a YouTube thumbnail about a bathroom", "", 1264, 1264, ctx=_R) == (1680, 944)
assert _S.fix_image_params("a vertical story poster", "", 1264, 1264, ctx=_R) == (944, 1680)
assert _S.fix_image_params("a square post", "", 1680, 944, given=True, ctx=_R)[0] ==     _S.fix_image_params("a square post", "", 1680, 944, given=True, ctx=_R)[1]
print("ok a named format shapes the size")
assert (_G(description="a square post", width=1264).height, _G(description="a YouTube cover", width=1680).height) == (1264, 944)
print("ok one side names the pair")
