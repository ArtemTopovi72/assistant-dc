"""A new format is a new generate_image: redraw keeps the size, so it must refuse, and
'горизонтально' must put generate_image in the agent's tool list."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from types import SimpleNamespace
from PIL import Image
import tool_retrieval, tool_image_handlers as h
import image_router  # the read itself: bench/removal_intent_live.py
image_router.EDIT_STUB = lambda t: {"reformats": "horizontal" in t}

img = os.path.join(tempfile.mkdtemp(), "a.png"); Image.new("RGB", (64, 96)).save(img)
ctx = SimpleNamespace(last_image_path=img, set_stage=lambda *a, **k: None)
r = h._handle_redraw_image(ctx, {"image_path": img}, {"instructions": "horizontal landscape for YouTube"})
assert r.startswith("[TOOL ERROR]") and "generate_image" in r, r

schemas = [{"type": "function", "function": {"name": n, "description": n, "parameters": {}}}
           for n in ("generate_image", "search", "calculate")]
os.environ["F5_TOOL_EMBED"] = "0"
print("ok")
from tool_args import GenerateImageArgs as _G
_a = _G(description="a wide landscape image for a YouTube banner with coffee", width=1920, height=3508)
assert _a.width > _a.height, _a
_b = _G(description="a vertical Instagram story, wide-angle lens", width=944, height=1680)
assert (_b.width, _b.height) == (944, 1680), _b
assert (_G(description="a square post", width=1264).height, _G(description="a YouTube cover", width=1680).height) == (1264, 944)
print("ok orientation follows the description")
assert (_G(description="a YouTube thumbnail about a bathroom").width, _G(description="a vertical story poster").height,
        _G(description="a cat").width) == (1680, 1680, None)
print("ok a size-less YouTube/story picture gets its format")
