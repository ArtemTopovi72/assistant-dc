import os, tempfile, types
import tool_image_handlers as H
import image_router

# The reads themselves (removes_region, former_place) run live in
# bench/removal_intent_live.py. Here: the wiring.
image_router.REGION_STUB = lambda i, r: True
H.image_mod._item_attributes = lambda ctx, ph: {"former_place": " where " in ph}


def _call(region, instructions):
    fd, p = tempfile.mkstemp(suffix=".png"); os.close(fd)
    try:
        ctx = types.SimpleNamespace(last_image_path=p, image_edit_engine="auto",
                                    set_stage=lambda *a, **k: None)
        return H._handle_inpaint_image(ctx, {"image_path": p},
                                       {"region": region, "instructions": instructions})
    finally:
        os.unlink(p)


def test_region_naming_an_absence_is_refused():
    out = _call("the area where the blue vase and flowers were located",
                "remove it, empty wooden table, no vase")
    assert out.startswith("[TOOL ERROR]") and "already removed" in out
