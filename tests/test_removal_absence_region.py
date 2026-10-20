import os, tempfile, types
import tool_image_handlers as H
import image_router

# The reads themselves (removes_region, names_former_place) run live in
# bench/removal_intent_live.py and bench/removal_target_place_live.py. Here: the wiring.


def _call(region, instructions, gone):
    fd, p = tempfile.mkstemp(suffix=".png"); os.close(fd)
    asked = []
    try:
        def yes_no(question, text, *a, **k):
            asked.append((question, text))
            return gone if "ALREADY GONE" in question else False
        ctx = types.SimpleNamespace(last_image_path=p, image_edit_engine="auto",
                                    set_stage=lambda *a, **k: None, yes_no=yes_no,
                                    deps=types.SimpleNamespace(region_read=lambda i, r: True))
        return H._handle_inpaint_image(ctx, {"image_path": p},
                                       {"region": region, "instructions": instructions}), asked
    finally:
        os.unlink(p)


def test_region_naming_an_absence_is_refused():
    out, asked = _call("the area where the blue vase and flowers were located",
                       "remove it, empty wooden table, no vase", gone=True)
    assert out.startswith("[TOOL ERROR]") and "already removed" in out
    assert any(t == "the area where the blue vase and flowers were located" for _q, t in asked)


def test_a_place_where_the_thing_still_is_goes_on_to_the_removal():
    # «the counter top where the grey cat sleeps» was read as a former place and the removal
    # refused as «already removed» while the cat lay there (live bench, 2026-10-21).
    out, _ = _call("the counter top where the grey cat sleeps", "remove the grey cat", gone=False)
    assert "already removed" not in out
