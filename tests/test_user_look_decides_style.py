"""The USER's words decide whether a picture is a photo or a drawing.

Live 10-03: «кибер-Ленин дерётся с чертями» came back as a cartoon every time,
while the owner wanted photorealism -- the agent's English rewrite said
«epic digital painting» and the planner followed it. Now: no drawn look in the
user's words -> a photograph (a logo stays graphic design); a named look wins.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")

import ideogram_layout as IL  # noqa: E402


def _painted():
    return {"art_style": "epic digital painting", "medium": "digital illustration",
            "aesthetics": "dramatic fantasy art", "photo": "", "lighting": "hellfire glow"}


def test_no_named_look_turns_the_agents_painting_into_a_photo():
    out = IL.enforce_user_look(_painted(), "none")
    assert out["art_style"] == "" and out["medium"] == "photography" and out["photo"]
    assert "photorealistic" in out["aesthetics"]


def test_asked_for_a_photo_is_a_photo():
    assert IL.enforce_user_look(_painted(), "photo")["medium"] == "photography"


def test_a_logo_stays_graphic_design():
    logo = {"art_style": "", "medium": "flat vector graphic design", "aesthetics": "minimal", "photo": ""}
    assert IL.enforce_user_look(dict(logo), "none") == logo


def test_a_named_drawn_look_wins():
    out = IL.enforce_user_look({"art_style": "", "medium": "photography", "photo": "85mm"}, "anime")
    assert out["art_style"] == "anime" and out["photo"] == ""
    kept = IL.enforce_user_look(_painted(), "oil painting")
    assert kept["art_style"] == "epic digital painting"


def test_no_user_words_leaves_the_layout_alone():
    assert IL.enforce_user_look(_painted(), "") == _painted()


def test_the_look_is_read_from_the_users_words_this_turn_first(monkeypatch):
    import tool_image_handlers as H
    looks = {"нарисуй кибер-Ленина": "none", "ещё раз": "none", "в стиле аниме кота": "anime",
             "сделай реалистично": "photo"}
    monkeypatch.setattr(IL, "_look", lambda t: looks.get(t, "none"))
    st = {"user_input_original": "ещё раз", "messages": [
        {"role": "user", "content": "в стиле аниме кота"}, {"role": "assistant", "content": "вот"},
        {"role": "user", "content": "ещё раз"}]}
    assert H._user_look(st) == "anime"                      # «ещё раз» after anime stays anime
    assert H._user_look({"user_input_original": "сделай реалистично"}) == "photo"
    assert H._user_look({"user_input_original": "нарисуй кибер-Ленина"}) == "none"
    assert H._user_look({}) == ""


def test_the_planner_applies_it(monkeypatch):
    import ideogram as I

    class Ctx:
        user_look = "none"
    monkeypatch.setattr(I, "_ask_planner", lambda ctx, p: dict(_painted(), elements=[
        {"desc": "Lenin punches a demon", "x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8}]))
    import draw_agent
    monkeypatch.setattr(draw_agent, "split_counted_boxes", lambda l, p: (l, []))
    monkeypatch.setattr(draw_agent, "merge_text_column", lambda l: l)
    layout = I.plan_layout(Ctx(), "cyber Lenin fights demons in hell, epic digital painting")
    assert layout["medium"] == "photography" and layout["art_style"] == ""


def test_a_redraw_restyles_only_when_the_user_names_a_look(monkeypatch, tmp_path):
    import image_router as R
    import draw_agent
    import image as image_mod
    cartoon = dict(_painted(), elements=[{"desc": "Lenin", "x": 0, "y": 0, "w": 1, "h": 1}])
    monkeypatch.setattr(image_mod, "load_layout_for", lambda p: {"layout": dict(cartoon), "prompt": "Lenin"})
    monkeypatch.setattr(draw_agent, "edit_layout", lambda ctx, l, i: (dict(l), ["the model proposed no change"]))
    seen = []
    monkeypatch.setattr(draw_agent, "run", lambda ctx, p, layout=None, **k: seen.append(layout) or {"image": None})
    pic = tmp_path / "p.png"
    from PIL import Image
    Image.new("RGB", (8, 8)).save(pic)

    class Ctx:
        user_look = "photo"
    R.edit_via_layout(Ctx(), str(pic), "make it realistic")
    assert seen and seen[-1]["medium"] == "photography" and seen[-1]["art_style"] == ""
    seen.clear()
    Ctx.user_look = "none"
    assert R.edit_via_layout(Ctx(), str(pic), "he is bald") is None and not seen   # nothing named: no restyle
