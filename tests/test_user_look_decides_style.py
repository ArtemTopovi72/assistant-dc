"""The USER's words decide whether a picture is a photo or a drawing.

«кибер-Ленин дерётся с чертями» came back as a cartoon every time,
while the owner wanted photorealism -- the agent's English rewrite said
«epic digital painting» and the planner followed it. No drawn look in the
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


def test_the_look_is_read_from_the_users_words_this_turn_first():
    import types
    import tool_image_handlers as H
    looks = {"нарисуй кибер-Ленина": "none", "ещё раз": "none", "в стиле аниме кота": "anime",
             "сделай реалистично": "photo"}
    ctx = types.SimpleNamespace(
        choice=lambda q, text, options, default: looks.get(text, "none"),
        intent_read=lambda c, text: {"redo": "same"} if text == "ещё раз" else {"redo": ""})
    st = {"user_input_original": "ещё раз", "messages": [
        {"role": "user", "content": "в стиле аниме кота"}, {"role": "assistant", "content": "вот"},
        {"role": "user", "content": "ещё раз"}]}
    assert H._user_look(ctx, st) == "anime"                 # «ещё раз» after anime stays anime
    st2 = {"user_input_original": "нарисуй кибер-Ленина", "messages": [
        {"role": "user", "content": "в стиле аниме кота"}, {"role": "assistant", "content": "вот"},
        {"role": "user", "content": "нарисуй кибер-Ленина"}]}
    assert H._user_look(ctx, st2) == "none"                 # a new request: its own words only
    assert H._user_look(ctx, {"user_input_original": "сделай реалистично"}) == "photo"
    assert H._user_look(ctx, {"user_input_original": "нарисуй кибер-Ленина"}) == "none"
    assert H._user_look(ctx, {}) == ""
