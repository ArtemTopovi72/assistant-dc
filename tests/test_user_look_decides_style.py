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
import types  # noqa: E402

# What the planner's style makes (photo / drawn / graphic) is the model's read; scripted.
_R = types.SimpleNamespace(choice=lambda q, text, options, default=None: (
    "graphic" if "vector" in text else "drawn" if "painting" in text else "photo"))


def _painted():
    return {"art_style": "epic digital painting", "medium": "digital illustration",
            "aesthetics": "dramatic fantasy art", "photo": "", "lighting": "hellfire glow"}


def test_no_named_look_turns_the_agents_painting_into_a_photo():
    out = IL.enforce_user_look(_painted(), "none", ctx=_R)
    assert out["art_style"] == "" and out["medium"] == "photography" and out["photo"]
    assert "photorealistic" in out["aesthetics"]


def test_asked_for_a_photo_is_a_photo():
    assert IL.enforce_user_look(_painted(), "photo")["medium"] == "photography"


def test_a_logo_stays_graphic_design():
    logo = {"art_style": "", "medium": "flat vector graphic design", "aesthetics": "minimal", "photo": ""}
    assert IL.enforce_user_look(dict(logo), "none", ctx=_R) == logo


def test_scene_with_logo_does_not_keep_invented_cartoon_style():
    ctx = types.SimpleNamespace(choice=lambda *a, **kw: 'graphic')
    layout = {'art_style': 'cartoon', 'medium': 'illustration', 'aesthetics': 'logo faces'}
    out = IL.enforce_user_look(layout, 'none', ctx=ctx)
    assert out['medium'] == 'photography' and not out['art_style']


def test_a_named_drawn_look_wins():
    out = IL.enforce_user_look({"art_style": "", "medium": "photography", "photo": "85mm"}, "anime")
    assert out["art_style"] == "anime" and out["photo"] == ""
    kept = IL.enforce_user_look(_painted(), "oil painting")
    assert kept["art_style"] == "epic digital painting"


def test_no_user_words_leaves_the_layout_alone():
    assert IL.enforce_user_look(_painted(), "") == _painted()


def test_funny_scene_does_not_implicitly_request_cartoon():
    import tool_image_handlers as H
    ctx = types.SimpleNamespace(choice=lambda *a, **k: 'cartoon',
                                yes_no=lambda *a, **k: False,
                                intent_read=lambda *a, **k: {'redo': ''})
    assert H._user_look(ctx, {'user_input_original': 'Нарисуй смешную сцену'}) == 'none'


def test_explicit_anime_survives_style_validator_failure():
    import tool_image_handlers as H
    def fail(*args, **kwargs):
        raise RuntimeError('validator unavailable')
    ctx = types.SimpleNamespace(choice=lambda *a, **kw: 'anime', yes_no=fail)
    assert H._user_look(ctx, {'user_input_original': 'Нарисуй кота в стиле аниме'}) == 'anime'
    ctx.choice = lambda *a, **kw: 'none'
    assert H._user_look(ctx, {'user_input_original': 'Нарисуй кота в стиле аниме'}) == 'anime'
    assert H._user_look(ctx, {'user_input_original': 'реалистичный кот в стиле аниме'}) == 'anime'


def test_anime_figurine_is_not_the_style_of_the_picture():
    import tool_image_handlers as H
    ctx = types.SimpleNamespace(choice=lambda *a, **kw: 'anime', yes_no=lambda *a, **kw: False,
                                intent_read=lambda *a, **kw: {'redo': ''})
    assert H._user_look(ctx, {'user_input_original': 'сфотографируй фигурку аниме на столе'}) == 'none'


def test_negated_or_object_style_does_not_override_photo():
    import tool_image_handlers as H
    ctx = types.SimpleNamespace(choice=lambda *a, **kw: 'photo', yes_no=lambda *a, **kw: False)
    for text in ('Нарисуй кота реалистично, не в стиле аниме',
                 'A photo, not anime style',
                 'сфотографируй человека с тату в стиле аниме'):
        assert H._user_look(ctx, {'user_input_original': text}) == 'photo'


def test_photo_default_removes_invented_cartoon_style_but_preserves_quoted_text():
    import tool_image_handlers as H
    text = 'A funny cartoon illustration of a man holding a sign "cartoon".'
    result = H._photographic_description(text)
    assert 'cartoon illustration' not in result
    assert '"cartoon"' in result
    assert 'cartoon illustration' not in H._photographic_description('A funny and dynamic cartoon illustration of two people.')
    assert 'cartoon-style' not in H._photographic_description('A funny cartoon-style scene featuring two people.')


def test_photo_default_preserves_cartoon_objects_in_the_scene():
    import tool_image_handlers as H
    text = 'A realistic photo of a comic book beside an anime figurine.'
    assert H._photographic_description(text) == text
    poster = 'A photo of a poster with a cartoon illustration of a cat.'
    assert H._photographic_description(poster) == poster
    poster_style = 'A realistic photo of a poster whose style is cartoon.'
    assert H._photographic_description(poster_style) == poster_style
    assert H._photographic_description('A photo of a poster with a cartoon-style character.') == 'A photo of a poster with a cartoon-style character.'
    assert H._photographic_description('A cartoon-style picture hangs on a wall in a photo.') == 'A cartoon-style picture hangs on a wall in a photo.'
    assert H._photographic_description('A cartoon illustration hangs on the wall in a photo.') == 'A cartoon illustration hangs on the wall in a photo.'
    assert H._photographic_description('A cartoon illustration of a cat hangs on a wall.') == 'A cartoon illustration of a cat hangs on a wall.'
    assert 'photorealistic' in H._photographic_description('Two people. The style is a cartoon.')


def test_the_look_is_read_from_the_users_words_this_turn_first():
    import types
    import tool_image_handlers as H
    looks = {"нарисуй кибер-Ленина": "none", "ещё раз": "none", "в стиле аниме кота": "anime",
             "сделай реалистично": "photo"}
    ctx = types.SimpleNamespace(
        choice=lambda q, text, options, default: looks.get(text, "none"),
        yes_no=lambda q, text, **kw: 'аниме' in text,
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
