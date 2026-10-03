"""Two deterministic guards on layout editing, driven offline.

Both were found by bench/draw_scenarios.py against the live model, and neither
can be pinned down there: the editor is sampled, so the destructive answer that
exposed the first guard appears in maybe one run out of three. The ops it
actually produced are replayed here instead, so the guard is exercised every
time and a regression cannot hide behind a lucky sampling.

  1. an ADD-only instruction may not delete or overwrite a box. Measured: told
     "добавь рыжего кота, который спит на диване", the editor answered
     `replace` on the sofa plus `delete` on the existing cat -- one thing asked
     for, two things lost.

  2. "put X on Y" may not be undone by the geometry repair. The editor sets the
     cat's box to the sofa's box exactly; the merge repair then pulled them
     apart, so the cat went back to the floor while the agent reported the move
     as done.

Run: venv/Scripts/python.exe -m pytest tests/test_draw_edit_guards.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import draw_agent as DA
import image_router
# Add-only is the model's read of the edit (edit_plan: one object_insert);
# the phrases run live in bench/intent_rest_live.py. Here: what the read means.
_ADDS = {"добавь рыжего кота", "дорисуй лампу", "add a lamp", "верни коробку обратно",
         "восстанови кружку", "put the box back", "bring back the cat", "put it back"}
image_router.EDIT_STUB = lambda t: {"kind": "object_insert"} if t in _ADDS else {"kind": "subject_edit"}
import draw_geometry as DG
import ideogram as IG


def _kitchen():
    return IG.normalize_layout({
        "background": "уютная кухня",
        "elements": [
            {"desc": "деревянный стол",         "x": 0.20, "y": 0.45, "w": 0.55, "h": 0.30},
            {"desc": "серый кот сидит на полу", "x": 0.30, "y": 0.70, "w": 0.20, "h": 0.20},
            {"desc": "синий диван у стены",     "x": 0.02, "y": 0.50, "w": 0.16, "h": 0.30},
        ]})


def _descs(layout):
    return [el["desc"] for el in layout["elements"]]


# --- 1. the additive guard --------------------------------------------------

def test_an_add_instruction_may_not_delete_a_box():
    layout, notes = DA.apply_ops(
        _kitchen(),
        [{"op": "delete", "target": "серый кот сидит на полу"}],
        additive_only=True)
    assert "серый кот сидит на полу" in _descs(layout), notes
    assert any("only asked" in n for n in notes), notes


def test_a_replace_under_an_add_instruction_becomes_an_add():
    """The description it wanted to overwrite WITH is the thing to add."""
    layout, notes = DA.apply_ops(
        _kitchen(),
        [{"op": "replace", "target": "синий диван у стены",
          "desc": "a ginger cat sleeping on a blue sofa"}],
        additive_only=True)
    assert "синий диван у стены" in _descs(layout), notes
    assert "a ginger cat sleeping on a blue sofa" in _descs(layout), notes
    assert len(layout["elements"]) == 4, _descs(layout)


def test_a_replace_with_nothing_to_add_is_simply_refused():
    layout, notes = DA.apply_ops(
        _kitchen(), [{"op": "replace", "target": "деревянный стол"}],
        additive_only=True)
    assert _descs(layout) == _descs(_kitchen())
    assert any("only asked" in n for n in notes), notes


def test_a_removal_instruction_still_removes():
    """The guard must not fire on the instructions that DO mean removal."""
    layout, _notes = DA.apply_ops(
        _kitchen(), [{"op": "delete", "target": "синий диван у стены"}],
        additive_only=False)
    assert "синий диван у стены" not in _descs(layout)


def test_which_instructions_count_as_add_only():
    for text in ("добавь рыжего кота", "дорисуй лампу", "add a lamp",
                 "включи в сцену стол"):
        if text.startswith("включи"):
            continue                      # not in the vocabulary; see below
        assert DA._additive_only(text), text
    for text in ("убери коробку", "замени кружку на бокал", "remove the dog",
                 "добавь лампу вместо кресла",   # add AND remove: not add-only
                 "передвинь кота на диван"):
        assert not DA._additive_only(text), text


# --- 2. "on the sofa" survives the geometry repair --------------------------

def _on_the_sofa():
    """The layout exactly as the editor leaves it after "передвинь кота на
    диван": the cat's box set to the sofa's box, to the digit."""
    return IG.normalize_layout({
        "background": "уютная кухня",
        "elements": [
            {"desc": "деревянный стол",         "x": 0.40, "y": 0.45, "w": 0.55, "h": 0.30},
            {"desc": "серый кот сидит на полу", "x": 0.02, "y": 0.50, "w": 0.16, "h": 0.30},
            {"desc": "синий диван у стены",     "x": 0.02, "y": 0.50, "w": 0.16, "h": 0.30},
        ]})


def _center(el):
    return (el["x"] + el["w"] / 2, el["y"] + el["h"] / 2)


def _sits_on(small, big):
    sx, sy = _center(small)
    bx, by = _center(big)
    return (abs(sx - bx) < (small["w"] + big["w"]) / 2 and
            abs(sy - by) < (small["h"] + big["h"]) / 2)


def test_a_cat_placed_on_a_sofa_is_still_on_it_after_the_repair():
    fixed, notes = DA.auto_fix_geometry(_on_the_sofa())
    cat = [el for el in fixed["elements"] if "кот" in el["desc"]][0]
    sofa = [el for el in fixed["elements"] if "диван" in el["desc"]][0]
    assert _sits_on(cat, sofa), (cat, sofa, notes)


def test_the_two_boxes_stop_being_identical():
    """Identical boxes are what the renderer fuses; nesting is the repair."""
    fixed, _ = DA.auto_fix_geometry(_on_the_sofa())
    cat = [el for el in fixed["elements"] if "кот" in el["desc"]][0]
    sofa = [el for el in fixed["elements"] if "диван" in el["desc"]][0]
    assert cat["w"] * cat["h"] < sofa["w"] * sofa["h"] * 0.9, (cat, sofa)


def test_the_nested_box_is_still_big_enough_to_render():
    fixed, _ = DA.auto_fix_geometry(_on_the_sofa())
    cat = [el for el in fixed["elements"] if "кот" in el["desc"]][0]
    assert cat["w"] * cat["h"] >= DG.MIN_AREA * 0.999, cat


def test_the_repair_is_idempotent():
    once, _ = DA.auto_fix_geometry(_on_the_sofa())
    twice, _ = DA.auto_fix_geometry(once)
    assert once["elements"] == twice["elements"]


def test_an_ordinary_collision_is_still_separated():
    """The exemption is narrow: two boxes on the same spot that do NOT say one
    sits on the other must still be pulled apart."""
    layout = IG.normalize_layout({"elements": [
        {"desc": "красный автомобиль", "x": 0.30, "y": 0.30, "w": 0.35, "h": 0.35},
        {"desc": "синий автобус",      "x": 0.31, "y": 0.31, "w": 0.35, "h": 0.35},
    ]})
    fixed, notes = DA.auto_fix_geometry(layout)
    a, b = fixed["elements"]
    assert DG._iou(a, b) < DG.MERGE_IOU, (a, b, notes)


def test_a_support_word_alone_does_not_excuse_a_collision():
    """"на полу" says it sits on the FLOOR, not on the box next to it: two
    half-overlapping subjects are still a collision."""
    layout = IG.normalize_layout({"elements": [
        {"desc": "кот сидит на полу",  "x": 0.30, "y": 0.30, "w": 0.35, "h": 0.35},
        {"desc": "картонная коробка",  "x": 0.36, "y": 0.36, "w": 0.35, "h": 0.35},
    ]})
    fixed, _ = DA.auto_fix_geometry(layout)
    a, b = fixed["elements"]
    assert DG._iou(a, b) < DG.MERGE_IOU, (a, b)


def test_a_mixed_instruction_is_not_treated_as_add_only():
    """The words only raise the suspicion; the ops settle it.

    "sign top right, lose the tree, add a bench" carries an add verb and a
    removal the vocabulary does not know. An answer that adds AND deletes is
    that instruction being carried out -- blocking the delete would silently
    drop the half of the request the word list failed to parse.
    """
    layout, notes = DA.apply_ops(
        _kitchen(),
        [{"op": "delete", "target": "синий диван у стены"},
         {"op": "add", "desc": "деревянная скамейка",
          "x": 0.05, "y": 0.78, "w": 0.25, "h": 0.18}],
        additive_only=True)
    assert "синий диван у стены" not in _descs(layout), notes
    assert "деревянная скамейка" in _descs(layout), notes
    assert not any("only asked" in n for n in notes), notes


def test_destroying_without_adding_is_still_blocked():
    """The same guard, on the answer that has no add in it at all."""
    layout, notes = DA.apply_ops(
        _kitchen(),
        [{"op": "delete", "target": "серый кот сидит на полу"},
         {"op": "replace", "target": "деревянный стол", "desc": "a marble table"}],
        additive_only=True)
    assert "серый кот сидит на полу" in _descs(layout), notes
    assert "деревянный стол" in _descs(layout), notes


# --- 3. growing a box must not lift it off what it stands on -----------------

def test_growing_a_small_box_keeps_its_bottom_edge():
    """Measured on a render: a mug whose bottom sat on the table's top edge came
    out of the MIN_AREA repair centred on its old middle, so it grew downward
    THROUGH the table and upward into the air -- and drew as a giant mug
    hovering above it."""
    layout = IG.normalize_layout({"elements": [
        {"desc": "деревянный стол",     "x": 0.20, "y": 0.45, "w": 0.55, "h": 0.30},
        {"desc": "белая кружка с чаем", "x": 0.34, "y": 0.31, "w": 0.12, "h": 0.14},
    ]})
    before = layout["elements"][1]
    bottom = before["y"] + before["h"]
    fixed, _ = DA.auto_fix_geometry(layout)
    mug = [el for el in fixed["elements"] if "кружка" in el["desc"]][0]
    assert mug["w"] * mug["h"] >= DG.MIN_AREA, mug          # it did grow
    assert abs((mug["y"] + mug["h"]) - bottom) < 1e-6, mug  # from the bottom
    table = [el for el in fixed["elements"] if "стол" in el["desc"]][0]
    assert mug["y"] + mug["h"] <= table["y"] + 0.05, (mug, table)


def test_growth_keeps_the_horizontal_centre():
    layout = IG.normalize_layout({"elements": [
        {"desc": "маленький шар", "x": 0.40, "y": 0.40, "w": 0.10, "h": 0.10},
    ]})
    cx = 0.45
    fixed, _ = DA.auto_fix_geometry(layout)
    el = fixed["elements"][0]
    assert abs((el["x"] + el["w"] / 2) - cx) < 1e-6, el


# --- 4. the caption's background stays scene CONTENT --------------------------

def test_the_background_carries_no_instructions_to_the_renderer():
    """A rejected fix, pinned so it does not come back.

    Ideogram invents lettering on objects, so "no letters, words or logos on any
    object" was appended to the caption background. Measured on renders: without
    the clause, a normal photograph; with it -- two phrasings, two runs -- the
    scene came back as cut-out objects floating on a transparency checkerboard,
    AND the invented lettering appeared anyway. An instruction of that shape
    reads as a product-photography brief and the renderer delivers one.

    background is scene content. Anything that is an instruction to the renderer
    has to reach it somewhere else.
    """
    cap = IG.layout_to_caption(IG.normalize_layout({
        "background": "уютная кухня",
        "elements": [{"desc": "белая кружка", "x": 0.3, "y": 0.3,
                      "w": 0.2, "h": 0.2}]}))
    bg = cap["compositional_deconstruction"]["background"]
    assert bg == "уютная кухня", bg
    for banned in ("no letters", "no text", "blank", "logos"):
        assert banned not in bg.lower(), bg


# --- 5. the loop must deliver its BEST attempt, not its last -----------------

def _att(image, ok=False, problems=(), failures=()):
    return {"layout": {"elements": []}, "image": image,
            "verdict": {"ok": ok, "problems": list(problems),
                        "text": {"failures": list(failures)}}}


def test_the_best_attempt_wins_not_the_newest():
    """The reported experience: five redraws, the second already good, and the
    fifth -- worse -- handed over. Each round re-rolls a scene the critic
    complained about, so the newest has no claim to being the best."""
    history = [_att("a.png", problems=["x", "y", "z"]),
               _att("b.png", problems=["x"]),
               _att("c.png", problems=["x", "y"])]
    assert DA._best_attempt(history)["image"] == "b.png"


def test_an_approved_attempt_beats_every_unapproved_one():
    history = [_att("a.png", ok=True, problems=["cosmetic"]),
               _att("b.png", problems=[])]
    assert DA._best_attempt(history)["image"] == "a.png"


def test_garbled_lettering_outweighs_a_list_of_complaints():
    """A sign reading AVCHKE is the most visible way a picture is wrong."""
    history = [_att("a.png", problems=["x", "y"], failures=[]),
               _att("b.png", problems=[], failures=[0])]
    assert DA._best_attempt(history)["image"] == "a.png"


def test_attempts_that_never_produced_an_image_are_not_candidates():
    history = [_att(None, ok=True), _att("b.png", problems=["x"])]
    assert DA._best_attempt(history)["image"] == "b.png"


def test_no_attempts_at_all_is_not_a_crash():
    assert DA._best_attempt([]) is None
    assert DA._best_attempt(None) is None


def test_run_hands_back_the_best_attempt(monkeypatch):
    """Driven through run() itself: the scoring function being right is not the
    same as the loop USING it, and the loop is what the user sees."""
    seq = ["one.png", "two.png", "three.png"]
    verdicts = [{"ok": False, "problems": ["a", "b", "c"]},
                {"ok": False, "problems": ["a"]},          # the good one
                {"ok": False, "problems": ["a", "b"]}]
    drawn = []

    def _gen(ctx, prompt, **kw):
        drawn.append(len(drawn))
        return seq[len(drawn) - 1]

    monkeypatch.setattr(DA.ideogram, "generate", _gen)
    monkeypatch.setattr(DA, "critique",
                        lambda *a, **k: dict(verdicts[len(drawn) - 1]))
    monkeypatch.setattr(DA, "verify_text", lambda *a, **k: {"checks": [], "failures": []})
    monkeypatch.setattr(DA, "text_problems", lambda *a, **k: [])
    monkeypatch.setattr(DA, "edit_layout", lambda ctx, lay, ins: (lay, ["noted"]))

    out = DA.run(None, layout=_kitchen(), rounds=2)
    assert len(drawn) == 3, drawn
    assert out["image"] == "two.png", out["image"]
    assert "best" in out["stopped"], out["stopped"]


def test_run_still_stops_the_moment_the_critic_is_happy(monkeypatch):
    """The best-of rule must not turn an early success into extra renders."""
    drawn = []
    monkeypatch.setattr(DA.ideogram, "generate",
                        lambda ctx, prompt, **kw: (drawn.append(1), "ok.png")[1])
    monkeypatch.setattr(DA, "critique", lambda *a, **k: {"ok": True, "problems": []})
    monkeypatch.setattr(DA, "verify_text", lambda *a, **k: {"checks": [], "failures": []})
    monkeypatch.setattr(DA, "text_problems", lambda *a, **k: [])

    out = DA.run(None, layout=_kitchen(), rounds=3)
    assert len(drawn) == 1, drawn
    assert out["stopped"] == "ok", out["stopped"]


# --- 6. no render may reach Ideogram without a style -------------------------

def test_an_edited_layout_still_carries_a_style():
    """The floor lived in plan_layout, so only a FRESHLY PLANNED layout had it.
    A hand-built or edited one arrived with an empty style_description -- and
    empty is not neutral, it lets the model pick. Measured: four boxes with no
    style came back as a collage of cut-out objects on a transparency
    checkerboard, with an invented product packet in the middle."""
    cap = IG.layout_to_caption(_kitchen())
    style = cap.get("style_description") or {}
    assert style, cap
    assert any(str(v).strip() for v in style.values()), style


def test_a_layout_that_names_its_own_style_keeps_it():
    lay = IG.normalize_layout({
        "background": "город",
        "art_style": "детская книжная иллюстрация",
        "medium": "illustration",
        "elements": [{"desc": "дом", "x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4}]})
    style = IG.layout_to_caption(lay).get("style_description") or {}
    assert "иллюстрац" in str(style), style
    assert "photography" not in str(style), style


def test_restoring_something_counts_as_adding_it():
    """Measured live: "верни коробку обратно" was answered with `replace` on an
    unrelated element, and the user lost their mug to a cardboard box. One
    instruction, two things wrong.

    Restoring is adding back, so the guard that turns a destructive answer into
    an add has to fire on this wording too."""
    for text in ("верни коробку обратно", "восстанови кружку",
                 "put the box back", "bring back the cat", "put it back"):
        assert DA._additive_only(text), text
    # ...but only when nothing is being taken away in the same breath.
    for text in ("верни коробку вместо кружки", "убери коробку"):
        assert not DA._additive_only(text), text


def test_a_restore_answered_as_a_replace_becomes_an_add():
    """The whole point: the box comes back and the mug survives."""
    layout = IG.normalize_layout({
        "background": "кухня",
        "elements": [
            {"desc": "белая кружка с чаем", "x": 0.30, "y": 0.30, "w": 0.20, "h": 0.20},
            {"desc": "деревянный стол",     "x": 0.20, "y": 0.55, "w": 0.55, "h": 0.30},
        ]})
    after, notes = DA.apply_ops(
        layout,
        [{"op": "replace", "target": "белая кружка с чаем", "desc": "a cardboard box"}],
        additive_only=True)
    descs = [el["desc"] for el in after["elements"]]
    assert "белая кружка с чаем" in descs, (descs, notes)
    assert "a cardboard box" in descs, (descs, notes)
    assert len(descs) == 3, descs


def test_a_reword_of_the_thing_that_was_named_is_left_alone():
    """The guard must not turn every restore into an add.

    "Верни фон белым" names the background and means change it. Only an op
    aimed at something the user did NOT name is collateral damage.
    """
    layout = IG.normalize_layout({
        "background": "кухня",
        "elements": [
            {"desc": "синий диван у стены", "x": 0.10, "y": 0.40, "w": 0.30, "h": 0.30},
            {"desc": "деревянный стол",     "x": 0.50, "y": 0.40, "w": 0.30, "h": 0.30},
        ]})
    after, notes = DA.apply_ops(
        layout,
        [{"op": "replace", "target": "синий диван у стены",
          "desc": "зелёный диван у стены"}],
        additive_only=True, instruction="верни дивану зелёный цвет")
    descs = [el["desc"] for el in after["elements"]]
    assert descs == ["зелёный диван у стены", "деревянный стол"], (descs, notes)


def test_without_the_instruction_the_guard_still_protects():
    """apply_ops is called directly by the critique loop with no instruction
    text; the conservative branch has to hold there."""
    layout = IG.normalize_layout({"elements": [
        {"desc": "белая кружка", "x": 0.3, "y": 0.3, "w": 0.2, "h": 0.2}]})
    after, _ = DA.apply_ops(
        layout, [{"op": "replace", "target": "белая кружка", "desc": "коробка"}],
        additive_only=True)
    descs = [el["desc"] for el in after["elements"]]
    assert "белая кружка" in descs and "коробка" in descs, descs


# --- 7. the minimum-area floor, as measured ----------------------------------

def test_a_prop_on_a_table_is_left_at_its_own_size():
    """The floor used to be 4% with the repair aiming at 6.4%, and that is what
    several bad renders were made of: a mug at 6% of a 1024px frame is the size
    of a hero product shot, and the renderer duly composed one -- a giant mug
    floating over the table, an inset panel, an advertising banner.

    Swept on real renders (bench/draw_minarea_probe.py): 0.6, 1.1, 2.0, 3.0 and
    4.0 % all came out clearly drawn with the scene coherent. The floor was
    buying a vanishing prop that does not vanish.
    """
    layout = IG.normalize_layout({
        "background": "кухня",
        "elements": [
            {"desc": "деревянный стол",     "x": 0.20, "y": 0.45, "w": 0.55, "h": 0.30},
            {"desc": "белая кружка с чаем", "x": 0.34, "y": 0.28, "w": 0.12, "h": 0.14},
        ]})
    fixed, notes = DA.auto_fix_geometry(layout)
    mug = [el for el in fixed["elements"] if "кружка" in el["desc"]][0]
    assert abs(mug["w"] * mug["h"] - 0.12 * 0.14) < 1e-6, (mug, notes)
    # The size is untouched. The mug does get lowered onto the table -- this
    # fixture is the layout the planner really produced, with the mug's box
    # ending at 0.42 and the table starting at 0.45 -- but that is the settle
    # pass, not the area floor, and nothing here may resize it.
    assert not [n for n in notes if "grew" in n], notes


def test_something_genuinely_tiny_is_still_grown():
    """The floor is lower, not gone: a box at 0.2% is below anything that was
    measured and still gets helped."""
    layout = IG.normalize_layout({"elements": [
        {"desc": "деревянный стол", "x": 0.20, "y": 0.45, "w": 0.55, "h": 0.30},
        {"desc": "крошечная монета", "x": 0.50, "y": 0.30, "w": 0.04, "h": 0.05},
    ]})
    fixed, _ = DA.auto_fix_geometry(layout)
    coin = [el for el in fixed["elements"] if "монета" in el["desc"]][0]
    assert coin["w"] * coin["h"] >= DG.MIN_AREA, coin
    # ...and not by miles: the overshoot clears the threshold, it does not sail
    # past it into hero-prop territory.
    assert coin["w"] * coin["h"] <= DG.MIN_AREA * 1.6, coin


# ── 3. the critic may not condemn a picture for a darkness the pixels deny ────
# Measured 2026-08-29 over fourteen bench renders: it called an ordinary lit
# kitchen "extremely dark and incoherent, making most elements unidentifiable".
# The verdict decides whether to redraw, so an imagined fault throws away a
# picture that was already right -- which is what a pointless redraw looks like
# from the outside.

def _img(tmp_path, name, value):
    from PIL import Image
    p = str(tmp_path / name)
    Image.new("L", (64, 64), value).convert("RGB").save(p)
    return p


def test_a_darkness_complaint_is_dropped_when_the_picture_is_lit(tmp_path):
    bright = _img(tmp_path, "bright.png", 200)
    kept = DA._drop_unfounded(
        bright, ["The image is extremely dark and mostly black silhouettes.",
                 "The cardboard box is missing."])
    assert kept == ["The cardboard box is missing."], kept


def test_a_genuinely_dark_picture_keeps_the_complaint(tmp_path):
    dark = _img(tmp_path, "dark.png", 12)
    probs = ["The image is extremely dark.", "The cat is missing."]
    assert DA._drop_unfounded(dark, probs) == probs


def test_nothing_else_the_critic_says_is_touched(tmp_path):
    bright = _img(tmp_path, "bright2.png", 200)
    probs = ["The mug is much larger than requested.",
             "The cat is not on the sofa."]
    assert DA._drop_unfounded(bright, probs) == probs


def test_an_unreadable_file_leaves_the_verdict_alone(tmp_path):
    probs = ["The image is extremely dark."]
    assert DA._drop_unfounded(str(tmp_path / "nope.png"), probs) == probs


# ── 4. a prop hanging just above a surface is rested on it ───────────────────
# Measured 2026-08-29: the planner gave the table y 0.45-0.75 and the mug
# 0.28-0.42, so the mug's box ended three hundredths of a frame ABOVE the table
# with nothing under it. Ideogram drew the sensible picture -- a mug ON the
# table -- and the critic then condemned the render for disobeying the layout,
# in seven of the fourteen bench cases. The layout was the thing that was wrong.

def _on_table(mug_y, mug_desc="белая кружка с чаем"):
    return IG.normalize_layout({
        "background": "кухня",
        "elements": [
            {"desc": "деревянный стол", "x": 0.20, "y": 0.45, "w": 0.55, "h": 0.30},
            {"desc": mug_desc,          "x": 0.34, "y": mug_y, "w": 0.12, "h": 0.14},
        ]})


def test_a_floating_mug_is_lowered_onto_the_table():
    fixed, notes = DA.auto_fix_geometry(_on_table(0.28))
    mug = [el for el in fixed["elements"] if "кружка" in el["desc"]][0]
    table = [el for el in fixed["elements"] if "стол" in el["desc"]][0]
    assert abs((mug["y"] + mug["h"]) - table["y"]) < 1e-6, (mug, table)
    assert any("rested" in n for n in notes), notes


def test_a_prop_already_resting_is_not_moved():
    layout = _on_table(0.31)            # bottom exactly on the table top
    before = [dict(el) for el in layout["elements"]]
    fixed, _ = DA.auto_fix_geometry(layout)
    assert [dict(el) for el in fixed["elements"]] == before


def test_something_far_above_is_a_different_composition_and_is_left_alone():
    fixed, notes = DA.auto_fix_geometry(_on_table(0.05))
    mug = [el for el in fixed["elements"] if "кружка" in el["desc"]][0]
    assert abs(mug["y"] - 0.05) < 1e-6, (mug, notes)


def test_a_hanging_lamp_is_not_dropped_onto_the_table():
    """The one thing this repair must never do."""
    fixed, notes = DA.auto_fix_geometry(_on_table(0.28, "люстра над столом"))
    lamp = [el for el in fixed["elements"] if "люстра" in el["desc"]][0]
    assert abs(lamp["y"] - 0.28) < 1e-6, (lamp, notes)


def test_a_prop_beside_the_table_is_not_pulled_onto_it():
    layout = IG.normalize_layout({
        "background": "кухня",
        "elements": [
            {"desc": "деревянный стол",     "x": 0.50, "y": 0.45, "w": 0.45, "h": 0.30},
            {"desc": "белая кружка с чаем", "x": 0.05, "y": 0.28, "w": 0.12, "h": 0.14},
        ]})
    fixed, _ = DA.auto_fix_geometry(layout)
    mug = [el for el in fixed["elements"] if "кружка" in el["desc"]][0]
    assert abs(mug["y"] - 0.28) < 1e-6, mug


def test_the_settle_pass_never_makes_the_report_worse():
    """It runs last, after the collision and lettering repairs have finished.

    Found by regression: resting a box on a surface pushed a sign back across
    the subject the lettering repair had just moved it off, and
    test_layout_text_collision went red. A move that raises the problem count
    is undone.
    """
    layout = IG.normalize_layout({
        "background": "улица",
        "elements": [
            {"desc": "a man in a red coat", "x": 0.25, "y": 0.40, "w": 0.50, "h": 0.50},
            {"desc": "вывеска", "text": "OPEN",
             "x": 0.26, "y": 0.20, "w": 0.40, "h": 0.10},
        ]})
    before = len([p for p in DG.geometry_report(layout) if p["kind"] != "crowded"])
    fixed, _ = DA.auto_fix_geometry(layout)
    after = len([p for p in DG.geometry_report(fixed) if p["kind"] != "crowded"])
    assert after <= before, (before, after, DG.geometry_report(fixed))


def test_lettering_is_never_settled():
    """Its position is the lettering repair's business, not this one's."""
    layout = IG.normalize_layout({
        "background": "кухня",
        "elements": [
            {"desc": "деревянный стол", "x": 0.20, "y": 0.45, "w": 0.55, "h": 0.30},
            {"desc": "надпись", "text": "MENU",
             "x": 0.30, "y": 0.28, "w": 0.12, "h": 0.14},
        ]})
    moved = DG._settle_onto_surfaces(layout["elements"])
    assert moved == [], moved
