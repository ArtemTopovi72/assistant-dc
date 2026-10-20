"""A picture button forces its tool.

A button payload names one tool ("[style_preset] call redraw_image with
mode=..."); that tool is forced on the first round, released once it has run,
and never forced when it is not on offer. Without the force the model may answer
in text with no tool call.

[style_preset] (tg_callbacks._cb_style_preset) is this suite's live example of a
bracket-tagged button.

Run: venv/Scripts/python.exe -m pytest tests/test_forced_button_tool.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
import graph_personality as G
import tg_bot as T

IMG = [{"function": {"name": "redraw_image"}}, {"function": {"name": "inpaint_image"}},
       {"function": {"name": "generate_image"}}]


def test_every_button_payload_names_a_tool_that_is_forced():
    for verb, payload in T._CB_CMDS.items():
        assert G._forced_button(payload, IMG, set()) in ("redraw_image", "generate_image", "inpaint_image"), verb


def test_style_preset_forces_redraw():
    assert G._forced_button("[style_preset] call redraw_image with mode=\"redraw\"", IMG, set()) == "redraw_image"


def test_regenerate_forces_generate():
    assert G._forced_button(T._CB_CMDS["regenerate"], IMG, set()) == "generate_image"


def test_released_once_the_tool_ran():
    assert G._forced_button("[style_preset] call redraw_image", IMG, {"redraw_image"}) == ""


def test_never_forces_a_tool_that_is_not_offered():
    assert G._forced_button("[style_preset] call redraw_image", [{"function": {"name": "search"}}], set()) == ""


def test_ordinary_text_is_not_forced():
    assert G._forced_button("восстанови старое фото", IMG, set()) == ""
    assert G._forced_button("", IMG, set()) == ""


# --- the loop actually uses it ---------------------------------------------
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_tgf", os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_graph_full.py"))
_TGF = _ilu.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_TGF)
except SystemExit:
    pass


def _rounds_for(text):
    llm = _TGF.LLM([
        _TGF._asst("", [_TGF._tc("redraw_image", {"mode": "redraw"}, "r1")]),
        _TGF._asst("Готово."),
    ])
    tools = _TGF.Tools({"redraw_image": "Image redrawn: x.png"})
    ctx = _TGF._ctx()
    _TGF._run(ctx, llm, tools, {"user_input": text, "messages": []}, fastpath=False)
    return [(c["kw"].get("tool_choice"),
             [t.get("function", {}).get("name") for t in (c["kw"].get("tools") or [])])
            for c in llm.calls]


def test_the_loop_forces_redraw_on_the_style_preset_button():
    rounds = _rounds_for("[style_preset] call redraw_image with mode=\"redraw\" instructions=\"anime\"")
    assert rounds, "no LLM call"
    # Forced = only that schema on offer; the first try is "auto" so the no-think
    # prefill can go with it ("required" is retried only on a text answer).
    assert rounds[0][0] == "auto" and rounds[0][1] == ["redraw_image"], rounds


def test_the_loop_answers_after_the_tool_ran():
    # released, and with nothing else on offer: a button is one action (live 2026-10-21)
    rounds = _rounds_for("[style_preset] call redraw_image with mode=\"redraw\" instructions=\"anime\"")
    assert len(rounds) > 1 and rounds[1] == ("none", []), rounds


def test_a_button_is_one_action_and_then_the_answer():
    """After the 🎨 «Реализм» redraw the model tried two hand inpaints of its own and answered «the
    change is not there» under the finished photo (live 2026-10-21)."""
    assert G._button_done("[style_preset] call redraw_image", {"redraw_image"})
    assert not G._button_done("[style_preset] call redraw_image", set())
    # the draw menu carries the user's own words: it may ask for more than one thing
    assert not G._button_done("generate an image of: a cat, then animate it", {"generate_image"})


def _edit_turn(text, script, calls, reverses=False):
    llm = _TGF.LLM([_TGF._asst("", [_TGF._tc(name, args, f"c{i}")]) for i, (name, args) in enumerate(calls)]
                   + [_TGF._asst("Не получилось.")])
    tools = _TGF.Tools(script)
    ctx = _TGF._ctx(yes_no=lambda *a, **k: reverses)
    _TGF._run(ctx, llm, tools, {"user_input": text, "messages": []}, fastpath=False)
    return [n for n, _a in tools.calls]


FAIL = "[TOOL ERROR] Could not edit the image — the edit of 'chest' was rejected by the check."
CHEST = {"region": "the chest area", "instructions": "a solid white bodysuit"}


def test_an_identical_failed_edit_is_not_run_again():
    """Three QA-rejected chest inpaints, two of them identical, a minute each (live 2026-10-21)."""
    ran = _edit_turn("убери с неё пятно", {"inpaint_image": [FAIL, FAIL, FAIL]},
                     [("inpaint_image", CHEST)] * 3)
    assert ran == ["inpaint_image"], ran


def test_two_failed_edits_end_the_editing():
    other = {"region": "the chest", "instructions": "a white top"}
    third = {"region": "the torso", "instructions": "white"}
    ran = _edit_turn("убери с неё пятно", {"inpaint_image": [FAIL, FAIL, FAIL]},
                     [("inpaint_image", CHEST), ("inpaint_image", other), ("inpaint_image", third)])
    assert ran == ["inpaint_image", "inpaint_image"], ran


def test_an_edit_that_reverses_the_request_is_not_run():
    """«remove …» rewritten into «a bodysuit covering her chest completely» (live 2026-10-21)."""
    ran = _edit_turn("remove from the image: the bodysuit", {}, [("inpaint_image", CHEST)], reverses=True)
    assert ran == [], ran
