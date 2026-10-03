"""A picture button forces its tool.

Live, 2026-09-12 (bench/live_tg_scenarios.py buttons): 🪄 Реставрация sent
"[restore] call redraw_image with mode='restore' ..." and the model answered
"Вы хотите восстановить? Подождите" -- no tool call, then the "actually
nothing happened" correction. The other five buttons happened to be obeyed.
A button payload names one tool; that tool is forced on the first round,
released once it has run, and never forced when it is not on offer.

[restore]/[upscale]/[enhance] no longer exist as buttons -- the old model, their
only engine, was removed along with its checkpoint files -- but the forcing
mechanism itself is still live for [style_preset] (tg_callbacks._cb_style_preset),
used below as this suite's live example of a bracket-tagged button.

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


def test_the_loop_releases_after_the_tool_ran():
    rounds = _rounds_for("[style_preset] call redraw_image with mode=\"redraw\" instructions=\"anime\"")
    assert len(rounds) > 1 and rounds[1][0] == "auto", rounds
