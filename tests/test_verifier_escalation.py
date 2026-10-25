"""Spend test-time compute only where the answer is in doubt.

The uncertain moment is precise and cheap to recognise: the model is about to
assert that a specific edit landed, and nothing in the turn checked whether it
did (`img_inpaint [wrong]` on the chaos bench — one inpaint_image call, a
plausible "Edit applied" back, no verification, "Готово! Я добавил ей очки.").

Escalating unconditionally — inspecting after every edit — buys the same
certainty and charges a vision call for it on every edit turn. The negatives
below are therefore the point of the suite: they are what keeps the cost on the
uncertain turns only.
"""
import sys, os, importlib.util as ilu
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import graph_personality as GP

CLAIM = "Готово! Я добавил ей очки."
STATE = {"image_path": "/tmp/x.png"}


_CTX = __import__("types").SimpleNamespace(reply_read=__import__("fakes").claims())


def _esc(draft=CLAIM, called=("inpaint_image",), state=None, done=False):
    return GP._should_escalate_to_verifier(
        draft, set(called), STATE if state is None else state, done, _CTX)


# --- when it must fire ------------------------------------------------------

def test_an_unverified_edit_claim_escalates():
    assert _esc() is True

def test_every_edit_tool_counts():
    for t in GP._EDIT_TOOLS:
        assert _esc(called=(t,)) is True, t


# --- when it must NOT (this is where the cost is controlled) ---------------

def test_a_turn_that_already_verified_pays_nothing():
    assert _esc(called=("inpaint_image", "inspect_image")) is False

def test_an_answer_that_asserts_nothing_pays_nothing():
    assert _esc(draft="Какую оправу ты хочешь?") is False

def test_generating_from_scratch_does_not_escalate():
    """The picture IS the deliverable; there is no separate change to confirm."""
    assert _esc(called=("generate_image",)) is False

def test_no_artifact_means_the_failure_nets_own_it():
    assert _esc(state={}) is False

def test_it_escalates_at_most_once_per_turn():
    """A second inspection cannot learn more than the first, and the budget is
    the user's."""
    assert _esc(done=True) is False

def test_an_empty_draft_does_not_escalate():
    assert _esc(draft="") is False


# --- the loop actually does it ---------------------------------------------

_spec = ilu.spec_from_file_location(
    "_tgf2", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "test_graph_full.py"))
_TGF = ilu.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_TGF)
except SystemExit:
    pass


def _run_edit_turn(second_reply):
    """inpaint runs, the model claims success, then whatever `second_reply` says."""
    llm = _TGF.LLM([
        _TGF._asst("", [_TGF._tc("inpaint_image", {"prompt": "очки"}, "e1")]),
        _TGF._asst(CLAIM),
        second_reply,
        _TGF._asst("Проверил: очки на месте."),
    ])
    tools = _TGF.Tools({
        "inpaint_image": "Edit applied; the rest of the image is unchanged.",
        "inspect_image": "Inspection: the glasses are present.",
    })
    ctx = _TGF._ctx()
    ctx.last_image_path = "/tmp/in.png"
    state = {"user_input": "добавь ей очки", "messages": [],
             "image_path": "/tmp/x.png"}
    out = _TGF._run(ctx, llm, tools, state, fastpath=False)
    return llm, tools, out


def test_the_loop_escalates_instead_of_shipping_the_claim():
    llm, tools, _ = _run_edit_turn(
        _TGF._asst("", [_TGF._tc("inspect_image", {}, "v1")]))
    called = [n for n, _ in tools.calls]
    assert "inspect_image" in called, f"never verified: {called}"


def test_the_escalated_round_forces_the_verifier():
    llm, tools, _ = _run_edit_turn(
        _TGF._asst("", [_TGF._tc("inspect_image", {}, "v1")]))
    forced = [(c["kw"].get("tool_choice"),
               [t.get("function", {}).get("name") for t in (c["kw"].get("tools") or [])])
              for c in llm.calls]
    # forced = only that schema; sent "auto" first by design (graph_personality),
    # "required" is the retry when the model answers in text
    assert any(ch in ("auto", "required") and names == ["inspect_image"]
               for ch, names in forced), forced
    assert all(isinstance(ch, str) for ch, _ in forced), forced


def test_it_does_not_escalate_twice():
    """The model ignoring the forced round must not start a loop.

    Counted on what the loop FORCED, not on what the scripted model chose to
    call: a model that keeps repeating the claim never calls the verifier, so
    counting tool calls scores zero either way and the once-per-turn cap goes
    untested.
    """
    llm, _tools, _ = _run_edit_turn(_TGF._asst(CLAIM))
    n = len([c for c in llm.calls
             if c["kw"].get("tool_choice") == "required"
             and [t.get("function", {}).get("name")
                  for t in (c["kw"].get("tools") or [])] == ["inspect_image"]])
    assert n <= 1, f"escalated {n} times — the cap is one per turn"


# --- a negative inspection is a verdict only about work done this turn ------
# find_content found the minaret photo, inspect_image was asked «mosque or minaret?» and
# «MOSQUE: MISSING» replaced a correct answer. The inspector's labels are no longer parsed
# by word lists: the model reads its text (bench/edit_judge_eval.py, 2026-10-25).

def test_a_search_answer_is_not_a_verdict():
    asked = []
    ctx = __import__("types").SimpleNamespace(choice=lambda *a, **k: asked.append(a) or "not")
    assert GP._inspection_misses_request("найди мечеть", "MINARET: PRESENT | MOSQUE: MISSING",
                                         {"find_content", "inspect_image"}, ctx=ctx) is False
    assert asked == [], "no edit or render this turn: nothing to read"


# Every label PRESENT, only the description off («blue and red armor» for «девушку в
# красных доспехах»): the swap landed, yet the reply said «он остался прежним» (2026-10-17).
def test_a_description_mismatch_is_partial_not_unchanged():
    from fakes import reads
    import graph_finalize as GF
    llm = _TGF.LLM([
        _TGF._asst("", [_TGF._tc("swap_person", {"person": 1}, "e1")]),
        _TGF._asst("", [_TGF._tc("inspect_image", {}, "v1")]),
        _TGF._asst("Готово! Заменил его на девушку в красных доспехах."),
        _TGF._asst("Готово! Заменил его на девушку в красных доспехах."),
    ])
    tools = _TGF.Tools({
        "swap_person": "Swap complete and saved: /tmp/x.png",
        "inspect_image": "A woman in blue and red armor stands in front. | Girl in red armor: PRESENT",
    })
    ctx = _TGF._ctx(**reads(pick=lambda q, t: "partly" if "blue and red" in t else "met"))
    ctx.last_image_path = "/tmp/in.png"
    out = _TGF._run(ctx, llm, tools, {"user_input": "замени мужика на девушку в красных доспехах",
                                      "messages": [], "image_path": "/tmp/x.png"}, fastpath=False)
    assert out["final_answer"] != GF._VERIFY_NEGATIVE_ANSWER, out["final_answer"]


# A girl ADDED beside the man to be dressed is not the change: each retry re-edited
# that result and she stayed through three passes. The retry starts from the picture
# before the rejected attempt (2026-10-17).
def test_a_rejected_edit_is_retried_from_the_picture_before_it():
    from fakes import reads
    sources = []
    llm = _TGF.LLM([
        _TGF._asst("", [_TGF._tc("redraw_image", {"instructions": "armour"}, "e1")]),
        _TGF._asst("", [_TGF._tc("inspect_image", {}, "v1")]),
        _TGF._asst("", [_TGF._tc("redraw_image", {"instructions": "armour"}, "e2")]),
        _TGF._asst("Не получилось."),
    ])
    def run(ctx, state, name, args):
        if name == "redraw_image":
            import tool_context
            sources.append(tool_context._edit_source(ctx, state))
            tool_context._set_current_image(ctx, state, f"/tmp/out{len(sources)}.png")
            return "Image redrawn and saved"
        return "A woman in armor was added in front. | Armor: PRESENT"
    ctx = _TGF._ctx(**reads(pick=lambda q, t: "not" if "woman" in t else "met"))
    ctx.last_image_path = "/tmp/orig.png"
    _TGF._run(ctx, llm, run, {"user_input": "одень его в доспехи", "messages": [],
                              "image_path": "/tmp/orig.png"}, fastpath=False)
    assert sources == ["/tmp/orig.png", "/tmp/orig.png"], sources
