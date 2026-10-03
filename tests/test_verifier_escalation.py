"""Stage 4: spend test-time compute only where the answer is in doubt.

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


def _esc(draft=CLAIM, called=("inpaint_image",), state=None, done=False):
    return GP._should_escalate_to_verifier(
        draft, set(called), STATE if state is None else state, done)


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
# Journey 31: find_content found the minaret photo, the model asked
# inspect_image "mosque or minaret?" and "MOSQUE: MISSING" replaced a correct
# answer with the edit-did-not-land text.

def test_missing_after_an_edit_is_a_negative_verdict():
    assert GP._inspection_contradicts_work(
        "GLASSES: MISSING.", {"inpaint_image", "inspect_image"}) is True

def test_missing_after_a_render_is_a_negative_verdict():
    assert GP._inspection_contradicts_work(
        "CAT: MISSING.", {"generate_image", "inspect_image"}) is True

def test_missing_in_a_search_answer_is_not_a_verdict():
    assert GP._inspection_contradicts_work(
        "MINARET: PRESENT. | MOSQUE: MISSING.",
        {"unpack_archive", "find_content", "inspect_image"}) is False

def test_a_positive_inspection_is_never_a_verdict():
    assert GP._inspection_contradicts_work(
        "GLASSES: PRESENT.", {"inpaint_image", "inspect_image"}) is False


# Live 2026-09-28: a visibly black cat, inspect said "PARTIAL -- dark
# grey/brown fur textures", and the reply became "the picture stayed as it was".
def test_partial_is_its_own_verdict():
    assert GP._inspection_contradicts_work(
        "Cat is solid black: PARTIAL — tabby markings.", {"inpaint_image"}) == "partial"
    assert GP._inspection_contradicts_work(
        "CAT: PARTIAL. HAT: MISSING.", {"inpaint_image"}) is True


def test_one_partial_among_presents_is_a_pass():
    assert GP._inspection_contradicts_work(
        "Watercolor: PARTIAL - sharp edges. | Paper texture: PRESENT. | Brushwork: PRESENT.",
        {"redraw_image"}) is False


def test_missing_is_success_for_a_removal():
    tc = {"inpaint_image"}
    r = "the dark bag hanging from the shoulder"
    assert GP._inspection_contradicts_work("Bag: MISSING | Traces: MISSING", tc, removal=r) is False
    assert GP._inspection_contradicts_work("Bag: NOT PRESENT", tc, removal=r) is False
    assert GP._inspection_contradicts_work("Dark bag: PRESENT", tc, removal=r) is True
    assert GP._inspection_contradicts_work(
        "Dark bag: MISSING | Area replacement: PRESENT (coat)", tc, removal=r) is False


def test_no_distortion_answer_is_a_pass():
    tc = {"inpaint_image"}
    v = ("Is the hat red? PRESENT | Is it still on the cat's head? PRESENT | "
         "Are there any major distortions to the cat or the background? ABSENT")
    assert not GP._inspection_contradicts_work(v, tc)
    assert GP._inspection_contradicts_work("Is the hat red? MISSING | distortions? ABSENT", tc)


def test_present_that_says_removed_is_a_clean_removal():
    # live 10-03: «убери кота» -> a clean table, judged «нужного изменения нет»
    v = ("A red apple floats above a wooden table. | Cat: PRESENT (The cat is not visible "
         "anywhere in the frame, meaning it has been removed from the scene). | "
         "Empty wooden table: PRESENT.")
    assert GP._inspection_contradicts_work(v, {"inpaint_image"}, removal="cat") is False
    assert GP._inspection_contradicts_work("Cat: PRESENT on the table", {"inpaint_image"},
                                           removal="cat") is True


def test_removal_is_judged_by_the_frame_description():
    # live 10-03: «Is the tabby cat completely removed?: PRESENT» over a clean table
    tc = {"inpaint_image"}
    v = ("A wooden table occupies the foreground, and a broken window frame looks out onto "
         "a dark scene. |  | Tabby cat: PRESENT | Empty wooden table: PRESENT")
    assert GP._inspection_contradicts_work(v, tc, removal="the tabby cat sitting on the table") is False
    v2 = "A tabby cat sits on a wooden table. | Cat: MISSING"
    assert GP._inspection_contradicts_work(v2, tc, removal="the tabby cat") is True


def test_removal_description_in_the_tool_result_format():
    # what the graph actually receives (tool_image_handlers): a header and newlines
    v = ("Inspection of the current image:\nA red apple floats above a large wooden table.\n\n"
         "Is the cat completely removed from the table?: PRESENT\n"
         "Is there only an empty wooden table?: PRESENT")
    assert GP._inspection_contradicts_work(v, {"inpaint_image"}, removal="cat") is False
