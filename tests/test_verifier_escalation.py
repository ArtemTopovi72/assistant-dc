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
# find_content found the minaret photo, the model asked
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


# A visibly black cat, inspect said "PARTIAL -- dark
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
    # «убери кота» -> a clean table, judged «нужного изменения нет»
    v = ("A red apple floats above a wooden table. | Cat: PRESENT (The cat is not visible "
         "anywhere in the frame, meaning it has been removed from the scene). | "
         "Empty wooden table: PRESENT.")
    assert GP._inspection_contradicts_work(v, {"inpaint_image"}, removal="cat") is False
    assert GP._inspection_contradicts_work("Cat: PRESENT on the table", {"inpaint_image"},
                                           removal="cat") is True


def test_removal_is_judged_by_the_frame_description():
    # «Is the tabby cat completely removed?: PRESENT» over a clean table
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


# A plainly cyberpunk picture, checked point by point against the spelled-out
# preset, had one detail MISSING and was answered «нужного изменения нет —
# остался прежним» (2026-10-11).
def test_one_missing_detail_among_presents_is_partial_not_unchanged():
    assert GP._inspection_contradicts_work(
        "Cyberpunk neon illustration: PRESENT. | Neon city at night: PRESENT. | "
        "Techwear clothes: PRESENT. | Chrome hardware: MISSING.", {"redraw_image"}) == "partial"
    assert GP._inspection_contradicts_work(
        "Cyberpunk: MISSING. | Neon city: MISSING. | Rain: PRESENT.", {"redraw_image"}) is True


# Shown BEFORE and AFTER, the inspector says the rest «remained unchanged»: an armor
# recolour with every label PRESENT was redone and answered «не полностью» (2026-10-17).
def test_before_after_unchanged_rest_is_a_pass():
    from tool_image_handlers import PAIR_INSPECTION
    v = (PAIR_INSPECTION + "\nThe armor of the girl on the right changed from dark blue to red. "
         "The girl on the left is unchanged.\n\nArmor of the girl on the right is red: PRESENT\n"
         "Rest of image is unchanged: PRESENT")
    assert GP._inspection_contradicts_work(v, {"inpaint_image"}) is False
    failed = (PAIR_INSPECTION + "\nNothing changed between them.\n\nArmor is red: MISSING")
    assert GP._inspection_contradicts_work(failed, {"inpaint_image"}) is True
    # a single picture saying «unchanged» still means the edit did not take
    assert GP._inspection_contradicts_work(
        "Inspection of the current image:\nThe picture is unchanged.", {"inpaint_image"}) is True


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


def test_a_partial_points_own_explanation_is_not_another_failure():
    v = ("Inspection of the current image (BEFORE vs AFTER): the man's clothing was replaced "
         "with red and black plate armor. | The man wearing red and black plate armor: PRESENT | "
         "He is still holding his shovel: PARTIAL (the handle is visible but the blade is missing) | "
         "He is standing by the car: PRESENT")
    assert GP._inspection_contradicts_work(v, {"transfer_image"}) is False
    assert GP._inspection_contradicts_work(v.replace("PARTIAL (the handle is visible but the blade is missing)",
                                                     "PARTIAL — the blade is missing/cut off"),
                                           {"transfer_image"}) is False


def test_a_change_note_naming_the_removed_thing_is_its_removal():
    """The BEFORE/AFTER inspector writes «The edit removed the glasses…»: the glasses named in
    the description were read as still there, and four clean removals (a watch twice, glasses,
    a beard) were answered «нужного изменения нет» (debug chat 2026-10-21)."""
    tc = {"inpaint_image"}
    for removal, v in (
            ("glasses", "Inspection of the current image (BEFORE vs AFTER):\nThe edit removed the glasses "
                        "from the man's face. The person remains the same. |  | Are the glasses gone? PRESENT"),
            ("the watch on his left wrist", "The edit removed the watch and its strap from the man's wrist, "
                                            "replacing them with skin. | Watch: PRESENT — the watch is MISSING"),
            ("beard", "The edit replaces the bottom half of the face (beard and mustache) with clean-shaven "
                      "skin. | Beard removed: PRESENT")):
        assert GP._inspection_contradicts_work(v, tc, removal=removal) is False, removal
    still = "The edit blurred the frame of the glasses, which are still on his face. | Glasses gone? MISSING"
    assert GP._inspection_contradicts_work(still, tc, removal="glasses") is True
    assert GP._inspection_contradicts_work("The man wears glasses and a white shirt. | x: PRESENT",
                                           tc, removal="glasses") is True
