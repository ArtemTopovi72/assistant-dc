"""A repeated request answered with «уже выполнил» and no tool call is a
refusal; a future-tense plan («применю … сформирую … будет готов») is a
promise. Whether a reply says either is the model's reading (agent/reply_read.py);
the phrases are scored live in bench/reply_read_set. Here: the loop acts on it.
"""
import os, sys, inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import graph_finalize as F
import graph_personality as G
import types
from fakes import reads


def _reads(read):
    return types.SimpleNamespace(**reads(read=read))

_H = [{"role": "user", "content": "Пересобери коллаж, удалив дубликаты"},
      {"role": "assistant", "content": "готово"},
      {"role": "user", "content": "Пересобери коллаж, удалив дубликаты"}]


def test_redo_detection():
    # the model's read (agent/intent.py)
    R = _reads({"заново": {"redo": "same"}, "сделай ещё раз": {"redo": "same"}}.get)
    assert F._is_redo_request("заново", [], ctx=R)
    assert F._is_redo_request("сделай ещё раз", [], ctx=R)
    assert F._is_redo_request("Пересобери коллаж, удалив дубликаты", _H, ctx=R)   # same text twice
    assert not F._is_redo_request("найди мост", [{"role": "user", "content": "найди мост"}], ctx=R)
    assert not F._is_redo_request("а теперь коллаж", _H, ctx=R)


def test_guard_is_wired_into_the_loop():
    src = inspect.getsource(G.personality_node)
    assert 'said["already"]' in src
    assert "_is_redo_request(user_input, messages, ctx=ctx)" in src
    assert "dedupe_photos" in src


def test_correction_names_file_tools_when_the_folder_has_files():
    src = inspect.getsource(G.personality_node)
    assert "dedupe_photos to drop re-shots" in src and "inpaint_image for a" in src


def test_promise_after_partial_work_is_still_caught():
    """find_content + dedupe_photos ran, then «Сейчас соберу из них коллаж!» ended
    the turn. The guard must not be disarmed by tools having run earlier in the
    turn."""
    src = inspect.getsource(G.personality_node)
    i = src.index('and said["promises"]):')
    cond = src[src.rfind("if (", 0, i):i]
    assert "_sandbox_touched_recently" in cond and "not tools_ran_now" not in cond, cond


def test_bare_redo_is_pinned_to_the_previous_request():
    """«заново» after the collage ask must not redo the cat edit from six turns
    earlier (redraw_image over a bridge photo)."""
    # the model reads the bare redo (agent/intent.py)
    R = _reads({"заново": {"redo": "same"}, "again": {"redo": "same"}, "ещё раз": {"redo": "same"},
                "пересобери коллаж без дубликатов": {"redo": "changed"}}.get)
    h = [{"role": "user", "content": "сделай кота чёрным"}, {"role": "assistant", "content": "не вышло"},
         {"role": "user", "content": "найди все мосты и собери коллаж"}, {"role": "assistant", "content": "вот"}]
    assert G._previous_request_for_redo("заново", h, ctx=R) == "найди все мосты и собери коллаж"
    assert G._previous_request_for_redo("again", h, ctx=R) == "найди все мосты и собери коллаж"
    # a chain of redos still points at the real request
    assert G._previous_request_for_redo("ещё раз", h + [{"role": "user", "content": "заново"},
                                                       {"role": "assistant", "content": "x"}], ctx=R) == "найди все мосты и собери коллаж"
    # a full request is not a bare redo; nothing before it -> nothing
    assert G._previous_request_for_redo("пересобери коллаж без дубликатов", h, ctx=R) == ""
    assert G._previous_request_for_redo("заново", [], ctx=R) == ""
    src = inspect.getsource(G.personality_node)
    assert "_previous_request_for_redo(user_input, clean_history, ctx=ctx)" in src


def test_collage_claim_needs_a_script_run():
    """«Коллаж готов!» after dedupe_photos with no run_code must not stand: the one
    found photo would go out as the collage."""
    assert "[\"collage\"]" in inspect.getsource(G)   # the ask itself: bench/intent_rest_live.py
    src = inspect.getsource(G.personality_node)
    i = src.index("tools_called_this_turn & _BUILDER_TOOLS")
    assert "claims_result(said)" in src[src.rfind("if (", 0, i):i]


def test_run_code_is_forced_once_the_photos_are_picked():
    """After find_content + dedupe_photos the model does not choose run_code by
    itself, even after two corrective rounds, so it is forced."""
    # the model's read is stubbed; phrases run live in bench/intent_rest_live.py
    R = _reads(lambda t: {"collage": "коллаж" in t.lower(), "dedupe": "дубл" in t.lower()})
    offered = [{"function": {"name": n}} for n in ("run_code", "dedupe_photos", "find_content")]
    ask = "найди все мосты, убери дубликаты и собери коллаж"
    assert G._forced_collage(ask, offered, {"find_content", "dedupe_photos"}, ctx=R) == "run_code"
    # dedupe was asked for and has not run yet: not before it
    assert G._forced_collage(ask, offered, {"find_content"}, ctx=R) == ""
    assert G._forced_collage("найди мосты и собери коллаж", offered, {"find_content"}, ctx=R) == "run_code"
    # already building, no collage asked, run_code not offered
    assert G._forced_collage(ask, offered, {"find_content", "dedupe_photos", "run_code"}, ctx=R) == ""
    assert G._forced_collage("найди мост", offered, {"find_content"}, ctx=R) == ""
    assert G._forced_collage(ask, offered[1:], {"find_content", "dedupe_photos"}, ctx=R) == ""
    assert "_forced_collage(" in inspect.getsource(G.personality_node)
