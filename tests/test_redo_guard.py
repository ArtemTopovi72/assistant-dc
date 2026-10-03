"""A repeated request answered with «уже выполнил» and no tool call is a
refusal; a future-tense plan («применю … сформирую … будет готов») is a
promise. Both seen live 2026-09-14 on the dedupe ask after a restart.
"""
import os, sys, inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import graph_finalize as F
import graph_personality as G
import intent

_H = [{"role": "user", "content": "Пересобери коллаж, удалив дубликаты"},
      {"role": "assistant", "content": "готово"},
      {"role": "user", "content": "Пересобери коллаж, удалив дубликаты"}]


def test_future_tense_plan_is_a_promise():
    d = ("В этот раз я применю еще более строгий алгоритм. Проведу сканирование "
         "и сформирую новый коллаж. Твой коллаж будет готов: bridge_ultimate.jpg.")
    assert F._FILE_PROMISE_RE.search(d)
    assert F._FILE_PROMISE_RE.search("I'll rebuild the collage now; it will be ready shortly.")


def test_a_finished_answer_is_not_a_promise():
    assert not F._FILE_PROMISE_RE.search("Готово: собрал коллаж из 4 уникальных мостов, отправляю.")
    assert not F._FILE_PROMISE_RE.search("В архиве 173 фото; мост есть на P1012796.JPG.")


def test_already_done_phrases():
    assert F._ALREADY_DONE_RE.search("Я уже выполнил эту задачу! В предыдущем шаге я пересобрал коллаж.")
    assert F._ALREADY_DONE_RE.search("I've already done that in my previous reply.")
    assert not F._ALREADY_DONE_RE.search("Коллаж готов, отправляю.")


def test_redo_detection():
    import intent   # the model's read (agent/intent.py)
    intent.STUB = {"заново": {"redo": "same"}, "сделай ещё раз": {"redo": "same"}}.get
    assert F._is_redo_request("заново", [])
    assert F._is_redo_request("сделай ещё раз", [])
    assert F._is_redo_request("Пересобери коллаж, удалив дубликаты", _H)   # same text twice
    assert not F._is_redo_request("найди мост", [{"role": "user", "content": "найди мост"}])
    assert not F._is_redo_request("а теперь коллаж", _H)


def test_guard_is_wired_into_the_loop():
    src = inspect.getsource(G.personality_node)
    assert "_ALREADY_DONE_RE.search(draft)" in src
    assert "_is_redo_request(user_input, messages)" in src
    assert "dedupe_photos" in src


def test_correction_names_file_tools_when_the_folder_has_files():
    src = inspect.getsource(G.personality_node)
    assert "dedupe_photos to drop re-shots" in src and "inpaint_image for a" in src


def test_promise_after_partial_work_is_still_caught():
    """Torture run 2026-09-14 19:46: find_content + dedupe_photos ran, then
    «Сейчас соберу из них коллаж!» ended the turn. The guard must not be
    disarmed by tools having run earlier in the turn."""
    assert F._FILE_PROMISE_RE.search("Осталось 4 уникальных фото. Сейчас соберу из них коллаж!")
    src = inspect.getsource(G.personality_node)
    i = src.index("_FILE_PROMISE_RE.search(draft)")
    cond = src[src.rfind("if (", 0, i):i]
    assert "not tools_ran_now" not in cond, cond


def test_bare_redo_is_pinned_to_the_previous_request():
    """Torture run 2026-09-14 19:47: «заново» after the collage ask redid the
    cat edit from six turns earlier (redraw_image over a bridge photo)."""
    import intent   # the model reads the bare redo (agent/intent.py)
    intent.STUB = {"заново": {"redo": "same"}, "again": {"redo": "same"}, "ещё раз": {"redo": "same"},
                   "пересобери коллаж без дубликатов": {"redo": "changed"}}.get
    h = [{"role": "user", "content": "сделай кота чёрным"}, {"role": "assistant", "content": "не вышло"},
         {"role": "user", "content": "найди все мосты и собери коллаж"}, {"role": "assistant", "content": "вот"}]
    assert G._previous_request_for_redo("заново", h) == "найди все мосты и собери коллаж"
    assert G._previous_request_for_redo("again", h) == "найди все мосты и собери коллаж"
    # a chain of redos still points at the real request
    assert G._previous_request_for_redo("ещё раз", h + [{"role": "user", "content": "заново"},
                                                       {"role": "assistant", "content": "x"}]) == "найди все мосты и собери коллаж"
    # a full request is not a bare redo; nothing before it -> nothing
    assert G._previous_request_for_redo("пересобери коллаж без дубликатов", h) == ""
    assert G._previous_request_for_redo("заново", []) == ""
    src = inspect.getsource(G.personality_node)
    assert "_previous_request_for_redo(user_input, clean_history)" in src


def test_collage_claim_needs_a_script_run():
    """Torture run 2026-09-14 20:00: «Коллаж готов!» after dedupe_photos with
    no run_code; the one found photo went out as the collage."""
    assert G._COLLAGE_DONE_RE.search("Я нашёл 8 мостов, теперь их 4. Коллаж готов!")
    assert G._COLLAGE_DONE_RE.search("Вот коллаж из 4 мостов.")
    assert G._COLLAGE_DONE_RE.search("The collage is ready.")
    assert not G._COLLAGE_DONE_RE.search("Осталось 4 фото. Сейчас соберу из них коллаж!")
    assert not G._COLLAGE_DONE_RE.search("В архиве 173 фото.")
    assert "[\"collage\"]" in inspect.getsource(G)   # the ask itself: bench/intent_rest_live.py
    src = inspect.getsource(G.personality_node)
    assert "tools_called_this_turn & _BUILDER_TOOLS" in src


def test_run_code_is_forced_once_the_photos_are_picked():
    """Torture runs #2-#3: after find_content + dedupe_photos the model never
    chose run_code by itself, even after two corrective rounds."""
    # the model's read is stubbed; phrases run live in bench/intent_rest_live.py
    intent.STUB = lambda t: {"collage": "коллаж" in t.lower(), "dedupe": "дубл" in t.lower()}
    offered = [{"function": {"name": n}} for n in ("run_code", "dedupe_photos", "find_content")]
    ask = "найди все мосты, убери дубликаты и собери коллаж"
    assert G._forced_collage(ask, offered, {"find_content", "dedupe_photos"}) == "run_code"
    # dedupe was asked for and has not run yet: not before it
    assert G._forced_collage(ask, offered, {"find_content"}) == ""
    assert G._forced_collage("найди мосты и собери коллаж", offered, {"find_content"}) == "run_code"
    # already building, no collage asked, run_code not offered
    assert G._forced_collage(ask, offered, {"find_content", "dedupe_photos", "run_code"}) == ""
    assert G._forced_collage("найди мост", offered, {"find_content"}) == ""
    assert G._forced_collage(ask, offered[1:], {"find_content", "dedupe_photos"}) == ""
    assert "_forced_collage(" in inspect.getsource(G.personality_node)
