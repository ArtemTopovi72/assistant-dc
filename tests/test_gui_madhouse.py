"""Drive the Madhouse tab for real: set a cast, force a turn, check who spoke.

docs/refactor_blockers.md section 4 says this suite does not exist, and that
its absence is why the Madhouse split had to be trusted on a structural
artifact instead of a test. Three mutations were planted in the moved code at
the time -- MadhouseGrid.set_characters returning early, choose_next_speaker
raising, MadhouseGrid.__init__ raising outright -- and test_gui_adversarial
and test_gui_tabs stayed exit 0 through all three. The room is constructed
defensively enough that a completely broken one reads as a passing suite.

So this file does the thing those two cannot: it drives the tab. It asserts on
STATE the tab produced (the transcript, the grid's sprite table, the combo
contents, who was picked), not merely that a call did not raise -- "did not
raise" is exactly the assertion that let those three mutations through.

Offline and deterministic by construction:
  * `llm` is replaced with a fake module for the router test, so no LM Studio
    call can happen even by accident;
  * MadhouseReplyWorker is stubbed with a class whose start() posts on the
    calling thread -- no QThread is ever started, so nothing can outlive the
    test and abort the process the way a dropped running QThread does;
  * ctx is None or a SimpleNamespace everywhere else, which routes
    generate_madhouse_reply down its no-model placeholder branch.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_madhouse.py
"""
import os
import sys
import types
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("USE_GUI", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QPixmap

import gui_madhouse_tab as MT
import gui_madhouse_brain as MB
import gui_madhouse_grid as MG

_APP = QApplication.instance() or QApplication(sys.argv)

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    assert cond, f"{name}: {detail}"


class _Host:
    """The slice of AssistantWindow the tab actually reaches for."""

    def __init__(self, ctx=None):
        self.ctx = ctx
        self.lines = []
        self.stage = types.SimpleNamespace(set_stage=lambda s: None)

    def _add_room_line(self, name, text, is_user=False):
        self.lines.append((name, text, is_user))


def _tab(ctx=None):
    tab = MT.MadhouseTab(_Host(ctx))
    return tab


def _cast(tab, *names):
    """Add characters through the real UI path (_add_character reads the widgets)."""
    for n in names:
        tab.name_in.setText(n)
        tab._add_character()
    return tab.characters


# --------------------------------------------------------------- construction
def test_tab_builds_every_widget_it_drives():
    """Every widget the tab's own methods touch must exist after construction.

    This is the guard on the UI-building code: _sync_controls, _rebuild_cast_views
    and _tick reach these by name, and a missing one is an AttributeError inside a
    Qt slot -- which is a native 0xC0000409 abort with no traceback, invisible
    under offscreen. Naming them here turns that into a normal test failure.
    """
    tab = _tab()
    for attr in ("name_in", "voice_combo", "pers_combo", "cast_list", "cast_lbl",
                 "start_btn", "pause_btn", "stop_btn", "hush_btn", "voice_mode",
                 "grid", "me_in", "router_box", "who", "say_in", "talk_btn",
                 "status_lbl", "vsplit"):
        check(f"widget_{attr}", getattr(tab, attr, None) is not None)
    # The "speak as" combo is seeded with the human before any cast exists.
    check("who_seeded_with_user", tab.who.count() == 1
          and tab.who.itemData(0) == tab.USER_ID)
    check("empty_room_status", tab.status_lbl.text() == "load a cast to begin")
    check("start_disabled_when_empty", not tab.start_btn.isEnabled())
    tab.shutdown()


def test_add_cast_populates_every_view():
    tab = _tab()
    _cast(tab, "Alice", "Bob")
    check("cast_len", len(tab.characters) == 2)
    check("cast_list_rows", tab.cast_list.count() == 2)
    check("cast_list_shows_name", "Alice" in tab.cast_list.item(0).text())
    check("cast_label", tab.cast_lbl.text() == "2 characters")
    # who = the human + both characters, in that order
    check("who_rows", tab.who.count() == 3)
    check("who_data", [tab.who.itemData(i) for i in (1, 2)]
          == [c["id"] for c in tab.characters])
    # The grid really took the cast -- set_characters returning early was one of
    # the three planted mutations that no suite caught.
    check("grid_took_cast", len(tab.grid._chars) == 2)
    check("grid_avatars_assigned", all(c.get("_avatar") for c in tab.characters))
    check("two_chars_enable_start", tab.start_btn.isEnabled())
    check("status_idle", tab.status_lbl.text() == "idle")
    tab.shutdown()


def test_duplicate_name_is_refused():
    tab = _tab()
    _cast(tab, "Alice")
    warned = []
    orig = QMessageBox.warning
    QMessageBox.warning = staticmethod(lambda *a, **k: warned.append(a))
    try:
        tab.name_in.setText("alice")        # same name, different case
        tab._add_character()
    finally:
        QMessageBox.warning = orig
    check("dup_refused", len(tab.characters) == 1)
    check("dup_warned", len(warned) == 1)
    tab.shutdown()


def test_remove_character_rebuilds_views():
    tab = _tab()
    _cast(tab, "Alice", "Bob")
    tab.cast_list.setCurrentRow(0)
    tab._remove_character()
    check("removed_one", [c["name"] for c in tab.characters] == ["Bob"])
    check("removed_list", tab.cast_list.count() == 1)
    check("removed_grid", len(tab.grid._chars) == 1)
    check("removed_status", tab.status_lbl.text()
          == "need at least 2 characters for auto-dialogue")
    tab.shutdown()


# ------------------------------------------------------------------ transcript
def test_post_writes_transcript_bubble_and_main_chat():
    tab = _tab()
    _cast(tab, "Alice", "Bob")
    tab._post(tab.characters[0], "hello room")
    check("post_transcript", [m["text"] for m in tab.messages] == ["hello room"])
    check("post_attributed", tab.messages[0]["character_id"] == tab.characters[0]["id"])
    check("post_bubble", tab.grid._chars[tab.characters[0]["id"]]["bubble"] == "hello room")
    check("post_mirrored", tab.host.lines == [("Alice", "hello room", False)])
    tab.shutdown()


def test_post_queues_character_lines_for_speech_but_never_yours():
    """You already said your line out loud; the room must not speak it back.

    _pump_speech is stubbed out so the queue can be inspected: with a real one
    the assertion would be reading whatever the pump happened to leave behind,
    which is a different fact from the one under test.
    """
    tab = _tab()
    _cast(tab, "Alice", "Bob")
    tab._pump_speech = lambda: None
    tab._post(tab.characters[0], "hello room")
    check("character_line_queued", [t for _v, t in tab._speech_q] == ["hello room"])
    tab.user["name"] = "Me"
    tab._post(tab.user, "hi back")
    check("user_line_mirrored", tab.host.lines[-1] == ("Me", "hi back", True))
    check("user_line_not_queued", [t for _v, t in tab._speech_q] == ["hello room"])
    tab.shutdown()


def test_no_model_drops_the_speech_backlog_instead_of_synthesising():
    """host.ctx is None => nothing to synthesise with. The room stays text-only.

    Pinned because the alternative is a real MadhouseSpeakWorker starting a real
    F5-TTS render from a test, which is exactly what these suites must never do.
    """
    tab = _tab()                    # host.ctx is None
    _cast(tab, "Alice", "Bob")
    tab._post(tab.characters[0], "say something")
    check("backlog_dropped", tab._speech_q == [])
    check("no_speak_worker_started", tab.speak_worker is None)
    # Silent voice mode does the same even when a ctx exists.
    tab2 = _tab(types.SimpleNamespace(cancel_event=threading.Event(), tts_disabled=False))
    _cast(tab2, "Alice", "Bob")
    tab2.voice_mode.setCurrentIndex(tab2.voice_mode.findData("off"))
    tab2._post(tab2.characters[0], "say something")
    check("silent_mode_drops_backlog", tab2._speech_q == [])
    check("silent_mode_no_worker", tab2.speak_worker is None)
    tab.shutdown(); tab2.shutdown()


def test_send_manual_posts_as_the_selected_speaker():
    tab = _tab()
    _cast(tab, "Alice", "Bob")
    tab.who.setCurrentIndex(tab.who.findData(tab.characters[1]["id"]))   # speak as Bob
    tab.say_in.setText("Bob speaking")
    tab._send_manual()
    check("manual_posted", tab.messages[-1]["name"] == "Bob")
    check("manual_input_cleared", tab.say_in.text() == "")
    tab.say_in.setText("   ")
    tab._send_manual()
    check("manual_blank_ignored", len(tab.messages) == 1)
    tab.shutdown()


def test_clear_history_empties_room_but_keeps_cast():
    tab = _tab()
    _cast(tab, "Alice", "Bob")
    tab._post(tab.characters[0], "something")
    tab._clear_history()
    check("cleared_messages", tab.messages == [])
    check("cleared_bubbles", not any(s.get("bubble") for s in tab.grid._chars.values()))
    check("cast_survives_clear", len(tab.characters) == 2)
    tab.shutdown()


# ----------------------------------------------------------------- a real turn
class _InstantReplyWorker:
    """Stands in for MadhouseReplyWorker without ever starting a QThread.

    start() runs the decision + emit synchronously on the calling thread. A real
    QThread here would be a dropped-running-thread abort waiting to happen; this
    also makes the turn deterministic.
    """
    LAST = {}

    def __init__(self, ctx, characters, history, last_id, use_router, human_name):
        self.ctx, self.characters, self.history = ctx, characters, history
        self.last_id, self.use_router, self.human_name = last_id, use_router, human_name
        self._done = self._failed = self._finished = None
        type(self).LAST = {"ctx": ctx, "last_id": last_id, "use_router": use_router,
                           "human_name": human_name, "history": list(history)}

    class _Sig:
        def __init__(self): self.fn = None
        def connect(self, fn): self.fn = fn
        def emit(self, *a):
            if self.fn: self.fn(*a)

    def __getattr__(self, name):
        if name in ("done", "failed", "finished"):
            sig = _InstantReplyWorker._Sig()
            object.__setattr__(self, name, sig)
            return sig
        raise AttributeError(name)

    def start(self):
        who = MB.choose_next_speaker(self.ctx, self.characters, self.history,
                                     self.last_id, self.use_router, self.human_name)
        self.done.emit(who["id"], f"{who['name']} took the turn")
        self.finished.emit()

    def wait(self, _ms=0): return True


def test_tick_runs_a_turn_and_posts_the_chosen_speaker():
    """The end-to-end assertion docs/refactor_blockers.md asked for."""
    tab = _tab()
    _cast(tab, "Alice", "Bob")
    tab._auto = True                       # _tick returns early otherwise
    orig = MT.MadhouseReplyWorker
    MT.MadhouseReplyWorker = _InstantReplyWorker
    try:
        tab._tick(tab._run_id)
    finally:
        MT.MadhouseReplyWorker = orig
        tab._auto = False
    check("turn_posted_a_line", len(tab.messages) == 1)
    speaker = tab.messages[0]["name"]
    check("turn_speaker_in_cast", speaker in ("Alice", "Bob"), speaker)
    check("turn_text_matches_speaker", tab.messages[0]["text"] == f"{speaker} took the turn")
    check("turn_reached_main_chat", tab.host.lines[-1][0] == speaker)
    check("turn_reached_grid",
          tab.grid._chars[tab.messages[0]["character_id"]]["bubble"] != "")
    tab.shutdown()


def test_no_new_line_while_the_last_one_is_still_being_spoken():
    """Text used to run ahead of the voice without end: a line was generated as
    soon as the previous one was synthesized, while it still waited to play."""
    tab = _tab()
    _cast(tab, "Alice", "Bob")
    tab._auto = True
    orig = MT.MadhouseReplyWorker
    MT.MadhouseReplyWorker = _InstantReplyWorker
    try:
        tab._speech_q.append((None, "still queued"))
        tab._tick(tab._run_id)
        check("queued_speech_holds_the_turn", len(tab.messages) == 0)
        tab._speech_q.clear()
        tab._tick(tab._run_id)
        check("turn_runs_once_the_room_is_quiet", len(tab.messages) == 1)
    finally:
        MT.MadhouseReplyWorker = orig
        tab._auto = False
    tab.shutdown()


def test_tick_hands_the_worker_a_scoped_ctx_not_the_hosts():
    """The main chat's Stop must not be able to mute the room.

    _ScopedCtx exists because one shared cancel_event across surfaces was a real
    bug: a Stop press in the main chat muted the Madhouse permanently. This
    asserts the worker gets a view whose cancel token is the ROOM's.
    """
    host_cancel = threading.Event()
    ctx = types.SimpleNamespace(cancel_event=host_cancel, model_name="m")
    tab = _tab(ctx)
    _cast(tab, "Alice", "Bob")
    tab._auto = True
    orig = MT.MadhouseReplyWorker
    MT.MadhouseReplyWorker = _InstantReplyWorker
    try:
        tab._tick(tab._run_id)
    finally:
        MT.MadhouseReplyWorker = orig
        tab._auto = False
    seen = _InstantReplyWorker.LAST["ctx"]
    check("scoped_not_host_ctx", seen is not ctx)
    check("scoped_uses_room_token", seen.cancel_event is tab._room_cancel)
    host_cancel.set()                      # main chat pressed Stop
    check("host_stop_does_not_mute_room", not seen.cancel_event.is_set())
    tab.shutdown()


def test_stale_tick_from_a_stopped_run_is_a_no_op():
    tab = _tab()
    _cast(tab, "Alice", "Bob")
    tab._auto = True
    stale = tab._run_id
    tab._stop_auto()                       # bumps _run_id, clears _auto
    orig = MT.MadhouseReplyWorker
    MT.MadhouseReplyWorker = _InstantReplyWorker
    try:
        tab._tick(stale)
    finally:
        MT.MadhouseReplyWorker = orig
    check("stale_tick_posted_nothing", tab.messages == [])
    check("stop_set_room_cancel", tab._room_cancel.is_set())
    tab.shutdown()


def test_start_auto_needs_two_characters():
    tab = _tab()
    _cast(tab, "Solo")
    tab._start_auto()
    check("no_auto_with_one", tab._auto is False)
    _cast(tab, "Duo")
    tab._start_auto()
    check("auto_with_two", tab._auto is True)
    check("start_clears_room_cancel", not tab._room_cancel.is_set())
    check("running_status", tab.status_lbl.text() == "running")
    tab._toggle_pause()
    check("paused_status", tab.status_lbl.text() == "paused")
    check("pause_btn_says_resume", tab.pause_btn.text() == "▶  Resume")
    tab._stop_auto()
    check("stopped_status", tab.status_lbl.text() == "idle")
    tab.shutdown()


# ---------------------------------------------------------------------- brain
def test_direct_address_beats_everything():
    cast = [{"id": "a", "name": "Alice", "prompt": ""},
            {"id": "b", "name": "Bob", "prompt": ""},
            {"id": "c", "name": "Carol", "prompt": ""}]
    hist = [{"character_id": "a", "name": "Alice", "text": "Carol, what do you think?"}]
    # ctx=None would already skip the router; pass one and assert the name still
    # wins WITHOUT any model call (a fake llm that raises proves it never ran).
    fake_llm = types.ModuleType("llm")
    def _boom(*a, **k):
        raise AssertionError("the router was called even though a name was addressed")
    fake_llm.send_to_lm_studio = _boom
    old = sys.modules.get("llm")
    sys.modules["llm"] = fake_llm
    try:
        for _ in range(8):                 # the fallback is random; addressing is not
            who = MB.choose_next_speaker(types.SimpleNamespace(), cast, hist, "a",
                                         use_router=True)
            check_once = who["name"] == "Carol"
            if not check_once:
                break
    finally:
        if old is None: sys.modules.pop("llm", None)
        else: sys.modules["llm"] = old
    check("addressed_always_answers", check_once)


def test_router_pick_is_honoured_and_unknown_names_fall_back():
    cast = [{"id": "a", "name": "Alice", "prompt": ""},
            {"id": "b", "name": "Bob", "prompt": ""}]
    hist = [{"character_id": "a", "name": "Alice", "text": "so anyway"}]
    fake_llm = types.ModuleType("llm")
    fake_llm.send_to_lm_studio = lambda *a, **k: {"content": "Bob"}
    old = sys.modules.get("llm")
    sys.modules["llm"] = fake_llm
    try:
        who = MB.choose_next_speaker(types.SimpleNamespace(), cast, hist, "a")
        check("router_pick_honoured", who["name"] == "Bob")
        # An unrecognisable answer must not raise and must still return someone.
        fake_llm.send_to_lm_studio = lambda *a, **k: {"content": "Zaphod"}
        who = MB.choose_next_speaker(types.SimpleNamespace(), cast, hist, "a")
        check("router_unknown_falls_back", who in cast)
        # A raising router is caught and falls back, never propagates into the slot.
        def _raise(*a, **k): raise RuntimeError("LM Studio is down")
        fake_llm.send_to_lm_studio = _raise
        who = MB.choose_next_speaker(types.SimpleNamespace(), cast, hist, "a")
        check("router_exception_falls_back", who in cast)
    finally:
        if old is None: sys.modules.pop("llm", None)
        else: sys.modules["llm"] = old


def test_speaker_never_repeats_when_someone_else_can_go():
    cast = [{"id": "a", "name": "Alice", "prompt": ""},
            {"id": "b", "name": "Bob", "prompt": ""}]
    hist = [{"character_id": "a", "name": "Alice", "text": "mmm"}]
    picks = {MB.choose_next_speaker(None, cast, hist, "a", use_router=False)["id"]
             for _ in range(20)}
    check("no_immediate_repeat", picks == {"b"}, str(picks))
    # With a cast of one the pool falls back to everybody rather than returning None.
    solo = cast[:1]
    check("solo_cast_still_speaks",
          MB.choose_next_speaker(None, solo, hist, "a", use_router=False)["id"] == "a")
    check("empty_cast_returns_none",
          MB.choose_next_speaker(None, [], hist, None, use_router=False) is None)


def test_name_is_addressed_tolerates_russian_case_endings():
    check("addr_exact", MB._name_is_addressed("Alice", "hey Alice, look"))
    check("addr_ru_case", MB._name_is_addressed("Лёха", "Лёхе привет"))
    check("addr_not_substring", not MB._name_is_addressed("Bob", "bobsleigh champion"))
    check("addr_short_name", not MB._name_is_addressed("Al", "although not"))


def test_no_model_reply_is_a_placeholder_not_a_crash():
    line = MB.generate_madhouse_reply(None, {"id": "a", "name": "Alice", "prompt": ""}, [])
    check("placeholder_names_character", line.startswith("Alice"))


# ----------------------------------------------------------------------- grid
def test_grid_renders_with_a_cast_and_a_bubble():
    """MadhouseGrid.__init__ raising outright was a planted mutation nothing caught.

    Painting to a real pixmap is what makes that visible: a broken grid cannot
    render.
    """
    grid = MG.MadhouseGrid()
    grid.resize(320, 240)
    grid.set_characters([{"id": "a", "name": "Alice", "_avatar": "🧑"},
                         {"id": "b", "name": "Bob", "_avatar": "🧔"}])
    check("grid_two_chars", len(grid._chars) == 2)
    grid.show_speech("a", "hello")
    grid.show_note("something happened")
    pm = QPixmap(grid.size())
    grid.render(pm)
    check("grid_rendered", not pm.isNull() and pm.width() == 320)
    grid.clear_all()
    check("grid_cleared_bubbles", not any(s.get("bubble") for s in grid._chars.values()))


def test_shutdown_is_safe_twice():
    tab = _tab()
    _cast(tab, "Alice", "Bob")
    tab._start_auto()
    tab.shutdown()
    tab.shutdown()                          # closeEvent can reach it more than once
    check("shutdown_idempotent", tab._auto is False and tab.recorder is None)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as exc:
            failed += 1
            import traceback; traceback.print_exc()
            print(f"  ERROR in {fn.__name__}: {type(exc).__name__}: {exc}")
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(fns) - failed}/{len(fns)} madhouse functions passed "
          f"({sum(1 for _, ok in RESULTS if ok)}/{len(RESULTS)} checks)")
    if bad:
        print("FAILED CHECKS: " + ", ".join(bad))
    sys.exit(1 if (failed or bad) else 0)
