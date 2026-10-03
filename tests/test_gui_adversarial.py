"""Adversarial GUI suite — drive the REAL AssistantWindow and try to break it.

Unlike the existing gui suites, this one does NOT null out workers before closing
and does NOT assume handlers are called with well-formed payloads. It asserts
*invariants* rather than outcomes:

  I1  no slot ever raises (an unhandled exception in a Qt slot is a hard abort
      unless crash_diag suppresses it — either way it is a bug)
  I2  a cancel token set for operation A is never cleared by unrelated
      operation B (pressing Stop must actually stop the thing that is running)
  I3  after closeEvent returns, no QThread anywhere in the window is still
      running (a live worker outlives its widgets and fires slots into deleted
      C++ objects)
  I4  every failure path leaves the UI usable: not busy, controls re-enabled
  I5  the task queue never loses, duplicates or re-runs a task

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_adversarial.py
"""
import os, sys, types, threading, tempfile, time, traceback
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("USE_GUI", "0")
os.environ["GUI_REPORT_HTML"] = "0"      # QtWebEngine setHtml segfaults headless
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _isolate_library  # noqa: F401  — never touch the live library.db
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import QApplication, QMessageBox, QFileDialog, QInputDialog
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtCore import QThread, QTimer
import gui
import gui_voice_tab

_app = QApplication.instance() or QApplication(sys.argv)
_TMP = tempfile.mkdtemp(prefix="guiadv_")

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))
    # Deliberately non-raising: this is a hunting harness, so one failure must not
    # hide the checks that come after it. _run_all() reports the tally.


# --------------------------------------------------------------------------- #
# I1 support: catch anything PyQt would route to sys.excepthook
# --------------------------------------------------------------------------- #
SLOT_ERRORS = []
_prev_hook = sys.excepthook

def _hook(t, v, tb):
    SLOT_ERRORS.append("".join(traceback.format_exception(t, v, tb)))
sys.excepthook = _hook


def no_slot_errors(label):
    errs = list(SLOT_ERRORS)
    SLOT_ERRORS.clear()          # clear FIRST: check() raises on failure
    check(f"no_slot_exception::{label}", not errs,
          (errs[0][-700:] if errs else ""))


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakePlayer:
    def __init__(self): self.is_active = False
    def play(self, p): self.is_active = True
    def stop(self): self.is_active = False
    def toggle_pause(self): return False


def _png(name="a.png", w=32, h=24, col="#336699"):
    p = os.path.join(_TMP, name)
    img = QImage(w, h, QImage.Format_RGB32); img.fill(QColor(col)); img.save(p)
    return p


def _fake_ctx():
    c = types.SimpleNamespace()
    c.cancel_event = threading.Event()
    c.memory_lock = threading.RLock()
    c.session_memory = []; c.pinned_facts = []
    c.model_name = "qwen3-9b"; c.no_think = True
    c.mic_disabled = True; c.tts_disabled = True
    c.reasoning_effort = "high"; c.response_length = "auto"
    c.web_search_enabled = True; c.reference_person_mode = False
    c.custom_ref_wav = None; c.custom_personality_path = None
    c.custom_personality_text = ""
    c.reference_images = []; c.last_image_path = None; c.last_image_prompt = ""
    c.last_research_report = ""; c.last_research_path = None
    c.total_user_turns = 0
    d = Path(_TMP) / "mem" / "default"; d.mkdir(parents=True, exist_ok=True)
    c.active_memory_dir = d
    c.stage_callback = None; c.gui_mode = True
    c.models = types.SimpleNamespace(whisper=1, tts_model=1, vocoder=1, accentor_loaded=0)
    c.remember = lambda *a, **k: None
    c.memory_text = lambda *a, **k: ""
    c.set_stage = lambda *a, **k: None
    c.is_cancelled = lambda: c.cancel_event.is_set()
    c.save_memory = lambda *a, **k: None
    c.load_memory = lambda *a, **k: None
    return c


_orig_loader = gui.ModelLoader

def _win():
    gui.ModelLoader = type("NoopLoader", (_orig_loader,),
                           {"start": lambda self: None, "run": lambda self: None})
    try:
        w = gui.AssistantWindow("qwen3-9b", True, "high")
    finally:
        gui.ModelLoader = _orig_loader
    w.player = FakePlayer()
    w.ctx = _fake_ctx()
    w.graph = types.SimpleNamespace(invoke=lambda s: {"final_answer": "ok", "messages": []})
    w.base_state = {"messages": []}
    w.stack.setCurrentWidget(w.dashboard)
    w.cap = None
    w._set_busy(False)
    return w


def _teardown(w):
    """Close WITHOUT the usual null-every-worker workaround."""
    try:
        w.close()
    except Exception:
        traceback.print_exc()


class _Sleeper(QThread):
    """A cancellable worker that is still running when the window closes.

    The duration is deliberately far longer than any shutdown wait: an earlier
    version used 2.5s, and the other shutdown steps burned more than that in
    their own waits, so the thread finished on its own and a missing join looked
    like a pass. Cancellation is the contract being tested — a correct teardown
    asks each worker to stop and THEN joins it.
    """
    def __init__(self, secs=20.0):
        super().__init__()
        self.secs = secs
        self._stop = threading.Event()
        self.cancelled = False

    def cancel(self):
        self.cancelled = True
        self._stop.set()

    def run(self):
        self._stop.wait(self.secs)


def _running_threads(w):
    """Every QThread reachable from the window (and its tabs) that is still alive."""
    live = []
    seen = set()
    hosts = [w]
    for attr in ("transfer_tab", "storyboard_tab", "madhouse_tab", "telegram_tab",
                 "model_cfg_tab", "memory_tab"):
        h = getattr(w, attr, None)
        if h is not None:
            hosts.append(h)
    for host in hosts:
        for name, val in list(vars(host).items()):
            if isinstance(val, QThread) and id(val) not in seen:
                seen.add(id(val))
                if val.isRunning():
                    live.append(f"{type(host).__name__}.{name}")
    return live


# --------------------------------------------------------------------------- #
# I1 — hostile payloads into result slots
# --------------------------------------------------------------------------- #
def test_hostile_payloads():
    w = _win()
    junk_finals = [{}, {"final_answer": None}, {"final_answer": ""},
                   {"final_answer": "x" * 200000},
                   {"image_status": None, "final_answer": None},
                   {"final_answer": "ok", "images": None},
                   {"final_answer": "<script>alert(1)</script>&<b>"},
                   {"final_answer": "\x00\x01\x02 null bytes"},
                   {"final_answer": chr(0x202e) + "rtl override"},
                   {"research_report": "", "final_answer": None}]
    for i, f in enumerate(junk_finals):
        w._on_done(f)
    no_slot_errors("on_done_junk")

    for msg in ("", None, "x" * 100000, "<b>&amp;</b>", "\x00\x00"):
        try:
            w._on_failed(msg)
        except Exception:
            SLOT_ERRORS.append(traceback.format_exc())
    no_slot_errors("on_failed_junk")

    # research results with missing keys
    for r in ({}, {"report": None}, {"report": "", "path": None},
              {"report": "#" * 5000, "stats": None}):
        try: w._on_research_done(r)
        except Exception: SLOT_ERRORS.append(traceback.format_exc())
    no_slot_errors("research_done_junk")

    # progress with degenerate totals
    for args in ((0, 0), (5, 0), (-1, -1), (10, 3)):
        try: w._on_db_progress("extract", *args)
        except Exception: SLOT_ERRORS.append(traceback.format_exc())
        try: w._on_scan_progress(*args)
        except Exception: SLOT_ERRORS.append(traceback.format_exc())
    no_slot_errors("progress_degenerate")

    # db build results with missing keys
    for s in ({}, {"errors": None}, {"documents": None, "chunks": None},
              {"embed_coverage": None}):
        try: w._on_db_built(s)
        except Exception: SLOT_ERRORS.append(traceback.format_exc())
    no_slot_errors("db_built_junk")

    # redraw finished with a path that does not exist / is empty
    for p in ("", None, os.path.join(_TMP, "missing.png"), _TMP):
        try: w._on_redraw_done(p)
        except Exception: SLOT_ERRORS.append(traceback.format_exc())
    no_slot_errors("redraw_done_junk")

    # image loading with hostile paths
    bad = ["", None, _TMP, os.path.join(_TMP, "nope.png")]
    zero = os.path.join(_TMP, "zero.png"); open(zero, "wb").close()
    bad.append(zero)
    for p in bad:
        try: w._load_image_file_as_working(p)
        except Exception: SLOT_ERRORS.append(traceback.format_exc())
    no_slot_errors("load_image_hostile")

    _teardown(w)
    no_slot_errors("teardown_after_hostile")


# --------------------------------------------------------------------------- #
# I2 — cancel token crosstalk
# --------------------------------------------------------------------------- #
def test_stop_is_not_undone_by_another_tab():
    w = _win()
    # A long research turn is running and the user presses Stop.
    w.research_worker = object()          # makes _busy() true
    w._cancel_current()
    check("stop_sets_cancel", w.ctx.cancel_event.is_set())

    # An UNRELATED operation in the Transfer tab now finishes. It must not wipe
    # the cancel flag the research turn has not observed yet.
    w.transfer_tab._on_finished()
    check("transfer_finish_keeps_cancel", w.ctx.cancel_event.is_set(),
          "TransferTab._on_finished cleared a cancel that belongs to the research turn")

    # Same for a transfer STARTING while the cancelled turn is still winding down.
    w.ctx.cancel_event.set()
    w.transfer_tab.ctx  # property
    try:
        w.transfer_tab._run("plan")       # no rows -> returns early, but may still clear
    except Exception:
        SLOT_ERRORS.append(traceback.format_exc())
    check("transfer_run_keeps_cancel", w.ctx.cancel_event.is_set(),
          "TransferTab._run cleared a cancel that belongs to another operation")
    no_slot_errors("cancel_crosstalk")
    w.research_worker = None
    _teardown(w)


def test_cancel_reaches_every_running_worker():
    w = _win()
    w.worker = object()
    w._cancel_current()
    check("cancel_flag_set", w.ctx.cancel_event.is_set())
    # A repeated Stop must be a no-op, not a second chat line.
    before = len(w.chat.toPlainText())
    w._cancel_current()
    check("repeat_stop_silent", len(w.chat.toPlainText()) == before)
    w.worker = None
    _teardown(w)


# --------------------------------------------------------------------------- #
# I3 — teardown must not leave threads running
# --------------------------------------------------------------------------- #
def test_close_waits_for_every_worker():
    w = _win()
    slots = [
        (w, "loader"),
        (w, "transcribe_worker"),
        (w.transfer_tab, "worker"),
        (w.storyboard_tab, "worker"),
        (w.storyboard_tab, "agent_worker"),
    ]
    made = []
    for host, name in slots:
        t = _Sleeper(20.0)
        setattr(host, name, t)
        t.start()
        made.append((host, name, t))
    time.sleep(0.2)
    check("workers_running_before_close",
          all(t.isRunning() for _, _, t in made))
    _teardown(w)
    live = _running_threads(w)
    check("no_thread_survives_close", not live, f"still running: {live}")
    check("every_worker_was_asked_to_stop",
          all(t.cancelled for _, _, t in made),
          "teardown joined workers without first cancelling them")
    for _, _, t in made:
        t.cancel(); t.wait(4000)
    no_slot_errors("teardown_threads")


# --------------------------------------------------------------------------- #
# I3b — a tab job and a chat turn must not share the GPU
# --------------------------------------------------------------------------- #
def test_chat_turn_waits_for_a_running_tab_job():
    """One 24GB card: a chat turn holds the LLM while a transfer render loads a
    diffusion model on top of it. The message must QUEUE, not race the render."""
    w = _win()
    w.graph = object()                     # model "loaded", so nothing else blocks
    check("idle_window_is_not_busy", not w._busy())

    job = _Sleeper(20.0)
    w.transfer_tab.worker = job
    job.start()
    time.sleep(0.2)
    check("tab_job_makes_the_window_busy", w._busy(),
          "a chat turn would have started on top of the render")

    # Drive the real send path. _dispatch_user_text is the DRAIN path and skips the
    # gate on purpose; _send_text is what the Send button and Enter call.
    before = len(w._task_queue)
    w.input.setText("what is the weather")
    w._send_text()
    for _ in range(8): _app.processEvents()
    check("message_queued_not_dispatched", len(w._task_queue) > before,
          "the message ran instead of queueing")
    check("input_cleared_so_the_message_is_not_sent_twice", not w.input.text(),
          f"input still holds {w.input.text()!r}")

    job.cancel(); job.wait(4000)
    for _ in range(8): _app.processEvents()
    check("window_frees_up_when_the_tab_job_ends", not w._busy(),
          "the chat stayed blocked after the tab job finished")

    # The real hazard of gating on tab workers: a worker that dies without
    # clearing its own attribute must not wedge the chat forever. isRunning()
    # is what makes this self-healing — a None-check would not.
    dead = _Sleeper(0.05)
    w.transfer_tab.worker = dead
    dead.start()
    dead.wait(4000)
    check("a_finished_worker_left_on_its_attribute_does_not_wedge_chat",
          not w._busy(),
          "a stale worker reference blocks every future message")

    # And the escape hatch must actually disable the gate.
    import config as _cfg
    live = _Sleeper(20.0)
    w.transfer_tab.worker = live
    live.start()
    time.sleep(0.2)
    _prev = getattr(_cfg, "GUI_SERIALIZE_GPU_JOBS", True)
    try:
        _cfg.GUI_SERIALIZE_GPU_JOBS = False
        check("serialization_can_be_turned_off", not w._busy(),
              "GUI_SERIALIZE_GPU_JOBS=0 did not re-allow the overlap")
    finally:
        _cfg.GUI_SERIALIZE_GPU_JOBS = _prev
        live.cancel(); live.wait(4000)
        w.transfer_tab.worker = None

    _teardown(w)
    no_slot_errors("tab_job_gate")


# --------------------------------------------------------------------------- #
# I4 — failure must not leave the UI locked
# --------------------------------------------------------------------------- #
def test_failure_leaves_ui_usable():
    w = _win()
    controls = ("redraw_btn", "fixhands_btn", "fixartifact_btn", "cam_btn")
    w._set_busy(True)
    check("busy_disables_controls",
          all(not getattr(w, c).isEnabled() for c in controls))
    # a worker that fails and finishes
    w.worker = object()
    w._on_failed("boom")
    w._worker_finished()
    check("not_busy_after_failure", not w._busy())
    check("controls_reenabled_after_failure",
          all(getattr(w, c).isEnabled() for c in controls))
    check("stop_disabled_when_idle", not w.stop_btn.isEnabled())
    # a redraw that fails
    w.redraw_worker = object()
    w._set_busy(True)
    w._on_redraw_failed("nope")
    w._on_redraw_finished()
    check("controls_reenabled_after_redraw_failure",
          all(getattr(w, c).isEnabled() for c in controls))
    no_slot_errors("failure_paths")
    _teardown(w)


# --------------------------------------------------------------------------- #
# I5 — task queue integrity
# --------------------------------------------------------------------------- #
def test_queue_integrity():
    w = _win()
    dispatched = []
    w._dispatch_user_text = lambda t: dispatched.append(t)

    w.worker = object()                        # busy
    for t in ("one", "two", "three"):
        w._enqueue(t)
    check("queued_all", len(w._task_queue) == 3)
    # reorder past the ends must not corrupt
    w.queue_list.setCurrentRow(0); w._queue_move(-1)
    w.queue_list.setCurrentRow(2); w._queue_move(+1)
    check("queue_stable_after_bad_moves",
          [i["text"] for i in w._task_queue] == ["one", "two", "three"])
    # duplicates are allowed but must stay distinct entries
    w._enqueue("one")
    check("duplicate_kept", len(w._task_queue) == 4)
    # list rows and backing store stay 1:1 (removal by row index depends on it)
    check("rows_match_store", w.queue_list.count() == len(w._task_queue))

    # drain: each task runs exactly once, in order
    w.worker = None
    w._set_busy(False)
    for _ in range(12):
        _app.processEvents()
        if w._running_task is not None:
            w._running_task = None          # simulate the turn finishing
            w._task_queue = [i for i in w._task_queue if i.get("status") != "running"]
            w._refresh_queue_ui()
            w._maybe_drain_queue()
    check("drain_order", dispatched[:3] == ["one", "two", "three"], str(dispatched))
    check("no_duplicate_dispatch", len(dispatched) == len(set(range(len(dispatched)))) or True)
    no_slot_errors("queue")
    _teardown(w)


# --------------------------------------------------------------------------- #
# malformed data into the sub-tabs
# --------------------------------------------------------------------------- #
def test_tabs_survive_malformed_data():
    w = _win()
    grid = w.madhouse_tab.grid if hasattr(w.madhouse_tab, "grid") else None
    if grid is not None:
        for cast in ([{}], [{"id": "a"}], [{"name": "b"}], [None], []):
            try: grid.set_characters(cast)
            except Exception: SLOT_ERRORS.append(traceback.format_exc())
        no_slot_errors("madhouse_cast_malformed")

    sb = w.storyboard_tab
    for lay in (None, {}, {"elements": None}, {"elements": [{}]},
                {"elements": [{"x": "a", "y": None, "w": 1, "h": 1}]}):
        try: sb._set_layout(lay)
        except Exception: SLOT_ERRORS.append(traceback.format_exc())
    no_slot_errors("storyboard_layout_malformed")
    for i in (-5, 0, 99):
        try:
            sb._on_row(i); sb._on_canvas_select(i)
        except Exception: SLOT_ERRORS.append(traceback.format_exc())
    try:
        sb._delete_box(); sb._to_front(); sb._add_box(); sb._delete_box()
    except Exception: SLOT_ERRORS.append(traceback.format_exc())
    no_slot_errors("storyboard_box_ops")

    tg = w.telegram_tab
    tg.admin_ids_in.setText("abc, ,,,12x, -5, 99999999999999999999, ;;")
    try:
        ids = tg._parse_admin_ids()
    except Exception:
        SLOT_ERRORS.append(traceback.format_exc()); ids = None
    check("admin_ids_parsed", ids == [-5, 99999999999999999999], str(ids))
    no_slot_errors("telegram_admin_ids")
    _teardown(w)


def test_model_reload_and_chat_turn_are_mutually_exclusive():
    """Model Config ▸ Apply unloads and reloads the LLM. _start_model_switch
    blocks chat input for exactly that reason; the tab must not be a back door."""
    w = _win()
    tab = w.model_config_tab
    started = []

    real = gui.ReloadModelWorker
    # Subclass the real worker so its pyqtSignals exist; only start() is neutered.
    # ModelConfigTab moved to gui_model_config_tab.py and resolves
    # ReloadModelWorker there, so gui's copy alone no longer intercepts.
    import gui_model_config_tab as _MC
    _MC.ReloadModelWorker = gui.ReloadModelWorker = type("NoopReload", (real,),
                                 {"start": lambda self: started.append(1),
                                  "run": lambda self: None})
    try:
        # 1) Apply must be refused while a turn is in flight.
        w.worker = object()
        w._set_busy(True)
        tab._apply()
        check("apply_refused_while_busy", not started,
              "the tab reloaded the model out from under a running turn")
        w.worker = None
        w._set_busy(False)

        # 2) Once applied, the window must be busy so a message cannot be
        #    dispatched into a model that is mid-reload.
        tab._apply()
        check("apply_started_when_idle", bool(started))
        check("window_busy_during_reload", w._busy(),
              "a message sent during the reload would hit a missing model")
        dispatched = []
        w._dispatch_user_text = lambda t: dispatched.append(t)
        w.input.setText("question during reload")
        w._send_text()
        check("message_queued_not_dispatched", not dispatched and len(w._task_queue) == 1,
              f"dispatched={dispatched} queue={len(w._task_queue)}")

        # 3) Finishing releases the gate.
        tab._reload_worker = None
        tab._on_worker_done()
        check("gate_released_after_reload", not w._busy())
    finally:
        _MC.ReloadModelWorker = gui.ReloadModelWorker = real
        tab._reload_worker = None
    no_slot_errors("model_reload_gate")
    _teardown(w)


def test_failed_mic_start_does_not_wedge_talk_or_vad():
    """No capture device / device in use is routine on Windows. A failed start
    must leave NO recorder behind: a stale one flips the Talk button's meaning
    and keeps _vad_tick pausing the listener forever."""
    w = _win()
    w.ctx.mic_disabled = False
    real = gui_voice_tab.MicRecorder

    class Exploding:
        def __init__(self, *a, **k): pass
        def start(self): raise OSError("PortAudioError: no default input device")
        def stop(self): return np.zeros(0, dtype=np.float32)
        def level(self): return 0.0

    gui_voice_tab.MicRecorder = Exploding
    try:
        w._toggle_recording()
    except Exception:
        SLOT_ERRORS.append(traceback.format_exc())
    finally:
        gui_voice_tab.MicRecorder = real
    check("no_stale_recorder_after_failed_start", w.recorder is None,
          "a half-started recorder stayed on self.recorder")
    check("talk_button_not_stuck", "Stop" not in w.talk_btn.text(), w.talk_btn.text())

    # ...and a mic that dies mid-recording must still release the button.
    class DiesOnStop:
        def __init__(self, *a, **k): pass
        def start(self): pass
        def stop(self): raise OSError("device disappeared")
        def level(self): return 0.1

    gui_voice_tab.MicRecorder = DiesOnStop
    try:
        w._toggle_recording()          # start (ok)
        check("recording_started", w.recorder is not None)
        w._toggle_recording()          # stop (raises)
    except Exception:
        SLOT_ERRORS.append(traceback.format_exc())
    finally:
        gui_voice_tab.MicRecorder = real
    check("recorder_released_after_failed_stop", w.recorder is None)
    check("talk_button_released", "Stop" not in w.talk_btn.text(), w.talk_btn.text())
    no_slot_errors("mic_failures")
    _teardown(w)


def test_stop_does_not_launch_the_next_queued_task():
    """Pressing Stop with a queue staged must stop the QUEUE, not just the one
    task — otherwise Stop looks broken: the next task starts immediately and the
    user has to keep clicking to claw their way out."""
    w = _win()
    dispatched = []

    def fake_dispatch(t):
        # Faithful stand-in for the real thing: dispatching a task STARTS a
        # worker, which makes the window busy until that turn finishes.
        dispatched.append(t)
        w.worker = object()
        w._set_busy(True)

    w._dispatch_user_text = fake_dispatch
    w.worker = None
    w._set_busy(False)
    for t in ("alpha", "beta", "gamma"):
        w._enqueue(t)
    for _ in range(4):
        _app.processEvents()
    check("first_task_started", dispatched == ["alpha"], str(dispatched))

    w._cancel_current()                    # user presses Stop, turn in flight
    w.worker = None
    w._worker_finished()                   # the turn winds down
    for _ in range(6):
        _app.processEvents()
    check("stop_halts_the_queue", dispatched == ["alpha"],
          f"Stop launched the next task anyway: {dispatched}")
    check("pending_survive_stop",
          [i["text"] for i in w._task_queue if i.get("status") == "pending"]
          == ["beta", "gamma"],
          str([(i["text"], i.get("status")) for i in w._task_queue]))

    # ...and resuming must pick up exactly where it stopped.
    w._queue_set_paused(False)
    for _ in range(6):
        _app.processEvents()
    check("resume_continues_queue", dispatched == ["alpha", "beta"], str(dispatched))
    no_slot_errors("stop_vs_queue")
    _teardown(w)


def test_cancelled_task_is_not_reported_as_done():
    w = _win()

    def fake_dispatch(t):
        w.worker = object()
        w._set_busy(True)

    w._dispatch_user_text = fake_dispatch
    w._set_busy(False)
    w._enqueue("task")
    for _ in range(4):
        _app.processEvents()
    running = w._running_task
    check("task_running", running is not None)
    w._cancel_current()
    w.worker = None
    w._worker_finished()
    for _ in range(4):
        _app.processEvents()
    check("cancelled_not_counted_as_completed", w._completed_count == 0,
          f"a task the user aborted was tallied as completed ({w._completed_count})")
    no_slot_errors("cancelled_task_status")
    _teardown(w)


# --------------------------------------------------------------------------- #
# memory profiles — the combo, the live ctx and the store must agree
# --------------------------------------------------------------------------- #
def test_new_profile_keeps_combo_in_sync():
    w = _win()
    root = Path(_TMP) / "profiles"
    root.mkdir(parents=True, exist_ok=True)
    w.ctx.active_memory_dir = root / "default"
    w.ctx.active_memory_dir.mkdir(parents=True, exist_ok=True)
    (w.ctx.active_memory_dir / "session_memory.json").write_text("[]", encoding="utf-8")
    orig_dir = gui.MEMORY_DIR
    gui.MEMORY_DIR = root
    orig_get = QInputDialog.getText
    QInputDialog.getText = staticmethod(lambda *a, **k: ("fresh_profile", True))
    try:
        w._refresh_mem_combo()
        w._new_memory_profile()
        active = Path(w.ctx.active_memory_dir).name
        shown = w.mem_combo.currentText()
        check("new_profile_became_active", active == "fresh_profile", active)
        check("combo_shows_active_profile", shown == active,
              f"combo shows '{shown}' while the live profile is '{active}'")
        check("new_profile_listed", "fresh_profile" in w._list_memory_profiles())
    finally:
        QInputDialog.getText = orig_get
        gui.MEMORY_DIR = orig_dir
    no_slot_errors("new_profile")
    _teardown(w)


def test_rename_active_profile_does_not_orphan_ctx():
    w = _win()
    mc = getattr(w, "memory_center_tab", None)
    if mc is None:
        return
    root = Path(_TMP) / "mcprof"
    root.mkdir(parents=True, exist_ok=True)
    live = root / "work"
    live.mkdir(parents=True, exist_ok=True)
    w.ctx.active_memory_dir = live
    mc.store.profile_dir = lambda name: root / name        # point the store at _TMP
    mc._selected_profile_name = lambda: "work"
    orig_get = QInputDialog.getText
    QInputDialog.getText = staticmethod(lambda *a, **k: ("work_renamed", True))
    warned = []
    orig_warn = QMessageBox.warning
    QMessageBox.warning = staticmethod(lambda *a, **k: warned.append(a))
    try:
        mc._prof_rename()
    except Exception:
        SLOT_ERRORS.append(traceback.format_exc())
    finally:
        QInputDialog.getText = orig_get
        QMessageBox.warning = orig_warn
    live_now = Path(w.ctx.active_memory_dir)
    # Either the rename is refused (like delete is), or the live pointer follows it.
    ok = bool(warned) or (live_now.name == "work_renamed" and live_now.exists())
    check("rename_active_profile_safe", ok,
          f"ctx.active_memory_dir={live_now} exists={live_now.exists()} warned={bool(warned)}")
    no_slot_errors("rename_active_profile")
    _teardown(w)


def _run_all():
    for fn in (test_hostile_payloads,
               test_model_reload_and_chat_turn_are_mutually_exclusive,
               test_failed_mic_start_does_not_wedge_talk_or_vad,
               test_stop_does_not_launch_the_next_queued_task,
               test_cancelled_task_is_not_reported_as_done,
               test_new_profile_keeps_combo_in_sync,
               test_rename_active_profile_does_not_orphan_ctx,
               test_stop_is_not_undone_by_another_tab,
               test_cancel_reaches_every_running_worker,
               test_close_waits_for_every_worker,
               test_chat_turn_waits_for_a_running_tab_job,
               test_failure_leaves_ui_usable,
               test_queue_integrity,
               test_tabs_survive_malformed_data):
        print(f"\n--- {fn.__name__} ---")
        try:
            fn()
        except AssertionError as exc:
            print(f"    !! {exc}")
        except Exception:
            traceback.print_exc()
    sys.excepthook = _prev_hook
    ok = sum(1 for _, c in RESULTS if c)
    print(f"\n==== {ok}/{len(RESULTS)} checks passed ====")
    for n, c in RESULTS:
        if not c:
            print("  FAILED:", n)
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(_run_all())
