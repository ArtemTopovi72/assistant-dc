"""How far a render is, wherever it runs: one progress line fed by ComfyUI and by the venv
workers, drawn under the status the user sees and beside the stage in the admin panel.

Before it, a ten-minute video showed «Генерирую видео…» from start to end, the same as a
hang; the step counter existed but only the desktop storyboard ever asked for it.
"""
import os
import sys
import threading
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "core", "imaging", "media", "bot", "agent"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ.setdefault("F5_TEST_RUN", "1")

import comfy_client  # noqa: E402
import music  # noqa: E402
import progress  # noqa: E402


class _Ctx:
    def __init__(self):
        self.got = []
        self.cancel = threading.Event()

    def set_progress(self, done, total, what):
        self.got.append((done, total, what))

    def is_cancelled(self):
        return self.cancel.is_set()


def test_the_line_reads_as_a_bar_or_as_a_phase():
    assert progress.bar(3, 10) == "▰▰▰▱▱▱▱▱▱▱ 30%"
    assert progress.line(6, 20, "Step", "ru") == "▰▰▰▱▱▱▱▱▱▱ 30% · шаг 6/20"
    assert progress.line(0, 0, "Loading the model", "ru") == "⏳ Загружаю модель"
    assert progress.bar(0, 0) == ""


def test_a_workers_own_progress_bar_is_read_from_its_log():
    log = "loading\n 10%|█         | 2/20 [00:01<00:09]\r 45%|████▌     | 9/20 [00:04<00:05, 2.1it/s]"
    assert progress.from_log(log) == (9, 20)
    assert progress.from_log("no counter here, 3/4 of the way") is None


def test_a_context_without_a_progress_line_is_left_alone():
    progress.report(types.SimpleNamespace(), 1, 2)          # no set_progress: nothing happens
    given = lambda k, n: None
    assert comfy_client._ctx_progress(types.SimpleNamespace(), {}, given) == (given, None)


def test_a_comfy_render_reports_its_steps_and_the_model_loading():
    ctx, own = _Ctx(), []
    wf = {"1": {"class_type": "UNETLoader"}, "8": {"class_type": "KSampler"},
          "9": {"class_type": "VAEDecode"}}
    on_progress, on_node = comfy_client._ctx_progress(ctx, wf, lambda k, n: own.append((k, n)))
    on_node("1")
    on_progress(3, 8)
    on_node("9")
    assert ctx.got == [(0, 0, "Loading the model"), (3, 8, "Step"), (0, 0, "Decoding the result")]
    assert own == [(3, 8)]                     # the caller's own counter still gets its steps


def test_a_venv_workers_steps_reach_the_context():
    ctx = _Ctx()
    script = ("import sys, time\n"
              "for i in range(1, 6):\n"
              "    sys.stderr.write(f'\\r {i * 20}%|##| {i}/5 [00:0{i}<00:01]'); sys.stderr.flush(); time.sleep(0.6)\n")
    ok, _tail = music.run_gpu_worker(ctx, sys.executable, "", {}, "t", timeout=60,
                                     cmd=[sys.executable, "-c", script])
    assert ok
    steps = [(d, t) for d, t, w in ctx.got if w == "Step"]
    assert steps and all(t == 5 for _, t in steps) and len(steps) >= 2


def test_the_status_line_carries_the_bar_and_a_new_stage_drops_it():
    import tg_tasks
    bot = types.SimpleNamespace(_stages_lock=threading.Lock(), _active_progress={})
    bot._with_progress = types.MethodType(tg_tasks.TaskRunnerMixin._with_progress, bot)
    shown = []
    status, on_progress = bot._with_progress(7, "ru", shown.append)
    status("🎬 <b>Генерирую видео…</b>")
    on_progress(2, 20, "Step")
    assert shown[-1] == "🎬 <b>Генерирую видео…</b>\n▰▱▱▱▱▱▱▱▱▱ 10% · шаг 2/20"
    assert bot._active_progress[7] == "▰▱▱▱▱▱▱▱▱▱ 10% · step 2/20"      # the admin panel's line
    on_progress(3, 20, "Step")                  # a second later: not another edit
    assert len(shown) == 2
    on_progress(20, 20, "Step")                 # the last step always shows
    assert shown[-1].endswith("100% · шаг 20/20")
    status("🔊 <b>Озвучиваю…</b>")
    assert shown[-1] == "🔊 <b>Озвучиваю…</b>" and 7 not in bot._active_progress


def test_a_button_job_draws_its_bar_under_its_own_note_not_the_desktops():
    """A ⛔ job (an audiobook, lyrics) copied the app's context and with it the desktop's bar."""
    import tg_queue
    import tg_tasks
    desktop, edits = [], []
    class Base:                                  # the app's Context: set_progress calls ITS callback
        progress_callback = staticmethod(lambda *a: desktop.append(a))

        def set_progress(self, done, total, what):
            self.progress_callback(done, total, what)
    base = Base()
    bot = types.SimpleNamespace(_stages_lock=threading.Lock(), _active_progress={},
                                _get_ctx=lambda: base, _send_get_id=lambda *a, **k: 42,
                                _edit_text=lambda chat, mid, text, **k: edits.append((mid, text)),
                                _api_post=lambda *a, **k: None, _run_busy=lambda chat, fn: fn())
    bot._with_progress = types.MethodType(tg_tasks.TaskRunnerMixin._with_progress, bot)

    def job(ctx):
        progress.report(ctx, 4, 4, "Piece")
    tg_queue.QueueMixin._run_cancellable(bot, 7, "📖 Читаю книгу…", job, lang="ru")
    assert edits == [(42, "📖 Читаю книгу…\n▰▰▰▰▰▰▰▰▰▰ 100% · фрагмент 4/4")]
    assert not desktop


def test_a_render_waiting_for_the_card_says_so():
    """Another job holding the card was minutes of the same «Generating…» as a hang."""
    ctx, held, go = _Ctx(), threading.Event(), threading.Event()

    def other():
        with comfy_client._gpu_slot(label="other"):
            held.set()
            go.wait(10)
    hogs = [threading.Thread(target=other) for _ in range(comfy_client.COMFY_MAX_CONCURRENT)]
    for t in hogs:
        t.start()
    held.wait(10)
    time.sleep(0.2)
    threading.Timer(0.5, go.set).start()
    with comfy_client._gpu_slot(label="mine", progress_ctx=ctx):
        pass
    for t in hogs:
        t.join(10)
    assert ctx.got == [(0, 0, "Waiting for the video card")]


def test_a_part_of_a_long_video_keeps_its_number_on_the_status():
    """generate_video set the bare «Generating a video» over the caller's «(2/4)», also after
    a «Preparing the references» in between (live 2026-10-20: «(6/6)» then the bare label)."""
    import video
    ctx = types.SimpleNamespace(video_part="2/4")
    ctx.set_stage = lambda s: None
    assert video._video_stage(ctx) == "Generating a video (2/4)"
    ctx.video_part = ""
    assert video._video_stage(ctx) == "Generating a video"
    assert video._video_stage(None) == "Generating a video"


def test_a_quiet_spell_on_the_socket_does_not_end_the_step_counter():
    """The connect timeout is the recv timeout too: a model loading longer than it ended the
    counter for the whole render -- 16 minutes of video with no step shown."""
    import json

    class WebSocketTimeoutException(Exception):
        pass
    frames = [WebSocketTimeoutException(), TimeoutError(),
              json.dumps({"type": "progress_state", "data": {"prompt_id": "p1", "nodes": {
                  "8": {"value": 3, "max": 8, "state": "running"}}}})]

    class Ws:
        def recv(self):
            if not frames:
                raise ConnectionError("closed")
            f = frames.pop(0)
            if isinstance(f, Exception):
                raise f
            return f

        def close(self):
            pass
    got = []
    stub = types.SimpleNamespace(create_connection=lambda *a, **k: Ws())
    comfy_client._ComfyProgress("cid", lambda k, n: got.append((k, n)), {"id": "p1"},
                                modules={"websocket": stub})._run()
    assert got == [(3, 8)]


def _runner(card_busy, last_step_ago):
    import tg_tasks
    comfy = types.SimpleNamespace(card_in_use=lambda: card_busy, server_busy=lambda: False)
    bot = types.SimpleNamespace(modules={"comfy_client": comfy},
                                _progress_at={} if last_step_ago is None else {7: time.monotonic() - last_step_ago})
    bot._still_rendering = types.MethodType(tg_tasks.TaskRunnerMixin._still_rendering, bot)
    return bot


def test_a_render_still_counting_steps_outlives_the_task_ceiling():
    """An approved six-part talking head with its retakes ran past the hour and the watchdog
    cancelled part 6: a render that still reports steps is not a runaway."""
    assert _runner(True, 30)._still_rendering(7)
    assert not _runner(True, 3600)._still_rendering(7)        # the steps stopped: stuck
    assert not _runner(False, 30)._still_rendering(7)         # the card is idle
    assert not _runner(True, None)._still_rendering(7)        # never rendered at all


def test_the_queue_watchdog_spares_a_task_that_is_still_rendering():
    import tg_queue
    import tg_tasks
    now = time.monotonic()
    rendering, stuck = types.SimpleNamespace(task_id="r" * 8), types.SimpleNamespace(task_id="s" * 8)
    comfy = types.SimpleNamespace(card_in_use=lambda: True, server_busy=lambda: False)
    cancels = {t.task_id: threading.Event() for t in (rendering, stuck)}
    bot = types.SimpleNamespace(
        modules={"comfy_client": comfy}, _task_lock=threading.Lock(),
        _running_task={1: [rendering], 2: [stuck]}, _progress_at={1: now - 20, 2: now - 2000},
        _task_started={t.task_id: now - 4000 for t in (rendering, stuck)}, _cancelled=set(),
        _task_cancels=cancels, _activity=types.SimpleNamespace(log=lambda *a, **k: None))
    bot._still_rendering = types.MethodType(tg_tasks.TaskRunnerMixin._still_rendering, bot)
    tg_queue.QueueMixin._cancel_runaways(bot, 3600)
    assert cancels[stuck.task_id].is_set()
    assert not cancels[rendering.task_id].is_set()
