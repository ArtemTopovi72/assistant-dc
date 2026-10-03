"""Build and drive the REAL window on the NATIVE Qt platform, not offscreen.

Every other gui suite sets QT_QPA_PLATFORM=offscreen. That backend never creates
a real window, never runs the platform's event delivery, and never exercises the
native title-bar filter — so a slot that crashes on the real platform passes
offscreen. The one native crash this project has actually shipped (0xC0000409,
an AttributeError inside eventFilter during construction) was invisible to the
offscreen suites for exactly that reason.

This runs in a SUBPROCESS so the exit code is the assertion: a PyQt abort kills
the interpreter, and no in-process test can observe that. The window is created
off-screen-positioned and closed immediately, so it does not steal focus for
more than a moment.

Run: venv/Scripts/python.exe tests/test_gui_native_platform.py
"""
import os, sys, subprocess, tempfile, textwrap
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Off Windows/macOS the native platform is X11 or Wayland: with neither there
# is no native platform to test (run it under xvfb-run instead).
if (sys.platform.startswith("linux")
        and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))):
    print("SKIP: no display (run under xvfb-run to exercise the native platform)")
    sys.exit(0)

CHILD = textwrap.dedent(r'''
    import os, sys, types, threading, time
    # NO QT_QPA_PLATFORM — use whatever the platform really is.
    os.environ["USE_GUI"] = "0"
    os.environ["GUI_REPORT_HTML"] = "0"
    sys.path.insert(0, %(root)r)
    import logging; logging.basicConfig(level=logging.CRITICAL)
    from pathlib import Path
    import crash_diag; crash_diag.install()
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QThread
    import gui

    app = QApplication.instance() or QApplication(sys.argv)

    def fake_ctx():
        c = types.SimpleNamespace()
        c.cancel_event = threading.Event(); c.memory_lock = threading.RLock()
        c.session_memory = []; c.pinned_facts = []
        c.model_name = "m"; c.no_think = True
        c.mic_disabled = True; c.tts_disabled = True
        c.reasoning_effort = "high"; c.response_length = "auto"
        c.web_search_enabled = True; c.reference_person_mode = False
        c.custom_ref_wav = None; c.custom_personality_path = None
        c.custom_personality_text = ""
        c.reference_images = []; c.last_image_path = None; c.last_image_prompt = ""
        c.last_research_report = ""; c.last_research_path = None
        c.total_user_turns = 0
        d = Path(%(tmp)r) / "mem" / "default"; d.mkdir(parents=True, exist_ok=True)
        c.active_memory_dir = d
        c.stage_callback = None; c.gui_mode = True
        c.models = types.SimpleNamespace(whisper=1, tts_model=1, vocoder=1, accentor_loaded=0)
        for n in ("remember", "set_stage", "save_memory", "load_memory"):
            setattr(c, n, lambda *a, **k: None)
        c.memory_text = lambda *a, **k: ""
        c.is_cancelled = lambda: c.cancel_event.is_set()
        return c

    real = gui.ModelLoader
    gui.ModelLoader = type("NoopLoader", (real,),
                           {"start": lambda self: None, "run": lambda self: None})
    w = gui.AssistantWindow("m", True, "high")
    gui.ModelLoader = real
    print("STAGE constructed", flush=True)

    class P:
        is_active = False
        def play(self, p): pass
        def stop(self): pass
        def toggle_pause(self): return False
    w.player = P()
    w.ctx = fake_ctx()
    w.graph = types.SimpleNamespace(invoke=lambda s: {"final_answer": "ok", "messages": []})
    w.base_state = {"messages": []}
    w.cap = None

    # A real, mapped window — this is the part offscreen cannot reproduce.
    w.show()
    app.processEvents()
    print("STAGE shown", flush=True)

    w.stack.setCurrentWidget(w.dashboard)
    w._set_busy(False)
    app.processEvents()

    # Walk every tab: construction is lazy in places and paintEvents only run
    # for real when the widget is actually mapped.
    for i in range(w.tabs.count()):
        w.tabs.setCurrentIndex(i)
        app.processEvents()
    print("STAGE tabs walked", flush=True)

    # Resize: FlowLayout / MadhouseGrid / ImagesPanel all do arithmetic on the
    # viewport size, and a zero-width viewport only happens on a real platform.
    for size in ((1280, 800), (420, 360), (240, 200), (1600, 900)):
        w.resize(*size)
        app.processEvents()
    print("STAGE resized", flush=True)

    w._add_user("hello"); w._add_assistant("hi"); w._add_system("note")
    w._on_done({"final_answer": "done"})
    app.processEvents()

    # Close with a worker still running: the teardown join must handle it.
    class Slow(QThread):
        def run(self):
            t0 = time.time()
            while time.time() - t0 < 1.5:
                time.sleep(0.02)
    w.transcribe_worker = Slow(); w.transcribe_worker.start()
    time.sleep(0.1)
    print("STAGE closing", flush=True)
    w.close()
    app.processEvents()
    live = [t for t in w._live_threads()]
    print("STAGE closed live=%%d" %% len(live), flush=True)
    assert not live, "threads survived close: %%r" %% live
    print("STAGE ok", flush=True)
''')


def main():
    tmp = tempfile.mkdtemp(prefix="guinative_")
    crash_log = os.path.join(ROOT, "crash.log")
    before = os.path.getsize(crash_log) if os.path.exists(crash_log) else 0

    src = CHILD % {"root": ROOT, "tmp": tmp}
    script = os.path.join(tmp, "child.py")
    with open(script, "w", encoding="utf-8") as fh:
        fh.write(src)

    env = dict(os.environ)
    env.pop("QT_QPA_PLATFORM", None)          # force the real platform
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run([sys.executable, script], capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=300, env=env, cwd=ROOT)
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out.strip()[-3000:])

    results = []
    def check(name, cond, detail=""):
        results.append((name, bool(cond)))
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))

    reached = [ln.split(" ", 1)[1] for ln in out.splitlines() if ln.startswith("STAGE ")]
    check("native_window_constructed", any(s.startswith("constructed") for s in reached))
    check("native_window_shown", any(s.startswith("shown") for s in reached))
    check("native_tabs_walked", any(s.startswith("tabs walked") for s in reached))
    check("native_resized", any(s.startswith("resized") for s in reached))
    check("native_closed_clean", any(s.startswith("ok") for s in reached),
          f"last stage: {reached[-1] if reached else 'none'}")
    check("native_exit_code_zero", proc.returncode == 0, f"rc={proc.returncode}")
    # A PyQt slot abort is 0xC0000409 -> -1073740791 on Windows.
    check("no_native_abort", proc.returncode not in (-1073740791, 3221226505),
          f"rc={proc.returncode}")

    after = os.path.getsize(crash_log) if os.path.exists(crash_log) else 0
    new = ""
    if after > before:
        with open(crash_log, "r", encoding="utf-8", errors="replace") as fh:
            fh.seek(before); new = fh.read()
    check("crash_log_not_appended", after == before, new[:800])

    ok = sum(1 for _, c in results if c)
    print(f"\n==== {ok}/{len(results)} checks passed ====")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
