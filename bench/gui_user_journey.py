"""Drive the REAL window the way a person does: type, press Enter, wait, look.

Every other gui suite pokes methods directly. That verifies the code path and
misses the interface: whether the keystroke reaches the handler, whether the
button is enabled when it should be, whether a second message typed while the
assistant is busy is dropped or queued. This suite types with QTest.keyClicks
and presses Enter, then waits on the UI for the result — no handler is called by
name.

It is a REAL end-to-end run: the live LM Studio plans the picture and the live
ComfyUI renders it. Nothing is stubbed. Consequently it is slow (a render is
~40s) and it needs both services up; it says so and exits 2 rather than pretend
to pass when they are down.

Runs in a SUBPROCESS on the NATIVE Qt platform. Offscreen never creates a real
window and has hidden a shipped crash before (0xC0000409 in an event filter), and
a PyQt abort kills the interpreter, so the child's exit code is the assertion.

SAFETY: redirect_settings + redirect_data_dir are called BEFORE the window is
built, so this never writes to the live QSettings scope (a previous suite
overwrote the real BotFather token that way) or the live user store.

Run: venv/Scripts/python.exe tests/test_gui_user_journey.py
"""
import os, sys, subprocess, tempfile, textwrap, json, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def _up(url, timeout=6):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


import config
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

if not _up(config.COMFY_URL.rstrip("/") + "/system_stats"):
    print(f"CANNOT RUN: ComfyUI is not answering at {config.COMFY_URL}. "
          "This suite draws for real; a stubbed render would not test the interface.")
    sys.exit(2)
if not _up(config.LM_STUDIO_URL.replace("/chat/completions", "/models")):
    print("CANNOT RUN: LM Studio is not answering.")
    sys.exit(2)

# Both answering is not the same as both FITTING. This suite makes the live
# model plan a picture and the live renderer draw it, one after the other, so
# they need the card at the same time -- and on a 24 GB card a 20.5 GB LLM
# leaves Ideogram nothing. Twice measured: the run waited its full ten minutes
# for a render that could never start, then reported "a new image file
# appeared" as a failure. That is a machine budget, not a defect, so say which
# it is instead of spending ten minutes to report the wrong one.
_NEED_FREE_MIB = 18000
try:
    import subprocess as _sp
    _free = int(_sp.run(["nvidia-smi", "--query-gpu=memory.free",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=20,
                        stdin=_sp.DEVNULL).stdout.strip().splitlines()[0])
except Exception:
    _free = _NEED_FREE_MIB          # cannot tell -- go ahead and try
if _free < _NEED_FREE_MIB:
    print("CANNOT RUN: only %d MiB of VRAM free and the renderer needs about "
          "%d. Free the card first (unload the LLM, or POST /free to ComfyUI); "
          "both services answering is not enough, they have to fit together."
          % (_free, _NEED_FREE_MIB))
    sys.exit(2)

CHILD = textwrap.dedent(r'''
    import os, sys, types, threading, time, json, glob
    # NO QT_QPA_PLATFORM: use the real platform, like the user's own session.
    os.environ["USE_GUI"] = "0"
    os.environ["GUI_REPORT_HTML"] = "0"
    sys.path.insert(0, %(root)r)
    import logging
    logging.basicConfig(level=logging.WARNING,
                        format="%%(levelname)s %%(name)s: %%(message)s")

    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt, QTimer, QElapsedTimer, QEvent
    from PyQt5.QtGui import QKeyEvent
    from PyQt5.QtTest import QTest
    import crash_diag; crash_diag.install()
    import gui, config

    TMP = %(tmp)r
    # BEFORE the window exists: never touch the live settings scope or user store.
    gui.redirect_settings(TMP)
    try:
        import tg_bot as _tg
        _tg.redirect_data_dir(TMP)
    except Exception:
        pass

    _n = {"ok": 0, "bad": 0}
    def check(label, cond, detail=""):
        if cond:
            _n["ok"] += 1
            print(f"  ok   {label}", flush=True)
        else:
            _n["bad"] += 1
            print(f"  FAIL {label}\n         {detail}", flush=True)

    app = QApplication.instance() or QApplication(sys.argv)
    # The real startup path asks the user to pick a model in a dialog; supply the
    # choice directly so the window comes up exactly as it would after that pick.
    win = gui.AssistantWindow(config.MODEL_NAME, False, "high")
    win.move(-4000, -4000)          # real window, off the user's screen
    win.show()
    app.processEvents()

    def pump(ms):
        t = QElapsedTimer(); t.start()
        while t.elapsed() < ms:
            app.processEvents()
            QTest.qWait(20)

    def wait_until(pred, timeout_ms, label=""):
        t = QElapsedTimer(); t.start()
        while t.elapsed() < timeout_ms:
            app.processEvents()
            if pred():
                return True
            QTest.qWait(50)
        return False

    def _type(widget, text):
        """Send one real QKeyEvent per character.

        NOT QTest.keyClicks: it maps each character to a Qt::Key code, has no
        mapping for Cyrillic, and ABORTS the process — verified on a bare
        QLineEdit with no event filter, so it is a harness limit, not an app
        bug. These are genuine key events delivered through the application's
        event system, so the installed eventFilter and the widget's own
        keyPressEvent both see them exactly as they see a person's typing.
        """
        for ch in text:
            for typ in (QEvent.KeyPress, QEvent.KeyRelease):
                ev = QKeyEvent(typ, 0, Qt.NoModifier, ch)
                QApplication.sendEvent(widget, ev)
            app.processEvents()

    def type_and_send(text):
        """Exactly what a person does: click the field, type, press Enter."""
        win.input.setFocus()
        QTest.mouseClick(win.input, Qt.LeftButton)
        _type(win.input, text)
        app.processEvents()
        typed = win.input.text()
        QTest.keyClick(win.input, Qt.Key_Return)   # ASCII key: keyClick is safe
        app.processEvents()
        return typed

    def chat_text():
        for attr in ("chat", "chat_view", "chat_box", "transcript"):
            w = getattr(win, attr, None)
            if w is not None:
                for meth in ("toPlainText", "toHtml"):
                    if hasattr(w, meth):
                        try:
                            return getattr(w, meth)()
                        except Exception:
                            pass
        return ""

    def newest_render():
        # Use the app's own predicate, not a hand-rolled "_INTERMEDIATE_" check:
        # it also knows _QA_cutout_ and _firered_tile_, and a harness that only
        # knew the one prefix latched onto a QA cutout and reported a false
        # failure for a render that was actually correct.
        import image as _img
        out = glob.glob(os.path.join(str(config.OUTPUT_DIR), "*.png"))
        out = [p for p in out if not _img.is_intermediate_artifact(p)]
        return max(out, key=os.path.getmtime) if out else None

    # ---------------------------------------------------------------- typing
    print("\n1. THE KEYSTROKES REACH THE FIELD AND THE FIELD CLEARS ON ENTER")
    before = newest_render()
    typed = type_and_send("нарисуй рыжего кота в скафандре на Марсе")
    check("the characters actually landed in the input",
          typed == "нарисуй рыжего кота в скафандре на Марсе", repr(typed))
    check("Enter cleared the field", win.input.text() == "", repr(win.input.text()))
    check("and the turn started (the UI went busy)",
          wait_until(lambda: win._busy(), 8000), "the assistant never went busy")

    # -------------------------------------------------- typing while busy
    print("\n2. TYPING WHILE BUSY QUEUES INSTEAD OF DROPPING OR INTERLEAVING")
    q_before = len(win._task_queue)
    t2 = type_and_send("сколько будет 17 умножить на 23")
    t3 = type_and_send("а столица Австралии какая")
    check("the second message was typed", t2.startswith("сколько"), t2)
    check("neither was silently dropped",
          len(win._task_queue) >= q_before + 2,
          f"queue went {q_before} -> {len(win._task_queue)}")
    check("they are PENDING, not running alongside the first",
          sum(1 for it in win._task_queue if it.get("status") == "pending") >= 2,
          [it.get("status") for it in win._task_queue])
    check("the queue is visible to the user",
          "17" in str(chat_text()) or any("17" in str(it.get("text", ""))
                                          for it in win._task_queue),
          "the queued message is not shown anywhere")

    # --------------------------------------------------------- the drawing
    print("\n3. THE PICTURE IS ACTUALLY DRAWN (live ComfyUI)")
    drew = wait_until(lambda: newest_render() not in (None, before), 600000)
    img = newest_render()
    check("a new image file appeared", drew and img != before, f"{before} -> {img}")
    if img:
        print(f"     rendered: {os.path.basename(img)}", flush=True)
        check("it was drawn by Ideogram, not the old model fallback",
              "ideogram" in os.path.basename(img).lower(), os.path.basename(img))
        check("and it carries a layout, so it can be edited by its boxes",
              os.path.exists(img + ".layout.json"), img + ".layout.json")

    check("the turn finished", wait_until(lambda: not win._busy(), 300000),
          "still busy long after the render")

    # -------------------------------------------------- the queue drains
    print("\n4. THE QUEUED MESSAGES RUN AFTERWARDS, IN ORDER, WITHOUT INTERFERENCE")
    drained = wait_until(
        lambda: not any(it.get("status") == "pending" for it in win._task_queue)
                and not win._busy(), 600000)
    check("the queue drained on its own", drained,
          [ (it.get("status"), str(it.get("text"))[:24]) for it in win._task_queue ])
    txt = str(chat_text())
    check("the arithmetic question was answered (391)", "391" in txt,
          txt[-400:])
    check("and the capital question too", "анберр" in txt or "anberr" in txt.lower(),
          txt[-400:])
    check("the picture was not redrawn by the queued text questions",
          newest_render() == img, f"{img} -> {newest_render()}")

    # ------------------------------------------------------------ the edit
    print("\n5. EDITING THROUGH THE INTERFACE USES THE BOXES")
    pre_edit = newest_render()
    pre_layout = {}
    if pre_edit and os.path.exists(pre_edit + ".layout.json"):
        pre_layout = json.load(open(pre_edit + ".layout.json", encoding="utf-8"))
    type_and_send("сделай кота чёрным")
    check("the edit turn started", wait_until(lambda: win._busy(), 15000),
          "the edit never started")
    edited = wait_until(lambda: newest_render() not in (None, pre_edit), 600000)
    new_img = newest_render()
    check("an edited image came back", edited and new_img != pre_edit,
          f"{pre_edit} -> {new_img}")
    if new_img and os.path.exists(new_img + ".layout.json"):
        post = json.load(open(new_img + ".layout.json", encoding="utf-8"))
        check("the edit went through the LAYOUT (a box changed)",
              post.get("layout") != pre_layout.get("layout"),
              "the boxes came back identical")
        check("and the seed was reused, so it is an edit not a re-roll",
              post.get("seed") == pre_layout.get("seed"),
              f"{pre_layout.get('seed')} -> {post.get('seed')}")
    else:
        check("the edited image carries a layout too",
              new_img and os.path.exists(str(new_img) + ".layout.json"),
              f"no sidecar for {new_img}")

    check("the app is still alive and idle at the end",
          wait_until(lambda: not win._busy(), 300000) and win.isVisible())

    print(f"\n{_n['ok']}/{_n['ok'] + _n['bad']} checks passed", flush=True)

    win.close()
    app.processEvents()
    gui.redirect_settings(None)
    sys.exit(1 if _n["bad"] else 0)
''')


def main():
    tmp = tempfile.mkdtemp(prefix="gui_journey_")
    src = CHILD % {"root": ROOT, "tmp": tmp}
    path = os.path.join(tmp, "child.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("QT_QPA_PLATFORM", None)
    print("Driving the real window with real keystrokes; this draws for real "
          "and takes several minutes.\n")
    p = subprocess.run([sys.executable, path], cwd=ROOT, env=env)
    if p.returncode < 0 or p.returncode == 3221225477:
        print(f"\nNATIVE CRASH: the child died with {p.returncode} — a Qt slot aborted "
              "the interpreter. That is the class of bug offscreen tests cannot see.")
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
