"""Plan a layout, DRAW it with Ideogram, DRAG a box, draw again — for real.

The claim under test is the one the whole storyboard rests on: the picture is a
layout you can still adjust. That means three things must hold together, and only
an end-to-end run can show it:

  1. it renders through IDEOGRAM (not the old model fallback), from the boxes;
  2. the box coordinates are REMEMBERED — on the canvas, in the tab's layout, and
     in the caption that is submitted;
  3. dragging a box with the mouse MOVES it, and the next draw uses the moved
     coordinates and the SAME seed (a new seed would be a different picture, not
     an edit).

Nothing is stubbed except the wall clock: the live LLM plans the scene and the
live ComfyUI renders it. Two renders, so it takes a few minutes.

The drag is real mouse events (QMouseEvent through the application), not a call to
mouseMoveEvent — a handler that works when called directly and never receives an
event is the exact failure this suite exists to catch.

Runs in a SUBPROCESS on the NATIVE Qt platform: offscreen has hidden a shipped
0xC0000409 abort before, and a PyQt abort kills the interpreter, so the child's
exit code is part of the assertion.

SAFETY: redirect_settings before the window is built, so the live QSettings scope
(which holds the real BotFather token) is never written.

Lives in bench/, not tests/: it plans with the LIVE model and renders on the
LIVE card, so running it from the suite meant run_all quietly started GPU work --
once, measured, CONCURRENTLY with a bench render that was already using the card.
tests/ stays offline and deterministic; live end-to-end runs belong here.

Run: venv/Scripts/python.exe bench/storyboard_drag.py
"""
import os, sys, subprocess, tempfile, textwrap, urllib.request

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
    print(f"CANNOT RUN: ComfyUI is not answering at {config.COMFY_URL}. This suite "
          "draws for real; a stubbed render would not test that Ideogram is used.")
    sys.exit(2)
if not _up(config.LM_STUDIO_URL.replace("/chat/completions", "/models")):
    print("CANNOT RUN: LM Studio is not answering; the layout is planned by the model.")
    sys.exit(2)

CHILD = textwrap.dedent(r'''
    import os, sys, json, copy
    os.environ["USE_GUI"] = "0"
    os.environ["GUI_REPORT_HTML"] = "0"
    sys.path.insert(0, %(root)r)
    import logging
    logging.basicConfig(level=logging.WARNING,
                        format="%%(levelname)s %%(name)s: %%(message)s")

    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt, QPoint, QElapsedTimer, QEvent
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtTest import QTest
    import crash_diag; crash_diag.install()
    import gui, config, ideogram
    # StoryboardTab moved to gui_storyboard_tab.py, and it resolves
    # IdeogramWorker as a global of THAT module. Patching gui.IdeogramWorker
    # alone would no longer intercept the render — the spy would go silently
    # dead and this suite would drive a real Ideogram job.
    import gui_storyboard_tab as _SB

    TMP = %(tmp)r
    gui.redirect_settings(TMP)

    _n = {"ok": 0, "bad": 0}
    def check(label, cond, detail=""):
        if cond:
            _n["ok"] += 1; print(f"  ok   {label}", flush=True)
        else:
            _n["bad"] += 1; print(f"  FAIL {label}\n         {detail}", flush=True)

    app = QApplication.instance() or QApplication(sys.argv)
    win = gui.AssistantWindow(config.MODEL_NAME, False, "high")
    win.move(-4000, -4000)
    win.show()
    app.processEvents()

    def wait_until(pred, timeout_ms):
        t = QElapsedTimer(); t.start()
        while t.elapsed() < timeout_ms:
            app.processEvents()
            if pred():
                return True
            QTest.qWait(50)
        return False

    tab = win.storyboard_tab
    canvas = tab.canvas

    # host.ctx does not exist until the model-loader thread finishes (_on_runtime_ready);
    # the tab reads it through a property, so planning before that just prints "No model
    # loaded yet" and produces an empty layout. Wait for the real thing.
    print("waiting for the model to finish loading…", flush=True)
    if not wait_until(lambda: getattr(win, "ctx", None) is not None, 600000):
        print("CANNOT RUN: the model never finished loading; the layout is planned by it.")
        sys.exit(2)
    print(f"model ready: {getattr(win.ctx, 'model_name', '?')}", flush=True)

    # Capture what is actually submitted to the renderer, without changing it.
    submitted = []
    _real_worker = _SB.IdeogramWorker
    class SpyWorker(_real_worker):
        def __init__(self, ctx, caption, w, h, seed, *a, **kw):
            submitted.append({"caption": copy.deepcopy(caption), "w": w, "h": h,
                              "seed": seed})
            super().__init__(ctx, caption, w, h, seed, *a, **kw)
    _SB.IdeogramWorker = SpyWorker
    gui.IdeogramWorker = SpyWorker

    # ───────────────────────────────────────────────────────────── 1. plan
    print("\n1. THE MODEL PLANS A LAYOUT OF BOXES")
    tab.prompt_in.setText("a red tractor on the left, a wooden barn on the right, "
                          "a sign above the barn saying FARM")
    tab.plan_btn.click()
    planned = wait_until(lambda: len(tab.layout_data.get("elements") or []) > 0, 240000)
    els = tab.layout_data.get("elements") or []
    check("the model produced boxes", planned and len(els) >= 2, f"{len(els)} elements")
    # `all()` over an EMPTY list is True — without the len() guard this check passed
    # while the planner had produced nothing at all, which is exactly the state it is
    # supposed to catch.
    check("every box has coordinates inside the frame",
          len(els) >= 2 and all(0.0 <= e["x"] <= 1.0 and 0.0 <= e["y"] <= 1.0
                                and 0.0 < e["w"] <= 1.0 and 0.0 < e["h"] <= 1.0
                                for e in els),
          [(e["x"], e["y"], e["w"], e["h"]) for e in els])
    if not els:
        print("\nABORT: no layout to draw, drag or verify — every check below "
              "would be vacuous.", flush=True)
        sys.exit(1)
    check("the canvas shows the SAME box objects the tab holds — one layout, "
          "not a copy that silently diverges",
          canvas.elements is tab.layout_data["elements"],
          f"{id(canvas.elements)} vs {id(tab.layout_data['elements'])}")

    # ───────────────────────────────────────────────────────────── 2. draw
    print("\n2. IT RENDERS THROUGH IDEOGRAM, FROM THOSE BOXES")
    tab.draw_btn.click()
    drew = wait_until(lambda: tab._last_image and os.path.exists(str(tab._last_image)),
                      900000)
    check("a picture came back", drew, f"_last_image={tab._last_image!r}")
    check("it went through the Ideogram path",
          len(submitted) == 1 and "compositional_deconstruction" in submitted[0]["caption"],
          list(submitted[0]["caption"].keys()) if submitted else "nothing submitted")
    if not submitted:
        print("\nABORT: nothing reached the renderer — the drag checks below would "
              "not be testing the drawing path.", flush=True)
        sys.exit(1)
    cap_els = (submitted[0]["caption"].get("compositional_deconstruction") or {}
               ).get("elements") or []
    check("the submitted caption carries a bounding box per element",
          len(cap_els) >= 2 and all("bbox" in e for e in cap_els),
          cap_els[:2])
    check("the boxes are in Ideogram's 0-1000 grid",
          all(all(0 <= v <= 1000 for v in (e["bbox"] or [])) for e in cap_els),
          [e.get("bbox") for e in cap_els])
    if tab._last_image:
        print(f"     rendered: {os.path.basename(str(tab._last_image))}", flush=True)
    check("the render became the canvas backdrop, so the next pass is a repaint "
          "of a composition you can still see",
          canvas.backdrop is not None and not canvas.backdrop.isNull())
    seed_1 = submitted[0]["seed"]
    check("the draw used a real seed", seed_1 > 0, seed_1)

    # ────────────────────────────────────────────────────────────── 3. drag
    print("\n3. DRAGGING A BOX WITH THE MOUSE MOVES IT")
    canvas.resize(700, 700)
    app.processEvents()
    idx = 0
    before = dict(tab.layout_data["elements"][idx])
    r = canvas._box_rect(tab.layout_data["elements"][idx])
    start = r.center()
    # Drag toward the frame centre-ish, far enough to be unambiguous.
    f = canvas.frame_rect()
    end = QPoint(min(f.right() - r.width(), start.x() + int(f.width() * 0.25)),
                 min(f.bottom() - r.height(), start.y() + int(f.height() * 0.2)))

    def send(kind, pos, buttons):
        ev = QMouseEvent(kind, pos, Qt.LeftButton, buttons, Qt.NoModifier)
        QApplication.sendEvent(canvas, ev)
        app.processEvents()

    moved_signals = []
    canvas.changed.connect(lambda: moved_signals.append(1))
    send(QEvent.MouseButtonPress, start, Qt.LeftButton)
    check("the press selected the box under the cursor", canvas.sel == idx,
          f"sel={canvas.sel}")
    steps = 8
    for i in range(1, steps + 1):
        send(QEvent.MouseMove,
             QPoint(start.x() + (end.x() - start.x()) * i // steps,
                    start.y() + (end.y() - start.y()) * i // steps),
             Qt.LeftButton)
    send(QEvent.MouseButtonRelease, end, Qt.NoButton)

    after = dict(tab.layout_data["elements"][idx])
    check("the drag emitted live change signals", len(moved_signals) >= 4,
          f"{len(moved_signals)} signals")
    check("the box's stored coordinates changed",
          (round(after["x"], 4), round(after["y"], 4))
          != (round(before["x"], 4), round(before["y"], 4)),
          f"{before['x']:.3f},{before['y']:.3f} -> {after['x']:.3f},{after['y']:.3f}")
    # Compare against the ACTUAL mouse delta, not the intended one: `end` is
    # clamped so the box stays on screen, and for a tall box that clamp can invert
    # the direction. Asserting "it went down" when the mouse went UP tested my
    # arithmetic, not the widget.
    dx, dy = end.x() - start.x(), end.y() - start.y()
    moved_x, moved_y = after["x"] - before["x"], after["y"] - before["y"]
    check("it moved in the direction the mouse actually went",
          (dx == 0 or (moved_x > 0) == (dx > 0))
          and (dy == 0 or (moved_y > 0) == (dy > 0)),
          f"mouse ({dx:+d},{dy:+d}) px -> box ({moved_x:+.3f},{moved_y:+.3f})")
    check("and it moved by the same fraction of the frame the mouse crossed",
          abs(moved_x - dx / f.width()) < 0.02
          and abs(moved_y - dy / f.height()) < 0.02,
          f"mouse ({dx / f.width():+.3f},{dy / f.height():+.3f}) "
          f"vs box ({moved_x:+.3f},{moved_y:+.3f})")
    check("dragging did not resize it",
          abs(after["w"] - before["w"]) < 1e-6 and abs(after["h"] - before["h"]) < 1e-6,
          f"{before['w']:.4f}x{before['h']:.4f} -> {after['w']:.4f}x{after['h']:.4f}")
    check("the box stayed inside the frame",
          0.0 <= after["x"] <= 1.0 - after["w"] + 1e-9
          and 0.0 <= after["y"] <= 1.0 - after["h"] + 1e-9, after)
    check("the OTHER boxes were not disturbed",
          all(tab.layout_data["elements"][j]["x"] == els[j]["x"]
              for j in range(len(els)) if j != idx),
          [(e["x"], e["y"]) for e in tab.layout_data["elements"]])
    check("the release ended the drag, so the box does not follow the cursor after",
          canvas._mode is None)

    # A box dragged past the edge must clamp, not escape the picture.
    send(QEvent.MouseButtonPress, canvas._box_rect(
        tab.layout_data["elements"][idx]).center(), Qt.LeftButton)
    send(QEvent.MouseMove, QPoint(f.right() + 400, f.bottom() + 400), Qt.LeftButton)
    send(QEvent.MouseButtonRelease, QPoint(f.right() + 400, f.bottom() + 400), Qt.NoButton)
    edge = tab.layout_data["elements"][idx]
    check("dragging past the edge clamps inside the frame",
          edge["x"] <= 1.0 - edge["w"] + 1e-9 and edge["y"] <= 1.0 - edge["h"] + 1e-9,
          edge)

    # ─────────────────────────────────────────── 4. the next draw uses the move
    print("\n4. THE NEXT DRAW USES THE MOVED BOX AND THE SAME SEED")
    dragged = dict(tab.layout_data["elements"][idx])
    prev_image = str(tab._last_image)
    tab.draw_btn.click()
    drew2 = wait_until(lambda: len(submitted) >= 2, 60000)
    check("a second render was submitted", drew2, f"{len(submitted)} submissions")
    if drew2:
        cap2 = (submitted[1]["caption"].get("compositional_deconstruction") or {}
                ).get("elements") or []
        bb1 = cap_els[idx].get("bbox")
        bb2 = cap2[idx].get("bbox") if idx < len(cap2) else None
        check("the submitted bounding box changed with the drag", bb1 != bb2,
              f"{bb1} -> {bb2}")
        expect_x = int(round(dragged["x"] * 1000))
        expect_y = int(round(dragged["y"] * 1000))
        # Ideogram bbox is [ymin, xmin, ymax, xmax] on a 0-1000 grid.
        check("and it matches where the box was actually dropped",
              bb2 is not None and abs(bb2[1] - expect_x) <= 2
              and abs(bb2[0] - expect_y) <= 2,
              f"dropped at x={expect_x} y={expect_y}, submitted {bb2}")
        check("the seed was REUSED — this is an edit of the composition, "
              "not a fresh picture",
              submitted[1]["seed"] == seed_1,
              f"{seed_1} -> {submitted[1]['seed']}")
        check("Repaint is what asks for a new seed, and it is a different button",
              tab.repaint_btn is not tab.draw_btn)
        finished2 = wait_until(
            lambda: tab._last_image and str(tab._last_image) != prev_image, 900000)
        check("the second picture was actually rendered", finished2,
              f"still {tab._last_image!r}")

    check("the app is alive and idle at the end",
          wait_until(lambda: not (tab.worker and tab.worker.isRunning()), 300000)
          and win.isVisible())

    print(f"\n{_n['ok']}/{_n['ok'] + _n['bad']} checks passed", flush=True)
    _SB.IdeogramWorker = _real_worker
    gui.IdeogramWorker = _real_worker
    win.close()
    app.processEvents()
    gui.redirect_settings(None)
    sys.exit(1 if _n["bad"] else 0)
''')


def main():
    tmp = tempfile.mkdtemp(prefix="storyboard_drag_")
    src = CHILD % {"root": ROOT, "tmp": tmp}
    path = os.path.join(tmp, "child.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("QT_QPA_PLATFORM", None)
    print("Planning, drawing twice and dragging on the real window — several minutes.\n")
    p = subprocess.run([sys.executable, path], cwd=ROOT, env=env)
    if p.returncode < 0 or p.returncode == 3221225477:
        print(f"\nNATIVE CRASH: the child died with {p.returncode} — a Qt slot aborted "
              "the interpreter.")
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
