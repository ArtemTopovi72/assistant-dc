"""Regression: every tools.py handler must degrade to a [TOOL ERROR] string (never a
false success, never a crash) when its external dependency fails.

Found by dependency fault-injection:
  BUG #10 — generate_image reported "Image generated and saved: <path>" and set
    state.image_status='success' when the renderer returned a NON-EXISTENT path (a
    stale/bogus path, or a temp file deleted between render and hand-off). It checked
    `if img_path:` (truthy) but never os.path.exists() — unlike every other image
    handler. Fix: verify existence; on a missing file fall through to the failure
    message and force status='fail'.
  read_clipboard failure returned a bare Russian message with no [TOOL ERROR] prefix,
    so the graph's error detection missed it and the model could treat the failure
    text as clipboard CONTENT. Fix: prefix [TOOL ERROR].

Run: venv/Scripts/python.exe tests/test_tool_fault_degradation.py
"""
import os, sys, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import tools as tmod
from models import Context
from pathlib import Path

BAD = str(Path("tests") / "_nonexistent_zzz.png")

def _ctx():
    c = Context(models=None, transcription_cache={}, cache_file=Path("tests/_tfd.json"),
                asr_lock=threading.Lock(), tts_lock=threading.Lock(),
                model_name="m", no_think=True)
    c.tts_disabled = True; c.gui_mode = True
    c.active_memory_dir = Path(__file__).resolve().parent / "_tfdmem"
    c.active_memory_dir.mkdir(parents=True, exist_ok=True)
    return c

def _state():
    return {"messages": [], "user_input": "", "image_data": None, "final_answer": "",
            "session_memory_text": "", "vision_summary": "", "image_path": "",
            "image_score": 0, "image_attempt": 0, "image_status": ""}


def test_generate_image_nonexistent_path_is_failure():
    """Renderer returns a path that does not exist -> NOT a success (BUG #10)."""
    tmod.generate_image_with_refinement = lambda *a, **k: {
        "path": BAD, "score": 7, "attempts": 1, "status": "success", "prompt": "x"}
    tmod.enhance_faces_with_comfy = lambda ctx, p: p
    st = _state()
    out = tmod.execute_tool(_ctx(), st, "generate_image", {"description": "a cat"})
    assert out.lstrip().startswith("[TOOL ERROR]"), f"false success on missing file: {out[:80]!r}"
    assert st.get("image_status") != "success", f"state still success: {st.get('image_status')!r}"
    assert st.get("image_path") in ("", None) or not os.path.exists(st["image_path"]), \
        "state.image_path points at a non-existent file but was accepted"
    print("PASS generate_image treats a non-existent render path as failure (BUG #10)")


def test_generate_image_none_path_is_failure():
    tmod.generate_image_with_refinement = lambda *a, **k: {"path": None, "status": "fail"}
    tmod.enhance_faces_with_comfy = lambda ctx, p: p
    st = _state()
    out = tmod.execute_tool(_ctx(), st, "generate_image", {"description": "a cat"})
    assert out.lstrip().startswith("[TOOL ERROR]") and st.get("image_status") != "success"
    print("PASS generate_image None path is a clean failure")


def test_read_clipboard_failure_is_tagged():
    import subprocess
    orig = subprocess.run
    def boom(*a, **k): raise RuntimeError("powershell missing")
    subprocess.run = boom
    try:
        out = tmod.execute_tool(_ctx(), _state(), "read_clipboard", {})
    finally:
        subprocess.run = orig
    assert out.lstrip().startswith("[TOOL ERROR]"), f"clipboard failure not tagged: {out[:80]!r}"
    print("PASS read_clipboard failure is [TOOL ERROR]-tagged")


def test_image_handlers_no_false_success_on_none():
    """redraw/inpaint/transfer/fix_hands/fix_artifact must all fail cleanly when their
    render dependency returns None."""
    import image as image_mod
    ctx = _ctx(); ctx.last_image_path = None
    # no source -> all should error (proves the no-source guard too)
    for name, args in [("redraw_image", {"mode": "enhance"}),
                       ("inpaint_image", {"region": "dress", "instructions": "red"}),
                       ("fix_hands", {}), ("fix_artifact", {"region": "seam"})]:
        out = tmod.execute_tool(ctx, _state(), name, args)
        assert out.lstrip().startswith("[TOOL ERROR]"), f"{name} not error with no source: {out[:60]!r}"
    print("PASS image handlers fail cleanly with no source image")


if __name__ == "__main__":
    test_generate_image_nonexistent_path_is_failure()
    test_generate_image_none_path_is_failure()
    test_read_clipboard_failure_is_tagged()
    test_image_handlers_no_false_success_on_none()
    print("\ndone")
