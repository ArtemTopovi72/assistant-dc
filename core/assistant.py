#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import logging
import os
import shutil
import threading
import time

import cv2
import numpy as np

from config import SAMPLE_RATE, OUTPUT_DIR, MEMORY_DIR
from models import Models, Context, AgentState
from prompts import SYSTEM_PROMPT_PERSONALITY
from utils import load_transcription_cache, ensure_required_files
from audio import (
    record_audio_until_enter,
    MicRecorder,
    transcribe_audio_array,
    transcribe_audio_file,
)
from graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("assistant")

# Under pythonw there is no console and no persistent record of "assistant.*"
# log traffic once gui.py's _install_log_handler() sets this logger's
# propagate=False (to stop a crash-prone root StreamHandler some library adds
# via its own basicConfig() call) — the GUI's log panel is in-memory only, so
# a restart wipes every trace of what happened right before it. A small
# rotating file, attached directly to THIS logger rather than root, keeps
# receiving every "assistant.*" record regardless of that propagate flag.
# encoding="utf-8", errors="replace" for the same reason the crash fix above
# exists: a non-ASCII log char must never be able to take the process down.
from logging.handlers import RotatingFileHandler
_file_handler = RotatingFileHandler(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assistant_app.log"),
    maxBytes=2_000_000, backupCount=3, encoding="utf-8", errors="replace")
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(_file_handler)


_CAM_LABELS = {"idle": "READY", "recording": "● REC", "processing": "… THINKING"}
_CAM_COLORS = {"idle": (0, 200, 0), "recording": (0, 0, 235), "processing": (0, 200, 235)}
_CAM_HINTS = {
    "idle": "SPACE: start talking    ESC: exit",
    "recording": "SPACE: send    ESC: cancel",
    "processing": "please wait...",
}


# Runtime construction moved to app_runtime.py so that gui.py can build a
# runtime without importing this entry-point module (that was the last
# import cycle). Re-exported here: assistant.main() uses them, and so may
# anything that already imports them from this module.
from app_runtime import make_base_state, build_runtime


def _draw_camera_overlay(display, state: str, status: str) -> None:
    """Draw a translucent status bar + state chip onto the live frame."""
    h, w = display.shape[:2]
    bar = display.copy()
    cv2.rectangle(bar, (0, h - 44), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.55, display, 0.45, 0, display)
    cv2.putText(display, _CAM_LABELS.get(state, ""), (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _CAM_COLORS.get(state, (255, 255, 255)), 2, cv2.LINE_AA)
    cv2.putText(display, _CAM_HINTS.get(state, ""), (185, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    if status:
        cv2.putText(display, status, (15, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2, cv2.LINE_AA)
    if state == "recording":
        cv2.rectangle(display, (3, 3), (w - 3, h - 3), (0, 0, 235), 3)


def camera_mode(ctx: Context, base_state: AgentState, assistant_graph) -> None:
    """Live camera with non-blocking capture: SPACE to talk, the stream stays
    live while the question is transcribed/answered on a background thread.
    """
    logger.info("Opening camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logger.error("Camera not available")
        print("Camera not available.")
        return

    win = "Assistant Camera"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    state = "idle"            # idle -> recording -> processing -> idle
    captured_frame = None
    recorder = None
    worker = None
    status = ""
    status_until = 0.0

    def set_status(msg: str, secs: float = 3.0) -> None:
        nonlocal status, status_until
        status, status_until = msg, time.time() + secs

    def process(frame_bytes: bytes, audio: np.ndarray) -> None:
        try:
            question = transcribe_audio_array(ctx, audio)
            if not question:
                set_status("Didn't catch that")
                return
            logger.info("Camera question: %s", question)
            ctx.remember("user", question, {"source": "camera"})
            st = copy.deepcopy(base_state)
            st["user_input"] = question
            st["image_data"] = frame_bytes
            st["session_memory_text"] = ctx.memory_text()
            final = assistant_graph.invoke(st)
            from graph import compact_history_if_needed
            base_state["messages"] = compact_history_if_needed(
                ctx, copy.deepcopy(final.get("messages", base_state["messages"])))
        except Exception as exc:
            logger.error("Camera processing error: %s", exc)
            set_status("Error during processing")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if state == "processing" and worker is not None and not worker.is_alive():
                worker = None
                state = "idle"

            display = frame.copy()
            _draw_camera_overlay(display, state, status if time.time() < status_until else "")
            cv2.imshow(win, display)
            key = cv2.waitKey(20) & 0xFF

            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break

            if key == 27:  # ESC
                if state == "recording":
                    recorder.stop()
                    recorder, captured_frame, state = None, None, "idle"
                    set_status("Cancelled")
                elif state == "idle":
                    break
                # processing: can't cancel mid-inference, ignore

            elif key == 32:  # SPACE
                if state == "idle":
                    captured_frame = frame.copy()
                    recorder = MicRecorder(fs=SAMPLE_RATE, gain_db=20.0)
                    recorder.start()
                    state = "recording"
                elif state == "recording":
                    audio = recorder.stop()
                    recorder = None
                    if len(audio) < SAMPLE_RATE * 0.5:
                        captured_frame, state = None, "idle"
                        set_status("Too short")
                    else:
                        ok, enc = cv2.imencode(".jpg", captured_frame)
                        captured_frame = None
                        if not ok:
                            state = "idle"
                            set_status("Capture failed")
                        else:
                            worker = threading.Thread(
                                target=process, args=(enc.tobytes(), audio), daemon=True
                            )
                            worker.start()
                            state = "processing"
    finally:
        if recorder is not None:
            recorder.stop()
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    from config import USE_GUI, LM_STUDIO_BASE, MODEL_NAME
    # Install crash instrumentation FIRST: enables faulthandler (native C-stack on a
    # real fault) and a custom sys.excepthook that logs the traceback of any
    # exception escaping a Qt slot — which otherwise makes PyQt5 abort() the process
    # with exit code 0xC0000409 and no traceback. Must run before Qt is imported.
    import crash_diag
    # Under pythonw every console child (lms, ffmpeg) flashes its own window
    # over the GUI at each model swap; hide them all in one place.
    try:
        import no_console_windows
        no_console_windows.install()
    except Exception:
        pass
    crash_diag.install()
    crash_diag.log_stage("assistant.main() start")
    ensure_required_files()

    if USE_GUI:
        from gui import run_gui
        run_gui()
        return

    # Terminal mode: pick the LLM + thinking mode, then load models and loop.
    from model_selector import choose_model_interactive
    model_name, no_think = choose_model_interactive(LM_STUDIO_BASE, MODEL_NAME)
    ctx, base_state, assistant_graph = build_runtime(model_name, no_think)

    print("=" * 60)
    print("ASSISTANT DC (LangGraph + tools + vision + TTS)")
    print(f"   Model: {ctx.model_name}  (thinking {'OFF' if ctx.no_think else 'ON'})")
    print("   Commands: c (camera), v (custom voice), m (switch model),")
    print("             Enter (voice), q (exit), or type any text")
    print("=" * 60)

    try:
        while True:
            raw = input("\nCommand (c / v / m / Enter / q / text): ").strip()
            cmd_lower = raw.lower() if raw else ""

            if cmd_lower in {"q", "exit"}:
                print("Goodbye!")
                break

            if cmd_lower == "c":
                camera_mode(ctx, base_state, assistant_graph)
                continue

            if cmd_lower == "v":
                path = input("Reference voice file (any audio format; blank = default): ").strip().strip('"')
                if not path:
                    ctx.custom_ref_wav = None
                    print("→ Reverted to the default voice.")
                elif os.path.exists(path):
                    ctx.custom_ref_wav = path
                    print(f"→ Custom voice set: {os.path.basename(path)}")
                else:
                    print("File not found; keeping the current voice.")
                continue

            if cmd_lower == "m":
                from config import LM_STUDIO_BASE
                from model_selector import choose_model_interactive
                from lmstudio import ensure_exclusive
                new_model, ctx.no_think = choose_model_interactive(LM_STUDIO_BASE, ctx.model_name)
                if new_model != ctx.model_name:
                    ctx.model_name = new_model
                    print(f"Loading '{new_model}' (unloading the previous model)…")
                    ok, msg = ensure_exclusive(LM_STUDIO_BASE, new_model)
                    print(f"  {'✓ ' + new_model if ok else '⚠ ' + msg}")
                continue

            if cmd_lower == "":
                logger.info("Voice input...")
                audio = record_audio_until_enter()
                if len(audio) < SAMPLE_RATE * 0.5:
                    logger.warning("Recording too short")
                    continue
                text = transcribe_audio_array(ctx, audio)
                if not text:
                    continue
            else:
                text = raw
                logger.info("Text input: %s", text)

            ctx.remember("user", text, {"source": "voice" if not raw else "keyboard"})
            state = copy.deepcopy(base_state)
            state["user_input"] = text
            state["image_data"] = None
            state["session_memory_text"] = ctx.memory_text()
            final_state = assistant_graph.invoke(state)
            from graph import compact_history_if_needed
            base_state["messages"] = compact_history_if_needed(
                ctx, copy.deepcopy(final_state.get("messages", base_state["messages"])))
    finally:
        ctx.save_memory(ctx.active_memory_dir)
        print(f"Memory saved → '{ctx.active_memory_dir.name}'.")


if __name__ == "__main__":
    main()
