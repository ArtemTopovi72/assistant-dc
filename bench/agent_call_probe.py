"""Which tools the REAL agent calls, with which arguments, for a request -- on live
LM Studio, with every tool recorded and answered "done" instead of run: no render,
no web, no card.

    venv/Scripts/python.exe bench/agent_call_probe.py [--picture <path>] "<request>" ["<request>" ...]
"""
import json
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != ROOT / "bench"]   # bench/knowledge.py shadows the package
for d in ("", "agent", "imaging", "media", "bot", "core", "voice", "knowledge", "research", "services", "sandbox", "personalities"):
    sys.path.insert(0, str(ROOT / d))
os.chdir(ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")

import graph
from app_runtime import make_base_state
from models import Context


def run(request: str, picture: str = "") -> None:
    calls = []

    def record(ctx, state, name, args):
        calls.append((name, args))
        return f"[{name} done]"

    ctx = Context(models=None, transcription_cache={}, cache_file=None,
                  asr_lock=threading.Lock(), tts_lock=threading.Lock())
    ctx.no_think, ctx.tts_disabled, ctx.gui_mode = True, True, True
    ctx.run_tool = record
    if picture:
        ctx.last_image_path = picture
    state = make_base_state()
    state["user_input"] = request
    final = graph.build_graph(ctx).invoke(state)
    print("=" * 100)
    print("REQUEST:", request)
    for name, args in calls:
        print(f"  CALL {name} {json.dumps(args, ensure_ascii=False)[:600]}")
    if not calls:
        print("  (no tool call)")
    print("  REPLY:", (final.get("final_answer") or "")[:300].replace("\n", " | "))


if __name__ == "__main__":
    args = sys.argv[1:]
    pic = ""
    if args[:1] == ["--picture"]:
        pic, args = args[1], args[2:]
    for req in args:
        run(req, pic)
