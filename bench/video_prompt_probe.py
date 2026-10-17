"""What the video model is actually SENT for a request -- the real agent, the real
plan/validate/prepare_prompt chain and real LM Studio, with ComfyUI replaced by a
recorder: no render, no card. Each H3 job's prompt is printed as it would go out.

    venv/Scripts/python.exe bench/video_prompt_probe.py <picture> "<request>" ["<request>" ...]

The live app shares LM Studio; nothing here touches ComfyUI or the bot's chats.
"""
import os
import shutil
import sys
import threading
import types
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

import video as V
import graph
from app_runtime import make_base_state
from models import Context

SENT = []
_STUB = ROOT / "runtime" / "tmp" / "probe_stub.mp4"


def _stub_clip() -> str:
    if not _STUB.exists():
        import subprocess
        _STUB.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "color=c=gray:s=320x320:d=5", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                        "-t", "5", "-shortest", str(_STUB)], check=True)
    out = ROOT / "runtime" / "tmp" / f"probe_{len(SENT)}.mp4"
    shutil.copy(_STUB, out)
    return str(out)


def _record(ctx, wf, **kw):
    texts = [n["inputs"][k] for n in wf.values() for k in ("text", "prompt", "description")
             if isinstance(n, dict) and isinstance(n.get("inputs", {}).get(k), str)
             and len(n["inputs"][k]) > 80]
    SENT.append({"label": kw.get("label", ""), "prompt": max(texts, key=len) if texts else ""})
    return _stub_clip()


class _Engine(types.ModuleType):
    """The real video module, its ComfyUI calls recorded instead of sent."""
    def __getattr__(self, name):
        return getattr(V, name)


def _engine():
    eng = _Engine("video_probe")
    eng.comfy_client = types.SimpleNamespace(
        _submit_and_poll=_record, last_failure=lambda: "",
        **{k: getattr(V.comfy_client, k) for k in dir(V.comfy_client)
           if not k.startswith("__") and k not in ("_submit_and_poll", "last_failure")})
    eng.engine_available = lambda ctx, *a, **k: (True, "")
    return eng


def run(picture: str, request: str) -> None:
    ctx = Context(models=None, transcription_cache={}, cache_file=None,
                  asr_lock=threading.Lock(), tts_lock=threading.Lock())
    ctx.no_think = True
    ctx.tts_disabled = True
    ctx.gui_mode = True
    ctx.video_engine = _engine()
    ctx.last_image_path = picture
    state = make_base_state()
    state["user_input"] = request
    SENT.clear()
    final = graph.build_graph(ctx).invoke(state)
    print("=" * 100)
    print("REQUEST:", request)
    for i, s in enumerate(SENT, 1):
        print(f"--- H3 job {i} ({s['label']}):\n{s['prompt']}\n")
    if not SENT:
        print("--- no H3 job was sent")
    print("REPLY:", (final.get("final_answer") or "")[:400])


if __name__ == "__main__":
    for req in sys.argv[2:]:
        run(sys.argv[1], req)
