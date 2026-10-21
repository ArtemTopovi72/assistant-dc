"""ObjectClear vs the contained instruction edit for removing a worn thing (a watch, glasses,
a beard): ObjectClear drew a new watch onto the wrist twice of twice (debug chat 2026-10-21).
    venv/Scripts/python.exe bench/removal_engine_ab.py      (holds runtime/gpu.lock)"""
import os, sys, threading, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for d in ("", "agent", "imaging", "media", "bot", "core", "voice"):
    sys.path.insert(0, str(ROOT / d))
os.chdir(ROOT); sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from PIL import Image
import gpu_lock, image_objects as O
from models import Context
ctx = Context(models=None, transcription_cache={}, cache_file=None, asr_lock=threading.Lock(), tts_lock=threading.Lock())
P = ROOT / "tmp" / "judge_photos"
CASES = [("watch1.jpg", "watch"), ("watch0.jpg", "watch"), ("glasses0.jpg", "glasses"), ("beard1.jpg", "beard")]
with gpu_lock.hold("removal engine A/B", wait=1800):
    for img, target in CASES:
        for engine in ("objectclear", "firered"):
            os.environ["OBJECT_REMOVE_ENGINE"] = engine
            t = time.time()
            out = O.remove_object_with_comfy(ctx, str(P / img), target, seed=7)
            if out:
                Image.open(out).convert("RGB").save(P / f"ab_{img[:-4]}_{engine}.png")
            print("OUT", img, engine, round(time.time() - t), out, flush=True)
