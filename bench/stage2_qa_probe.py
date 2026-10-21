"""What the contained edit's STAGE-2 QA sees when it rejects a plain face/clothes edit
(«сделай нос побольше», «галстук красный» both refused as WRONG, debug chat 2026-10-21).
Saves every QA preview with its verdict.  venv/Scripts/python.exe bench/stage2_qa_probe.py"""
import os, sys, threading, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for d in ("", "agent", "imaging", "media", "bot", "core", "voice"):
    sys.path.insert(0, str(ROOT / d))
os.chdir(ROOT); sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import gpu_lock, image as I, image_maskqa as Q
from models import Context
ctx = Context(models=None, transcription_cache={}, cache_file=None, asr_lock=threading.Lock(), tts_lock=threading.Lock())
P = ROOT / "tmp" / "judge_photos"
n = [0]
real = Q._vlm_qa_verdict


def spy(c, png, question, labels):
    v = real(c, png, question, labels)
    n[0] += 1
    if png and os.path.exists(str(png)):
        shutil.copy(png, P / f"qa_{n[0]}_{v}.png")
    print("QA", n[0], v, question[:120], flush=True)
    return v


Q._vlm_qa_verdict = spy
I._vlm_qa_verdict = spy
CASES = [("beard0.jpg", "nose", "a larger, more prominent nose"),
         ("glasses0.jpg", "tie", "a red tie")]
with gpu_lock.hold("stage2 QA probe", wait=1800):
    for img, region, instr in CASES:
        out = I.inpaint_region_with_comfy(ctx, str(P / img), region, instr, seed=11)
        print("OUT", img, region, out, flush=True)
