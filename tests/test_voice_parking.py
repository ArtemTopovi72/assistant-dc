"""Whisper and F5 leave the card while a render owns it, and come back after.

With the chat model unloaded nobody answers by voice, yet the two speech models
kept ~1.5 GB of the 24 GB card beside every render. Parking moves them to RAM;
a voice note that arrives mid-render must still transcribe (the wrapper wakes
Whisper on use), and a reply must still speak (wake_tts under tts_lock).
"""
import os, sys, threading
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import torch
import models as M
from types import SimpleNamespace as NS

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

if not torch.cuda.is_available():
    print("SKIP  no CUDA: parking is a card-memory concern")
    sys.exit(0)

from faster_whisper import WhisperModel
from config import DEVICE

wh = M._ParkableWhisper(WhisperModel("base", device="cuda", compute_type="int8_float16"))
tts = torch.nn.Linear(256, 256).to(DEVICE)
voc = torch.nn.Linear(16, 16).to(DEVICE)
ctx = NS(models=NS(whisper=wh, tts_model=tts, vocoder=voc),
         asr_lock=threading.Lock(), tts_lock=threading.Lock())
check("park reports it parked", M.park_voice(ctx))
check("Whisper left the card", not wh.model.model_is_loaded)
check("F5 left the card", next(tts.parameters()).device.type == "cpu"
      and next(voc.parameters()).device.type == "cpu")

wav = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DC_short.wav")
segs, info = wh.transcribe(wav, beam_size=1)
text = " ".join(s.text for s in segs).strip()
check("a voice note mid-render still transcribes", bool(text), text)

M.park_voice(ctx)
with ctx.tts_lock:
    M.wake_tts(ctx.models)
check("a reply mid-render puts F5 back first", next(tts.parameters()).device.type == "cuda")

M.park_voice(ctx)
M.wake_voice(ctx)
check("give-back restores both", wh.model.model_is_loaded
      and next(voc.parameters()).device.type == "cuda")

with ctx.tts_lock:
    M.park_voice(ctx, wait_s=0.1)
check("speech in progress is never moved under it", next(tts.parameters()).device.type == "cuda")

print(f"\n{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
