"""Voice output died silently: every synthesis returned None.

f5_tts's preprocess_ref_audio_text runs its OWN ASR whenever the reference
transcript is empty, and that decodes audio through torchaudio -> torchcodec.
torchcodec is version-locked to BOTH torch and the installed FFmpeg; with
torch pinned at 2.8.0+cu128 and only FFmpeg 8 shared libraries present, it
raised on import. synth_single_segment caught the exception and returned None,
so the text reply still arrived and only the voice went missing — the failure
looked like "TTS randomly stopped working" rather than a crash.

The configured assistant actor has ref_text = "", so this fired on EVERY turn,
not in some edge case.

Fix: transcribe the reference with our own faster-whisper path (PyAV, no
torchcodec coupling) and hand the text to f5_tts, which then skips its ASR
entirely. The result is cached in transcriptions.json, so it also stops
re-transcribing the same reference every turn.

Run: venv/Scripts/python.exe tests/test_tts_reference_transcript.py
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import audio as A
import config as C

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


class _Ctx:
    """Enough context to reach the reference-transcript step, no models."""
    def __init__(self):
        self.transcription_cache = {}
        self.cache_file = None
        self.tts_lock = threading.Lock()
        self.models = None
        self.custom_ref_wav = None
    def save_cache(self): pass


print("=" * 70)
print("1. The configured actor really does ship an EMPTY reference transcript")
print("=" * 70)
# This is what made the bug universal instead of an edge case. If someone later
# fills the transcript in, the guard below simply stops being needed.
# The actor's own clip is a real voice and not in git; f5_tts's sample stands in.
import f5_tts
A.DC_REF_WAV = os.path.join(list(f5_tts.__path__)[0], "infer", "examples", "basic", "basic_ref_en.wav")
_wav, _text, _speed = A.get_actor_ref_and_speed(C.ASSISTANT_ACTOR)
check("the assistant actor has no configured ref_text", _text == "",
      repr(_text))
check("and it does have a reference wav", bool(_wav), _wav)

print()
print("=" * 70)
print("2. An empty transcript is filled in by OUR whisper, not f5_tts's")
print("=" * 70)

calls = {"our_asr": 0, "f5_ref_text": None, "engine": None}

def _fake_transcribe(ctx, path, engine="auto"):
    calls["our_asr"] += 1
    calls["engine"] = engine
    return "эталонная расшифровка"

def _fake_preprocess(ref_wav, ref_text):
    # f5_tts transcribes internally IFF ref_text is falsy — that is the branch
    # that reaches torchcodec. Record what we were handed.
    calls["f5_ref_text"] = ref_text
    return ref_wav, ref_text

def _fake_infer(*a, **k):
    import numpy as np
    return np.zeros(2400, dtype="float32"), 24000, None

import f5_tts.infer.utils_infer as U
_saved = (A.transcribe_audio_file, U.preprocess_ref_audio_text, U.infer_process)
A.transcribe_audio_file = _fake_transcribe
U.preprocess_ref_audio_text = _fake_preprocess
U.infer_process = _fake_infer
try:
    ctx = _Ctx()
    class _M: tts_model = None; vocoder = None
    ctx.models = _M()
    out = A.synth_single_segment(
        ctx=ctx, idx=-1, actor=C.ASSISTANT_ACTOR, raw_text="привет",
        out_stem=None, use_censoring=False, apply_stress=False)
finally:
    A.transcribe_audio_file, U.preprocess_ref_audio_text, U.infer_process = _saved

# A reference clip is NOT known to be Russian -- the Mantella speaker library
# is read in English -- so this one call must never be routed to the
# Russian-only engine, whatever the installation has configured.
check("the reference is transcribed by Whisper, never the Russian-only engine",
      calls["engine"] == "whisper", calls["engine"])
check("our own ASR was used for the reference", calls["our_asr"] == 1,
      calls["our_asr"])
check("f5_tts received a NON-empty transcript, so it never runs its own ASR",
      bool(calls["f5_ref_text"]), repr(calls["f5_ref_text"]))
check("synthesis produced a file", bool(out) and os.path.exists(out), out)
if out and os.path.exists(out):
    try: os.remove(out)
    except OSError: pass

print()
print("=" * 70)
print("3. The audio stack this depends on actually loads")
print("=" * 70)
# The regression was environmental, so assert the environment too: a mismatched
# torchcodec/FFmpeg/torch trio is exactly what broke it.
try:
    import torch, torchaudio
    w, sr = torchaudio.load(str(A.DC_REF_WAV))
    check("torchaudio can decode the reference wav",
          hasattr(w, "shape") and w.numel() > 0 and sr > 0, (tuple(w.shape), sr))
    check("torch is still the pinned CUDA build (pip must not swap it)",
          torch.__version__.startswith("2.8.0") and ("cu" in torch.__version__ or os.getenv("CI")),
          torch.__version__)
except Exception as exc:
    check("torchaudio can decode the reference wav", False, f"{type(exc).__name__}: {exc}")

print()
print(f"{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
