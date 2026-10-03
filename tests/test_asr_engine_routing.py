"""GigaAM is Russian-only, and nothing must reach it that is not Russian.

It does not detect an unsupported language and it does not refuse one: it
renders it as confident Russian, with nothing in the output to say so. The same
function that transcribes a user's voice note also transcribes F5 reference
clips, and the Mantella speaker library is read in ENGLISH -- so a global
switch would quietly poison 160 reference transcripts and the voices built
from them.

No model is loaded here and nothing touches the GPU.

Run: venv/Scripts/python.exe tests/test_asr_engine_routing.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import logging; logging.basicConfig(level=logging.CRITICAL)

import numpy as np

import audio as A
import config as C

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:200])


_saved = (C.ASR_ENGINE, A.WHISPER_LANGUAGE)


def route(engine, language):
    C.ASR_ENGINE = engine
    A.WHISPER_LANGUAGE = language
    return A._use_gigaam()


# ── the shipped default ──────────────────────────────────────────────────────
# GigaAM belongs to the Skyrim stack only. Pinning this is the point: the app
# takes voice notes from arbitrary Telegram users, and the failure mode of a
# Russian-only engine on another language is silent nonsense, not an error.
check("the app ships on Whisper", str(_saved[0]).lower() == "whisper", _saved[0])
check("and pinning Russian alone does NOT switch it",
      route("whisper", "ru") is False)

# ── when the Russian-only engine may be used ─────────────────────────────────
check("auto + no language pinned -> Whisper (the app asked for detection)",
      route("auto", None) is False)
check("auto + Russian pinned -> GigaAM", route("auto", "ru") is True)
check("auto + English pinned -> Whisper", route("auto", "en") is False)
check("explicit whisper wins even with Russian pinned",
      route("whisper", "ru") is False)
check("explicit gigaam is honoured with no language pinned",
      route("gigaam", None) is True)
check("a junk setting falls back to the safe branch",
      route("nonsense", None) is False)

C.ASR_ENGINE, A.WHISPER_LANGUAGE = _saved

# ── the forced path, which is what protects the references ───────────────────
import inspect
src = inspect.getsource(A.synth_single_segment)
check("the reference transcript is taken with engine=\"whisper\"",
      'engine="whisper"' in src, src[src.find("ref_text ="):][:200])

wsrc = inspect.getsource(A._whisper_transcribe)
check("_whisper_transcribe honours a forced engine",
      'engine != "whisper"' in wsrc, wsrc[:300])

fsrc = inspect.getsource(A.transcribe_audio_file)
check("the cache key separates the two engines",
      '"whisper" if engine == "whisper" else "asr"' in fsrc, fsrc[:400])

# The Skyrim voice server lives only in a local checkout (mantella/ is not in git).
_srv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "mantella", "mantella_f5_server.py")
if os.path.exists(_srv):
    srv = open(_srv, encoding="utf-8").read()
    # It loads no Whisper at all (Models.load(whisper=False)): its Russian
    # game-actor references go through GigaAM directly.
    check("the Mantella server transcribes references with GigaAM, no Whisper loaded",
          "_gigaam_transcribe(" in srv and "whisper=False" in srv)

# ── silence must not become words ────────────────────────────────────────────
ctx = types.SimpleNamespace(asr_lock=__import__("threading").Lock())
check("digital silence returns nothing",
      A._gigaam_transcribe(ctx, np.zeros(16000, dtype=np.float32)) == "")
check("room tone returns nothing",
      A._gigaam_transcribe(ctx, (np.random.randn(16000) * 0.001).astype(np.float32)) == "")
check("an empty array returns nothing",
      A._gigaam_transcribe(ctx, np.zeros(0, dtype=np.float32)) == "")

# ── a failure must fall through, not take the turn down ──────────────────────
_real = A._gigaam_transcribe
try:
    def boom(ctx, src):
        raise RuntimeError("onnxruntime is unhappy")
    A._gigaam_transcribe = boom
    C.ASR_ENGINE = "gigaam"
    seen = {}

    class _W:
        def transcribe(self, source, **kw):
            seen["used"] = True
            seg = types.SimpleNamespace(text="из виспера")
            return [seg], types.SimpleNamespace(language="ru", language_probability=1.0)

    ctx2 = types.SimpleNamespace(asr_lock=__import__("threading").Lock(),
                                 models=types.SimpleNamespace(whisper=_W()))
    out = A._whisper_transcribe(ctx2, np.zeros(16000, dtype=np.float32))
    check("a GigaAM crash falls back to Whisper instead of losing the turn",
          seen.get("used") and out == "из виспера", (seen, out))
finally:
    A._gigaam_transcribe = _real
    C.ASR_ENGINE = _saved[0]

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
