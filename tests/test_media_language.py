"""A video's ASR language is detected from the clip, not taken from the chat.

An English video in a Russian chat was re-read as Russian whenever Whisper's
guess fell under ASR_LANG_CONFIDENCE, because the session language was the hint.
No model is loaded; Whisper is a fake.

Run: venv/Scripts/python.exe tests/test_media_language.py
"""
import os
import sys
import threading
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import audio as A

BAD = 0


def check(name, cond, extra=""):
    global BAD
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else "   " + str(extra)))
    BAD += 0 if cond else 1


def ctx_with(lang, prob):
    w = types.SimpleNamespace(detect_language=lambda wav, **kw: (lang, prob, []))
    return types.SimpleNamespace(asr_lock=threading.Lock(), models=types.SimpleNamespace(whisper=w))


import faster_whisper
_real_decode, _real_pin = faster_whisper.decode_audio, A.WHISPER_LANGUAGE
faster_whisper.decode_audio = lambda path, sampling_rate=16000: np.zeros(16000, dtype=np.float32)
A.WHISPER_LANGUAGE = None
try:
    check("a confident English clip is English", A.detect_media_language(ctx_with("en", 0.7), "x.mp4") == "en")
    check("an unsure guess gives no hint", A.detect_media_language(ctx_with("en", 0.3), "x.mp4") == "")
    boom = types.SimpleNamespace(asr_lock=threading.Lock(), models=None)
    check("a failure gives no hint, not an exception", A.detect_media_language(boom, "x.mp4") == "")
    A.WHISPER_LANGUAGE = "ru"
    check("a pinned WHISPER_LANGUAGE wins", A.detect_media_language(ctx_with("en", 0.99), "x.mp4") == "ru")
finally:
    faster_whisper.decode_audio, A.WHISPER_LANGUAGE = _real_decode, _real_pin

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(root, "bot/tg_tasks.py"), encoding="utf-8").read()
check("tg_tasks detects the language of a video", 'if media == "video"' in src and "detect_media_language(ctx, src)" in src)
check("diarized turns use the clip's language, labels the chat's", "asr_lang=lang_hint" in src and "lang=label_lang" in src)
check("long-video portions use one detected language",
      "lang_hint=asr_lang" in open(os.path.join(root, "bot/tg_video.py"), encoding="utf-8").read())

sys.exit(1 if BAD else 0)
