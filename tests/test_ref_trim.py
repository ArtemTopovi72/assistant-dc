"""The reference transcript must describe the audio F5 actually hears.

F5 clips any reference over 12 s (utils_infer.py:324) and carries on, but it
is handed the transcript of the WHOLE file. The model then still has text left
when the reference audio ends and finishes it — the tail of the reference is
spoken into the reply. A 20.3 s Stepan_short.wav put "это довольно длинное
сообщение" into a sentence about roubles and minutes; measured by synthesising
and transcribing, 0/6 phrases came back recognisable. After trimming: 5/6.

Run: venv/Scripts/python.exe tests/test_ref_trim.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

from pydub import AudioSegment
from pydub.generators import Sine

import audio

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:200])


TMP = tempfile.mkdtemp(prefix="reftrim_")


def make(ms, path):
    """Speech-shaped: tone, silence, tone… so the silence split has seams."""
    seg = AudioSegment.silent(duration=0)
    while len(seg) < ms:
        seg += Sine(220).to_audio_segment(duration=2000).apply_gain(-6)
        seg += AudioSegment.silent(duration=1200)
    seg[:ms].export(path, format="wav")
    return path


check("the cap matches F5's own", audio.REF_CAP_MS == 12_000, audio.REF_CAP_MS)

short = make(8_000, os.path.join(TMP, "short.wav"))
check("a reference already under the cap is returned untouched",
      audio.trim_ref_to_cap(short) == short)

long_ = make(20_300, os.path.join(TMP, "long.wav"))
out = audio.trim_ref_to_cap(long_)
check("a long reference is replaced", out != long_, out)
dur = len(AudioSegment.from_file(out))
check("and the result is within the cap", dur <= audio.REF_CAP_MS, dur)
check("but not trimmed to nothing", dur >= 6_000, dur)
check("the trimmed copy sits beside the original",
      os.path.dirname(out) == os.path.dirname(long_), out)

again = audio.trim_ref_to_cap(long_)
check("a second call reuses the cached copy", again == out, (out, again))

check("an unreadable path is passed through rather than raising",
      audio.trim_ref_to_cap(os.path.join(TMP, "nope.wav"))
      == os.path.join(TMP, "nope.wav"))

# The order matters as much as the trim: taking the transcript first would
# describe audio the model never hears.
import inspect
src = inspect.getsource(audio.synth_single_segment)
check("the reference is trimmed BEFORE it is transcribed",
      src.index("trim_ref_to_cap(") < src.index("transcribe_audio_file("), src)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
