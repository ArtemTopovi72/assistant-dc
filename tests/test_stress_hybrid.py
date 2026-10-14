"""RUAccent's line stands; only its own homographs take silero-stress's choice."""
import os, sys, types
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(R, "voice"), os.path.join(R, "core")]
import stress as S

ru = types.SimpleNamespace(omographs={"замок": ["з+амок", "зам+ок"]})
a = S.BilingualAccentor()
a.silero = lambda t: "зам+ок на двер+и, электроуд+очник"   # silero: right homograph, wrong rare word
out = a._silero_homographs(ru, "з+амок на двер+и, электро+удочник")
assert out == "зам+ок на двер+и, электро+удочник", out
a.silero = lambda t: "совсем другой текст"                     # misaligned: RUAccent kept
assert a._silero_homographs(ru, "з+амок") == "з+амок"
print("ok")
