import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_songs

t = "[verse]\nа\nб\n\n[pre-chorus]\nэй\nэй\n\n[verse]\nв\nг\n\n[pre-chorus]\nэй\nэй\n"
out = tg_songs._repeats_are_chorus(t)
assert out.count("[chorus]") == 2 and out.count("[verse]") == 2 and "эй" in out, out
assert tg_songs._repeats_are_chorus("[verse]\nа\n\n[bridge]\nб\n") == "[verse]\nа\n\n[bridge]\nб\n"
print("PASS a block sung twice is the chorus")

_own = "а\nб\n\nэй\nэй\n\nв\nг\n\nэй\nэй"
_t = tg_songs.tag_lyrics(_own)
assert _t.count("[chorus]") == 2 and _t.count("эй") == 4, _t        # nothing added
print("PASS the author's own refrain is tagged, not repeated again")
