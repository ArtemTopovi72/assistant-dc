"""YuE2 has no notion of stress: 'анализы' came out аналИзы, 'гнилое' гнИлое
(user, 2026-09-28). A/B of three lyric spellings, same seed, judged by ear:
plain / acute accent (а́) / capital stressed vowel (анАлизы)."""
import os, re, shutil, sys, threading, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import music
from stress import BilingualAccentor

LYRICS = """[bridge]
Там анализы теряют, в карты пишут лютый бред,
У них на каждый геморрой — один тупой ответ.

[chorus]
Никто ни хера не делает, лишь хамят в лицо,
Собрали в поликлинике сплошное гнилое яйцо!"""
STYLE = "russian punk rock, angry female vocal, fast drums, distorted guitars"
V = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"


def marked(text: str) -> str:
    return BilingualAccentor(enable_english=False)._accent_russian(text)


def variant(plus: str, how: str) -> str:
    def word(m):
        w = m.group(0)
        if "+" not in w:
            return w
        if sum(c in V for c in w) < 2:            # one vowel: nothing to disambiguate
            return w.replace("+", "")
        if how == "acute":
            return re.sub(r"\+(.)", "\\1\u0301", w)
        return re.sub(r"\+(.)", lambda v: v.group(1).upper(), w)
    return re.sub(r"[\w+\u0301]+", word, plus)


if __name__ == "__main__":
    plus = marked(LYRICS)
    print(plus)
    ctx = types.SimpleNamespace(set_stage=lambda *a, **k: None, cancel_event=threading.Event())
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_stress_ab")
    os.makedirs(out_dir, exist_ok=True)
    for name, lyr in [("1_plain", LYRICS), ("2_acute", variant(plus, "acute")),
                      ("3_caps", variant(plus, "caps"))]:
        print("==", name, "\n" + lyr, flush=True)
        f = music._generate_yue2(ctx, lyr, STYLE, 1234)
        shutil.copy(f, os.path.join(out_dir, name + os.path.splitext(f)[1]))
        print("->", name, flush=True)
