"""Nothing unpronounceable may reach F5-TTS.

Found live: the bot answered Vlad with "Да, я разбираюсь в Arduino и помогу с
написанием кода", and the voice produced a noise burst where the Latin word
was -- Whisper transcribed it as "разбираюсь в 1-0 с написанием кода". The
Latin tokens EXIST in vocab.txt (the F5 base vocabulary is multilingual), so
nothing failed and nothing warned; the Russian fine-tune simply never heard
them against Russian audio, so their pronunciation is undefined.

Run: venv/Scripts/python.exe tests/test_spoken_form.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import audio

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:200])


LATIN_OR_DIGIT = re.compile(r"[A-Za-z0-9]")

# ── detection ────────────────────────────────────────────────────────────────
check("Latin is detected", audio.needs_spoken_form("Да, я разбираюсь в Arduino"))
check("digits are detected", audio.needs_spoken_form("версия 2.4.1"))
check("plain Russian is left alone", not audio.needs_spoken_form("просто русский текст"))
check("an empty string is not flagged", not audio.needs_spoken_form(""))

# ── the fallback table (no ctx = no LLM) ─────────────────────────────────────
out = audio.to_spoken_form(None, "Да, я разбираюсь в Arduino и USB")
check("the fallback leaves no Latin behind", not re.search(r"[A-Za-z]", out), out)
check("a known word is spelled, not letter-by-letter", "ардуино" in out, out)
check("an acronym is read letter by letter", "ю-эс-би" in out, out)

check("an unknown Latin word still becomes Cyrillic",
      not re.search(r"[A-Za-z]", audio.to_spoken_form(None, "открой Quixotic сейчас")),
      audio.to_spoken_form(None, "открой Quixotic сейчас"))

# ── the LLM reply is not trusted blindly ─────────────────────────────────────
check("a reply that still contains Latin is rejected",
      not audio._spoken_form_is_sane("Arduino", "the Arduino board"))
check("a reply that runs away in length is rejected",
      not audio._spoken_form_is_sane("USB", "ю-эс-би " * 200))
check("an empty reply is rejected", not audio._spoken_form_is_sane("USB", ""))
check("a good reply is accepted",
      audio._spoken_form_is_sane("Да, я разбираюсь в Arduino",
                                 "Да, я разбираюсь в ардуино"))

# ── the whole preprocessing chain ────────────────────────────────────────────
prepped = audio.preprocess_text_for_synthesis(
    None, "Да, я разбираюсь в Arduino и помогу с написанием кода.",
    apply_stress=False)
check("nothing unpronounceable survives preprocessing",
      not LATIN_OR_DIGIT.search(prepped), prepped)
check("the end padding is still appended", prepped.endswith("."), prepped)

# ── the rewrite must happen BEFORE the accentor ──────────────────────────────
import inspect
src = inspect.getsource(audio.preprocess_text_for_synthesis)
check("the spoken-form pass runs before stress is applied",
      src.index("to_spoken_form(") < src.index("stress_plus("), src)

# ── caching ──────────────────────────────────────────────────────────────────
a = audio.to_spoken_form(None, "USB кабель")
b = audio.to_spoken_form(None, "USB кабель")
check("repeated text is answered from the cache", a == b, (a, b))

# ── digits must not reach the voice either, even without the LLM ─────────────
d = audio.to_spoken_form(None, "версия 2.4.1 займёт 15 минут")
check("the fallback spells digits out", not re.search(r"[0-9]", d), d)
check("and it separates letters from a following number",
      audio.to_spoken_form(None, "ESP32") == "и-эс-пи тридцать два",
      audio.to_spoken_form(None, "ESP32"))

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
