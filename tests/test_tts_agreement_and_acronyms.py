"""What the voice said wrong on the live drive of 2026-09-12.

  "На этикетке 2 строки"           -> voiced "два строки"    (must be "две строки")
  "по данным ЦБ РФ"                -> voiced as one garbled word ("цбров")

Numbers one and two agree in gender with the noun that follows; a Cyrillic
acronym with no vowel is read letter by letter. Both run on TTS input only,
never on displayed text. Pure; no model.

Run: venv/Scripts/python.exe tests/test_tts_agreement_and_acronyms.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
os.environ["F5_TEST_RUN"] = "1"
import logging; logging.basicConfig(level=logging.CRITICAL)

import audio as A

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:200])


def v(t):
    return A.post_process_answer(t)


# -- gender agreement of one and two ------------------------------------------
check("two lines", v("2 строки") == "две строки", v("2 строки"))
check("two minutes", v("2 минуты назад") == "две минуты назад")
check("twenty-two lines", v("22 строки") == "двадцать две строки", v("22 строки"))
check("two years stays masculine", v("2 года") == "два года")
check("two cats stays masculine", v("2 кота") == "два кота")
check("twelve is not two", v("12 строк") == "двенадцать строк")
check("one week", v("1 неделя") == "одна неделя")
check("one window", v("1 окно") == "одно окно")
check("one cat", v("1 кот") == "один кот")
check("hundred and one picture", v("101 картина") == "сто одна картина", v("101 картина"))
check("eleven windows untouched", v("11 окон") == "одиннадцать окон")
check("a decimal is not touched", "восемьдесят пять целых" in v("85,6 рубля"))
check("a bare number is unchanged", v("ответ: 2") == "ответ: два")

# a day of the month is an ordinal, not a count
check("first of January", v("1 января") == "первое января", v("1 января"))
check("eighth of March", v("8 марта") == "восьмое марта")
check("full dotted dates still go the date way",
      v("12.09.2026") == "двенадцатое сентября две тысячи двадцать шестого года")

# the LLM-free fallback path agrees too
check("fallback path agrees", A._digits_to_words("2 строки") == "две строки",
      A._digits_to_words("2 строки"))

# -- Cyrillic acronyms ----------------------------------------------------------
check("ЦБ РФ is read letter by letter", v("по данным ЦБ РФ") == "по данным цэ-бэ эр-эф", v("по данным ЦБ РФ"))
check("ФСБ, МВД", v("ФСБ и МВД") == "эф-эс-бэ и эм-вэ-дэ")
check("США is a word and stays", v("США") == "США")
check("ООН, СМИ stay", v("ООН, СМИ") == "ООН, СМИ")
check("ГИБДД is spelled despite its vowel", v("ГИБДД") == "гэ-и-бэ-дэ-дэ")
check("a capitalised word is not an acronym", v("Москва") == "Москва")
check("an ALL-CAPS word is brought down to ordinary case", v("вывеска У ОЛЬГИ") == "вывеска У Ольги", v("вывеска У ОЛЬГИ"))
check("НАТО is a word and is spoken as one", v("НАТО") == "Нато")
check("idempotent", A.spell_cyrillic_acronyms(A.spell_cyrillic_acronyms("ЦБ")) == "цэ-бэ")
check("the synthesis path applies it too (Skyrim goes straight there)",
      "цэ-бэ" in A.preprocess_text_for_synthesis(None, "ЦБ сказал", apply_stress=False))

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
