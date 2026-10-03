"""«answer in English from now on» pins English; a later Russian line still gets English."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")
import graph_language as GL
import tg_tasks as T, intent
intent.STUB = {"translate it into Russian": {"translate": True}}.get   # the model's read


class _Ctx:
    reply_lang = "en"
    def is_cancelled(self): return False


GL.call_llm_simple = lambda *a, **k: "Try The Silk Roads by Peter Frankopan."
ru = "Тогда прочитай «Шелковые пути» Питера Франкопана, это масштабный эпос."
assert GL._match_reply_language(_Ctx(), ru, "а еще одну?").startswith("Try")
assert GL._match_reply_language(_Ctx(), ru, "translate it into Russian") == ru
en = "The capital of Australia is Canberra."
assert GL._match_reply_language(_Ctx(), en, "какая столица Австралии?") == en   # not translated back
wall = "Сегодня был долгий день. " * 300 + "Сколько раз я написал слово день?"
assert GL._translate_to_english(_Ctx(), wall) == wall.strip()        # not translated (70 s, wrong count)
print("PASS lang pin")
assert T._message_script("Qual è la capitale del Portogallo?") == "en"   # accented Latin
assert "{lang}" in GL._TRANSLATE_OUT_PROMPT
assert GL._english_words("Was ist die Hauptstadt in Deutschland? Qual é a capital do Brasil?") == 0
assert GL._english_words("what's the capital of Peru?") >= 2
calls = []
GL.call_llm_simple = lambda ctx, sysp, *a, **k: calls.append(sysp) or (
    "Spanish" if sysp == GL._LANG_DETECT_PROMPT else "Madrid tiene unos 3,3 millones de habitantes.")
class _Es(_Ctx): reply_lang = "en"
assert GL._match_reply_language(_Es(), "Madrid has approximately 3.3 million inhabitants.",
                                "¿Cuántos habitantes tiene Madrid?").startswith("Madrid tiene")
assert GL._match_reply_language(_Es(), "Lima is the capital of Peru.", "what's the capital of Peru?") \
    == "Lima is the capital of Peru."
assert "into Spanish" in calls[-1]
print("PASS spanish not answered in English")

import prompts as _p
leak = _p._LANGUAGE_RULES["ru"][:600]
assert GL._prompt_leak_guard(_Ctx(), leak).startswith("I don't share")
assert GL._prompt_leak_guard(_Ctx(), "Столица Перу — Лима, это крупный город.") == "Столица Перу — Лима, это крупный город."
print("PASS prompt leak guard")
_ok = "I cannot send SMS, messages or email to other people, make calls, order a taxi or food."
assert GL._prompt_leak_guard(_Ctx(), _ok) == _ok      # an honest refusal echoing one rule line
print("PASS refusal is not a leak")
assert "2,6 ст.л., соль 0,875 ч.л." in GL._ru_thousands("Рецепт: мука 437,5 г, сахар 2.6 ст.л., соль 0.875 ч.л.")
print("PASS recipe decimals")
assert GL._ru_thousands("Население Казани составляет примерно 1250000 человек.") == "Население Казани составляет примерно 1 250 000 человек."
assert GL._ru_thousands("индекс 420111 Казань") == "индекс 420111 Казань"
print("PASS bare big numbers grouped")
_calls = []
GL.call_llm_simple = lambda ctx, sysp, *a, **k: _calls.append(sysp) or "Я проверил ещё раз: 17 × 24 = 408."
_q = "я уверен, проверь ещё раз, мой калькулятор показывает 418"
assert GL._unparrot(_Ctx(), "Я уверен, проверь ещё раз, мой калькулятор показывает 408.", _q).startswith("Я проверил")
assert "408" in _calls[-1]
assert GL._unparrot(_Ctx(), "17 умножить на 24 будет 408.", _q) == "17 умножить на 24 будет 408."
print("PASS near-copy echo regenerated")
_fix = "исправь ошибки: я вчира ходил в магазин и купил хлеп и малако"
assert GL._unparrot(_Ctx(), "Я вчера ходил в магазин и купил хлеб и молоко.", _fix) == "Я вчера ходил в магазин и купил хлеб и молоко."
print("PASS a proofread is not an echo")
_h = [{"role": "user", "content": "сколько будет 17*24?"}, {"role": "assistant", "content": "17 умножить на 24 будет 408."},
      {"role": "user", "content": _q}, {"role": "assistant", "content": "Я уверен ... 408."}]
GL._unparrot(_Ctx(), "Я уверен, проверь ещё раз, мой калькулятор показывает 408.", _q, _h)
assert "будет 408" in _calls[-1]
print("PASS echo retry sees the previous reply")
