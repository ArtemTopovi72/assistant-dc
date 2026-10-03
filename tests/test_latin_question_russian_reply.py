"""A Latin-script question answered in Cyrillic is put back into the user's language."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sweep_stub  # noqa: F401  the model's narrow reads, stubbed
import intent
intent.STUB = lambda t: {"reply_language": "ru"} if "in Russian" in t else None
import graph_language as gl


class _Ctx:
    def is_cancelled(self):
        return False


calls = []
gl.call_llm_simple = lambda ctx, sysp, *a, **k: (calls.append(1), "German" if sysp == gl._LANG_DETECT_PROMPT
                                                 else "Es ist jetzt 18:37 in Berlin.")[1]
out = gl._match_reply_language(_Ctx(), "Сейчас в Берлине 18:37, понедельник.", "Wie spät ist es jetzt in Berlin?")
assert out.startswith("Es ist"), out
calls.clear()
out = gl._match_reply_language(_Ctx(), "Привет! Как дела у тебя сегодня?", "answer me in Russian please now")
assert out.startswith("Привет") and not calls, out
out = gl._match_reply_language(_Ctx(), "Кот", "ok")
assert out == "Кот"
print("PASS test_latin_question_russian_reply")

# An answer that only echoes the user's message is regenerated.
gl.call_llm_simple = lambda *a, **k: "Звучит разумно! О чём поговорим?"
q = "Я следую общим принципам полезности, стабильности и контекста диалога."
assert gl._unparrot(_Ctx(), q, q).startswith("Звучит")
assert gl._unparrot(_Ctx(), "Да.", "да") == "Да."
print("PASS unparrot")

assert gl._ru_thousands("17% от 2,345,678 рублей — 399,000 рублей") == "17% от 2 345 678 рублей — 399 000 рублей"
assert gl._ru_thousands("It costs 2,345 dollars") == "It costs 2,345 dollars"
assert gl._ru_thousands("Числа 1,5 и 3,14 и 1,234.5 остаются") == "Числа 1,5 и 3,14 и 1,234.5 остаются"
print("PASS ru thousands")
assert gl._ru_thousands("составляет 398 765.26 рублей, курс 84.40 руб, Python 3.11") == "составляет 398 765,26 рублей, курс 84,40 руб, Python 3.11"
print("PASS ru decimal")

assert gl._fix_totals("• a — 100\n• b — 200\n• c — 300\nВсего 900 руб.") == "• a — 100\n• b — 200\n• c — 300\nВсего 600 руб."
assert gl._fix_totals("• a — 1 000\n• b — 2 000\n• c — 3 000\nИтого: 6 000 рублей") == "• a — 1 000\n• b — 2 000\n• c — 3 000\nИтого: 6 000 рублей"
print("PASS totals")

assert gl._ru_thousands("Вы заплатили 1088.20, по 362.73 на каждого; Python 3.11, версия 2.10") == "Вы заплатили 1088,20, по 362,73 на каждого; Python 3.11, версия 2.10"
print("PASS money")

calls.clear()
_de = "Sehr geehrter Herr Iwan Petrowitsch, bitte verschieben Sie das Treffen auf Freitag."
import intent   # the model's read of the words (agent/intent.py); phrases: bench/intent_lang_live.py
intent.STUB = lambda t: {"translate": True} if t.startswith("переведи") else None
assert gl._match_reply_language(_Ctx(), _de, "переведи на немецкий: Уважаемый Иван Петрович, прошу перенести встречу") == _de
print("PASS translate request keeps its target language")
