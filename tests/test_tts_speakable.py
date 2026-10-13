"""graph.speakable: a written answer is rewritten into speech before TTS, and
the rewrite is thrown away when it drops a number.
Run: venv/Scripts/python.exe tests/test_tts_speakable.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import graph as G

calls = []


def fake(reply):
    def _f(ctx, system, user, **kw):
        calls.append((system, user, kw))
        return reply
    return _f


bad = 0


def check(name, cond, extra=""):
    global bad
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"   {extra}"))
    bad += not cond


from types import SimpleNamespace as NS
_T = [None]   # the text model the context hands out
K = NS(text_model=lambda *a, **k: _T[0](*a, **k))

written = "Итог: да, можно. Условие — температура не выше 40 °C."
_T[0] = fake("Итог такой: да, можно, если температура не выше 40 °C.")
out = G.speakable(K, written)
check("a written answer is rewritten for the voice", out.startswith("Итог такой"), out)
check("...with reasoning switched off", calls[-1][2].get("prefill") == "<think></think>", calls[-1][2])

_T[0] = fake("Итог такой: да, можно, если не слишком жарко.")
check("a rewrite that drops a number is rejected", G.speakable(K, written) == written)

_T[0] = fake("")
check("an empty rewrite falls back to the text", G.speakable(K, written) == written)

calls.clear()
check("plain speech is not sent to the model",
      G.speakable(K, "Да, конечно, приходи вечером") == "Да, конечно, приходи вечером" and not calls)

_T[0] = fake("Добавь count = 0 перед циклом, ссылка в сообщении.")
out = G.speakable(K, "Решение → добавить `count = 0` перед циклом. https://x.ru/a1")
check("backticks dropped, the URL's digits are not demanded", out == "Добавь count = 0 перед циклом, ссылка в сообщении.", out)
_T[0] = fake("Нужны, во-первых, паспорт, во-вторых, билеты на 14:30.")
out = G.speakable(K, "Нужно:\n1. Паспорт\n2. Билеты на 14:30")
check("list numbering is not a number to keep", out.startswith("Нужны, во-первых"), out)
_T[0] = fake("Для начала мониторинг, затем база знаний, ну и контракты, и 90% успеха.")
out = G.speakable(K, "План:\n**1. Мониторинг**\n### 2. База знаний\n**3. Контракты**\n90% успеха")
check("bold or heading numbering is not a number to keep", out.startswith("Для начала"), out)
sys.exit(1 if bad else 0)
