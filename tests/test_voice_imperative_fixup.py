"""A voice note misheard as a first-person statement is read as the command.

Live 2026-09-13 (mega run 3, step 25): the ASR turned «Нарисуй синего слона
на пляже под пальмой» into «Нарисую синего слона…» — "I will draw" — and the
model, taking it as a statement, replied with filler about the session memory
and drew nothing. Nobody tells the bot what they are about to draw: a leading
first-person future verb in a voice transcript is the imperative.
"""
import os, sys, inspect
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import audio as A

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

f = A.fix_voice_imperative
for heard, meant in [
    ("Нарисую синего слона на пляже под пальмой.", "Нарисуй синего слона на пляже под пальмой."),
    ("Сделаю слона розовым.", "Сделай слона розовым."),
    ("напишу текст песни про осень", "напиши текст песни про осень"),
    ("  Найду самую высокогорную дорогу", "  Найди самую высокогорную дорогу"),
    ("Переведу на английский: привет", "Переведи на английский: привет"),
]:
    check(f"{heard!r} -> {meant!r}", f(heard) == meant, f(heard))

for keep in [
    "Нарисуй кота",                     # already the imperative
    "Я нарисую тебе кота сам",          # first person NOT at the start — a real statement
    "Привет! Расскажи, почему кошки мурлыкают.",
    "Спасибо, пока!",
    "",
    "Draw a cat",
]:
    check(f"untouched: {keep!r}", f(keep) == keep, f(keep))

import tg_tasks
src = inspect.getsource(tg_tasks.TaskRunnerMixin._transcribe)
check("the Telegram transcript goes through the fix-up", "fix_voice_imperative(" in src)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
