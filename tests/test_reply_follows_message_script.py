"""An English message in a Russian-language chat is answered in English.

Live 2026-09-13 (mega journey, steps 31/33/34): "hi! what's the capital of
Australia? one line" got "Столица Австралии — Канберра", and the upscale /
outpaint buttons after "draw a lighthouse..." came back with Russian captions.
The reply language is now flipped per turn by the script the user wrote in;
button payloads, emoji and links inherit the last typed script.
"""
import os, sys, inspect
os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_tasks as T
import tg_sessions as S
import intent   # the model's read of the words (agent/intent.py); phrases: bench/intent_lang_live.py
intent.STUB = {"и то же самое по-английски": {"reply_language": "en"},
               "ответь на этот вопрос по-английски, пожалуйста": {"reply_language": "en"}}.get

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

ms = T._message_script
check("English question -> en", ms("hi! what's the capital of Australia? one line") == "en")
check("English draw -> en", ms("draw a lighthouse on a cliff during a storm") == "en")
check("Russian question -> ru", ms("за сколько дней надо предупредить о выезде?") == "ru")
check("button payload inherits", ms("[upscale]") == "")
check("one-word command inherits", ms("upscale") == "")
check("emoji inherits", ms("👍") == "")
check("link inherits", ms("https://ru.wikipedia.org/wiki/Омлет") == "")
check("mixed text inherits", ms("замени надпись на «BREAD AND SALT» please now") == "")
check("Russian with a quoted brand is still ru", ms("нарисуй витрину пекарни с вывеской «ХЛЕБ У ДОМА»") == "ru")
check("empty inherits", ms("") == "")
check("an explicit ask for English wins over the alphabet", ms("и то же самое по-английски") == "en")
check("...also 'ответь по-английски'", ms("ответь на этот вопрос по-английски, пожалуйста") == "en")

sess = S._Session({})
check("session carries turn_lang", sess is not None and hasattr(sess, "turn_lang"))
if sess is not None:
    sess.turn_lang = "en"
    check("turn_lang persists", sess.to_dict().get("turn_lang") == "en")

src = inspect.getsource(T.TaskRunnerMixin._execute_task)
check("the task applies the per-turn override",
      "ctx.reply_lang = _turn_lang" in src and "_message_script(task.user_text)" in src)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
