"""One model read per message: every read of this turn's words -- as written, translated, or
the two joined, with or without a context of its own -- is the turn's one read.

About twenty places read the same message, each with its own context, and a message cost 4-7
model calls of ~2 s before any work began (log 2026-10-26: 5 per task was the mode, 10 the worst).
"""
import json, os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["F5_TEST_RUN"] = "1"
os.environ["INTENT_LIVE"] = "1"
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import intent

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

calls = []
def model(ctx, system, user, **k):
    calls.append(user)
    return json.dumps({"needs_tool": True, "wants": ["generate_image"]})
ctx = types.SimpleNamespace(text_model=model)

intent._CACHE.clear()
intent.bind_turn(["нарисуй Пушкина при свече", "  нарисуй Пушкина при свече  "], "Вот кот.", "a photo")
intent.turn_alias("draw Pushkin by candlelight")
intent.read(ctx, "нарисуй Пушкина при свече")
intent.read(ctx, "draw Pushkin by candlelight", "Вот кот.")
intent.read(ctx, "нарисуй Пушкина при свече draw Pushkin by candlelight")
intent.read(ctx, "нарисуй Пушкина при свече", "", "a photo")
check("the words, their translation and the two joined are one read", len(calls) == 1, calls)
check("...read with the turn's context", calls and "Вот кот." in calls[0] and "a photo" in calls[0], calls)
check("...of the turn's own words", calls and "нарисуй Пушкина при свече" in calls[0], calls)

intent.read(ctx, "нарисуй Пушкина при свече", "", "several pictures earlier in the chat")
check("a fact of the caller's own gets its own read", len(calls) == 2 and "several pictures" in calls[1], calls)
intent.read(ctx, "что такое фотосинтез")
check("other words are read as before", len(calls) == 3, calls)

intent.bind_turn(["а теперь собаку"], "", "")
intent.read(ctx, "нарисуй Пушкина при свече")
check("the next turn unbinds the last one's words", len(calls) == 4 and "Вот кот." not in calls[3], calls[-1:])

# What the message asks the program to do exactly (sort, count, a dose) rode a second
# model call of its own, ~1.5 s per message; it is a field of the one read now.
import ask_read
def model_ask(ctx, system, user, **k):
    calls.append(user)
    return json.dumps({"needs_tool": False, "wants": [],
                       "ask": {"op": "sort", "items": ["груша", "абрикос", "вишня"], "detailed": True}})
calls.clear(); intent._CACHE.clear(); ask_read._CACHE.clear()
ctx2 = types.SimpleNamespace(text_model=model_ask)
said = "отсортируй по алфавиту: груша, абрикос, вишня"
intent.bind_turn([said], "", "")
intent.read(ctx2, said)
got = ask_read.read(said, ctx=ctx2)
check("the exact-operation read rides the message's one read", len(calls) == 1, calls)
check("...and keeps its fields", got["op"] == "sort" and got["items"] == ["груша", "абрикос", "вишня"]
      and got["detailed"], got)

# Telegram read one message six times: three threads before the graph, the text bare, with a
# link note under it, collapsed -- each its own key (trace 2026-10-26).
import threading
calls.clear(); intent._CACHE.clear(); intent._BY_WORDS.clear()
said = "сколько стоит этот рюкзак? https://ozon.ru/t/LCabAMR"
note = "\n[The user's message contains an Ozon product link (https://ozon.ru/t/LCabAMR). Read that card.]"
arrive = threading.Thread(target=intent.bind_turn, args=([said], "Вот рюкзаки.", ""), kwargs={"entry": True})
arrive.start(); arrive.join()
def other_threads():
    intent.read(ctx, said)                                   # the debounce loop
    intent.read(ctx, said + note)                            # the task, link note under it
    intent.read(ctx, " ".join((said + note).split()))       # collapsed
    intent.bind_turn([said + note], "", "data with many numbers earlier in the chat")   # the graph
    intent.read(ctx, said + note, "", "data with many numbers earlier in the chat")
t = threading.Thread(target=other_threads); t.start(); t.join()
check("a message read across threads and shapes is one read", len(calls) == 1, calls)
check("...with the context it was bound with on arrival", calls and "Вот рюкзаки." in calls[0], calls)
calls.clear()
intent.bind_turn([said], "Другой ответ.", "", entry=True)   # the same words sent again
intent.read(ctx, said)
check("the same words in a new message are a new read", len(calls) == 1, calls)

# «этот же, но быстрее» went to the analogue shopper and «найди аналоги» to same-item, by
# turns, whatever the two descriptions said: one narrow question picks between them.
asked = []
def model_same(ctx, system, user, **k):
    return json.dumps({"needs_tool": True, "wants": ["ozon_shop", "ozon_reviews"], "must_call": "ozon_shop"})
ctx3 = types.SimpleNamespace(text_model=model_same, yes_no=lambda q, text, default=False, wait=True, ctx=None:
                              asked.append(text) or "этот же" in text)
intent._CACHE.clear(); intent._BY_WORDS.clear(); intent.bind_turn(["x"])
got = intent.read(ctx3, "этот же, но чтобы привезли быстрее https://ozon.ru/t/dw4OyhY")
check("the exact item of a linked card goes to ozon_same",
      got["must_call"] == "ozon_same" and got["wants"][:2] == ["ozon_same", "ozon_shop"], got)
got = intent.read(ctx3, "найди аналоги подешевле https://ozon.ru/t/dw4OyhY")
check("analogues of a linked card go to ozon_shop", got["must_call"] == "ozon_shop", got)
n = len(asked)
intent.read(ctx3, "подбери робот-пылесос")
check("no link, no question", len(asked) == n, asked)

print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
