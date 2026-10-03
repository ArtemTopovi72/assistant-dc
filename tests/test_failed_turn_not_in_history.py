"""A turn that ends in the "I produced no answer" placeholder must not be
committed to the chat history.

Live 2026-09-12 (journey 5): two stuck turns in a row left two UNANSWERED
questions in the transcript ("можно ли с собакой?", "за сколько дней…"), each
followed by "Повтори вопрос, пожалуйста". When the user then asked a third,
different question ("а залог возвращают?") the model answered the OLDEST
open one — about the dog. The user was told to ask again, so nothing of the
failed exchange belongs in the history.
"""
import os, sys, time, threading, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
os.environ.setdefault("F5_TEST_RUN", "1")

_DATA = tempfile.mkdtemp(prefix="tg_failedturn_")
import tg_bot as T
T.redirect_data_dir(_DATA)
import graph as graph_mod
import graph_finalize as GF

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")


class FakeCtx:
    def __init__(self):
        self.session_memory = []; self.pinned_facts = []
        self.cancel_event = threading.Event()
        self.last_image_path = ""; self.last_image_prompt = ""
        self.stage_callback = None; self.total_user_turns = 0
        self.tts_disabled = True; self.memory_lock = threading.Lock()
    def memory_text(self): return ""
    def set_stage(self, s): pass


CTX = FakeCtx()
STUCK = {"можно ли с собакой?", "за сколько дней надо предупредить?"}


class StubGraph:
    def __init__(self, ctx): self.ctx = ctx
    def invoke(self, base):
        text = base.get("user_input", "")
        msgs = list(base.get("messages", []))
        msgs.append({"role": "user", "content": text})
        if text in STUCK:
            # what graph_finalize produces when the model emitted nothing
            msgs.append({"role": "assistant", "content": GF._STUCK_RU})
            return {"final_answer": GF._STUCK_RU, "messages": msgs}
        msgs.append({"role": "assistant", "content": f"echo: {text}"})
        return {"final_answer": f"echo: {text}", "messages": msgs}


graph_mod.build_graph = lambda ctx: StubGraph(ctx)

bot = T.TelegramBot("123:TEST", lambda: CTX, lambda: object(), lambda: {"messages": []}, silent_mode=True)
bot._backend = T.InMemoryBackend()
bot.sent = []
bot._send_text = lambda cid, t, **kw: (bot.sent.append((cid, t)), 1)[1]
bot._send_get_id = lambda cid, t, **kw: (bot.sent.append((cid, t)), len(bot.sent))[1]
bot._edit_text = lambda *a, **kw: None
bot._delete = lambda *a, **kw: None
bot._api_post = lambda *a, **k: {}
bot._api_get = lambda *a, **k: {}
bot._activity.log = lambda *a, **k: None
bot._running = True
th = threading.Thread(target=bot._consumer_loop, daemon=True); th.start()

CID = 777
bot._user_store.put(T._User(chat_id=CID, name="Tester", status="approved", is_admin=False))
sess = bot._get_session(CID); sess.clear_context(); sess.lang = "ru"; sess.reg_state = ""; bot._store.put(sess)

_uid = [30_000_000]
def send(text):
    _uid[0] += 1
    n_before = len(bot.sent)
    bot._dispatch({"update_id": _uid[0],
                        "message": {"message_id": _uid[0], "chat": {"id": CID, "type": "private"},
                                    "from": {"id": CID}, "text": text, "date": int(time.time())}})
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if len(bot.sent) > n_before and not (bot._running_task.get(CID) or []):
            break
        time.sleep(0.1)
    time.sleep(0.3)


check("placeholder is recognised", GF.is_failure_placeholder(GF._STUCK_RU) and GF.is_failure_placeholder(GF._STUCK_EN) and GF.is_failure_placeholder(""))
check("a real answer is not", not GF.is_failure_placeholder("Залог возвращается при выезде."))

send("привет")
h1 = bot._get_session(CID).get_history()
check("a normal turn is committed (user + assistant)", len(h1) == 2, len(h1))

send("можно ли с собакой?")
send("за сколько дней надо предупредить?")
h2 = bot._get_session(CID).get_history()
check("two stuck turns add NOTHING to the history", len(h2) == 2, [m.get("content", "")[:40] for m in h2])
check("the user still got the honest placeholder each time",
      sum(1 for c, t in bot.sent if GF._STUCK_RU in t) == 2, bot.sent)

send("а залог возвращают?")
h3 = bot._get_session(CID).get_history()
check("the next good turn lands on a history free of open questions",
      len(h3) == 4 and h3[2]["content"] == "а залог возвращают?", [m.get("content", "")[:40] for m in h3])
check("no stuck question survives in the transcript",
      not any(m.get("content") in STUCK for m in h3))

bot._running = False
print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
