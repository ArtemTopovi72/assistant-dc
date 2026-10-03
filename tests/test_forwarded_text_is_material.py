"""A forwarded TEXT reaches the agent framed as someone else's words.

Live 2026-09-13 (mega journey, journey 25 inside the long chat): a forwarded
channel post ("Внимание! В субботу ... перекрывают движение") went in as if
the user had said it; with document search on the bot answered "the documents
say nothing about a road closure", and "о чём это?" afterwards was answered
about the last picture. Forwarded text is now flagged at dispatch, framed as
a quotation by the resolver, and never wrapped into the document prompt.
"""
import os, sys, time, threading, tempfile
os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA = tempfile.mkdtemp(prefix="fwdtext_")
import logging; logging.basicConfig(level=logging.CRITICAL)
import tg_bot as T
T.redirect_data_dir(_DATA)
import graph as graph_mod
import tg_tasks

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

INVOKES = []
class FakeCtx:
    def __init__(self):
        self.session_memory = []; self.pinned_facts = []
        self.cancel_event = threading.Event(); self.last_image_path = ""
        self.last_image_prompt = ""; self.stage_callback = None
        self.total_user_turns = 0; self.tts_disabled = True
        self.memory_lock = threading.Lock()
    def memory_text(self): return ""
    def set_stage(self, s): pass
class StubGraph:
    def __init__(self, ctx): self.ctx = ctx
    def invoke(self, base):
        INVOKES.append(base.get("user_input", ""))
        return {"final_answer": "echo", "messages": base.get("messages", [])}
graph_mod.build_graph = lambda ctx: StubGraph(ctx)
SHARED = FakeCtx()

bot = T.TelegramBot("123:TEST", lambda: SHARED, lambda: object(), lambda: {"messages": []}, silent_mode=True)
bot._backend = T.InMemoryBackend(); bot.sent = []
bot._send_text = lambda cid, t, **kw: (bot.sent.append((cid, t)), 1)[1]
bot._send_get_id = lambda cid, t, **kw: (bot.sent.append((cid, t)), len(bot.sent))[1]
bot._edit_text = lambda *a, **kw: None; bot._delete = lambda *a, **kw: None
bot._api_post = lambda *a, **k: {}; bot._api_get = lambda *a, **k: {}
bot._activity.log = lambda *a, **k: None
bot._running = True
thr = threading.Thread(target=bot._consumer_loop, daemon=True); thr.start()

CID = 9_300_001
bot._user_store.put(T._User(chat_id=CID, name="Fwd", status="approved", is_admin=False))
sess = bot._get_session(CID); sess.clear_context(); sess.lang = "ru"; sess.reg_state = ""; bot._store.put(sess)

_seq = [40_000_000]
def _nid(): _seq[0] += 1; return _seq[0]
def upd(text, forwarded=False):
    m = {"message_id": _nid(), "chat": {"id": CID, "type": "private"}, "from": {"id": CID},
         "text": text, "date": int(time.time())}
    if forwarded:
        m["forward_from_chat"] = {"id": -1001234567, "type": "channel", "title": "Новости города"}
        m["forward_date"] = int(time.time()) - 3600
    return {"update_id": _nid(), "message": m}
def settle(pred, timeout=8.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred(): return True
        time.sleep(0.1)
    return pred()

POST = "Внимание! В субботу 20 сентября перекрывают движение из-за марафона."
bot._dispatch(upd(POST, forwarded=True))
check("the forwarded post reaches the agent", settle(lambda: len(INVOKES) >= 1))
got = INVOKES[-1] if INVOKES else ""
check("...framed as someone else's words", got.startswith("<<<QUOTED MESSAGE -- the user forwarded this message"), got[:80])
check("...with the post quoted verbatim", POST in got)
check("the frame is a RAG bypass", tg_tasks._plainly_not_a_doc_question(got))

settle(lambda: any("echo" in t for _, t in bot.sent)); time.sleep(0.5)
bot._dispatch(upd("о чём это?"))
check("a plain message is still bare", settle(lambda: len(INVOKES) >= 2) and INVOKES[-1] == "о чём это?", INVOKES[-1:])

bot._running = False; thr.join(timeout=8)
print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
