"""📚 Audiobook flow in the bot, and its ISOLATION from the document library (RAG), the sandbox and the agent.

Armed: the voice sample, then the book, are consumed by the button -- the library is not indexed, nothing
reaches the agent. Not armed (or after another button): a document goes the ordinary way, untouched.
The voice is a fake: a tone per letter."""
import os, sys, time, threading, tempfile, subprocess
os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA = tempfile.mkdtemp(prefix="abflow_")
import logging; logging.basicConfig(level=logging.CRITICAL)
import tg_bot as T
T.redirect_data_dir(_DATA)
import graph as graph_mod
import voice_clone

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

BOOK = os.path.join(_DATA, "book.txt")
SENT = "Он шёл по улице. Было холодно, и снег падал медленно. Она ждала у окна. "
open(BOOK, "w", encoding="utf-8").write(
    "Глава 1\nНачало\n\n" + SENT * 12 + "\n\nГлава 2\n\n" + SENT * 12)
SAMPLE = os.path.join(_DATA, "sample.wav")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=200:duration=12", SAMPLE], check=True)

INVOKES, INDEXED, SPOKEN, VOICES = [], [], [], []
class FakeCtx:
    def __init__(self):
        self.session_memory = []; self.pinned_facts = []
        self.cancel_event = threading.Event(); self.last_image_path = ""
        self.last_image_prompt = ""; self.stage_callback = None
        self.total_user_turns = 0; self.tts_disabled = True
        self.memory_lock = threading.Lock()
    def memory_text(self): return ""
    def set_stage(self, s): pass
    def is_cancelled(self): return self.cancel_event.is_set()
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
bot._index_document = lambda *a, **k: INDEXED.append(a)
CAPS = []
bot._send_voice_from_wav = lambda cid, wav, keyboard=None, caption="": (VOICES.append(wav), CAPS.append(caption), True)[1]
bot._running = True
threading.Thread(target=bot._consumer_loop, daemon=True).start()

def fake_prepare(ctx, src, out_dir, lang="ru"):
    return SAMPLE, "образец голоса"
def fake_speak(ctx, ref, ref_text, text, out_dir):
    SPOKEN.append((ref, text))
    wav = os.path.join(out_dir, f"s{len(SPOKEN)}.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"sine=frequency=300:duration={max(0.5, len(text) * 0.06):.2f}", wav], check=True)
    return wav
voice_clone.prepare_reference = fake_prepare
voice_clone.speak = fake_speak

CID = 9_300_099
bot._user_store.put(T._User(chat_id=CID, name="Bk", status="approved", is_admin=False))
sess = bot._get_session(CID); sess.clear_context(); sess.lang = "ru"; sess.lang_chosen = True; sess.reg_state = ""; bot._store.put(sess)
bot._dl_bytes = lambda fid, *a, **k: open(BOOK if fid.startswith("BK") else SAMPLE, "rb").read()
_seq = [44_000_000]
def _nid(): _seq[0] += 1; return _seq[0]
def m(**kw):
    d = {"message_id": _nid(), "chat": {"id": CID, "type": "private"}, "from": {"id": CID}, "date": int(time.time())}
    d.update(kw); return {"update_id": _nid(), "message": d}
def wait(pred, t=20):
    e = time.monotonic() + t
    while time.monotonic() < e:
        if pred(): return True
        time.sleep(0.1)
    return pred()
def doc(name, fid="BK1"): return {"file_id": fid, "file_name": name, "mime_type": "text/plain", "file_size": 1000}
st = lambda: bot._get_session(CID).book_state

# not armed: a document goes the ordinary way (the button does not eat it)
seen_items = []
orig_enqueue = bot._enqueue_item
bot._enqueue_item = lambda cid, item: (seen_items.append(item), orig_enqueue(cid, item))[1]
bot._dispatch(m(document=doc("notes.txt")))
check("NOT armed: a .txt document is an ordinary document item (library/sandbox path intact)",
      any(i.get("type") == "document" for i in seen_items) and not VOICES and not SPOKEN, seen_items[-1:])
check("...and it was indexed as before", wait(lambda: len(INDEXED) >= 1, 15), INDEXED)
INDEXED.clear()

# armed
bot._dispatch(m(text="📚 Аудиокнига"))
check("the button asks for the narrator's voice", wait(lambda: st() == "want_voice"), bot.sent[-2:])
n_items = len(seen_items)
bot._dispatch(m(document=doc("notes.txt")))
wait(lambda: False, 1)
check("armed for the VOICE: a book file is not read as a voice and does not reach the library either",
      st() == "want_voice" and not INDEXED and len(seen_items) == n_items, (st(), INDEXED))
bot._dispatch(m(voice={"file_id": "VO1", "duration": 12, "mime_type": "audio/ogg"}))
check("the voice sample is taken: now asks for the book", wait(lambda: st() == "want_book"), bot.sent[-2:])
check("its reference is stored for this chat", bool(bot._get_session(CID).book_ref))

bot._dispatch(m(text="привет, как дела"))
wait(lambda: False, 1)
check("a short text while waiting for the book is a hint, never the agent", st() == "want_book" and not INDEXED and
      not any("привет" in i for i in INVOKES), INVOKES[-2:])
bot._dispatch(m(document={"file_id": "PK1", "file_name": "photo.exe", "mime_type": "application/octet-stream", "file_size": 10}))
wait(lambda: False, 1)
check("a non-book file is refused and the mode stays armed", st() == "want_book" and not INDEXED)
bot._dispatch(m(document=dict(doc("huge.txt"), file_size=5 * 1024 ** 3)))
wait(lambda: False, 1)
check("a file over the bot's receive limit is refused by name, the mode stays armed",
      st() == "want_book" and any("МБ" in t and "принимаю до" in t for _, t in bot.sent[-3:]), bot.sent[-2:])
n_items = len(seen_items)
bot._dispatch(m(document=doc("Война и мир.txt")))
check("the book is read: voice messages arrive, one per chapter", wait(lambda: len(VOICES) >= 2, 40), (len(VOICES), bot.sent[-3:]))
check("each voice message carries a caption: which chapter of how many, and its name",
      len(CAPS) >= 2 and "1/2" in CAPS[0] and "Начало" in CAPS[0] and "2/2" in CAPS[1], CAPS)
check("every piece used the SAME reference voice (the narrator copy kept for resuming)", len({r for r, _ in SPOKEN}) == 1
      and os.path.basename(SPOKEN[0][0]).startswith("narrator"), {r for r, _ in SPOKEN})
check("chapter 1 opens with its spoken title, chapter 2 too",
      SPOKEN and SPOKEN[0][1].startswith("Глава 1. Начало") and any(t.startswith("Глава 2") for _, t in SPOKEN))
check("ISOLATION: the book was not indexed, not queued as a document, not given to the agent",
      not INDEXED and len(seen_items) == n_items and not any("Он шёл" in i for i in INVOKES), (INDEXED, INVOKES[-1:]))
check("the mode ended: a book file afterwards is an ordinary document again", wait(lambda: st() == "", 5))
bot._dispatch(m(document=doc("again.txt")))
wait(lambda: False, 1)
check("...and goes the ordinary way", len(seen_items) > n_items)
check("a finished note is sent", wait(lambda: any("Готово" in t for _, t in bot.sent), 10), bot.sent[-3:])
_bd = os.path.join(_DATA, "audiobook", str(CID))
check("no chapter files are left behind", not [d for d in os.listdir(_bd) if d.startswith("run_")], os.listdir(_bd))

# another button leaves the mode
bot._dispatch(m(text="📚 Аудиокнига"))
wait(lambda: st() == "want_voice")
bot._dispatch(m(text="🎙 Клонировать голос"))
check("another button disarms the audiobook", wait(lambda: st() == ""), st())


# resume after a restart: the job file says chapter 1 is done, the button offers to finish
import json
check("a finished book leaves no job file", not os.path.exists(os.path.join(_bd, "job.json")))
nar = os.path.join(_bd, "narrator.wav")
json.dump({"chapters": [["Глава 1", "Раз. " * 40], ["Глава 2", "Два. " * 40]], "ref": nar, "ref_text": "образец", "next": 1},
          open(os.path.join(_bd, "job.json"), "w", encoding="utf-8"), ensure_ascii=False)
VOICES.clear(); CAPS.clear(); SPOKEN.clear()
kb = []
bot._send_text = lambda cid, t, **kw: (bot.sent.append((cid, t)), kb.append(kw.get("keyboard")), 1)[1]
bot._dispatch(m(text="📚 Аудиокнига"))
check("an unfinished book: the button offers to finish it (1 chapter left)",
      wait(lambda: any("недочитанная" in t and "осталось глав 1" in t for _, t in bot.sent[-3:]), 10), bot.sent[-2:])
check("...with Finish / New buttons", any(k and "bk:resume" in json.dumps(k) and "bk:new" in json.dumps(k) for k in kb[-3:]))
bot._handle_callback(CID, "bk:resume") if hasattr(bot, "_handle_callback") else bot._cb_book(CID, "bk:resume")
check("resuming voices ONLY the remaining chapter", wait(lambda: len(VOICES) == 1, 40) and CAPS and "2/2" in CAPS[0]
      and SPOKEN and SPOKEN[0][1].startswith("Глава 2"), (CAPS, SPOKEN[:1]))
check("and the job file is gone afterwards", wait(lambda: not os.path.exists(os.path.join(_bd, "job.json")), 10))


# a YouTube link as the narrator's sample (live 10-04: it was retold as a video instead)
import tg_links
tg_links.fetch_video = lambda url, *a, **k: {"data": open(SAMPLE, "rb").read(), "seconds": 12, "title": "t", "description": ""}
before = len(INVOKES)
bot._dispatch(m(text="📚 Аудиокнига"))
check("the button is armed again for a new narrator", wait(lambda: st() == "want_voice", 10), st())
bot._dispatch(m(text="https://www.youtube.com/watch?v=UhPKSA2UhnU"))
check("a video link is taken as the narrator's sample (then asks for the book), not retold",
      wait(lambda: st() == "want_book", 20) and len(INVOKES) == before, (st(), INVOKES[before:]))

print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
