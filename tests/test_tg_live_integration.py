"""Live integration checks for the tg_bot hardening pass.

Needs LM Studio up (embeddings). Exercises the code paths the offline suite can
only stub: real document indexing + retrieval, the full _run_task_inner turn
(fact/memory/image swaps, RAG injection, cancellation, delivery), and the
/status health probe.
"""
import os
import sys
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tg_bot as T

# Before any bot is built — otherwise fixture users land in the LIVE store.
import tempfile
_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_live_")
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1
        print(f"PASS  {name}")
    else:
        BAD += 1
        print(f"FAIL  {name}   {extra}")


CID = 999201
SAMPLE = """The Kessler Bridge was completed in 1904 by the engineer Marta Ilves.
It spans 412 metres across the Varda river and carries two tram lines.
The bridge was closed for restoration between 1998 and 2001.
Its distinctive green railings were added in 1954."""


import models as _models


class Ctx(_models.Context):
    """The real Context, with only the bits this suite needs pre-set.

    It used to be a hand-rolled stand-in listing a handful of attributes. That
    drifts: the agent core grew `facts_text()` and `custom_personality_text`, the
    stub did not, and SIX checks failed with "'Ctx' object has no attribute …" —
    failures that said nothing about the behaviour under test and hid whatever
    those checks were actually meant to catch. Deriving from the real class means
    this suite tracks the real contract for free and can only fail for real
    reasons. Nothing here loads a model: Context(models=None) is inert.
    """

    def __init__(self):
        super().__init__(models=None, transcription_cache={}, cache_file=None,
                         asr_lock=threading.Lock(), tts_lock=threading.Lock())
        # A real Context defaults model_name to "", which the bot now reads as
        # the deliberate "started without a model" state and refuses every turn
        # for. A running app always has a name here, so say so.
        self.model_name = "house-model"
        self.tts_disabled = False
        self.last_image_path = None
        self.last_image_prompt = ""
        self.stage_callback = None
        self.total_user_turns = 0

    def set_stage(self, s):
        pass


def make_bot(ctx, graph):
    bot = T.TelegramBot("123:TEST", lambda: ctx, lambda: graph,
                        lambda: {"messages": []}, silent_mode=True)
    bot.sent = []
    bot._send_text = lambda cid, text, **kw: bot.sent.append(text) or 1
    bot._send_get_id = lambda cid, text, **kw: (bot.sent.append(text), 1)[1]
    bot._edit_text = lambda *a, **kw: None
    bot._delete = lambda *a, **kw: None
    bot._send_document = lambda *a, **kw: True
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    return bot


def texts(bot):
    return " || ".join(bot.sent)


print("=" * 62)
print("REAL DOCUMENT INDEXING + RETRIEVAL")
print("=" * 62)

ctx = Ctx()
seen_prompt = {}


class Graph:
    """Reports on the context it was BUILT with, not the module-level one.

    That distinction is the whole subject of this file: _execute_task scopes a
    context per task and builds the graph around it, so a graph closing over
    the shared `ctx` would see the desktop app's facts and write its own fact
    back into them -- which is the bug the scoping exists to prevent, and a
    fake that does it cannot tell the two apart.
    """
    def __init__(self, bound_ctx=None):
        self.ctx = bound_ctx if bound_ctx is not None else ctx

    def invoke(self, base):
        c = self.ctx
        seen_prompt["user_input"] = base["user_input"]
        seen_prompt["facts"] = list(c.pinned_facts)
        seen_prompt["memory"] = list(c.session_memory)
        c.pinned_facts.append({"ts": 1, "text": "fact written by this turn"})
        return {"messages": base.get("messages", []),
                "final_answer": "answered"}


bot = make_bot(ctx, Graph())
# A task builds its OWN graph now -- build_graph closes over the context, and
# the prebuilt one would run every node against the shared context, which is
# the isolation this file exists to check. So the graph handed to make_bot is
# no longer the one _execute_task runs, and without this the REAL model
# answered the question: five checks looked at a prompt nobody had captured.
import graph as _graph_mod
import intent as _intent
_intent.STUB = lambda t: {"doc_question": "Kessler" in t}
_graph_mod.build_graph = lambda _ctx: Graph(_ctx)
sess = bot._get_session(CID)
sess.clear_context()
sess.set_tg_facts([])
sess.lang = "en"
bot._store.put(sess)

# wipe any previous library for this test chat
bot._clear_library(CID)
p = bot._library_path(CID)
if p.exists():
    try:
        p.unlink()
    except Exception:
        pass

import tempfile
import os

tmpdir = tempfile.mkdtemp(prefix="tglive_")
doc = os.path.join(tmpdir, "kessler_bridge.txt")
with open(doc, "w", encoding="utf-8") as fh:
    fh.write(SAMPLE)

bot._index_document(CID, sess, doc, "kessler_bridge.txt")
check("indexing reported success", "indexed" in texts(bot).lower(), texts(bot)[:300])
check("document search auto-enabled after an upload", sess.use_docs is True)

st = bot._library_stats(CID)
check("document appears in the library", st["documents"] == 1, st)
check("document produced passages", st["chunks"] >= 1, st)

bot.sent.clear()
bot._send_library_list(CID, sess)
check("library listing names the document", "kessler_bridge" in texts(bot), texts(bot)[:200])

print()
print("=" * 62)
print("FULL TURN — RAG injection, fact isolation, delivery")
print("=" * 62)

# GUI-side state that must survive the turn untouched
ctx.session_memory.extend(["GUI MEMORY"])
ctx.pinned_facts.extend([{"ts": 0, "text": "GUI FACT — must not leak to Telegram"}])
ctx.last_image_prompt = "gui prompt"
sess.set_tg_facts([{"ts": 1, "text": "telegram fact"}])
sess.last_image_prompt = "tg prompt"
bot._store.put(sess)

bot.sent.clear()
task = T._Task(task_id="live1", chat_id=CID,
               user_text="Who built the Kessler Bridge and when?")
bot._execute_task(task)

check("retrieved passages were injected into the prompt",
      "Kessler Bridge" in seen_prompt.get("user_input", "")
      and "Retrieved passages" in seen_prompt.get("user_input", ""),
      seen_prompt.get("user_input", "")[:200])
check("the agent saw only the Telegram facts",
      seen_prompt.get("facts") == [{"ts": 1, "text": "telegram fact"}],
      seen_prompt.get("facts"))
check("the agent saw only the Telegram memory",
      "GUI MEMORY" not in str(seen_prompt.get("memory")), seen_prompt.get("memory"))
check("GUI facts restored after the turn",
      ctx.pinned_facts == [{"ts": 0, "text": "GUI FACT — must not leak to Telegram"}],
      ctx.pinned_facts)
# session_memory is a deque, and deque(["x"]) != ["x"].
check("GUI memory restored after the turn",
      list(ctx.session_memory) == ["GUI MEMORY"], ctx.session_memory)
check("GUI image prompt restored", ctx.last_image_prompt == "gui prompt",
      ctx.last_image_prompt)
check("fact written during the turn is saved to the session",
      any(f.get("text") == "fact written by this turn"
          for f in bot._get_session(CID).get_tg_facts()),
      bot._get_session(CID).get_tg_facts())
check("reply was delivered", "answered" in texts(bot), texts(bot)[:200])

# same question with document search OFF must not carry passages
sess = bot._get_session(CID)
sess.use_docs = False
bot._store.put(sess)
seen_prompt.clear()
bot.sent.clear()
bot._execute_task(T._Task(task_id="live2", chat_id=CID, user_text="plain question"))
check("no RAG wrapping when document search is off",
      seen_prompt.get("user_input") == "plain question",
      seen_prompt.get("user_input", "")[:120])

# a cancelled turn must not deliver the half-finished answer
sess.use_docs = False
bot._store.put(sess)
bot.sent.clear()
task3 = T._Task(task_id="live3", chat_id=CID, user_text="cancel me")
bot._cancelled.add("live3")
bot._execute_task(task3)
check("cancelled turn reports cancellation instead of the answer",
      "cancelled" in texts(bot).lower() and "answered" not in texts(bot),
      texts(bot)[:200])

print()
print("=" * 62)
print("HEALTH PROBE")
print("=" * 62)

bot.sent.clear()
bot._send_status(CID, bot._get_session(CID))
out = texts(bot)
check("status reports the LLM endpoint", "LM Studio" in out, out[:300])
check("status reports ComfyUI", "ComfyUI" in out, out[:300])
check("status reports queue and usage", "Queue" in out and "usage" in out.lower(),
      out[:400])
check("status shows the document count", "documents" in out, out[:400])

# cleanup
bot._clear_library(CID)
import shutil

shutil.rmtree(tmpdir, ignore_errors=True)

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
