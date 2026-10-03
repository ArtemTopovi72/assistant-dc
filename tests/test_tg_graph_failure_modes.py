"""Adversarial coverage of the graph.invoke() failure-handling contract.

The agent core (graph.py / tools.py) is a separate layer from the queueing /
delivery machinery, and the existing harnesses stub it out entirely for
determinism -- so nothing exercises what happens when THAT layer misbehaves.
This drives the REAL TelegramBot (real consumer threads, real _execute_task /
_run_task_inner, real InMemoryBackend) with a StubGraph forced to misbehave
four ways, and checks the user is never left in silence and the chat/consumer
pool is never permanently wedged:

  1. graph.invoke() raises a plain Exception partway through.
  2. graph.invoke() returns a malformed/unexpected result shape (None, a bare
     string, an empty dict, a dict with a None value where an image path is
     expected).
  3. graph.invoke() hangs indefinitely. graph.py/tools.py never check
     ctx.cancel_event on the ordinary-turn path (only deep_research's own
     progress loop does), so this used to have NO enforced ceiling anywhere:
     the consumer thread blocked forever, and once _task_started's slot never
     cleared, that chat's admission gate (tg_queue.py's `_chat_busy`) was
     wedged permanently too. Fixed by running graph.invoke() on its own
     thread and joining it with config.TG_INVOKE_TIMEOUT_S.
  4. graph.invoke() succeeds with a false "done" claim -- final_answer says
     the artifact is ready but image_status/document_status say otherwise, or
     the reported path does not exist on disk. Same shape as the historical
     "bot claims an edit but sends no image" bug; checked here for both
     images and documents.

Run: venv/Scripts/python.exe tests/test_tg_graph_failure_modes.py
"""
import os, sys, time, threading, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

_DATA = tempfile.mkdtemp(prefix="tg_failmode_")
import tg_bot as T
T.redirect_data_dir(_DATA)
import graph as graph_mod
import config as _cfg

OK = BAD = 0
FAILURES = []


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1
        print(f"PASS  {name}")
    else:
        BAD += 1
        FAILURES.append(name)
        print(f"FAIL  {name}   {extra}")


def section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


class FakeCtx:
    def __init__(self):
        self.session_memory = []
        self.pinned_facts = []
        self.cancel_event = threading.Event()
        self.last_image_path = ""
        self.last_image_prompt = ""
        self.stage_callback = None
        self.total_user_turns = 0
        self.tts_disabled = True
        self.memory_lock = threading.Lock()

    def memory_text(self): return ""
    def set_stage(self, s): pass


SHARED_CTX = FakeCtx()
BEHAVIOR = {}   # user_input text -> callable(base) -> result | raises | blocks


class StubGraph:
    def __init__(self, ctx):
        self.ctx = ctx

    def invoke(self, base):
        text = base.get("user_input", "")
        fn = BEHAVIOR.get(text)
        if fn is None:
            return {"final_answer": f"echo: {text}", "messages": base.get("messages", [])}
        return fn(base)


def _stub_build_graph(ctx):
    return StubGraph(ctx)


graph_mod.build_graph = _stub_build_graph


def make_bot(n_consumers=2):
    bot = T.TelegramBot("123:TEST", lambda: SHARED_CTX, lambda: object(),
                        lambda: {"messages": []}, silent_mode=True,
                        # Voice off: this suite tests graph failures, not TTS.
                        # With it on, the echo reply cold-imports f5_tts/torch
                        # (~10s+) before the text goes out -- flaky vs settle().
                        tts_enabled_fn=lambda: False)
    bot._backend = T.InMemoryBackend()
    bot.sent = []
    bot._send_text = lambda cid, t, **kw: (bot.sent.append((cid, t, kw.get("keyboard"))), 1)[1]
    bot._send_get_id = lambda cid, t, **kw: (bot.sent.append((cid, t, kw.get("keyboard"))), len(bot.sent))[1]
    bot._edit_text = lambda *a, **kw: None
    bot._delete = lambda *a, **kw: None
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._dl_bytes = lambda fid: b"\xff\xd8\xff" + b"0" * 100
    bot._running = True
    bot._consumers = [threading.Thread(target=bot._consumer_loop, daemon=True,
                                       name=f"c{i}") for i in range(n_consumers)]
    for t in bot._consumers:
        t.start()
    return bot


def stop_bot(bot):
    bot._running = False
    for t in bot._consumers:
        t.join(timeout=8)


def approve(bot, cid, name="Tester"):
    bot._user_store.put(T._User(chat_id=cid, name=name, status="approved", is_admin=False))
    sess = bot._get_session(cid)
    sess.clear_context(); sess.lang = "en"; sess.reg_state = ""
    bot._store.put(sess)
    return sess


_UPDATE_ID_SEQ = [20_000_000]


def _next_update_id():
    _UPDATE_ID_SEQ[0] += 1
    return _UPDATE_ID_SEQ[0]


def msg_update(cid, text):
    return {"update_id": _next_update_id(),
            "message": {"message_id": _next_update_id(),
                        "chat": {"id": cid, "type": "private"},
                        "from": {"id": cid}, "text": text,
                        "date": int(time.time())}}


def settle(predicate, timeout=10.0, interval=0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def texts_for(bot, cid):
    return [t for c, t, k in bot.sent if c == cid]


def n_running(bot, cid):
    return len(bot._running_task.get(cid) or [])


def settle_done(bot, cid, done_timeout=15.0):
    """Wait until the task has visibly completed for this chat.

    n_running==0 is trivially true before the debounced task has even been
    admitted, so settling on THAT alone races the debounce window. These stub
    behaviors (raise / None / bad shape) resolve near-instantly once admitted
    -- often within a single poll tick -- so gating on an observed
    running>=1 transition first is itself flaky for them (the task can start
    and finish between two 0.1s polls). Instead wait for the debounce window
    to have plausibly elapsed (TG_DEBOUNCE_S) AND running==0 AND at least one
    reply landed for this chat -- three independent signals that the task
    genuinely ran to completion, not that it simply never started.
    """
    time.sleep(getattr(T, "_DEBOUNCE_S", 1.8) + 0.2)
    return settle(lambda: n_running(bot, cid) == 0 and len(texts_for(bot, cid)) > 0,
                  timeout=done_timeout)


# ══════════════════════════════════════════════════════════════════════════
section("1. graph.invoke() raises a plain Exception partway through")
# ══════════════════════════════════════════════════════════════════════════
CID1 = 910001
bot = make_bot()
approve(bot, CID1)
BEHAVIOR["boom"] = lambda base: (_ for _ in ()).throw(RuntimeError("tool exploded"))
bot._dispatch(msg_update(CID1, "boom"))
ok = settle_done(bot, CID1)
check("consumer thread survives an invoke() exception (task slot clears)", ok)
check("all consumer threads are still alive after the crash",
      all(t.is_alive() for t in bot._consumers),
      [t.is_alive() for t in bot._consumers])
check("the user gets a clear failure message, not silence",
      any("Error" in t or "Ошибка" in t for t in texts_for(bot, CID1)),
      texts_for(bot, CID1))
check("a retry button is offered",
      any(k and "retry:" in str(k) for c, _, k in bot.sent if c == CID1),
      [k for c, t, k in bot.sent if c == CID1])
bot.sent.clear()
bot._dispatch(msg_update(CID1, "still alive?"))
ok2 = settle(lambda: any("echo: still alive?" in t for t in texts_for(bot, CID1)))
check("the chat is not wedged after the crash -- next message still answers",
      ok2, texts_for(bot, CID1))
stop_bot(bot)
print(f"\n{OK}/{OK+BAD} passed so far")


# ══════════════════════════════════════════════════════════════════════════
section("2. graph.invoke() returns a malformed/unexpected result shape")
# ══════════════════════════════════════════════════════════════════════════

# 2a. returns None outright
CID2A = 910010
bot = make_bot()
approve(bot, CID2A)
BEHAVIOR["give me none"] = lambda base: None
bot._dispatch(msg_update(CID2A, "give me none"))
ok = settle_done(bot, CID2A)
check("2a. task slot clears when invoke() returns None", ok)
check("2a. consumer threads survive a None result",
      all(t.is_alive() for t in bot._consumers))
check("2a. the user gets a clear failure message, not silence",
      any("Error" in t or "Ошибка" in t for t in texts_for(bot, CID2A)),
      texts_for(bot, CID2A))
stop_bot(bot)

# 2b. returns a bare string instead of a dict
CID2B = 910011
bot = make_bot()
approve(bot, CID2B)
BEHAVIOR["give me a string"] = lambda base: "not a dict"
bot._dispatch(msg_update(CID2B, "give me a string"))
ok = settle_done(bot, CID2B)
check("2b. task slot clears when invoke() returns a bare string", ok)
check("2b. consumer threads survive", all(t.is_alive() for t in bot._consumers))
check("2b. the user gets a clear failure message, not silence",
      any("Error" in t or "Ошибка" in t for t in texts_for(bot, CID2B)),
      texts_for(bot, CID2B))
stop_bot(bot)

# 2c. dict missing every expected key -- must not crash, some reply goes out
CID2C = 910012
bot = make_bot()
approve(bot, CID2C)
BEHAVIOR["give me empty dict"] = lambda base: {}
bot._dispatch(msg_update(CID2C, "give me empty dict"))
check("2c. an empty dict does not crash delivery -- some reply is sent",
      settle(lambda: len(texts_for(bot, CID2C)) > 0, timeout=8),
      texts_for(bot, CID2C))
stop_bot(bot)

# 2d. malformed image delivery -- image_status success but image_path is None
CID2D = 910013
bot = make_bot()
approve(bot, CID2D)
BEHAVIOR["make a picture of a cat"] = lambda base: {
    "final_answer": "done!", "messages": [],
    "image_path": None, "image_status": "success"}
bot._dispatch(msg_update(CID2D, "make a picture of a cat"))
ok = settle_done(bot, CID2D)
check("2d. a None image_path with status=success does not crash delivery", ok)
check("2d. consumer threads survive a None image_path",
      all(t.is_alive() for t in bot._consumers))
stop_bot(bot)

print(f"\n{OK}/{OK+BAD} passed so far")


# ══════════════════════════════════════════════════════════════════════════
section("3. graph.invoke() hangs indefinitely -- is there an ENFORCED timeout?")
# ══════════════════════════════════════════════════════════════════════════
# Use a short TG_INVOKE_TIMEOUT_S so the test doesn't take the full production
# default.
_orig_timeout = _cfg.TG_INVOKE_TIMEOUT_S
_cfg.TG_INVOKE_TIMEOUT_S = 2

# The deadline now EXTENDS while the render server is genuinely working, so a
# picture queued behind a ~70-minute music job is not killed for waiting its
# turn (see tests/test_tg_invoke_deadline.py). That means this section has to
# say which world it is testing: here the call is truly hung and the server is
# idle, so the deadline must fire. Without this stub the check would ask the
# REAL ComfyUI -- making an offline suite depend on whether the machine happens
# to be rendering, which is exactly how it went red on a healthy box.
import comfy_client as _comfy_stub
_orig_busy = _comfy_stub.server_busy
_comfy_stub.server_busy = lambda: False

_release = threading.Event()   # never set -- simulates a truly hung call


def _hang(base):
    _release.wait()
    return {"final_answer": "should never be seen", "messages": []}


CID3 = 910020
bot = make_bot()
approve(bot, CID3)
BEHAVIOR["hang forever"] = _hang
bot._dispatch(msg_update(CID3, "hang forever"))

t0 = time.monotonic()
ok = settle_done(bot, CID3, done_timeout=15.0)
elapsed = time.monotonic() - t0
check("3. a hung invoke() is abandoned near TG_INVOKE_TIMEOUT_S, not never",
      ok and elapsed < 10, f"elapsed={elapsed:.1f}s running={n_running(bot, CID3)}")
check("3. the consumer thread is freed (not permanently blocked) after the hang",
      all(t.is_alive() for t in bot._consumers))
check("3. the user is told the request timed out, not left in silence",
      any("took too long" in t.lower() or "долго" in t.lower()
          for t in texts_for(bot, CID3)),
      texts_for(bot, CID3))

bot.sent.clear()
bot._dispatch(msg_update(CID3, "are you back"))
ok2 = settle(lambda: any("echo: are you back" in t for t in texts_for(bot, CID3)), timeout=8)
check("3. the chat is NOT permanently wedged -- a later message still gets answered",
      ok2, texts_for(bot, CID3))

_release.set()   # let the abandoned background thread finish (best-effort
                 # cleanup; nothing reads its result any more)
_cfg.TG_INVOKE_TIMEOUT_S = _orig_timeout
_comfy_stub.server_busy = _orig_busy
stop_bot(bot)
print(f"\n{OK}/{OK+BAD} passed so far")


# ══════════════════════════════════════════════════════════════════════════
section("4. false 'done' claim -- partial-success honesty")
# ══════════════════════════════════════════════════════════════════════════

# 4a. image: model claims success, tool reports failure via image_status
CID4A = 910030
bot = make_bot()
approve(bot, CID4A)
BEHAVIOR["draw me a cat"] = lambda base: {
    "final_answer": "Here is your cat!", "messages": [],
    "image_path": "", "image_status": "fail"}
bot._dispatch(msg_update(CID4A, "draw me a cat"))
ok = settle_done(bot, CID4A)
check("4a. a false-success image claim is corrected with an explicit "
      "'missing' notice, not left as a bare 'Here is your cat!'",
      any(T._t("img_missing", "en").split("<")[0][:10] in t
          for t in texts_for(bot, CID4A)),
      texts_for(bot, CID4A))
stop_bot(bot)

# 4b. document: model claims success, document_status=success but the file
# path does not exist on disk (tool lied about writing it)
CID4B = 910031
bot = make_bot()
approve(bot, CID4B)
BEHAVIOR["create a presentation about: bees"] = lambda base: {
    "final_answer": "Your presentation is ready!", "messages": [],
    "document_path": os.path.join(_DATA, "nonexistent_deck.pptx"),
    "document_status": "success"}
bot._dispatch(msg_update(CID4B, "create a presentation about: bees"))
ok = settle_done(bot, CID4B)
check("4b. a false-success document claim (path reported but missing on "
      "disk) is corrected, not left as a bare success claim",
      any("failed" in t.lower() or "missing" in t.lower() or "no file" in t.lower()
          or "не удал" in t.lower() or "нет" in t.lower() for t in texts_for(bot, CID4B)),
      texts_for(bot, CID4B))
stop_bot(bot)

print(f"\n{OK}/{OK+BAD} passed so far")
print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
