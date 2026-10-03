"""A forwarded voice/video note landing in the same debounce window as
another item silently ate the whole batch.

_resolve_and_push's per-item loop `return`s unconditionally for a "fwd_voice"
item -- on both the "already answered by caption intent" path and the normal
"what should I do with it?" ask path (tg_resolve.py's `elif t == "fwd_voice"`
block). It never falls through to the merge-and-push code at the bottom of
_resolve_and_push. A batch is only ONE call: if a photo (or plain text, or a
second forwarded note) shares the same ~1.8s debounce window as a forwarded
voice note, the two items are queued together and resolved as a single
`batch` list. Whichever position the fwd_voice item sits at, its `return`
exits the WHOLE function -- so an item processed before it (bytes already
downloaded into the local `img_bytes`/`texts` accumulator) is thrown away
with them, and an item still to come is never even looked at. No error, no
trace: the picture (or message) just vanishes.

Reproduced live via the adversarial-style real-consumer-thread harness below,
in both orderings (photo-then-forward, forward-then-photo), plus a sibling
check that ordinary text sharing the window survives the same way. Fixed by
folding "batch contains a fwd_voice item alongside anything else" into the
existing genuine-conflict split in _resolve_and_push (the same mechanism
already used for conflicting image targets and multiple image-bearing items
in one batch): each item gets pushed through its own single-item call instead
of being silently merged into one that only the fwd_voice item's `return`
ever reaches.

Run: venv/Scripts/python.exe tests/test_tg_fwd_voice_batch.py
"""
import os, sys, time, threading, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

_DATA = tempfile.mkdtemp(prefix="tg_fwdvoice_")
import tg_bot as T
T.redirect_data_dir(_DATA)
import graph as graph_mod
# The intent read is a model call (_fwdv_intent); its own quality is checked
# live by bench/fwdv_intent_live.py -- here only the routing it drives.
import llm as _llm0
_real_simple0 = _llm0.call_llm_simple
def _intent_or_real(ctx, system, user, *a, **k):
    if system == T._FWDV_SYSTEM:
        return '{"label": "%s"}' % ("sum" if user.strip().lower() in ("перескажи", "who drives? в двух словах суть") else "other")
    return _real_simple0(ctx, system, user, *a, **k)
_llm0.call_llm_simple = _intent_or_real

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


INVOKES = []


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


class StubGraph:
    def __init__(self, ctx):
        self.ctx = ctx

    def invoke(self, base):
        text = base.get("user_input", "")
        INVOKES.append((text, getattr(self.ctx, "last_image_path", "")))
        return {"final_answer": f"echo: {text}", "messages": base.get("messages", [])}


def _stub_build_graph(ctx):
    return StubGraph(ctx)


graph_mod.build_graph = _stub_build_graph
SHARED_CTX = FakeCtx()


def make_bot(n_consumers=2):
    bot = T.TelegramBot("123:TEST", lambda: SHARED_CTX, lambda: object(),
                        lambda: {"messages": []}, silent_mode=True)
    bot._backend = T.InMemoryBackend()
    bot.sent = []
    bot._send_text = lambda cid, t, **kw: (bot.sent.append((cid, t)), 1)[1]
    bot._send_get_id = lambda cid, t, **kw: (bot.sent.append((cid, t)), len(bot.sent))[1]
    bot._edit_text = lambda *a, **kw: None
    bot._delete = lambda *a, **kw: None
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._dl_bytes = lambda fid: b"\xff\xd8\xff" + b"P" * 80
    bot._transcribe = lambda ctx, fid, media="voice", **kw: "the transcribed forwarded content"
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


_UPDATE_ID_SEQ = [30_000_000]


def _next_update_id():
    _UPDATE_ID_SEQ[0] += 1
    return _UPDATE_ID_SEQ[0]


def msg_update(cid, text):
    return {"update_id": _next_update_id(),
            "message": {"message_id": _next_update_id(),
                        "chat": {"id": cid, "type": "private"},
                        "from": {"id": cid}, "text": text,
                        "date": int(time.time())}}


def photo_update(cid, fid):
    return {"update_id": _next_update_id(),
            "message": {"message_id": _next_update_id(),
                        "chat": {"id": cid, "type": "private"},
                        "from": {"id": cid},
                        "photo": [{"file_id": fid, "file_size": 100,
                                   "width": 100, "height": 100}],
                        "caption": "", "date": int(time.time())}}


def fwd_voice_update(cid, fid):
    return {"update_id": _next_update_id(),
            "message": {"message_id": _next_update_id(),
                        "chat": {"id": cid, "type": "private"},
                        "from": {"id": cid},
                        "voice": {"file_id": fid, "duration": 5},
                        "forward_origin": {"type": "user"},
                        "date": int(time.time())}}


def settle(predicate, timeout=8.0, interval=0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# ══════════════════════════════════════════════════════════════════════════
section("FORWARD then PHOTO in the same debounce window")
# ══════════════════════════════════════════════════════════════════════════

CID_FP = 900300
bot = make_bot()
approve(bot, CID_FP)
bot.sent.clear(); INVOKES.clear()

bot._dispatch(fwd_voice_update(CID_FP, "voicefid1"))
bot._dispatch(photo_update(CID_FP, "photofid1"))

ok = settle(lambda: len(bot._get_session(CID_FP).image_log) >= 1)
sess = bot._get_session(CID_FP)
check("forward-then-photo: the photo is not silently dropped",
      ok and len(sess.image_log) == 1, sess.image_log)
check("forward-then-photo: the forwarded note still asked its question",
      any("Forwarded voice" in t or "forward" in t.lower() for t in
          [x for _, x in bot.sent]), bot.sent)
stop_bot(bot)

# ══════════════════════════════════════════════════════════════════════════
section("PHOTO then FORWARD in the same debounce window (reverse order)")
# ══════════════════════════════════════════════════════════════════════════

CID_PF = 900301
bot = make_bot()
approve(bot, CID_PF)
bot.sent.clear(); INVOKES.clear()

bot._dispatch(photo_update(CID_PF, "photofid2"))
bot._dispatch(fwd_voice_update(CID_PF, "voicefid2"))

ok = settle(lambda: len(bot._get_session(CID_PF).image_log) >= 1)
sess = bot._get_session(CID_PF)
check("photo-then-forward: the photo is not silently dropped",
      ok and len(sess.image_log) == 1, sess.image_log)
stop_bot(bot)

# ══════════════════════════════════════════════════════════════════════════
section("FORWARD then plain TEXT in the same debounce window")
# ══════════════════════════════════════════════════════════════════════════

CID_FT = 900302
bot = make_bot()
approve(bot, CID_FT)
bot.sent.clear(); INVOKES.clear()

bot._dispatch(fwd_voice_update(CID_FT, "voicefid3"))
bot._dispatch(msg_update(CID_FT, "what time is it"))

ok = settle(lambda: any("what time is it" in t for t, _ in INVOKES))
check("forward-then-text: the text runs through the graph", ok, INVOKES)
# The words typed next to a forwarded note are the instruction FOR the note
# (live 2026-09-17: «Что думаешь?» beside a forwarded кружок was answered as
# small talk while the note asked "what do I do with it?"). One turn: the
# transcript quoted, the words as the request, no ask.
_turn = next((t for t, _ in INVOKES if "what time is it" in t), "")
check("forward-then-text: the transcript rides in the same turn as the words",
      "the transcribed forwarded content" in _turn and _turn.rstrip().endswith("what time is it"), _turn)
check("forward-then-text: no 'what do I do with it?' ask",
      not any("Forwarded voice" in t for _, t in bot.sent), bot.sent)
stop_bot(bot)

# ══════════════════════════════════════════════════════════════════════════
section("TEXT then FORWARD (reverse order) -- same single turn")
# ══════════════════════════════════════════════════════════════════════════

CID_TF = 900304
bot = make_bot()
approve(bot, CID_TF)
bot.sent.clear(); INVOKES.clear()
bot._dispatch(msg_update(CID_TF, "what do you think"))
bot._dispatch(fwd_voice_update(CID_TF, "voicefid6"))
ok = settle(lambda: any("what do you think" in t for t, _ in INVOKES))
_turn = next((t for t, _ in INVOKES if "what do you think" in t), "")
check("text-then-forward: one turn with the transcript and the words",
      ok and "the transcribed forwarded content" in _turn and len(INVOKES) == 1, INVOKES)
# an explicit intent word still takes the transcript/summary shortcut
# The retelling is a one-shot llm.call_llm_simple outside the agent (see
# _retell_transcript), so it is stubbed here; unstubbed it waited on the real
# model and the suite read the silence as a missing shortcut.
import llm as _llm_mod
_real_simple = _llm_mod.call_llm_simple
_llm_mod.call_llm_simple = lambda ctx, system, *a, **k: (
    '{"label": "sum"}' if system == T._FWDV_SYSTEM else "Summary: stub retelling")
try:
    bot.sent.clear(); INVOKES.clear()
    bot._dispatch(msg_update(CID_TF, "перескажи"))
    bot._dispatch(fwd_voice_update(CID_TF, "voicefid7"))
    ok = settle(lambda: any("Summary" in t for _, t in bot.sent), timeout=6)
finally:
    _llm_mod.call_llm_simple = _real_simple
check("text-then-forward: an intent word ('перескажи') still runs the shortcut, not a graph turn",
      ok and not INVOKES, (bot.sent, INVOKES))
stop_bot(bot)

# ══════════════════════════════════════════════════════════════════════════
section("A forwarded CONVERSATION: voices, a кружок and texts from two people")
# ══════════════════════════════════════════════════════════════════════════
# Several forwarded pieces are one conversation: read in order, each line
# under its author, each note labelled, ONE ask -- not an ask per piece.


def fwd_from(cid, name, **body):
    return {"update_id": _next_update_id(),
            "message": {"message_id": _next_update_id(),
                        "chat": {"id": cid, "type": "private"}, "from": {"id": cid},
                        "forward_origin": {"type": "user",
                                           "sender_user": {"id": 1, "first_name": name}},
                        "date": int(time.time()), **body}}


CID_FF = 900303
bot = make_bot()
approve(bot, CID_FF)
_heard = {"v1": "can you pick me up at six", "v2": "sure, near the station",
          "c1": "here is the parking lot"}
bot._transcribe = lambda ctx, fid, media="voice", **kw: _heard[fid]
bot._look_video = lambda ctx, fid, **kw: {"description": "0:00 — a parking lot at dusk\n0:05 — a red car",
                                          "sheet": "", "times": [0, 5]}
_kbs = []
_plain_send = bot._send_text
bot._send_text = lambda cid, t, **kw: (bot.sent.append((cid, t)), _kbs.append(kw.get("keyboard")), 1)[2]
bot.sent.clear(); INVOKES.clear()

bot._dispatch(fwd_from(CID_FF, "Anna", voice={"file_id": "v1", "duration": 3}))
bot._dispatch(fwd_from(CID_FF, "Boris", voice={"file_id": "v2", "duration": 2}))
bot._dispatch(fwd_from(CID_FF, "Anna", text="ok, waiting"))
bot._dispatch(fwd_from(CID_FF, "Boris", video_note={"file_id": "c1", "duration": 6}))
# A menu button pressed while the forward lands must not break the batch apart
# (live 2026-10-01 11:37: «↩ Назад» among a voice note and nine кружки).
bot._dispatch(msg_update(CID_FF, T._b("back", "en")))
ok = settle(lambda: any("Forwarded conversation" in t for _, t in bot.sent))
asks = [t for _, t in bot.sent if "Forwarded conversation" in t or "Forwarded voice" in t]
check("a forwarded batch gets ONE ask", ok and len(asks) == 1, bot.sent)
check("...naming both senders", ok and "Anna" in asks[0] and "Boris" in asks[0], asks)
held = bot._get_session(CID_FF).fwd_transcript
_order = [held.find(x) for x in ("Anna [🎤 voice 1]: can you pick me up",
                                 "Boris [🎤 voice 2]: sure",
                                 "Anna: ok, waiting",
                                 "Boris [🎥 video 1]: here is the parking lot")]
check("the conversation is held in order, each line under its author", -1 not in _order
      and _order == sorted(_order), held)
check("the storyboard says which clip it came from", "🎥 video 1 · Boris\n0:00" in held, held)
check("the agent was not run on it", not INVOKES, INVOKES)
_kb = next((k for k in reversed(_kbs) if k and "inline_keyboard" in k), {})
_datas = [b["callback_data"] for r in _kb.get("inline_keyboard", []) for b in r]
check("the ask offers the storyboard and an own request",
      any(d.startswith("fwdv:board") for d in _datas) and any(d.startswith("fwdv:own") for d in _datas),
      _datas)

# 📝 Transcript: the words only; 👁 Storyboard: the frames, labelled and bold.
fid = bot._get_session(CID_FF).fwd_transcript_id
bot.sent.clear()
bot._cb_forwarded_voice(CID_FF, "fwdv:text:" + fid)
_txt = "\n".join(t for _, t in bot.sent)
check("transcript button shows the words without the storyboard",
      "Anna" in _txt and "parking lot at dusk" not in _txt, _txt)
bot.sent.clear()
bot._cb_forwarded_voice(CID_FF, "fwdv:board:" + fid)
_brd = "\n".join(t for _, t in bot.sent)
check("storyboard button shows the frames under their clip",
      "<b>🎥 video 1 · Boris</b>" in _brd and "<b>0:05</b>" in _brd and "Anna" not in _brd, _brd)

# ✍️ My own request: the next text is the request, the conversation rides along.
bot._cb_forwarded_voice(CID_FF, "fwdv:own:" + fid)
check("own request arms the session", bool(bot._get_session(CID_FF).fwd_own))
# live 16:03: ✍ pressed straight away (no button before it), so the typed
# question met an un-acted-on transcript and was taken for "retell".
_s = bot._get_session(CID_FF); _s.fwd_transcript_done = False; bot._store.put(_s)
INVOKES.clear()
bot._dispatch(msg_update(CID_FF, "who drives? в двух словах суть"))
ok = settle(lambda: any("who drives? в двух словах суть" in t for t, _ in INVOKES))
_turn = next((t for t, _ in INVOKES if "who drives? в двух словах суть" in t), "")
check("the typed request runs with the conversation quoted",
      ok and "Boris [🎤 voice 2]: sure" in _turn, _turn)
check("...and the own request is spent", not bot._get_session(CID_FF).fwd_own)
stop_bot(bot)

# ══════════════════════════════════════════════════════════════════════════
section("A single forwarded VIDEO: the storyboard waits for its button")
# ══════════════════════════════════════════════════════════════════════════
CID_V = 900305
bot = make_bot()
approve(bot, CID_V)
bot._look_video = lambda ctx, fid, **kw: {"description": "0:00 — a cat on a sofa",
                                          "sheet": "", "times": [0]}
bot.sent.clear()
bot._dispatch(fwd_from(CID_V, "Anna", video_note={"file_id": "c9", "duration": 4}))
ok = settle(lambda: any("Forwarded video" in t for _, t in bot.sent))
_ask = next((t for _, t in bot.sent if "Forwarded video" in t), "")
check("the video ask does not dump the storyboard", ok and "cat on a sofa" not in _ask, _ask)
stop_bot(bot)

print(f"\n{OK}/{OK + BAD} passed")
if BAD:
    print("FAILURES:", FAILURES)
    sys.exit(1)
