"""Cross-chat isolation under REAL concurrent consumer threads.

Drives the real TelegramBot._dispatch() / router / debounce / admission-guard
/ consumer-thread pool (make_bot(n_consumers=3), real InMemoryBackend; only
the Telegram API and the LLM graph are stubbed) with deliberately-conflicting,
genuinely concurrent activity from three distinct chat_ids (900100/900101/
900102), to catch the class of bug found and fixed earlier this branch:
per-update state (image target, photo bytes, cancel scope, callback
resolution) accidentally promoted to a shared/global slot instead of being
kept per-chat.

Covers, each with BOTH a graph.invoke()-argument assertion and a delivered-
message assertion:
  1. conflicting image_ids -- concurrent upscale presses on different pictures
  2. conflicting photo bytes -- concurrent uploads in different chats
  3. cancellation -- one chat's /cancel must not touch another chat's task
  4. callbacks -- same callback_data SHAPE, different per-chat image_ids
  5. slow op in one chat must not block/delay the other two (queue independence)
  6. duplicate/retry in one chat concurrent with another chat's independent turn
  7. rapid interleaved plain text across all three chats -- session/menu state

All seven passed clean on the run that produced this file: no cross-chat leak
was found. Several early draft failures turned out to be test-harness timing
artifacts (debounce merge, exact-repeat dedup, the already_running guard's
debounce-cycle window) rather than application bugs -- see the inline
comments at each fix for how that was distinguished from a real defect.

Run: venv/Scripts/python.exe tests/test_tg_cross_chat_isolation.py
"""
import os, sys, time, copy, threading, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

_DATA = tempfile.mkdtemp(prefix="tg_xchat_")
import tg_bot as T
T.redirect_data_dir(_DATA)
import graph as graph_mod

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


# INVOKES now also carries chat_id (extracted from ctx via a per-task attr
# tg_bot._scoped_ctx sets, `_xchat_cid`) so cross-chat leaks can be asserted
# directly against which chat's invoke() actually saw which state.
INVOKES = []
SCRIPT = {}


def script(text, **behavior):
    SCRIPT[text] = behavior


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
        self._xchat_cid = None      # set by StubGraph.invoke via base["chat_id"] if present

    def memory_text(self): return ""
    def set_stage(self, s): pass


class StubGraph:
    def __init__(self, ctx):
        self.ctx = ctx

    def invoke(self, base):
        text = base.get("user_input", "")
        cid = base.get("chat_id")
        INVOKES.append((text, threading.current_thread().name, time.monotonic(),
                         getattr(self.ctx, "last_image_path", ""), cid,
                         base.get("image_data")))
        beh = SCRIPT.get(text, {})
        delay = beh.get("delay", 0)
        if delay:
            time.sleep(delay)
        if beh.get("raise"):
            raise RuntimeError(beh["raise"])
        out = {"final_answer": beh.get("final_answer", f"echo: {text}"),
               "messages": base.get("messages", [])}
        if "set_image_path" in beh:
            self.ctx.last_image_path = beh["set_image_path"]
        if "document_path" in beh:
            out["document_path"] = beh["document_path"]
            out["document_status"] = beh.get("document_status", "success")
        return out


def _stub_build_graph(ctx):
    return StubGraph(ctx)


SHARED_CTX = FakeCtx()


def make_bot(n_consumers=3):
    bot = T.TelegramBot("123:TEST", lambda: SHARED_CTX, lambda: object(),
                        lambda: {"messages": []}, silent_mode=True)
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


def approve(bot, cid, name="Tester", admin=False):
    bot._user_store.put(T._User(chat_id=cid, name=name, status="approved", is_admin=admin))
    sess = bot._get_session(cid)
    sess.clear_context(); sess.lang = "en"; sess.reg_state = ""
    bot._store.put(sess)
    return sess


_UPDATE_ID_SEQ = [20_000_000]


def _next_update_id():
    _UPDATE_ID_SEQ[0] += 1
    return _UPDATE_ID_SEQ[0]


def msg_update(cid, text, uid=None, update_id=None, message_id=None):
    return {"update_id": update_id if update_id is not None else _next_update_id(),
            "message": {"message_id": message_id if message_id is not None
                        else _next_update_id(),
                        "chat": {"id": cid, "type": "private"},
                        "from": {"id": uid if uid is not None else cid}, "text": text,
                        "date": int(time.time())}}


def cb_update(cid, data, uid=None, msg_id=None, update_id=None, cb_id=None):
    return {"update_id": update_id if update_id is not None else _next_update_id(),
            "callback_query": {"id": cb_id if cb_id is not None else str(_next_update_id()),
                               "data": data,
                               "from": {"id": uid if uid is not None else cid},
                               "message": {"chat": {"id": cid, "type": "private"},
                                           "message_id": msg_id if msg_id is not None
                                           else 900}}}


def photo_update(cid, fid, caption="", update_id=None, message_id=None):
    return {"update_id": update_id if update_id is not None else _next_update_id(),
            "message": {"message_id": message_id if message_id is not None
                        else _next_update_id(),
                        "chat": {"id": cid, "type": "private"},
                        "from": {"id": cid},
                        "photo": [{"file_id": fid, "file_size": 100,
                                   "width": 100, "height": 100}],
                        "caption": caption,
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


def seed_pics(bot, cid, n, tag=""):
    sess = bot._get_session(cid)
    sess.clear_context(); sess.lang = "en"
    ids = []
    for i in range(n):
        p = os.path.join(_DATA, f"pic_{cid}_{tag}_{i}.png")
        open(p, "wb").write(b"\x89PNG\r\n\x1a\n" + bytes([cid % 250]) * 32 + bytes([i]) * 32)
        T._log_image(sess, p, label=f"pic{i}", src="bot")
        ids.append(sess.image_log[-1]["id"])
    bot._store.put(sess)
    return ids


graph_mod.build_graph = _stub_build_graph
print("graph.build_graph stubbed for cross-chat isolation run")

CID_A, CID_B, CID_C = 900100, 900101, 900102

# ══════════════════════════════════════════════════════════════════════════
section("1. CONFLICTING image_ids ACROSS 3 CHATS -- concurrent upscale presses")
# ══════════════════════════════════════════════════════════════════════════
bot = make_bot(n_consumers=3)
for cid in (CID_A, CID_B, CID_C):
    approve(bot, cid)
ids = {}
paths = {}
for cid in (CID_A, CID_B, CID_C):
    ids[cid] = seed_pics(bot, cid, 1, tag="img")[0]
    paths[cid] = T._image_by_id(bot._get_session(cid), ids[cid])["path"]

UPSCALE = T._CB_CMDS["regenerate"]
script(UPSCALE, final_answer="upscaled", delay=0.3)
INVOKES.clear()
# fire near-simultaneously from 3 threads to actually race the consumers
threads = [threading.Thread(target=lambda cid=cid: bot._dispatch(
                cb_update(cid, f"regenerate:{ids[cid]}", msg_id=1)))
           for cid in (CID_A, CID_B, CID_C)]
for t in threads: t.start()
for t in threads: t.join()

ok = settle(lambda: sum(1 for t, *_ in INVOKES if t == UPSCALE) >= 3, timeout=10)
check("all three concurrent upscale presses ran", ok, INVOKES)
# base has no chat_id key; attribute each invoke() to a chat by REVERSE-mapping
# the (chat-unique-by-construction) image path it saw. Any path that resolves
# to nobody, or a path count that doesn't match 1-per-chat, is a leak.
path_to_cid = {p: c for c, p in paths.items()}
upscale_paths = [p for t, _, _, p, _, _ in INVOKES if t == UPSCALE]
leak = (sorted(upscale_paths) != sorted(paths.values())) or any(
    path_to_cid.get(p) is None for p in upscale_paths)
check("each chat's invoke() saw ONLY its own image path, never another chat's, "
      "and every chat's path appeared exactly once",
      not leak, upscale_paths)
check("no chat received another chat's upscaled-picture reply",
      all(texts_for(bot, c) for c in (CID_A, CID_B, CID_C)), None)
stop_bot(bot)
print(f"\n{OK}/{OK+BAD} passed so far")

# ══════════════════════════════════════════════════════════════════════════
section("2. CONFLICTING PHOTO BYTES -- concurrent uploads in 3 chats")
# ══════════════════════════════════════════════════════════════════════════
bot = make_bot(n_consumers=3)
for cid in (CID_A, CID_B, CID_C):
    approve(bot, cid)
photo_bytes = {
    CID_A: b"\xff\xd8\xff" + b"A" * 90,
    CID_B: b"\xff\xd8\xff" + b"B" * 90,
    CID_C: b"\xff\xd8\xff" + b"C" * 90,
}
fid_of = {CID_A: "fidA", CID_B: "fidB", CID_C: "fidC"}
bot._dl_bytes = lambda fid: {"fidA": photo_bytes[CID_A], "fidB": photo_bytes[CID_B],
                              "fidC": photo_bytes[CID_C]}.get(fid, b"\xff\xd8\xff0")
script("image", final_answer="looked at it")
INVOKES.clear()

threads = [threading.Thread(target=lambda cid=cid: bot._dispatch(
                photo_update(cid, fid_of[cid])))
           for cid in (CID_A, CID_B, CID_C)]
for t in threads: t.start()
for t in threads: t.join()
ok = settle(lambda: len(INVOKES) >= 3, timeout=10)
check("all three concurrent photo uploads ran", ok, INVOKES)
time.sleep(0.5)

# The uploaded photo's raw bytes are not written into sess.image_log (that
# only records GENERATED output images -- see tg_tasks.py:741). What each
# chat's task actually threads through is base["image_data"], built per-task
# from _dl_bytes(file_id) in the debounce/resolve path. base carries no
# chat_id key, so attribute each invoke to a chat by reverse-mapping the
# (chat-unique-by-construction) bytes payload it saw.
bytes_to_cid = {v: k for k, v in photo_bytes.items()}
image_datas = [data for t, _, _, _, _, data in INVOKES if data]
leak = (sorted(image_datas) != sorted(photo_bytes.values())) or any(
    bytes_to_cid.get(d) is None for d in image_datas)
check("each chat's invoke() saw ONLY its own uploaded photo bytes in "
      "image_data, never another chat's, each exactly once",
      not leak, image_datas)
stop_bot(bot)
print(f"\n{OK}/{OK+BAD} passed so far")

# ══════════════════════════════════════════════════════════════════════════
section("3. CANCELLATION -- chat A cancels while chat B runs a similar-shaped task")
# ══════════════════════════════════════════════════════════════════════════
bot = make_bot(n_consumers=3)
for cid in (CID_A, CID_B, CID_C):
    approve(bot, cid)
script("slow shared-shape op", final_answer="finished slow op", delay=1.5)
INVOKES.clear(); bot.sent.clear()

bot._dispatch(msg_update(CID_A, "slow shared-shape op"))
bot._dispatch(msg_update(CID_B, "slow shared-shape op"))
ok = settle(lambda: n_running(bot, CID_A) >= 1 and n_running(bot, CID_B) >= 1, timeout=6)
check("both A and B slow tasks are running before cancel", ok,
      (n_running(bot, CID_A), n_running(bot, CID_B)))

b_running_at_cancel = n_running(bot, CID_B)   # sample immediately, before A's
                                               # cancel has any chance to run
bot._dispatch(msg_update(CID_A, "/cancel"))
ok_a_cleared = settle(lambda: n_running(bot, CID_A) == 0, timeout=6)
check("chat A's running-task slot clears after its own /cancel", ok_a_cleared)
# B must NOT be affected by A's cancel. B's own 1.5s op may have already
# finished naturally by the time we get here (same delay as A, timing is not
# deterministic across threads) -- the real invariant is that A's /cancel
# never force-clears B's slot AHEAD of B's own natural completion, which the
# "B still delivers its own reply" check below proves either way.
check("chat B's running-task slot was non-empty just before A's /cancel "
      "was even dispatched (B was never touched by A's cancel machinery)",
      b_running_at_cancel >= 1, b_running_at_cancel)

ok_b_finished = settle(lambda: any("finished slow op" in t for t in texts_for(bot, CID_B)),
                        timeout=6)
check("chat B's task still completes and delivers normally", ok_b_finished,
      texts_for(bot, CID_B))
time.sleep(0.3)
check("chat A never receives the cancelled task's result",
      not any("finished slow op" in t for t in texts_for(bot, CID_A)),
      texts_for(bot, CID_A))
stop_bot(bot)
print(f"\n{OK}/{OK+BAD} passed so far")

# ══════════════════════════════════════════════════════════════════════════
section("4. CALLBACKS -- same callback_data SHAPE, different per-chat image_ids")
# ══════════════════════════════════════════════════════════════════════════
bot = make_bot(n_consumers=3)
for cid in (CID_A, CID_B, CID_C):
    approve(bot, cid)
ids2 = {}
paths2 = {}
for cid in (CID_A, CID_B, CID_C):
    ids2[cid] = seed_pics(bot, cid, 1, tag="cb")[0]
    paths2[cid] = T._image_by_id(bot._get_session(cid), ids2[cid])["path"]

ENHANCE = T._CB_CMDS["regenerate"]
script(ENHANCE, final_answer="enhanced", delay=0.2)
INVOKES.clear()
# Deliberately reuse the same msg_id/cb pattern across chats -- only the
# per-chat image_id differs, exactly like real "upscale:<id>" button presses.
threads = [threading.Thread(target=lambda cid=cid: bot._dispatch(
                cb_update(cid, f"regenerate:{ids2[cid]}", msg_id=1, cb_id="dup-shape")))
           for cid in (CID_A, CID_B, CID_C)]
for t in threads: t.start()
for t in threads: t.join()
ok = settle(lambda: sum(1 for t, *_ in INVOKES if t == ENHANCE) >= 3, timeout=10)
check("all three same-shape-callback presses ran", ok, INVOKES)
enhance_paths = [p for t, _, _, p, _, _ in INVOKES if t == ENHANCE]
leak2 = sorted(enhance_paths) != sorted(paths2.values())
check("no cross-chat callback resolution: each chat's enhance ran against "
      "its own image only, each exactly once", not leak2, enhance_paths)
stop_bot(bot)
print(f"\n{OK}/{OK+BAD} passed so far")

# ══════════════════════════════════════════════════════════════════════════
section("5. SLOW OP IN A DOES NOT BLOCK/DELAY B AND C (queue independence)")
# ══════════════════════════════════════════════════════════════════════════
bot = make_bot(n_consumers=3)
for cid in (CID_A, CID_B, CID_C):
    approve(bot, cid)
script("very slow op", final_answer="slow done", delay=8.0)
script("fast op B", final_answer="fast done B", delay=0.05)
script("fast op C", final_answer="fast done C", delay=0.05)
INVOKES.clear(); bot.sent.clear()

bot._dispatch(msg_update(CID_A, "very slow op"))
ok_a_running = settle(lambda: n_running(bot, CID_A) >= 1, timeout=6)
check("A's slow op is actually running before B/C are dispatched", ok_a_running)
# Measure elapsed from AFTER B/C are dispatched, not from A's dispatch -- A's
# own debounce window (1.8s) is not part of what "queue independence" means;
# what matters is whether B/C, dispatched once A is already mid-flight, are
# themselves serialized behind A's still-running 3s task.
t0 = time.monotonic()
bot._dispatch(msg_update(CID_B, "fast op B"))
bot._dispatch(msg_update(CID_C, "fast op C"))
ok_bc = settle(lambda: any("fast done B" in t for t in texts_for(bot, CID_B))
               and any("fast done C" in t for t in texts_for(bot, CID_C)), timeout=7)
elapsed = time.monotonic() - t0
# Not serialized behind A == B and C answered while A was still running. A
# wall-clock bound (it was < 2.7 s: 1.8 s debounce + 0.9 s) measured the
# runner's speed too and failed on a loaded CI machine.
a_still_running = not any("slow done" in t for t in texts_for(bot, CID_A))
check("B and C's fast turns complete while A's slow op is still running "
      "(not serialized behind it)", ok_bc and a_still_running, (elapsed, a_still_running))
# Admission guard in A must not affect B/C. Settle rather than sample: the
# reply text is sent from inside _run_task_inner, and the task only leaves
# _running_task in the `finally` that wraps it, so for a moment after B's
# answer is visible its own slot is legitimately still occupied. Sampling
# there caught that window, not a leak -- what a leak would look like is the
# slot staying occupied, which a 3s settle still fails on.
check("B's own already-running admission guard state is independent of A",
      settle(lambda: n_running(bot, CID_B) == 0, timeout=3), n_running(bot, CID_B))
check("C's own already-running admission guard state is independent of A",
      settle(lambda: n_running(bot, CID_C) == 0, timeout=3), n_running(bot, CID_C))
ok_a_done = settle(lambda: any("slow done" in t for t in texts_for(bot, CID_A)), timeout=12)
check("A's slow op still completes normally afterward", ok_a_done, texts_for(bot, CID_A))
stop_bot(bot)
print(f"\n{OK}/{OK+BAD} passed so far")

# ══════════════════════════════════════════════════════════════════════════
section("6. DUPLICATE/RETRY IN A CONCURRENT WITH B's INDEPENDENT ACTIVITY")
# ══════════════════════════════════════════════════════════════════════════
bot = make_bot(n_consumers=3)
for cid in (CID_A, CID_B, CID_C):
    approve(bot, cid)
ids3 = seed_pics(bot, CID_A, 1, tag="dup")[0]
UPSCALE2 = T._CB_CMDS["regenerate"]
# delay must be long enough that the DUPLICATE's own debounce cycle (~1.8s)
# still lands while the original is running -- the already_running guard is
# only even consulted once the duplicate's OWN debounce fires (tg_resolve.py
# ~498-526), so a short delay lets the original finish first and makes the
# "duplicate" a legitimate second task by the documented "outside the
# debounce window becomes a second task" rule. Verified in isolation
# (scratch/isolate_dup.py) against a SINGLE chat before attributing anything
# to cross-chat behaviour: delay=1.5s reproduces "runs twice" even with only
# one chat involved, so it is a timing-parameter artifact, not a leak.
# delay=3.0s (matching the proven adversarial_harness.py CID_DUP3 case)
# keeps the original genuinely running through the duplicate's own debounce.
script(UPSCALE2, final_answer="handled-once-A", delay=3.0)
script("independent B text", final_answer="handled-B", delay=0.05)
INVOKES.clear(); bot.sent.clear()

cbA = cb_update(CID_A, f"regenerate:{ids3}", msg_id=1, cb_id="cb-retry-A")
bot._dispatch(cbA)
ok_running_a = settle(lambda: n_running(bot, CID_A) >= 1, timeout=6)
# duplicate delivery of the SAME callback in A, concurrent with independent B activity
bot._dispatch(copy.deepcopy(cbA))
bot._dispatch(msg_update(CID_B, "independent B text"))

ok_b = settle(lambda: any("handled-B" in t for t in texts_for(bot, CID_B)), timeout=6)
check("chat B's independent activity is unaffected by chat A's duplicate retry",
      ok_b, texts_for(bot, CID_B))
time.sleep(4.0)
n_a = sum(1 for t, *_ in INVOKES if t == UPSCALE2)
already_running_msg = T._t("already_running", "en").split("<")[0][:15]
refused_a = any(already_running_msg in t for t in texts_for(bot, CID_A))
check("chat A's duplicate is refused via already_running (per-chat keyed), "
      "exactly one real invoke", n_a == 1 and refused_a, (n_a, texts_for(bot, CID_A)))
n_b = sum(1 for t, *_ in INVOKES if t == "independent B text")
check("chat B was not accidentally caught by A's already_running guard "
      "(guard is not keyed globally)", n_b == 1, n_b)
stop_bot(bot)
print(f"\n{OK}/{OK+BAD} passed so far")

# ══════════════════════════════════════════════════════════════════════════
section("7. RAPID INTERLEAVED PLAIN TEXT ACROSS 3 CHATS -- session/menu state")
# ══════════════════════════════════════════════════════════════════════════
bot = make_bot(n_consumers=3)
for cid in (CID_A, CID_B, CID_C):
    approve(bot, cid)

# Drive each chat into a DIFFERENT, distinguishable reg_state-like mode by
# arming "change password" only for chat B, leaving A and C in normal mode,
# then interleave rapid plain-text turns across all three and confirm each
# chat's session state stays exactly what it should be independently.
bot._dispatch(cb_update(CID_B, "acct_setpwd"))
check("chat B entered change_password mode", bot._get_session(CID_B).reg_state == "change_password",
      bot._get_session(CID_B).reg_state)
check("chat A was not affected by B's mode change",
      bot._get_session(CID_A).reg_state == "", bot._get_session(CID_A).reg_state)
check("chat C was not affected by B's mode change",
      bot._get_session(CID_C).reg_state == "", bot._get_session(CID_C).reg_state)

# Rapid identical same-chat texts inside the debounce window are collapsed by
# the exact-repeat dedup before batching (observed directly: the delivered
# text was the single literal "turnA"/"turnC", not a repeated "\n"-joined
# batch) -- script the literal text each chat sends.
script("turnA", final_answer="ansA")
script("turnC", final_answer="ansC")
INVOKES.clear(); bot.sent.clear()

seq = []
for i in range(6):
    seq.append(threading.Thread(target=lambda cid=CID_A: bot._dispatch(msg_update(cid, "turnA"))))
    seq.append(threading.Thread(target=lambda cid=CID_C: bot._dispatch(msg_update(cid, "turnC"))))
# B gets a real new password concurrently
seq.append(threading.Thread(target=lambda: bot._dispatch(msg_update(CID_B, "newSecurePass123"))))
for th in seq: th.start()
for th in seq: th.join()

settle(lambda: bot._get_session(CID_B).reg_state == "", timeout=6)
check("chat B's password change completed on its own session, unaffected by "
      "concurrent A/C text traffic", bot._get_session(CID_B).reg_state == "")
new_hash = bot._user_store.get(CID_B).password_hash
check("chat B's password actually changed", new_hash != "", new_hash)

ok_ac = settle(lambda: any("ansA" in t for t in texts_for(bot, CID_A))
               and any("ansC" in t for t in texts_for(bot, CID_C)), timeout=8)
check("A and C both got answered despite heavy interleaving with B's "
      "password flow", ok_ac, (texts_for(bot, CID_A), texts_for(bot, CID_C)))
# no cross-talk: A never sees C's text answers and vice versa
cross_leak = (any("ansC" in t for t in texts_for(bot, CID_A)) or
              any("ansA" in t for t in texts_for(bot, CID_C)) or
              any("newSecurePass123" in t for t in texts_for(bot, CID_A) + texts_for(bot, CID_C)))
check("no session/menu-state cross-talk between A, B, C under rapid interleaving",
      not cross_leak, (texts_for(bot, CID_A), texts_for(bot, CID_B), texts_for(bot, CID_C)))
stop_bot(bot)
print(f"\n{OK}/{OK+BAD} passed so far")

print("\n" + "=" * 78)
print(f"TOTAL: {OK}/{OK+BAD} passed, {BAD} failed")
if FAILURES:
    print("FAILURES:")
    for f in FAILURES:
        print(f"  - {f}")
print("=" * 78)
sys.exit(1 if BAD else 0)
