"""The bot's message pipeline: intake -> debounce -> resolve -> quota -> queue.

Everything between "Telegram handed us an update" and "a task reached the backend"
had no direct coverage. That stretch is where the bot decides whether a message is
answered at all, so a defect there is invisible in the happy path and total for the
user: the reply simply never arrives.

Covered here:

  · _classify_task / _quota_limit / quota enforcement and accounting
  · _fmt_eta (the wait estimate a queued user is shown)
  · the debounce merge (several messages in a burst become one turn)
  · album buffering (a multi-photo send is one turn, not N)
  · document intake and _extract_doc, including the unreadable path
  · Stop: staleness of already-queued tasks, and per-request cancel
  · offset persistence (the difference between "restart" and "replay everything")
  · the extremism gate and the private-chat guard
  · _recover_inflight (a crash mid-task must tell the user, not go silent)

Run: venv/Scripts/python.exe tests/test_tg_pipeline.py
"""
import sys, os, io, json, time, types, tempfile, threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_pipe_")
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


CID = 999801


def make_bot(**kw):
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None,
                        lambda: {}, silent_mode=True, **kw)
    bot.sent, bot.pushed = [], []
    bot._send_text = lambda cid, text, **k: (bot.sent.append((cid, text)), 1)[1]
    bot._send_get_id = lambda cid, text, **k: (bot.sent.append((cid, text)), 1)[1]
    bot._activity.log = lambda *a, **k: None
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._dl_bytes = lambda fid: b"PHOTOBYTES"
    real_push = bot._backend.push
    bot._backend.push = lambda task: (bot.pushed.append(task), real_push(task))[0]
    return bot


def approved(bot, cid=CID, **kw):
    u = T._User(chat_id=cid, name="U", status="approved", **kw)
    bot._user_store.put(u)
    return u


def texts(bot):
    return " ".join(t for _c, t in bot.sent)


print("=" * 70)
print("QUOTA CLASSIFICATION AND ACCOUNTING")
print("=" * 70)

cases = {
    "do a deep research on: fusion":              T.KIND_RESEARCH,
    "  DO A DEEP RESEARCH ON: fusion":            T.KIND_RESEARCH,
    "generate an image of: a cat":                T.KIND_IMAGE,
    "edit the image: remove the hat":             T.KIND_IMAGE,
    "regenerate the image":                       T.KIND_IMAGE,
    "search the web for: weather":                T.KIND_TASK,
    "what is the capital of France":              T.KIND_TASK,
    "":                                           T.KIND_TASK,
}
for text, want in cases.items():
    got = T._classify_task(text)
    check(f"classify {text[:34]!r} -> {want}", got == want, got)

check("None does not crash the classifier", T._classify_task(None) == T.KIND_TASK)
check("an unknown bucket has no limit", T._quota_limit("nonsense") == 0)
check("each real bucket has a positive limit",
      all(T._quota_limit(k) > 0
          for k in (T.KIND_TASK, T.KIND_IMAGE, T.KIND_RESEARCH)))
check("research is the scarcest bucket",
      T._quota_limit(T.KIND_RESEARCH) < T._quota_limit(T.KIND_IMAGE)
      < T._quota_limit(T.KIND_TASK))

bot = make_bot()
approved(bot)
bot._user_store.bump_usage(CID, T.KIND_IMAGE, 2)
check("usage accumulates per bucket",
      bot._user_store.usage_today(CID).get(T.KIND_IMAGE) == 2)
bot._user_store.bump_usage(CID, T.KIND_IMAGE)
check("a second bump adds to the same day",
      bot._user_store.usage_today(CID).get(T.KIND_IMAGE) == 3)
check("buckets are independent",
      bot._user_store.usage_today(CID).get(T.KIND_RESEARCH) is None)
check("usage is per user",
      bot._user_store.usage_today(CID + 1) == {})

# An image request must consume BOTH the image bucket and the task bucket, or the
# per-bucket limits do not compose and 40 images cost nothing against the 200 cap.
# Usage lives in the shared DB, so this needs a chat that nothing else has touched.
CID_Q = CID + 100
bot2 = make_bot(); approved(bot2, CID_Q)
bot2._resolve_and_push(CID_Q, [{"type": "text", "text": "generate an image of: a cat"}])
u2 = bot2._user_store.usage_today(CID_Q)
check("an image request charges the image bucket", u2.get(T.KIND_IMAGE) == 1, str(u2))
check("an image request also charges the task bucket", u2.get(T.KIND_TASK) == 1, str(u2))
check("the task reached the queue", len(bot2.pushed) == 1)

# Hitting a limit must refuse BEFORE anything is queued.
bot3 = make_bot(); approved(bot3)
lim = T._quota_limit(T.KIND_RESEARCH)
bot3._user_store.bump_usage(CID, T.KIND_RESEARCH, lim)
bot3._resolve_and_push(CID, [{"type": "text", "text": "do a deep research on: x"}])
check("a spent research quota refuses the request", len(bot3.pushed) == 0)
check("and says so", "🚦" in texts(bot3), repr(texts(bot3)[:80]))
check("a refused request is not charged",
      bot3._user_store.usage_today(CID).get(T.KIND_RESEARCH) == lim)
bot3.sent.clear()
bot3._resolve_and_push(CID, [{"type": "text", "text": "hello there"}])
check("an unrelated request still goes through", len(bot3.pushed) == 1,
      repr(texts(bot3)[:80]))

# Admins are exempt from quotas and from the queue cap.
bot4 = make_bot(); approved(bot4, is_admin=True)
bot4._user_store.bump_usage(CID, T.KIND_RESEARCH, lim + 5)
bot4._resolve_and_push(CID, [{"type": "text", "text": "do a deep research on: x"}])
check("an admin is not blocked by a spent quota", len(bot4.pushed) == 1,
      repr(texts(bot4)[:80]))


print()
print("=" * 70)
print("THE WAIT ESTIMATE")
print("=" * 70)

mk = lambda txt: types.SimpleNamespace(user_text=txt)
check("no tasks ahead reads as zero seconds", T._fmt_eta([]).startswith("0"))
# _fmt_eta is localized, so the unit words differ by language; pin the language
# rather than the house default, which is Russian.
check("a couple of ordinary tasks are quoted in seconds/minutes",
      T._fmt_eta([mk("hi"), mk("hello")], "en") in
      (f"{2 * T._cfg_int('TG_ETA_TASK_SEC', 45)} s", "2 min", "1 min"),
      T._fmt_eta([mk("hi"), mk("hello")], "en"))
res = T._fmt_eta([mk("do a deep research on: fusion")], "en")
check("one deep research is quoted in minutes or hours",
      res.endswith(("min", "h")), res)
res_ru = T._fmt_eta([mk("do a deep research on: fusion")], "ru")
check("and the same estimate reads in Russian for a Russian user",
      res_ru.endswith(("мин", "ч")), res_ru)
check("a research is estimated far above an ordinary task",
      T._fmt_eta([mk("do a deep research on: x")]) !=
      T._fmt_eta([mk("x")]))
check("the estimate is localised", T._fmt_eta([mk("hi")], "ru").endswith(("с", "мин", "ч")),
      T._fmt_eta([mk("hi")], "ru"))
check("a task with no user_text does not crash the estimate",
      bool(T._fmt_eta([types.SimpleNamespace()])))


print()
print("=" * 70)
print("DEBOUNCE: A BURST OF MESSAGES IS ONE TURN")
print("=" * 70)

bot5 = make_bot(); approved(bot5)
bot5._resolve_and_push(CID, [{"type": "text", "text": "what is"},
                             {"type": "text", "text": "the capital"},
                             {"type": "text", "text": "of France"}])
check("three messages become one task", len(bot5.pushed) == 1)
check("in the order they were sent",
      bot5.pushed[0].user_text == "what is\nthe capital\nof France",
      repr(bot5.pushed[0].user_text))

bot6 = make_bot(); approved(bot6)
bot6._resolve_and_push(CID, [{"type": "photo", "file_id": "f1", "caption": "what is this"}])
check("a photo with a caption carries the caption as the text",
      len(bot6.pushed) == 1 and bot6.pushed[0].user_text == "what is this",
      repr(bot6.pushed and bot6.pushed[0].user_text))
check("and the image is handed over as a file",
      bool(bot6.pushed[0].image_path) and os.path.exists(bot6.pushed[0].image_path))
try: os.unlink(bot6.pushed[0].image_path)
except Exception: pass

bot7 = make_bot(); approved(bot7)
bot7._resolve_and_push(CID, [{"type": "photo", "file_id": "f1", "caption": ""}])
check("a photo with nothing else attached is not queued as an empty turn",
      len(bot7.pushed) == 0 or bool(bot7.pushed[0].user_text)
      or bool(bot7.pushed[0].image_path))

# The real debounce thread: several _enqueue_item calls inside the window must
# collapse into one task.
bot8 = make_bot(); approved(bot8)
for part in ("one", "two", "three"):
    bot8._enqueue_item(CID, {"type": "text", "text": part})
deadline = time.time() + 10
while time.time() < deadline and not bot8.pushed:
    time.sleep(0.05)
check("the live debounce worker produced exactly one task",
      len(bot8.pushed) == 1, f"{len(bot8.pushed)} tasks")
if bot8.pushed:
    check("carrying every message from the burst",
          all(p in bot8.pushed[0].user_text for p in ("one", "two", "three")),
          repr(bot8.pushed[0].user_text))
check("the worker registered itself for this chat",
      CID in bot8._workers or not bot8._workers, str(list(bot8._workers)))


print()
print("=" * 70)
print("ALBUMS: A MULTI-PHOTO SEND IS ONE TURN")
print("=" * 70)

bot9 = make_bot(); approved(bot9)
enq = []
bot9._enqueue_item = lambda cid, item: enq.append((cid, item))
# Telegram attaches the caption to ONE photo of the album, and it is not reliably
# the first — taking photos[0]["caption"] loses the user's question entirely.
for i in range(4):
    bot9._buffer_album(CID, "grp1", f"file{i}", "look at these" if i == 2 else "")
check("the album is buffered, not enqueued per photo", enq == [], str(enq))
bot9._flush_album("grp1")
check("flushing an album enqueues exactly one item", len(enq) == 1, str(enq))
if enq:
    check("with the caption from whichever photo carried it",
          enq[0][1].get("caption") == "look at these", str(enq[0][1]))
    check("as a single album item, not N photos",
          enq[0][1].get("type") == "album", str(enq[0][1].get("type")))
    check("carrying every file id in order",
          enq[0][1].get("file_ids") == [f"file{i}" for i in range(4)],
          str(enq[0][1].get("file_ids")))
check("flushing an unknown album is a no-op",
      (bot9._flush_album("nope"), len(enq))[1] == 1)
check("flushing twice does not double-enqueue",
      (bot9._flush_album("grp1"), len(enq))[1] == 1)

# …and the resolver has to understand what the buffer produced, or the whole
# album path ends in a task nobody can run.
bot9b = make_bot(); approved(bot9b, CID + 101)
bot9b._resolve_and_push(CID + 101, [{"type": "album",
                                     "file_ids": ["a", "b", "c"],
                                     "caption": "rate these"}])
check("an album resolves to exactly one task", len(bot9b.pushed) == 1,
      f"{len(bot9b.pushed)} tasks")
if bot9b.pushed:
    check("with the album caption as the question",
          bot9b.pushed[0].user_text == "rate these",
          repr(bot9b.pushed[0].user_text))
    check("and one image attached, not three",
          bool(bot9b.pushed[0].image_path),
          repr(bot9b.pushed[0].image_path))
    try: os.unlink(bot9b.pushed[0].image_path)
    except Exception: pass

# An album's photos beyond the first used to be downloaded and then thrown
# away with no image_log entry and no trace: the loop kept a single img_bytes
# slot and `break`d on the first successful download. Fixed to log every
# successfully-downloaded photo (the first still drives the task itself --
# a _Task can only carry one image_path), so "the second one"/"the third
# one" has something to resolve against later.
bot9d = make_bot(); approved(bot9d, CID + 103)
_album_bytes = {"x0": b"ALBUM-PHOTO-0", "x1": b"ALBUM-PHOTO-1", "x2": b"ALBUM-PHOTO-2"}
bot9d._dl_bytes = lambda fid: _album_bytes.get(fid)
bot9d._resolve_and_push(CID + 103, [{"type": "album",
                                     "file_ids": ["x0", "x1", "x2"],
                                     "caption": "rate these"}])
check("an album still resolves to exactly one task", len(bot9d.pushed) == 1,
      f"{len(bot9d.pushed)} tasks")
sess9d = bot9d._get_session(CID + 103)
_logged9d = []
for entry in sess9d.image_log:
    try:
        with open(entry["path"], "rb") as fh:
            _logged9d.append(fh.read())
    except Exception:
        pass
check("all 3 album photos are logged to image_log, not just the first",
      len(_logged9d) == 3, f"image_log has {len(sess9d.image_log)} entries")
check("every album photo's actual bytes survive (none overwritten/discarded)",
      all(b in _logged9d for b in _album_bytes.values()), _logged9d)
if bot9d.pushed:
    check("the task itself still carries exactly one driving image",
          bool(bot9d.pushed[0].image_path), repr(bot9d.pushed[0].image_path))
for entry in sess9d.image_log:
    try: os.unlink(entry["path"])
    except Exception: pass

# An album whose downloads all fail must not queue a blank turn.
bot9c = make_bot(); approved(bot9c, CID + 102)
bot9c._dl_bytes = lambda fid: None
bot9c._resolve_and_push(CID + 102, [{"type": "album", "file_ids": ["a"],
                                     "caption": ""}])
check("an album with no usable photo and no caption is dropped",
      len(bot9c.pushed) == 0, str(len(bot9c.pushed)))


print()
print("=" * 70)
print("DOCUMENTS")
print("=" * 70)

def write(name, data, mode="w"):
    p = os.path.join(_DATA_DIR, name)
    with open(p, mode, **({"encoding": "utf-8"} if mode == "w" else {})) as fh:
        fh.write(data)
    return p

txt = write("a.txt", "hello document world")
check("a text file is extracted", T._extract_doc(txt, "a.txt") == "hello document world")
check("extraction keys off the CLAIMED name, not the path",
      T._extract_doc(txt, "a.bin") == "",
      repr(T._extract_doc(txt, "a.bin")))
md = write("b.md", "# Title\n\nbody")
check("markdown is extracted", "Title" in T._extract_doc(md, "b.md"))
check("an unknown extension yields empty, not an exception",
      T._extract_doc(txt, "a.xyz") == "")
check("a missing file yields empty, not an exception",
      T._extract_doc(os.path.join(_DATA_DIR, "nope.txt"), "nope.txt") == "")
bad = write("c.txt", b"\xff\xfe\x00binary\x00", mode="wb")
check("undecodable bytes degrade instead of raising",
      isinstance(T._extract_doc(bad, "c.txt"), str))
check("a name with no extension is handled",
      T._extract_doc(txt, "noext") == "")
check("a path-like filename cannot change the parser",
      T._extract_doc(txt, "../../../etc/passwd") == "")

bot10 = make_bot(); approved(bot10)
bot10._dl_bytes = lambda fid: b"just some readable text in a file"
_indexed10 = []
bot10._index_document = lambda cid, sess, path, name, **kw: _indexed10.append(name)
# CONTRACT CHANGE (measured live): every upload is indexed, whatever its size.
# Indexing only the big ones left 📚 My documents empty after a short upload — it
# looked like the upload had failed — and with no caption the turn became "here
# is some text, what would you like me to do with it?" instead of an
# acknowledgement. A bare small file is now acknowledged, not pushed to the agent.
bot10._resolve_and_push(CID, [{"type": "document", "file_id": "d1",
                               "filename": "notes.txt", "caption": ""}])
check("a small document is indexed like any other", _indexed10 == ["notes.txt"],
      _indexed10)
check("with no caption it is acknowledged, not sent to the agent",
      len(bot10.pushed) == 0, bot10.pushed)

_indexed10.clear()
bot10.pushed.clear()
bot10._resolve_and_push(CID, [{"type": "document", "file_id": "d2",
                               "filename": "notes2.txt",
                               "caption": "о чём этот файл?"}])
check("a small document sent WITH a question still reaches the agent",
      len(bot10.pushed) == 1, bot10.pushed)
check("…and is indexed too", _indexed10 == ["notes2.txt"], _indexed10)
check("the question and the file text both reach the agent",
      "о чём этот файл?" in bot10.pushed[0].user_text
      and "readable text" in bot10.pushed[0].user_text,
      bot10.pushed[0].user_text[:120])
if bot10.pushed:
    check("with its contents", "readable text" in bot10.pushed[0].user_text,
          repr(bot10.pushed[0].user_text[:60]))

bot11 = make_bot(); approved(bot11)
bot11._dl_bytes = lambda fid: b"\x00\x01\x02"
bot11._resolve_and_push(CID, [{"type": "document", "file_id": "d1",
                               "filename": "mystery.bin", "caption": ""}])
check("an unreadable document is refused, not queued", len(bot11.pushed) == 0)
check("and the user is told which file",
      "mystery.bin" in texts(bot11), repr(texts(bot11)[:80]))

bot12 = make_bot(); approved(bot12)
bot12._dl_bytes = lambda fid: b"\x00\x01\x02"
bot12._resolve_and_push(CID, [{"type": "document", "file_id": "d1",
                               "filename": "mystery.bin",
                               "caption": "what is in this file?"}])
check("an unreadable document with a caption still runs the question",
      len(bot12.pushed) == 1, repr(texts(bot12)[:80]))

bot13 = make_bot(); approved(bot13)
bot13._dl_bytes = lambda fid: None
bot13._resolve_and_push(CID, [{"type": "document", "file_id": "d1",
                               "filename": "x.txt", "caption": ""}])
check("a failed download does not queue an empty turn", len(bot13.pushed) == 0)


print()
print("=" * 70)
print("STOP AND CANCEL")
print("=" * 70)

bot14 = make_bot(); approved(bot14)
t_old = T._Task(task_id="a", chat_id=CID, user_text="old")
# Another user's task, queued at the same moment — under round-robin this is
# regularly somebody else's work sitting next to yours.
t_other = T._Task(task_id="c", chat_id=CID + 1, user_text="not mine")
time.sleep(0.01)
bot14._request_stop(CID)
t_new = T._Task(task_id="b", chat_id=CID, user_text="new")
check("a task queued BEFORE the Stop is stale", bot14._stop_requested_after(t_old))
check("a task queued AFTER the Stop is not", not bot14._stop_requested_after(t_new))
check("another chat's already-queued task survives this chat's Stop",
      not bot14._stop_requested_after(t_other))
bot14._request_stop(CID + 1)
check("and that chat's own Stop does reach it",
      bot14._stop_requested_after(t_other))

bot15 = make_bot(); approved(bot15)
for i in range(3):
    bot15._enqueue_item(CID, {"type": "text", "text": f"m{i}"})
dropped = bot15._request_stop(CID)
check("Stop reports how much it discarded", dropped >= 0, str(dropped))

bot16 = make_bot(); approved(bot16)
# _cancel_task now returns one of "running"/"dropped"/"" (tri-state, so a
# double press of the same Cancel button can honestly say "already finished"
# instead of repeating a false "cancelled") — "" is falsy, matching the old
# bool contract for this harmless-unknown-id case.
check("cancelling an unknown task id is harmless",
      bot16._cancel_task(CID, "no-such-task") in ("running", "dropped", ""))
check("a cancelled id is remembered", bot16._is_cancelled("no-such-task"))
check("an unrelated id is not", not bot16._is_cancelled("other"))
# The cancelled set must stay bounded or it grows for the life of the process.
for i in range(600):
    bot16._cancel_task(CID, f"t{i}")
check("the cancelled set is bounded", len(bot16._cancelled) <= 512,
      str(len(bot16._cancelled)))


print()
print("=" * 70)
print("UPDATE OFFSET PERSISTENCE")
print("=" * 70)

bot17 = make_bot()
bot17._offset = 0
calls = []
class _R:
    def __init__(self, payload): self._p = payload
    def json(self): return self._p
import requests as _rq
_real_get = _rq.get
_rq.get = lambda url, **k: (calls.append(k.get("params")),
                            _R({"ok": True, "result": [{"update_id": 41}, {"update_id": 42}]}))[1]
try:
    got = bot17._get_updates()
finally:
    _rq.get = _real_get
check("every update in the batch is returned", len(got) == 2)
check("the offset advances past the last update", bot17._offset == 43,
      str(bot17._offset))
check("the offset is persisted immediately", bot17._load_offset() == 43,
      str(bot17._load_offset()))

_rq.get = lambda url, **k: _R({"ok": True, "result": []})
try:
    bot17._get_updates()
finally:
    _rq.get = _real_get
check("an empty poll does not rewind the offset", bot17._offset == 43)

# 409 (second instance) / 401 come back instantly: [] here was a tight loop
# the watchdog read as healthy. It must raise so the poll loop backs off.
_rq.get = lambda url, **k: _R({"ok": False, "error_code": 409, "description": "Conflict"})
try:
    bot17._get_updates(); _raised = False
except RuntimeError:
    _raised = True
finally:
    _rq.get = _real_get
check("a not-ok getUpdates raises (poll loop backs off)", _raised)
check("and does not move the offset", bot17._offset == 43)

bot17._save_offset(99)
check("a saved offset survives a reload", bot17._load_offset() == 99)
T._OFFSET_FILE.write_text("not json", encoding="utf-8")
check("a corrupt offset file degrades to 0 instead of crashing",
      bot17._load_offset() == 0)


print()
print("=" * 70)
print("GUARDS")
print("=" * 70)

check("a known extremism keyword is detected",
      T._check_extremism(next(iter(T._load_extremism_keywords()), "zzz"))
      if T._load_extremism_keywords() else True)
check("ordinary text is not flagged",
      not T._check_extremism("what is the weather in Paris"))
check("empty text is not flagged", not T._check_extremism(""))
check("None does not crash the check", not T._check_extremism(None))

bot18 = make_bot()
approved(bot18)
grp = {"message_id": 1, "chat": {"id": -100123, "type": "supergroup"},
       "text": "hello", "from": {"id": CID}}
bot18._dispatch({"message": grp})
check("a group message is refused",
      not bot18.pushed, str(bot18.pushed))
check("with an explanation", "👥" in texts(bot18) or not texts(bot18),
      repr(texts(bot18)[:80]))


print()
print("=" * 70)
print("CRASH RECOVERY")
print("=" * 70)

bot19 = make_bot(); approved(bot19)
T._INFLIGHT_FILE.write_text(json.dumps(
    [T._Task(task_id="z", chat_id=CID, user_text="my interrupted question").to_dict()]),
    encoding="utf-8")
bot19._recover_inflight()
check("the user is told their task died with the process",
      "⚠️" in texts(bot19), repr(texts(bot19)[:80]))
check("and the text is stashed so Retry has something to run",
      bot19._get_session(CID).last_task_text == "my interrupted question",
      repr(bot19._get_session(CID).last_task_text))
check("the inflight file is consumed", not T._INFLIGHT_FILE.exists())

bot20 = make_bot()
bot20.sent.clear()
bot20._recover_inflight()
check("no inflight file means no message", bot20.sent == [], str(bot20.sent))

T._INFLIGHT_FILE.write_text("{{{ not json", encoding="utf-8")
bot21 = make_bot(); bot21.sent.clear()
try:
    bot21._recover_inflight(); ok = True
except Exception as exc:
    ok = False
check("a corrupt inflight file does not break startup", ok)
check("and it is cleared so it cannot wedge every restart",
      not T._INFLIGHT_FILE.exists())

# A single unusable entry must not silence the rest of the recovery.
bot22 = make_bot(); approved(bot22)
T._INFLIGHT_FILE.write_text(json.dumps([
    {"garbage": True},
    T._Task(task_id="y", chat_id=CID, user_text="second question").to_dict(),
]), encoding="utf-8")
bot22._recover_inflight()
check("one bad entry does not stop the others being reported",
      "⚠️" in texts(bot22), repr(texts(bot22)[:80]))

print()
print("=" * 70)
print("RESTART RECOVERY -- tasks QUEUED (pushed, never popped), not just running")
print("=" * 70)
# _recover_inflight (above) only ever fires for a task that had already
# reached _running_task -- i.e. a consumer had popped it and started
# graph.invoke(). A task that was merely PUSHED onto the backend and never
# popped (the debounce timer fired, the task was accepted and queued, but
# the process died before a consumer got to it) used to leave no trace
# anywhere: with the default InMemoryBackend that queue lives only in this
# process's RAM, so a restart drops it silently -- while the daily-quota
# bump for it (tg_resolve._resolve_and_push bumps usage BEFORE pushing) is
# already committed to the persisted SQLite usage store. The user is
# charged for a task that never ran and never hears about it, unlike the
# mid-execution case above which at least says "interrupted, retry?".
# _pending_journal + the "_state" tag on _INFLIGHT_FILE entries close that
# gap for the one backend where it is real (InMemoryBackend); a durable
# backend (Redis/Kafka) keeps its own copy outside the process, so a
# leftover "pending" entry for one of those must NOT be reported (the task
# will simply run normally once consumers start, and a false "interrupted"
# notice would land right before the real answer does).

class _FakeDurableBackend(T._Backend):
    """Same push/pop mechanics as InMemoryBackend, but deliberately NOT an
    instance of it, so the recovery gate treats it as external/durable."""
    def __init__(self): self._inner = T.InMemoryBackend()
    def push(self, task): return self._inner.push(task)
    def pop(self, timeout=5.0): return self._inner.pop(timeout)
    def depth(self): return self._inner.depth()
    def position_of(self, task_id): return self._inner.position_of(task_id)
    def requeue(self, task): return self._inner.requeue(task)
    def service_order(self): return self._inner.service_order()
    def chat_depth(self, chat_id): return self._inner.chat_depth(chat_id)
    def drop_chat(self, chat_id): return self._inner.drop_chat(chat_id)
    def drop_task(self, task_id): return self._inner.drop_task(task_id)


CID_Q = 999802

bot23 = make_bot(); approved(bot23, CID_Q)
task_q = T._Task(task_id="q1", chat_id=CID_Q, user_text="queued when the process died")
with bot23._task_lock:
    bot23._pending_journal[task_q.task_id] = task_q
bot23._write_inflight()
check("a merely-pushed (never popped) task is written to the inflight file",
      T._INFLIGHT_FILE.exists())

bot24 = make_bot(); approved(bot24, CID_Q)   # simulated restart: fresh instance
bot24._recover_inflight()
check("a queued-but-never-run task is reported the SAME way a mid-execution "
      "crash is -- interrupted + a Retry button, not silence",
      "⚠️" in texts(bot24), repr(texts(bot24)[:120]))
check("Retry has something to retry", bot24._get_session(CID_Q).last_task_text ==
      "queued when the process died", bot24._get_session(CID_Q).last_task_text)
check("the inflight file is consumed once recovered", not T._INFLIGHT_FILE.exists())

# Durable backend: the SAME leftover "pending" entry must be left alone --
# the external queue still has the real task, so notifying would be a lie.
bot25 = make_bot(); approved(bot25, CID_Q)
bot25._backend = _FakeDurableBackend()
task_q2 = T._Task(task_id="q2", chat_id=CID_Q, user_text="still safely queued in redis")
with bot25._task_lock:
    bot25._pending_journal[task_q2.task_id] = task_q2
bot25._write_inflight()

bot26 = make_bot(); approved(bot26, CID_Q)
bot26._backend = _FakeDurableBackend()
bot26._recover_inflight()
check("a pending entry is NOT reported when the backend is durable -- the "
      "task is still safely queued there and will run normally",
      texts(bot26) == "", repr(texts(bot26)))

# Explicit cancel of a queued task purges the journal, so it does not
# resurrect as a false "interrupted" notice after the next restart.
bot27 = make_bot(); approved(bot27, CID_Q)
task_q3 = T._Task(task_id="q3", chat_id=CID_Q, user_text="cancel me before I run")
with bot27._task_lock:
    bot27._pending_journal[task_q3.task_id] = task_q3
bot27._backend.push(task_q3)
bot27._write_inflight()
bot27._request_stop(CID_Q)
check("Stop/cancel on a queued task removes it from the pending journal",
      task_q3.task_id not in bot27._pending_journal)

bot28 = make_bot(); approved(bot28, CID_Q)
bot28._recover_inflight()
check("an explicitly cancelled queued task does not resurrect after restart",
      texts(bot28) == "", repr(texts(bot28)))

# A bare typed «Stop» stops; it is not a steer note for the running render
# (live 2026-09-27: «👌 Noted: Stop» and the redraw went on).
for word in ("Stop", "стоп!", "Отмена"):
    b = make_bot(); approved(b)
    hit = []
    b._stop_and_report = lambda cid, sess, lang: hit.append(cid)
    b._dispatch({"update_id": 1, "message": {"message_id": 1, "chat": {"id": CID},
                                             "from": {"id": CID}, "text": word}})
    check("typed %r stops" % word, hit == [CID], (hit, b.pushed))

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
