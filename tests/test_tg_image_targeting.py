"""Point at a picture and get an answer about THAT picture.

The complaint: "the request to describe an image flies off to a random one from
the thread." It was not random — the bot kept ONE "current image" slot, so every
image instruction meant "whatever was newest", no matter which picture the user
was looking at or replying to. `reply_to_message` was never read anywhere in the
file, and every button under every picture sent a bare verb with no idea which
picture it sat under.

Three independent ways of pointing, each driven end to end through _dispatch —
the real update dicts Telegram sends, not method calls:

  A. REPLY to a picture  → that picture
  B. BUTTON under a picture → that picture, even a scrolled-up one
  C. AMBIGUOUS instruction → ask, and honour the answer

plus what must NOT change: one image, or a fresh upload, still goes straight
through without a question.

Run: venv/Scripts/python.exe tests/test_tg_image_targeting.py
"""
import sys, os, types, tempfile, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_target_")
import tg_bot as T
T.redirect_data_dir(_DATA_DIR)

# What the words ask of a picture is the model's read (agent/intent.py); the
# phrases run against the real model in bench/intent_picture_live.py.
import intent
intent.STUB = {
    "describe the image": {"is_question": True, "about_picture": True},
    "describe the first picture": {"is_question": True, "about_picture": True, "names_picture": 1},
    "make the cat from the first picture black": {"needs_tool": True, "wants": ["inpaint_image"],
                                                  "names_picture": 1},
    "опиши последнюю картинку": {"about_picture": True, "names_picture": -1},
    "restore quality": {"needs_tool": True, "wants": ["redraw_image"]},
    "hello, how are you": {"needs_tool": False},
}.get

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


CID = 999901
IMG_DIR = tempfile.mkdtemp(prefix="tgimgs_")


def make_png(name):
    """A real file on disk — the register drops entries whose file is gone, so a
    fake path would make every targeting check pass for the wrong reason."""
    p = os.path.join(IMG_DIR, name)
    with open(p, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + os.urandom(64))
    return p


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: object(), lambda: object(),
                        lambda: {}, silent_mode=True)
    bot.sent = []          # (text, keyboard)
    bot.pushed = []

    def _send(cid, text, **kw):
        bot.sent.append((text, kw.get("keyboard")))
        return 1
    bot._send_text = _send
    bot._send_get_id = lambda cid, text, **kw: (_send(cid, text, **kw), 1)[1]
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._backend = types.SimpleNamespace(
        push=lambda t: bot.pushed.append(t), depth=lambda: 0, name=lambda: "stub",
        drop_chat=lambda cid: 0, close=lambda: None)
    bot._activity.log = lambda *a, **k: None
    bot._dl_bytes = lambda fid: b"\x89PNG\r\n\x1a\n" + os.urandom(64)
    u = T._User(chat_id=CID, name="Tester", status="approved")
    bot._user_store.put(u)
    return bot


def texts(bot):
    return " ".join(t for t, _ in bot.sent)


def last_inline(bot):
    for _t_, kb in reversed(bot.sent):
        if isinstance(kb, dict) and "inline_keyboard" in kb:
            return kb
    return None


def msg(text, reply_to=None, **extra):
    m = {"chat": {"id": CID, "type": "private"}, "from": {"id": CID},
         "message_id": 500, "text": text}
    if reply_to is not None:
        m["reply_to_message"] = reply_to
    m.update(extra)
    return {"message": m}


def settle(bot, timeout=8.0):
    """Wait for the real debounce worker to hand the message on.

    _dispatch does NOT push synchronously — it buffers for _DEBOUNCE_S so two
    quick messages merge. Driving _resolve_and_push directly would skip the very
    layer where reply_to is resolved, so the wait is the price of testing the
    real path.
    """
    import time as _time
    # Baseline, not "is there anything": leftovers from an earlier step made this
    # return instantly and the check then read a queue that had not run yet.
    base = (len(bot.pushed), len(bot.sent))
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if (len(bot.pushed), len(bot.sent)) != base:
            _time.sleep(0.15)      # let a batch that has started finish
            return
        _time.sleep(0.05)


def press(bot, data):
    bot._dispatch({"callback_query": {"id": "1", "data": data, "from": {"id": CID},
                                      "message": {"chat": {"id": CID, "type": "private"},
                                                  "message_id": 900}}})
    # A button's instruction goes through the SAME debounce queue as typing.
    settle(bot)


def seed_two_images(bot):
    """Two pictures the bot 'sent', with the message ids they went out as."""
    s = bot._get_session(CID)
    s.clear_context()
    cat, dog = make_png("cat.png"), make_png("dog.png")
    id_cat = T._log_image(s, cat, msg_id=101, label="draw a cat", src="bot")
    id_dog = T._log_image(s, dog, msg_id=202, label="draw a dog", src="bot")
    s.last_image_path = dog          # the newest — what the old code always used
    s.lang = "en"
    bot._store.put(s)
    return cat, dog, id_cat, id_dog


bot = make_bot()

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("THE REGISTER")
print("=" * 70)

s = bot._get_session(CID); s.clear_context()
a, b = make_png("a.png"), make_png("b.png")
ia = T._log_image(s, a, msg_id=11, label="first")
ib = T._log_image(s, b, msg_id=22, label="second")
check("two pictures get two ids", ia != ib and len(s.image_log) == 2, s.image_log)
check("a picture is found by its id", T._image_by_id(s, ia)["path"] == a)
check("and by the message it was sent as", T._image_by_msg(s, 22)["path"] == b)
check("re-registering the same file does not duplicate it",
      T._log_image(s, a) == ia and len(s.image_log) == 2, s.image_log)
check("an unknown id resolves to nothing, not to the newest",
      T._image_by_id(s, "deadbeef") is None)
check("an unknown message id resolves to nothing", T._image_by_msg(s, 999) is None)
check("message id 0 (a send whose id we never learned) matches nothing",
      T._image_by_msg(s, 0) is None)
for i in range(T._IMAGE_LOG_MAX + 5):
    T._log_image(s, make_png(f"bulk{i}.png"), msg_id=1000 + i)
check("the register is capped and cannot grow forever",
      len(s.image_log) == T._IMAGE_LOG_MAX, len(s.image_log))
missing = T._log_image(s, os.path.join(IMG_DIR, "never_existed.png"), msg_id=7)
check("a registered file that is not on disk is not offered as a choice",
      all(e["id"] != missing for e in T._live_images(s)))
s.clear_context()
check("clearing the conversation clears the register too",
      s.image_log == [] and s.target_image == "", s.image_log)

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("A. REPLYING TO A PICTURE TARGETS THAT PICTURE")
print("=" * 70)

cat, dog, id_cat, id_dog = seed_two_images(bot)
bot.pushed.clear(); bot.sent.clear()
# Reply to the CAT (message 101) while the dog is the newest image.
bot._dispatch(msg("describe this picture",
                  reply_to={"message_id": 101, "chat": {"id": CID}}))
settle(bot)
s = bot._get_session(CID)
check("the reply armed the older picture, not the newest",
      s.target_image == id_cat, f"target={s.target_image!r} cat={id_cat}")
check("and the instruction still went through", len(bot.pushed) == 1,
      [p.user_text for p in bot.pushed])

# A reply to something with no picture behind it must change nothing.
s.target_image = ""; bot._store.put(s)
bot.pushed.clear()
bot._dispatch(msg("and this?", reply_to={"message_id": 55555, "chat": {"id": CID},
                                         "text": "some earlier sentence"}))
settle(bot)
s = bot._get_session(CID)
check("replying to a TEXT message does not fake an image target",
      s.target_image == "", s.target_image)
check("but the quoted line reaches the turn as context",
      bot.pushed and "some earlier sentence" in bot.pushed[-1].user_text,
      bot.pushed[-1].user_text if bot.pushed else "nothing pushed")
check("the quote is consumed, not left armed for the next message",
      bot._get_session(CID).quoted_text == "",
      bot._get_session(CID).quoted_text)

# Replying to a photo the bot never registered: re-fetch it from Telegram.
s = bot._get_session(CID); s.target_image = ""; bot._store.put(s)
bot.pushed.clear()
bot._dispatch(msg("what is in it?", reply_to={
    "message_id": 777, "chat": {"id": CID},
    "photo": [{"file_id": "F1", "width": 90, "height": 90},
              {"file_id": "F2", "width": 800, "height": 600}]}))
settle(bot)
s = bot._get_session(CID)
tgt = T._image_by_id(s, s.target_image)
check("replying to a photo we never stored re-fetches it", tgt is not None,
      s.target_image)
check("and the fetched file really exists on disk",
      tgt and os.path.exists(tgt["path"]), tgt)
check("it is recorded as the USER's picture, not ours",
      tgt and tgt.get("src") == "user", tgt)

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("B. THE BUTTONS UNDER A PICTURE BELONG TO THAT PICTURE")
print("=" * 70)

cat, dog, id_cat, id_dog = seed_two_images(bot)
kb = T._image_kb("en", id_cat)
datas = [b["callback_data"] for row in kb["inline_keyboard"] for b in row]
check("every button carries the picture's id",
      all(d.endswith(":" + id_cat) for d in datas), datas)
check("the labels are still localized",
      T._image_kb("ru", id_cat)["inline_keyboard"][0][0]["text"]
      != T._image_kb("en", id_cat)["inline_keyboard"][0][0]["text"])

bot.pushed.clear(); bot.sent.clear()
press(bot, "regenerate:" + id_cat)          # the OLDER picture
s = bot._get_session(CID)
check("pressing 🔄 under the older picture targets the older picture",
      s.target_image == id_cat, f"{s.target_image!r} != {id_cat}")
check("and the regenerate instruction was queued", len(bot.pushed) == 1,
      [p.user_text for p in bot.pushed])

# A bare verb (a keyboard sent before this change) must keep working.
s.target_image = ""; bot._store.put(s)
bot.pushed.clear()
# A DIFFERENT verb on purpose: two identical machine payloads in a row are a
# double-tap and the repeat guard drops the second — correct behaviour, and
# repeating "upscale" here would have been testing that guard, not this path.
press(bot, "regenerate")
check("an old bare-verb button still works", len(bot.pushed) == 1,
      [p.user_text for p in bot.pushed])
check("and it means 'the current one', as it always did",
      bot._get_session(CID).target_image == "")

# An old keyboard still identifies its picture: the callback names the MESSAGE it
# is attached to, and that message IS the picture.
cat, dog, id_cat, id_dog = seed_two_images(bot)
bot.pushed.clear(); bot.sent.clear()
bot._dispatch({"callback_query": {"id": "1", "data": "regenerate", "from": {"id": CID},
                                  "message": {"chat": {"id": CID, "type": "private"},
                                              "message_id": 101}}})   # the CAT
settle(bot)
check("a bare verb pressed under the older picture still resolves to it",
      bot._get_session(CID).target_image == id_cat,
      f"{bot._get_session(CID).target_image!r} != {id_cat}")
check("and it was queued, not turned into a question", len(bot.pushed) == 1,
      texts(bot)[:120])

# A button pointing at a file that is gone must SAY so, not act on another one.
s = bot._get_session(CID)
gone = T._log_image(s, os.path.join(IMG_DIR, "deleted.png"), msg_id=303)
bot._store.put(s)
bot.pushed.clear(); bot.sent.clear()
press(bot, "regenerate:" + gone)
check("a button for a deleted picture refuses instead of acting on another",
      not bot.pushed and T._t("img_gone", "en")[:20] in texts(bot),
      f"pushed={len(bot.pushed)} said={texts(bot)[:90]}")

# A colon in some OTHER callback must never be read as an image reference.
bot.pushed.clear(); bot.sent.clear()
press(bot, "depth:deep")
check("an unrelated callback with a colon is untouched",
      bot._get_session(CID).dr_depth == "deep",
      bot._get_session(CID).dr_depth)
check("a malformed image id is not accepted",
      T._IMAGE_ID_RE.fullmatch("zzz") is None
      and T._IMAGE_ID_RE.fullmatch(id_cat) is not None)

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("B2. TWO BUTTON PRESSES ON DIFFERENT PICTURES IN ONE DEBOUNCE WINDOW")
print("=" * 70)
# Found by an adversarial harness driving the real bot: tap [upscale] under an
# OLDER picture, then immediately tap [upscale] under a NEWER one, before the
# debounce timer fires. Both items land in the same batch. The batch used to
# be merged into ONE task with a single image_id (whichever item came last) —
# and since both presses produce byte-identical command text, the "collapse a
# double-tap" dedup then also folded the two text entries into one, so the
# FIRST picture's upscale request vanished with no error, no result, no trace.

cat, dog, id_cat, id_dog = seed_two_images(bot)
bot.pushed.clear(); bot.sent.clear()
# Two raw dispatches back to back, deliberately NOT settling in between, so
# both land in the same debounce batch.
bot._dispatch({"callback_query": {"id": "1", "data": "regenerate:" + id_cat,
                                  "from": {"id": CID},
                                  "message": {"chat": {"id": CID, "type": "private"},
                                              "message_id": 101}}})
bot._dispatch({"callback_query": {"id": "2", "data": "regenerate:" + id_dog,
                                  "from": {"id": CID},
                                  "message": {"chat": {"id": CID, "type": "private"},
                                              "message_id": 202}}})
settle(bot)
check("BOTH presses survive as separate requests, not one merged/lossy task",
      len(bot.pushed) == 2, [(p.user_text[:12], p.image_id) for p in bot.pushed])
check("the FIRST press still targets the picture it was pressed under",
      any(p.image_id == id_cat for p in bot.pushed),
      [(p.user_text[:12], p.image_id) for p in bot.pushed])
check("the SECOND press targets its own, different picture",
      any(p.image_id == id_dog for p in bot.pushed),
      [(p.user_text[:12], p.image_id) for p in bot.pushed])

# The SAME two presses on the SAME picture must still collapse (the original
# double-tap fix this guards against regressing).
bot.pushed.clear(); bot.sent.clear()
bot._dispatch({"callback_query": {"id": "3", "data": "regenerate:" + id_cat,
                                  "from": {"id": CID},
                                  "message": {"chat": {"id": CID, "type": "private"},
                                              "message_id": 101}}})
bot._dispatch({"callback_query": {"id": "4", "data": "regenerate:" + id_cat,
                                  "from": {"id": CID},
                                  "message": {"chat": {"id": CID, "type": "private"},
                                              "message_id": 101}}})
settle(bot)
check("a genuine double-tap on the SAME picture still collapses to one task",
      len(bot.pushed) == 1, [(p.user_text[:12], p.image_id) for p in bot.pushed])

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("C. WHEN IT CANNOT KNOW, IT ASKS — AND HONOURS THE ANSWER")
print("=" * 70)

cat, dog, id_cat, id_dog = seed_two_images(bot)
bot.pushed.clear(); bot.sent.clear()
bot._dispatch(msg("describe the image"))
settle(bot)
check("an ambiguous image request is NOT guessed", not bot.pushed,
      [p.user_text for p in bot.pushed])
check("the user is asked which one", "Which picture" in texts(bot), texts(bot)[:120])
kb = last_inline(bot)
picks = [b["callback_data"] for row in (kb or {}).get("inline_keyboard", [])
         for b in row]
check("the picker offers every live picture plus 'the latest'",
      "pick:latest" in picks and ("pick:" + id_cat) in picks
      and ("pick:" + id_dog) in picks, picks)
check("the labels say what each picture was", "draw a cat" in str(kb), str(kb)[:200])
check("the instruction is held, not thrown away",
      bot._get_session(CID).pending_instruction == "describe the image",
      bot._get_session(CID).pending_instruction)

bot.sent.clear()
press(bot, "pick:" + id_cat)
s = bot._get_session(CID)
check("picking re-runs the held instruction", len(bot.pushed) == 1,
      [p.user_text for p in bot.pushed])
check("the user does not have to retype it",
      bot.pushed and bot.pushed[0].user_text == "describe the image",
      [p.user_text for p in bot.pushed])
check("and the held instruction is cleared", s.pending_instruction == "",
      s.pending_instruction)
check("the re-pushed task carries the PICKED image's id on the task itself "
      "(captured at pick: time), not left to a lazy re-read of "
      "sess.target_image at execution time -- same 'captured at press time' "
      "contract as every button/reply path in this router",
      bot.pushed and bot.pushed[0].image_id == id_cat,
      [(p.user_text, p.image_id) for p in bot.pushed])

# Live 2026-09-14: «найди все мосты, убери дубликаты ... коллаж» + DCIM.zip
# got «Какую картинку? В этом чате их 9». The document IS what the
# instruction is about; the old collages in the chat are not candidates.
bot2 = make_bot(); seed_two_images(bot2); bot2.pushed.clear(); bot2.sent.clear()
import sandbox_access as _sa
_u = bot2._user_store.get(CID); _u.sandbox_access = _sa.FILES; bot2._user_store.put(_u)
bot2._resolve_and_push(CID, [{"type": "document", "file_id": "z1", "filename": "DCIM.zip",
                              "caption": "убери дубликаты и собери коллаж"}])
settle(bot2)
check("an instruction on an attached file is NOT a 'which picture?' question",
      "Which picture" not in texts(bot2) and "Какую картинку" not in texts(bot2), texts(bot2)[:120])
check("it is pushed to the agent", len(bot2.pushed) == 1, [p.user_text[:40] for p in bot2.pushed])

# Telegram's "which picture?" keyboard never expires. If picking an answer
# only set sess.target_image (the pre-fix behaviour) instead of stamping the
# re-pushed task's own image_id, a SECOND tap on that same stale keyboard --
# for a totally different, already-answered prompt, out of curiosity or by
# accident -- silently overwrote target_image before the first re-pushed
# task got a chance to run (it queues behind whatever else the chat is
# doing), so it executed against whichever picture was picked LAST, not the
# one it was actually re-armed for. Reproduced live via the adversarial
# harness (probe4.py in scratchpad): pick B for "upscale this", then --
# before that queued task runs -- tap the same stale keyboard for A; the
# task silently ran on A.
cat, dog, id_cat, id_dog = seed_two_images(bot)
bot.pushed.clear(); bot.sent.clear()
bot._dispatch(msg("describe the image"))
settle(bot)
press(bot, "pick:" + id_dog)          # answer: dog
check("first pick re-pushed exactly one task, targeting dog",
      len(bot.pushed) == 1 and bot.pushed[0].image_id == id_dog,
      [(p.user_text, p.image_id) for p in bot.pushed])
# A second tap on the SAME (never-expiring) keyboard, for the SAME
# already-answered prompt -- pending_instruction is empty now, so this must
# NOT re-push a second task, but it still moves sess.target_image.
press(bot, "pick:" + id_cat)
check("the stale second tap does not re-push another task",
      len(bot.pushed) == 1, [(p.user_text, p.image_id) for p in bot.pushed])
check("the FIRST task's stamped image_id is still dog, unaffected by the "
      "later stale tap for cat",
      bot.pushed[0].image_id == id_dog, bot.pushed[0].image_id)

# 'the latest' must mean the newest, not the first in the list.
cat, dog, id_cat, id_dog = seed_two_images(bot)
bot.pushed.clear(); bot.sent.clear()
bot._dispatch(msg("describe the image"))
settle(bot)
press(bot, "pick:latest")
check("'the most recent one' resolves to the newest picture",
      bot._get_session(CID).target_image == id_dog,
      f"{bot._get_session(CID).target_image} != {id_dog}")

# A stale "which picture?" prompt must not resurrect a forgotten instruction.
# Telegram buttons never expire, and pending_instruction used to be cleared
# ONLY by pressing pick: on the prompt it was armed for — so unrelated
# activity in between left it live, and tapping the old button later (out of
# curiosity, or by accident) silently re-ran whatever was asked long ago.
cat, dog, id_cat, id_dog = seed_two_images(bot)
bot.pushed.clear(); bot.sent.clear()
bot._dispatch(msg("restore quality"))
settle(bot)
check("the ambiguous instruction was held",
      bot._get_session(CID).pending_instruction == "restore quality",
      bot._get_session(CID).pending_instruction)
bot.pushed.clear()
bot._dispatch(msg("hello, how are you"))
settle(bot)
bot._dispatch(msg("tell me a joke"))
settle(bot)
check("the unrelated turns were queued normally",
      [p.user_text for p in bot.pushed] == ["hello, how are you", "tell me a joke"],
      [p.user_text for p in bot.pushed])
check("the abandoned instruction was dropped once other activity happened",
      bot._get_session(CID).pending_instruction == "",
      bot._get_session(CID).pending_instruction)
bot.pushed.clear()
press(bot, "pick:" + id_cat)
check("the stale button just picks the image — the forgotten instruction "
      "does NOT fire on its own", not bot.pushed,
      [p.user_text for p in bot.pushed])

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("D. A QUEUED REPLY SURVIVES A LATER REPLY TO A DIFFERENT PICTURE")
print("=" * 70)

# Reproduces a real defect: reply to picture A, then — BEFORE that task
# actually runs (it is still sitting in the queue behind whatever this chat
# is already doing) — reply to picture B. The second reply used to overwrite
# the single sess.target_image slot, and since only BUTTON presses captured
# their target at commit time (task.image_id), the queued reply-to-A task
# read the shared slot lazily at execution time and silently answered about
# B instead of A. The fix threads _resolve_reply_target's result into the
# enqueued item's "image_id", exactly like an image-verb button already does.
cat, dog, id_cat, id_dog = seed_two_images(bot)
bot.pushed.clear(); bot.sent.clear()

bot._dispatch(msg("describe this picture",
                  reply_to={"message_id": 101, "chat": {"id": CID}}))  # cat
settle(bot)
check("task A was queued", len(bot.pushed) == 1, [p.user_text for p in bot.pushed])
task_a = bot.pushed[-1]
check("task A carries the CAT's id at commit time, not a lazy live read",
      task_a.image_id == id_cat, f"task_a.image_id={task_a.image_id!r} cat={id_cat}")

bot._dispatch(msg("what is this",
                  reply_to={"message_id": 202, "chat": {"id": CID}}))  # dog
settle(bot)
check("task B was queued too, and the shared slot now points at the dog",
      len(bot.pushed) == 2 and bot._get_session(CID).target_image == id_dog,
      (len(bot.pushed), bot._get_session(CID).target_image))
task_b = bot.pushed[-1]
check("task B carries the DOG's id at commit time",
      task_b.image_id == id_dog, f"task_b.image_id={task_b.image_id!r} dog={id_dog}")

# task A must still resolve against the CAT when it finally runs, not the
# DOG that overwrote the shared slot in the meantime.
sess = bot._get_session(CID)
resolved_id = getattr(task_a, "image_id", "") or getattr(sess, "target_image", "")
resolved = T._image_by_id(sess, resolved_id)
check("task A resolves against the picture it was actually a reply to (the cat)",
      resolved and resolved["path"] == cat,
      f"resolved={resolved!r} expected cat={cat!r}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("WHAT MUST NOT CHANGE")
print("=" * 70)

# One picture in the chat: nothing to disambiguate, so do not nag.
s = bot._get_session(CID); s.clear_context(); s.lang = "en"
T._log_image(s, make_png("only.png"), msg_id=1)
bot._store.put(s)
bot.pushed.clear(); bot.sent.clear()
bot._dispatch(msg("describe the image"))
settle(bot)
check("with only one picture it just answers", len(bot.pushed) == 1,
      texts(bot)[:120])

# A photo sent WITH the instruction is unambiguous by construction.
cat, dog, id_cat, id_dog = seed_two_images(bot)
bot.pushed.clear(); bot.sent.clear()
bot._dispatch({"message": {"chat": {"id": CID, "type": "private"},
                           "from": {"id": CID}, "message_id": 601,
                           "caption": "describe this image",
                           "photo": [{"file_id": "F9", "width": 800, "height": 600}]}})
settle(bot)
check("a freshly attached photo is never questioned", len(bot.pushed) == 1,
      texts(bot)[:140])
check("and that upload joins the register for later reference",
      any(e.get("src") == "user" for e in bot._get_session(CID).image_log),
      bot._get_session(CID).image_log)

# Ordinary conversation must not trip the picker.
cat, dog, id_cat, id_dog = seed_two_images(bot)
for phrase in ("hello, how are you", "what is the capital of Australia",
               "напиши стих про осень", "describe the first picture"):
    bot.pushed.clear(); bot.sent.clear()
    bot._dispatch(msg(phrase))
    settle(bot)
    check(f"'{phrase[:34]}' is not treated as an image request",
          len(bot.pushed) == 1, texts(bot)[:100])

# Torture run 2026-09-14: «сделай кота из первой картинки чёрным» edited the
# photo sent LAST. Naming a picture by its place in the chat must pick it.
bot3 = make_bot(); cat3, dog3, id_cat3, id_dog3 = seed_two_images(bot3)
bot3.pushed.clear(); bot3.sent.clear()
bot3._dispatch(msg("make the cat from the first picture black")); settle(bot3)
s3 = bot3._get_session(CID)
check("'the first picture' targets the OLDEST register entry, and asks nothing",
      len(bot3.pushed) == 1 and s3.target_image == id_cat3 and "Which picture" not in texts(bot3),
      (s3.target_image, id_cat3, texts(bot3)[:80]))
s3.target_image = ""; bot3._store.put(s3); bot3.pushed.clear()
bot3._dispatch(msg("опиши последнюю картинку")); settle(bot3)
check("«последнюю картинку» targets the newest", bot3._get_session(CID).target_image == id_dog3,
      (bot3._get_session(CID).target_image, id_dog3))

# 🖼 Create picture: the description is a NEW picture, never «which one?»
# (live 10-03: a scene with a broken window got «Какую картинку? их 12»).
_scene = "cat sits on the table, the window glass is broken"
intent.STUB = (lambda t, _o=intent.STUB: {"about_picture": True, "wants": ["inpaint_image"]}
               if t == _scene else _o(t))
seed_two_images(bot)
bot.pushed.clear(); bot.sent.clear()
s = bot._get_session(CID); s.pending_prefix = T._PROMPT_KB["gen_image"]; bot._store.put(s)
bot._dispatch(msg(_scene)); settle(bot)
check("a Create-picture description is not asked «which picture?»",
      "Which picture" not in " ".join(t for t, _ in bot.sent), bot.sent)
check("...and goes on as a drawing request", len(bot.pushed) == 1
      and "generate an image of" in str(getattr(bot.pushed[0], "user_text", bot.pushed[0])), bot.pushed[:1])

# The target is a gesture about ONE message, never a sticky mode.
s = bot._get_session(CID)
check("the target is stored per turn and persisted",
      "target_image" in s.to_dict() and "image_log" in s.to_dict(),
      list(s.to_dict().keys())[:12])
raw = s.to_dict()
check("the register survives a storage round-trip",
      len(T._Session(CID, raw).image_log) == len(s.image_log))

print(f"\n{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
