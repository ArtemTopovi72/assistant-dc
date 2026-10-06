"""👗 Outfit: the answer can be words, a photo of the clothes, or both.

User 2026-10-06: «можно референс дать типо референс одежды + описание, либо
просто текст, либо пикчу». 👗 arms pending_outfit_target with the pressed
picture; the next photo is the CLOTHING reference (tg_resolve ->
_push_style_transfer_task(outfit_wish=caption)), never a new picture to edit.
Words alone still go through pending_prefix and disarm the photo wait.

Offline: bot._backend is a stub recording pushed tasks.
Run: venv/Scripts/python.exe tests/test_tg_outfit_reference.py
"""
# Script, not pytest: check() only counts, and the module-level sys.exit lives
# under `if __name__ == "__main__":` (indented), which tests/run_all.py's
# classify() cannot see -- undeclared, this file silently fell through to
# "skip" and never ran as part of the suite at all (caught 2026-09-18: it was
# missing from run_all's own summary).
RUN_AS_SCRIPT = True
import os, sys, tempfile, time, types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_outfit_")
import tg_bot as T
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")

CID = 999913
IMG_DIR = tempfile.mkdtemp(prefix="tgtest_outfit_imgs_")


def make_png(name):
    p = os.path.join(IMG_DIR, name)
    with open(p, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + os.urandom(64))
    return p


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: object(), lambda: object(),
                        lambda: {}, silent_mode=True)
    bot.sent = []
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
        chat_depth=lambda cid: 0, drop_chat=lambda cid: 0, close=lambda: None,
        tasks_ahead=lambda tid: [])
    bot._activity.log = lambda *a, **k: None
    # transfer_image is never actually called (offline); avoid a real download.
    bot._dl_bytes = lambda file_id: b"\x89PNG\r\n\x1a\n" + os.urandom(32)
    u = T._User(chat_id=CID, name="Tester", status="approved")
    bot._user_store.put(u)
    return bot


def seed_one_image(bot):
    s = bot._get_session(CID)
    s.clear_context()
    a = make_png("target.png")
    id_a = T._log_image(s, a, msg_id=101, label="target", src="bot")
    s.lang = "en"
    bot._store.put(s)
    return a, id_a


def press(bot, data, msg_id=900):
    bot._dispatch({"callback_query": {"id": "1", "data": data, "from": {"id": CID},
                                      "message": {"chat": {"id": CID, "type": "private"},
                                                  "message_id": msg_id}}})
    deadline = time.monotonic() + 5.0
    base_sent = len(bot.sent)
    while time.monotonic() < deadline:
        if len(bot.sent) != base_sent:
            time.sleep(0.05)
            return
        time.sleep(0.03)


def send_photo(bot, file_id="ref_file_1", caption=""):
    bot._dispatch({"message": {"chat": {"id": CID, "type": "private"},
                               "message_id": 901, "date": int(time.time()),
                               "photo": [{"file_id": file_id, "file_size": 1234,
                                         "width": 100, "height": 100}],
                               "caption": caption}})
    deadline = time.monotonic() + 5.0
    base = len(bot.pushed)
    while time.monotonic() < deadline:
        if len(bot.pushed) != base:
            time.sleep(0.1)
            return
        time.sleep(0.03)




for caption in ("", "только без шапки"):
    print("=" * 70)
    print(f"👗 -> photo of clothes (caption={caption!r}) -> transfer task on the ORIGINAL")
    print("=" * 70)
    bot = make_bot()
    path_a, id_a = seed_one_image(bot)
    press(bot, f"change_clothes:{id_a}")
    sess = bot._get_session(CID)
    check("pending_outfit_target armed on the pressed image",
          sess.pending_outfit_target == id_a, sess.pending_outfit_target)
    check("prompt mentions sending a photo", bool(bot.sent) and "photo" in bot.sent[-1][0], bot.sent)
    send_photo(bot, file_id="dress_1", caption=caption)
    check("exactly one task was pushed", len(bot.pushed) == 1, bot.pushed)
    if bot.pushed:
        t = bot.pushed[0]
        check("task targets the ORIGINAL picture", t.image_path == path_a, (t.image_path, path_a))
        check("the clothes photo rides as the reference",
              bool(t.style_ref_path) and os.path.exists(t.style_ref_path) and t.style_ref_path != path_a,
              t.style_ref_path)
        check("[outfit] transfer_image instruction", t.user_text.startswith("[outfit] call transfer_image"),
              t.user_text[:80])
        check("caption carried as the user's wish" if caption else "no wish block without a caption",
              (caption in t.user_text) if caption else ("wishes for the outfit" not in t.user_text),
              t.user_text)
        check("prefix not glued on", "change the outfit to:" not in t.user_text, t.user_text)
    s2 = bot._get_session(CID)
    check("waits cleared", s2.pending_outfit_target == "" and s2.pending_prefix == "",
          (s2.pending_outfit_target, s2.pending_prefix))

print("=" * 70)
print("👗 -> words only: the old prefix path, and the photo wait is disarmed")
print("=" * 70)
bot = make_bot()
path_a, id_a = seed_one_image(bot)
press(bot, f"change_clothes:{id_a}")
bot._dispatch({"message": {"chat": {"id": CID, "type": "private"}, "message_id": 902,
                           "date": int(time.time()), "text": "белое летнее платье"}})
deadline = time.monotonic() + 8.0
while time.monotonic() < deadline and not bot.pushed:
    time.sleep(0.05)
check("one task pushed", len(bot.pushed) == 1, bot.pushed)
if bot.pushed:
    check("text carries the outfit prefix", bot.pushed[0].user_text.startswith("change the outfit to: белое"),
          bot.pushed[0].user_text[:80])
    check("no reference image", not bot.pushed[0].style_ref_path)
check("photo wait disarmed", bot._get_session(CID).pending_outfit_target == "",
      bot._get_session(CID).pending_outfit_target)

print(f"\n{OK}/{OK+BAD} checks passed")
if __name__ == "__main__":
    sys.exit(0 if BAD == 0 else 1)
