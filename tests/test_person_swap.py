"""🔁 Swap a person under a picture: the people as a numbered collage, a number picks one,
the next photo is who goes in their place -- one transfer task naming THAT person.

The owner's idea (2026-10-12): Stalin erased Yezhov; here anyone can be swapped for
anyone (SpongeBob), drawn in the photo's own style.

Run: venv/Scripts/python.exe tests/test_person_swap.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import logging
logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T
from PIL import Image

T.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_swap_"))
import person_swap

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1
        print(f"PASS  {name}")
    else:
        BAD += 1
        print(f"FAIL  {name}   {extra}")


CID = 999877
bot = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
bot.sent, bot.pushed, bot.photos = [], [], []
bot._send_text = lambda cid, text, **k: (bot.sent.append((cid, text, k.get("keyboard"))), 1)[1]
bot._send_photo = lambda cid, path, keyboard=None: (bot.photos.append((path, keyboard)), True)[1]
bot._activity.log = lambda *a, **k: None
bot._api_post = lambda *a, **k: {}
bot._dl_bytes = lambda fid: open(SPONGE, "rb").read()
real_push = bot._backend.push
bot._backend.push = lambda task: (bot.pushed.append(task), real_push(task))[0]
bot._user_store.put(T._User(chat_id=CID, name="U", status="approved"))

d = tempfile.mkdtemp()
PIC, SPONGE = os.path.join(d, "four.jpg"), os.path.join(d, "sponge.jpg")
Image.new("RGB", (800, 500), "grey").save(PIC)
Image.new("RGB", (300, 300), "yellow").save(SPONGE)
# four men left to right; the detector is a model, the flow around it is what is tested
person_swap.people = lambda path: [(80, 100, 140, 170), (260, 90, 320, 160),
                                   (440, 95, 500, 165), (620, 110, 680, 180)]

sess = bot._get_session(CID)
img_id = T._log_image(sess, PIC, label="stalin", src="user")
sess.target_image = img_id
bot._store.put(sess)

bot._cb_swap_person(CID)
check("the people come back as one numbered collage", len(bot.photos) == 1
      and os.path.exists(bot.photos[0][0]), bot.photos)
cbs = [b["callback_data"] for row in (bot.photos[0][1] or {}).get("inline_keyboard", []) for b in row]
check("...with a button per person, numbered left to right", cbs == ["swp:1", "swp:2", "swp:3", "swp:4"], cbs)

bot._cb_swap_pick(CID, "swp:4")
s = bot._get_session(CID)
check("a number picks that person and asks for who goes instead",
      "right" in s.swap_where and "fourth of the 4" in s.swap_where, s.swap_where)

# The swap runs itself, not through the agent: its transfer_image re-rendered and
# coloured the whole photo, kept the man and answered «done» (live 2026-10-12).
CALLS = []
OUT = os.path.join(d, "swapped.png")
Image.new("RGB", (800, 500), "white").save(OUT)
person_swap.swap = lambda ctx, path, box, ref, wish="": (CALLS.append((path, box, ref, wish)), OUT)[1]
bot.photos.clear()
bot._resolve_and_push(CID, [{"type": "photo", "file_id": "sponge", "caption": "пусть курит"}])
import time
for _ in range(50):
    if bot.photos:
        break
    time.sleep(0.1)
check("the next photo swaps directly, no agent task", not bot.pushed and len(CALLS) == 1, (bot.pushed, CALLS))
c = CALLS[0] if CALLS else None
check("...the picked person of the picked picture, the photo as who goes in, the caption as wishes",
      c and c[0] == PIC and c[1] == (620, 110, 680, 180) and os.path.exists(c[2]) and c[3] == "пусть курит", c)
check("...and the result comes back as a picture with its own buttons",
      bot.photos and bot.photos[-1][0] == OUT and "swap_person:" in str(bot.photos[-1][1]), bot.photos)
s = bot._get_session(CID)
check("the swap is spent: the next photo is an ordinary photo again",
      not s.pending_swap_target and not s.swap_where, (s.pending_swap_target, s.swap_where))

print(f"\n{OK}/{OK + BAD} passed")
sys.exit(1 if BAD else 0)
