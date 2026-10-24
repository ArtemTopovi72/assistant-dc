"""A follow-up continues on the picture the previous turn was about.

In a chat that has drawn a car, read a receipt and just recoloured a dress (six
pictures, three roots), "теперь убери все надписи с фона" must run the edit,
not answer "Which picture? There are 6". The picture the previous turn
delivered (or the user sent, or pointed at) is "the picture" for a follow-up;
older ones stay one reply/button away.
"""
import sys, os, types, tempfile
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_inplay_")
import tg_bot as T
T.redirect_data_dir(_DATA_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakes import reads   # the model's read of the words (agent/intent.py)
# Whether the chat leaves the picture open is one read of the message with the chat lines
# before it (tg_bot._message_subject, scored on bench/which_picture_eval.py); scripted here.
_R = types.SimpleNamespace(**reads(read={"describe the image": {"is_question": True, "about_picture": True},
                                         "now remove all the lettering from the background":
                                             {"needs_tool": True, "wants": ["inpaint_image"]}}.get,
                                   pick=lambda q, blob: "unclear" if blob.startswith("MESSAGE: describe")
                                   else "current"))
# The dispatch harness from test_tg_image_targeting (that module exits at
# import, so its helpers are re-bound here by source, not imported).
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_tg_image_targeting.py"), encoding="utf-8").read()
_helpers = _src.split("bot = make_bot()")[0].split("OK = BAD = 0")[1]
H = types.ModuleType("H"); H.__dict__.update(dict(sys=sys, os=os, types=types, tempfile=tempfile, T=T, _R=_R))
exec(_helpers, H.__dict__)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

print("\n" + "=" * 70); print("PICTURE IN PLAY"); print("=" * 70)
bot = H.make_bot()
bot.reads = _R
s = bot._get_session(H.CID); s.clear_context()
car, receipt, girl, dress = (H.make_png(n) for n in ("car.png", "receipt.png", "girl.png", "dress.png"))
id_car = T._log_image(s, car, msg_id=1, label="draw a car", src="bot")
T._log_image(s, receipt, msg_id=2, label="receipt", src="user")
id_girl = T._log_image(s, girl, msg_id=3, label="photo", src="user")
id_dress = T._log_image(s, dress, msg_id=4, label="black dress", src="bot", parent=id_girl)
s.last_image_path = dress; s.lang = "en"
s.turn_image = id_dress            # the previous turn delivered the dress edit
bot._store.put(s)

bot.pushed.clear(); bot.sent.clear()
bot._dispatch(H.msg("now remove all the lettering from the background"))
H.settle(bot)
check("no 'Which picture?' when the newest picture is in play", "Which picture" not in H.texts(bot), H.texts(bot)[:120])
check("the instruction is pushed", len(bot.pushed) == 1, [p.user_text for p in bot.pushed])
# The chat leaves it open -> a real choice.
s = bot._get_session(H.CID); s.turn_image = ""; s.target_image = ""; bot._store.put(s)
bot.pushed.clear(); bot.sent.clear()
bot._dispatch(H.msg("describe the image")); H.settle(bot)
check("without a picture in play, several roots still ask", "Which picture" in H.texts(bot) and not bot.pushed)

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_tg_image_targeting.py"), encoding="utf-8").read()
_helpers = _src.split("bot = make_bot()")[0].split("OK = BAD = 0")[1]
H = types.ModuleType("H"); H.__dict__.update(dict(sys=sys, os=os, types=types, tempfile=tempfile, T=T, _R=_R))
exec(_helpers, H.__dict__)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

print("\n" + "=" * 70); print("PICTURE IN PLAY"); print("=" * 70)
bot = H.make_bot()
bot.reads = _R
s = bot._get_session(H.CID); s.clear_context()
car, receipt, girl, dress = (H.make_png(n) for n in ("car.png", "receipt.png", "girl.png", "dress.png"))
id_car = T._log_image(s, car, msg_id=1, label="draw a car", src="bot")
T._log_image(s, receipt, msg_id=2, label="receipt", src="user")
id_girl = T._log_image(s, girl, msg_id=3, label="photo", src="user")
id_dress = T._log_image(s, dress, msg_id=4, label="black dress", src="bot", parent=id_girl)
s.last_image_path = dress; s.lang = "en"
s.turn_image = id_dress            # the previous turn delivered the dress edit
bot._store.put(s)

bot.pushed.clear(); bot.sent.clear()
bot._dispatch(H.msg("now remove all the lettering from the background"))
H.settle(bot)
check("no 'Which picture?' when the newest picture is in play", "Which picture" not in H.texts(bot), H.texts(bot)[:120])
check("the instruction is pushed", len(bot.pushed) == 1, [p.user_text for p in bot.pushed])
# The chat leaves it open -> a real choice.
s = bot._get_session(H.CID); s.turn_image = ""; s.target_image = ""; bot._store.put(s)
bot.pushed.clear(); bot.sent.clear()
bot._dispatch(H.msg("describe the image")); H.settle(bot)
check("without a picture in play, several roots still ask", "Which picture" in H.texts(bot) and not bot.pushed)

# In play but NOT the newest (the user drew something else since) -> ask.
s = bot._get_session(H.CID); s.pending_instruction = ""; s.target_image = ""; s.turn_image = id_car; bot._store.put(s)
bot.pushed.clear(); bot.sent.clear()
bot._dispatch(H.msg("describe the image")); H.settle(bot)
check("a stale in-play id (not the newest) does not silently win", "Which picture" in H.texts(bot) and not bot.pushed)

# ✍️ after a forward: the text rides with the forwarded material, which IS its subject.
# «Что думаешь?» after «✍️ Напиши, что сделать с пересланным» got «Какую картинку?» (2026-10-22).
s = bot._get_session(H.CID); s.pending_instruction = ""; s.target_image = ""; s.turn_image = ""
s.fwd_own = "Alex: [video] a cafe receipt, CAJ MINT 6,50"; bot._store.put(s)
bot.pushed.clear(); bot.sent.clear()
bot._dispatch(H.msg("describe the image")); H.settle(bot)
check("a question about forwarded material never asks which picture",
      "Which picture" not in H.texts(bot) and len(bot.pushed) == 1 and "CAJ MINT" in bot.pushed[0].user_text,
      (H.texts(bot)[:120], [p.user_text[:80] for p in bot.pushed]))

src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(T.__file__))), "bot/tg_tasks.py"), encoding="utf-8").read()
check("a delivered picture becomes the one in play", "sess.turn_image = img_id" in src)
check("a turn without a picture clears it", 'sess.turn_image = (_target or {}).get("id", "") or (' in src)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
