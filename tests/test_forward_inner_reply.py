"""A reply INSIDE forwarded material is its author's, written inline; the user's
own reply keeps the quote frame (live 2026-10-01: Stepan's «Подтверди или
опровергни» quoting «Можно любой положить, это дефолтный» was framed as the
user replying, its close ended the forward's frame early, and the bot answered
about the Ozon cart)."""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import tg_bot as T
T.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_fwdreply_"))
from prompt_guard import QUOTE_OPEN, user_words

bot = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
bot._send_text = lambda *a, **k: 1
bot._activity.log = lambda *a, **k: None
CID = 777
fails = 0
def check(name, cond, extra=""):
    global fails
    print(("PASS  " if cond else "FAIL  ") + name, "" if cond else extra)
    fails += not cond

fwd = {"chat": {"id": CID}, "from": {"id": CID}, "message_id": 5, "text": "Подтверди или опровергни, плз",
       "forward_origin": {"type": "user", "sender_user": {"id": 42, "first_name": "Stepan"}},
       "reply_to_message": {"message_id": 3, "text": "Можно любой положить, это дефолтный"}}
bot._resolve_reply_target(CID, fwd)
check("forwarded reply is annotated inline, no nested frame",
      QUOTE_OPEN not in fwd["text"] and "Можно любой положить" in fwd["text"], fwd["text"])
check("and it never becomes the user's quote", not bot._get_session(CID).quoted_text)

own = {"chat": {"id": CID}, "from": {"id": CID}, "message_id": 6, "text": "что ответить?",
       "reply_to_message": {"message_id": 3, "text": "Чей голос?"}}
bot._resolve_reply_target(CID, own)
check("the user's own reply: words stay clean, the quote rides beside them",
      own["text"] == "что ответить?" and own.get("_quote") == "Чей голос?", own)
check("and user_words drops the quote", user_words(own["text"]) == "что ответить?", user_words(own["text"]))
sys.exit(1 if fails else 0)
