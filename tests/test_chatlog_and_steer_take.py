import os,sys,json,tempfile,logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_bot as T; d=tempfile.mkdtemp(); T.redirect_data_dir(d)
import chatlog, steer
bot=T.TelegramBot("1:T",lambda:object(),lambda:object(),lambda:{},silent_mode=True)
with chatlog.bind(5): logging.getLogger("assistant.x").warning("tool draw(...) -> ok")
chatlog.outgoing("sendMessage",{"chat_id":5,"text":"x"*5000})
m={"message_id":1,"text":"hunter2"}; m["_secret"]=True
bot._dispatch=lambda u: None
bot._dispatch_logged({"message":{"chat":{"id":5},"text":"hunter2","_secret":True}})
bot._dispatch_logged({"message":{"chat":{"id":5},"text":"привет"}})
rows=[json.loads(l) for l in open(os.path.join(chatlog._dir(),"5.jsonl"),encoding="utf-8")]
assert rows[0]["kind"]=="log" and "tool draw" in rows[0]["data"]
assert len(rows[1]["data"]["text"])==5000
assert "hunter2" not in json.dumps(rows) and rows[3]["data"]["text"]=="привет"
ib=steer.Inbox(); c=type("C",(),{})(); c.steer_inbox=ib; ib.put("сделай рок")
n=steer.take(c); assert n==["сделай рок"] and "сделай рок" in steer.with_notes("песня о коте",n) and steer.take(c)==[]
print("OK")
import keyboard_layout as KL
assert KL.fix("привет rfr дела?") == "привет как дела?" and KL.fix("RTX 3090") == "RTX 3090"
bot.sent = []; bot._send_text = lambda cid, t, **k: bot.sent.append(t)
seen = []; bot._dispatch = lambda u: seen.append(u["message"]["text"])
bot._dispatch_logged({"message": {"chat": {"id": 5}, "text": "ghbdtn rfr ltkf"}})
assert seen == ["привет как дела"] and "привет как дела" in bot.sent[0], (seen, bot.sent)
print("layout OK")
