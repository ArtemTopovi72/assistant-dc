"""Send files to the owner's Telegram chat with the bot's saved token (never printed).
    venv/Scripts/python.exe bench/tg_send.py file [file ...] [--caption "text"] [--chat ID]"""
import sys, requests
from pathlib import Path
from PyQt5.QtCore import QSettings
a = sys.argv[1:]
arg = lambda k, d=None: a[a.index(k) + 1] if k in a else d
chat, cap = arg("--chat", os.environ.get("TG_CHAT_ID", "")), arg("--caption", "")
files = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or not a[i - 1].startswith("--"))]
s = QSettings("AssistantApp", "TelegramBot")
tok = next(str(s.value(k)) for k in s.allKeys() if "token" in k.lower() and s.value(k))
for f in files:
    p = Path(f)
    m, field = (("sendAudio", "audio") if p.suffix in (".mp3", ".wav", ".m4a")
                else ("sendVideo", "video") if p.suffix == ".mp4" else ("sendDocument", "document"))
    for base in ("http://127.0.0.1:8081", "https://api.telegram.org"):
        try:
            r = requests.post(f"{base}/bot{tok}/{m}", data={"chat_id": chat, "caption": cap or p.name},
                              files={field: (p.name, open(p, "rb"))}, timeout=300)
            if r.ok:
                print("sent", p.name, base); break
            print(base, r.status_code, r.text[:150])
        except Exception as e:
            print(base, type(e).__name__)
