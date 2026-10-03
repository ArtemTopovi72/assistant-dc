"""A user who lives in Novosibirsk gets Novosibirsk time and 'at 09:00' reminders."""
import os, sys, datetime as dt, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")
from zoneinfo import ZoneInfo
import prompts, tools, reminders

nsk = dt.datetime.now(ZoneInfo("Asia/Novosibirsk"))
assert f"{nsk:%H:%M}" in prompts._now_line("Asia/Novosibirsk") and "UTC+07:00" in prompts._now_line("Asia/Novosibirsk")
assert "UTC+07:00" in prompts.build_system_prompt_lite(tz="Asia/Novosibirsk")

reminders.register(lambda *a: None, os.path.join(tempfile.mkdtemp(), "r.json"))
class C:
    user_tz = "Asia/Novosibirsk"; reply_lang = "ru"; reminder_owner = 42
out = tools._handle_set_reminder(C(), {}, {"text": "таблетка", "at": "09:00"})
due = reminders.pending(42)[0]["due"]
assert dt.datetime.fromtimestamp(due, ZoneInfo("Asia/Novosibirsk")).strftime("%H:%M") == "09:00", out
assert "09:00" in out, out
print("PASS user timezone")
