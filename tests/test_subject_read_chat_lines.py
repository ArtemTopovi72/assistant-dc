"""The «which picture?» read sees the chat BEFORE the message, not the message twice.

The message is logged to the chat before the read runs; with it as the last chat line the read
answered «unclear» 3/3 for «Сколько стоит чай» right after a forwarded receipt video, and the bot
asked «Какую картинку?» instead of answering (2026-10-24). Without the echo: «current» 3/3.
"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["F5_TEST_RUN"] = "1"
import tg_bot
from fakes import reads

seen = []
ctx = types.SimpleNamespace(**reads(pick=lambda q, blob: seen.append(blob) or "current"))
recent = ["bot: 🔎 итог 13,00 составит 1171 рубль", "user: [forwards a video note]",
          "bot: 🎥 Видео · 23 с Что с ним сделать?", "user: Сколько стоит чай"]
assert tg_bot._message_subject("Сколько стоит чай", recent, ctx=ctx) == "current"
chat = seen[-1].split("CHAT BEFORE IT:\n", 1)[1]
assert chat.splitlines()[-1] == "bot: 🎥 Видео · 23 с Что с ним сделать?", chat
assert "[forwards a video note]" in chat
# a same-worded EARLIER message stays: only the trailing echo of this one goes
tg_bot._message_subject("да", ["user: да", "bot: точно?"], ctx=ctx)
assert "user: да" in seen[-1].split("CHAT BEFORE IT:\n", 1)[1], seen[-1]
print("ok the subject read sees the chat before the message, not its echo")
