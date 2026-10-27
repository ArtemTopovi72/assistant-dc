"""A bare text follow-up about the picture ("Что там написано?" / "what's written
there?") is answered from the CURRENT photo.

`vision_agent_node` (graph.py) only runs vision analysis on the turn a NEW photo
arrives (`state["image_data"]` present). With no fresh photo the answer used to
come from `ctx.session_memory`, which holds ONE vision summary per photo ever
sent with nothing marking which is "current". `vision_agent_node` therefore
re-analyzes `ctx.last_image_path` FRESH whenever there is no new photo but the
question plainly needs a look at the picture, and tg_bot's `_needs_image_choice`
(the "which picture do you mean?" ambiguity gate for when 2+ images are live)
recognizes the same "written"/"написано" phrasing.

Run: venv/Scripts/python.exe tests/test_followup_image_reask.py
"""
import os, sys, tempfile, threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import graph as G
import tg_bot as T
import models

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


# Whether a message is about the picture on screen is the model's read
# (agent/intent.py about_picture); the phrases run live in bench/intent_live.py.
import types
from fakes import reads
_READ = reads(read=lambda t: ({"is_question": True, "about_picture": True}
                              if t in ("Что там написано?", "what is written there?") else None))
_R = types.SimpleNamespace(**_READ)


print()
print("=" * 70)
print("1. «Which picture?» is asked only when the chat leaves it open")
print("=" * 70)

# One read of the message WITH the chat lines before it (tg_bot._message_subject) replaced a
# word gate and four exceptions: a forwarded text, a new drawing, the bot's last line, a
# forwarded voice -- each exception was a patch and the next case asked anyway. Scored on
# bench/which_picture_eval.py; here: the lines reach the read, and only "unclear" asks.
import chatlog
_seen = []
def _pick(answer):
    return types.SimpleNamespace(**reads(
        read=lambda t: {"about_picture": True},
        pick=lambda q, text: (_seen.append(text), answer)[1]))
_fwd = chatlog.lines_of([
    {"kind": "in", "data": {"voice": {"file_id": "v"}, "forward_date": 1}},
    {"kind": "out", "data": {"method": "sendMessage", "text": "🎧 <b>Пересланное голосовое</b> · 12 с\nЧто с ним сделать?"}},
    {"kind": "out", "data": {"method": "sendMessage", "text": "⚙️ <b>Думаю…</b>"}},
])
check("the chat lines show the forward and the bot's question, not its progress lines",
      _fwd == ["user: [forwards a voice message]", "bot: 🎧 Пересланное голосовое · 12 с Что с ним сделать?"], _fwd)
check("«Что думаешь?» after a forwarded voice: no «which picture?»",
      not T._needs_image_choice("Что думаешь?", _fwd, ctx=_pick("forwarded")))
check("...and the read saw the message and the forward", "Что думаешь?" in _seen[-1]
      and "[forwards a voice message]" in _seen[-1], _seen[-1:])
_three = ["user: [sends a picture]", "user: [sends a picture]", "user: [sends a picture]"]
check("three photos and «убери фон»: asks which", T._needs_image_choice("убери фон", _three, ctx=_pick("unclear")))
check("an empty message never asks", not T._needs_image_choice("  ", _three, ctx=_pick("unclear")))


print()
print("=" * 70)
print("2. vision_agent_node re-analyzes the CURRENT image on a bare follow-up")
print("=" * 70)

_TMP = tempfile.mkdtemp(prefix="followup_reask_")
img1 = os.path.join(_TMP, "photo1.jpg")
img2 = os.path.join(_TMP, "photo2.jpg")
for p in (img1, img2):
    with open(p, "wb") as fh:
        fh.write(b"\xff\xd8\xff\xe0" + b"\x00" * 32)  # minimal fake jpeg bytes

_calls = []
def _fake_analyze(ctx, image_bytes, user_text, system_prompt):
    # Identify which "photo" was analyzed by which fake bytes came through —
    # downscale_image_bytes may transform them, so key off ctx.last_image_path
    # (what the node is supposed to be looking at) instead.
    _calls.append(getattr(ctx, "last_image_path", None))
    return f"description of {os.path.basename(getattr(ctx, 'last_image_path', ''))}"


ctx = models.Context(models=None, transcription_cache={}, cache_file=None,
                     asr_lock=threading.Lock(), tts_lock=threading.Lock())
ctx.intent_read = _READ["intent_read"]
ctx.vision_model = _fake_analyze
graph_obj = G.build_graph(ctx)
vision_node = graph_obj.nodes["vision_agent"]

def run_node(state):
    return vision_node.invoke(state)

# Simulate: photo2 was the LAST one sent (ctx.last_image_path already points to
# it, exactly as tg_bot.py sets it up before graph.invoke — see _run_task_inner).
ctx.last_image_path = img2

state = {"user_input": "Что там написано?"}  # no image_data: a plain follow-up
out = run_node(state)
check("vision_summary was populated for a bare follow-up asking about content",
      bool(out.get("vision_summary")), out)
check("...and it analyzed the CURRENT image (photo2), not a stale one",
      _calls and _calls[-1] == img2, _calls)

_calls.clear()
state2 = {"user_input": "hello how are you"}  # unrelated text — must NOT trigger
out2 = run_node(state2)
check("an unrelated follow-up does NOT trigger a re-look (no wasted vision call)",
      not _calls, _calls)
check("...and vision_summary stays empty for it",
      not out2.get("vision_summary"), out2)

_calls.clear()
ctx.last_image_path = None
state3 = {"user_input": "Что там написано?"}  # no image at all yet
out3 = run_node(state3)
check("no current image at all -> no crash, no call, empty summary",
      not _calls and not out3.get("vision_summary"), (_calls, out3))

print()
print(f"{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
