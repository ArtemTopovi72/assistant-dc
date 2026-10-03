"""Live-chat bug (2026-08-04): two photos sent back to back, then a plain
text follow-up ("Что там написано?" / "what's written there?") was answered
describing the FIRST photo's content, not the second (current) one.

Root cause: `vision_agent_node` (graph.py) only ever runs vision analysis on
the turn a NEW photo arrives (`state["image_data"]` present). A later text-only
question has nothing fresh to answer from — it falls back to whatever sits in
`ctx.session_memory`, which accumulates ONE vision summary per photo ever
sent, with nothing marking which is "current". The model just picked from
memory and got the wrong one.

Fix: `vision_agent_node` now re-analyzes `ctx.last_image_path` FRESH whenever
there's no new photo this turn but the question plainly needs a look at the
picture (`_ASKS_ABOUT_CURRENT_IMAGE_RE`), so the CURRENT working image is what
gets described, not a stale multi-photo memory dump. tg_bot.py's own
`_NEEDS_IMAGE_RE` (the separate "which picture do you mean?" ambiguity gate for
when 2+ images are live) had the same gap — "written"/"написано" phrasing
wasn't recognized — fixed alongside it.

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
import intent
intent.STUB = lambda t: ({"is_question": True, "about_picture": True}
                         if t in ("Что там написано?", "what is written there?") else None)


print()
print("=" * 70)
print("2. tg_bot's ambiguity-gate regex has the same 'written' coverage")
print("=" * 70)

for t in ("Что там написано?", "what is written there?"):
    check(f"tg_bot asks which picture for: {t!r}", T._needs_image_choice(t), t)


print()
print("=" * 70)
print("3. vision_agent_node re-analyzes the CURRENT image on a bare follow-up")
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

G.analyze_image_with_llm = _fake_analyze
G.downscale_image_bytes = lambda b: b  # passthrough, avoid needing a real image

ctx = models.Context(models=None, transcription_cache={}, cache_file=None,
                     asr_lock=threading.Lock(), tts_lock=threading.Lock())
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
