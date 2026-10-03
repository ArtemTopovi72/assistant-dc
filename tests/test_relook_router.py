"""A follow-up question about the user's photo looks at the photo again.

Live 2026-09-12 (journey 2): a receipt was photographed and summarised, then
"which item is the most expensive?" got "I need to see the receipt". Two
defects: `entry_router` never sent a turn without a new photo to
vision_agent_node, so its re-look branch was unreachable in the wired graph;
and the re-look gate only fired on questions that NAMED the picture. Now the
router is bound to ctx and any question is a re-look when the working image
is a user upload.
"""
import os, sys, tempfile, threading
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import graph as G, models, intent
# The model reads the words (agent/intent.py); phrases: bench/intent_picture_live.py
intent.STUB = {"which item is the most expensive?": {"is_question": True},
               "сколько стоит молоко": {"is_question": True},
               "how much is it?": {"is_question": True},
               "what is written on it?": {"is_question": True, "about_picture": True}}.get

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

tmp = tempfile.mkdtemp(prefix="relook_")
upload = os.path.join(tmp, "_working_input_1.jpg")
render = os.path.join(tmp, "ideogram_0001.png")
for p in (upload, render):
    open(p, "wb").write(b"\xff\xd8" + b"\0" * 16)

class C: pass
ctx = C(); ctx.last_image_path = upload

check("new photo -> vision_agent", G.entry_router({"image_data": b"x"}, ctx) == "vision_agent")
check("question about the user's photo -> vision_agent",
      G.entry_router({"user_input": "which item is the most expensive?"}, ctx) == "vision_agent")
check("Russian question word counts too",
      G.entry_router({"user_input": "сколько стоит молоко"}, ctx) == "vision_agent")
check("a statement is not a re-look",
      G.entry_router({"user_input": "make a shopping list from this"}, ctx) == "personality")
ctx.last_image_path = render
check("a question with only a RENDER on screen goes to the model",
      G.entry_router({"user_input": "how much is it?"}, ctx) == "personality")
check("...unless it names the picture",
      G.entry_router({"user_input": "what is written on it?"}, ctx) == "vision_agent")
ctx.last_image_path = os.path.join(tmp, "_working_input_gone.jpg")
check("a missing file never routes to vision", G.entry_router({"user_input": "what?"}, ctx) == "personality")
check("no ctx at all is safe", G.entry_router({"user_input": "what?"}) == "personality")

# The wired graph actually reaches the re-look branch.
calls = []
G.analyze_image_with_llm = lambda ctx, image_bytes, user_text, system_prompt: calls.append(user_text) or "milk 89"
G.downscale_image_bytes = lambda b: b
G.personality_node = lambda ctx, s: dict(s, final_answer="ok")
G.tts_node = lambda ctx, s: s
G._translate_to_english = lambda ctx, t: t
rctx = models.Context(models=None, transcription_cache={}, cache_file=None,
                      asr_lock=threading.Lock(), tts_lock=threading.Lock())
rctx.last_image_path = upload
out = G.build_graph(rctx).invoke({"messages": [], "user_input": "which item is the most expensive?"})
check("wired graph re-looked at the upload on a bare follow-up", calls == ["which item is the most expensive?"], calls)
check("and carried the fresh look into the turn", out.get("vision_summary") == "milk 89", out.get("vision_summary"))

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
