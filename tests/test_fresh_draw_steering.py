"""'нарисуй X' with a picture in the chat is steered to generate_image.

Live 2026-09-13 (mega journey re-run, step 48): 'нарисуй набережную Сочи в
такую погоду' after a lighthouse picture was redrawn ON the lighthouse — the
'previously created image' steering leans towards the edit tools. A fresh
drawing verb that does not point at the picture now gets a generate_image
steer; 'нарисуй ей шляпу' / 'draw a hat on it' still edit.
"""
import os, sys, types
os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import graph_compose as C

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

# New picture vs a change to the one on screen is the model's read
# (agent/intent.py `wants`); the phrases run live in bench/intent_picture_live.py.
import intent
NEW = {"needs_tool": True, "wants": ["generate_image"]}
EDIT = {"needs_tool": True, "wants": ["inpaint_image"]}
intent.STUB = {"нарисуй набережную Сочи в такую погоду": NEW, "сделай слона розовым": EDIT,
               "нарисуй ей шляпу": {"needs_tool": True, "wants": ["inpaint_image", "generate_image"]},
               "нарисуй кота": NEW}.get
_c = types.SimpleNamespace(last_image_path="x.png")
W = lambda o: C._wants_fresh_picture(_c, {"user_input": "x", "user_input_original": o})
check("a new picture", W("нарисуй набережную Сочи в такую погоду"))
check("an edit is not", not W("сделай слона розовым"))
check("drawing ON the picture is an edit", not W("нарисуй ей шляпу"))
check("no read, no steer", not W("непрочитанное"))

ctx = types.SimpleNamespace(last_image_path="C:/x/lighthouse.png", facts_text=lambda *a, **k: "", memory_text=lambda: "")
msg, _, _ = C._compose_user_message(ctx, {"user_input": "draw the Sochi embankment in this kind of weather",
                                          "user_input_original": "нарисуй набережную Сочи в такую погоду"})
check("fresh draw with a picture in the chat gets the generate_image steer", "NEW picture" in msg and "generate_image" in msg)
check("...and not the edit steer", "previously created image is available" not in msg)
msg, _, _ = C._compose_user_message(ctx, {"user_input": "make the elephant pink", "user_input_original": "сделай слона розовым"})
check("an edit keeps the edit steer", "previously created image is available" in msg)
msg, _, _ = C._compose_user_message(ctx, {"user_input": "draw a cat", "user_input_original": "нарисуй кота", "image_data": b"x"})
check("a picture sent THIS turn still wins", "just sent an image" in msg and "NEW picture" not in msg)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
