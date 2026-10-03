"""One picture in several states is not a choice.

Live, 2026-09-12: draw -> upscale -> outpaint, then "describe the picture"
-> "Which picture? I have 3 in this chat" -- after the user had pressed the
buttons themselves. Versions now record their parent; when every live
picture but one is a version of another, the latest is the picture.
Pure: no bot, no GPU.

Run: venv/Scripts/python.exe tests/test_image_lineage.py
"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:200])


d = tempfile.mkdtemp(prefix="lineage_")
def f(name):
    p = os.path.join(d, name); open(p, "wb").write(b"x"); return p

sess = types.SimpleNamespace(image_log=[])
a = T._log_image(sess, f("draw.png"), label="draw", src="bot")
b = T._log_image(sess, f("up.png"), label="[upscale]", src="bot", parent=a)
c = T._log_image(sess, f("out.png"), label="outpaint", src="bot", parent=b)
live = T._live_images(sess)
check("versions carry their parent", [e.get("parent") for e in live] == ["", a, b])
check("draw -> upscale -> outpaint is one lineage", T._one_lineage(live))
check("the latest is last", live[-1]["id"] == c)
check("re-registering keeps the parent", T._log_image(sess, live[1]["path"], msg_id=7) == b and T._image_by_id(sess, b)["parent"] == a)
check("_image_by_path finds it", (T._image_by_path(sess, live[2]["path"]) or {}).get("id") == c)

sess2 = types.SimpleNamespace(image_log=[])
x = T._log_image(sess2, f("cat.png"), src="bot")
y = T._log_image(sess2, f("dog.png"), src="bot")
check("two separate pictures are a real choice", not T._one_lineage(T._live_images(sess2)))
z = T._log_image(sess2, f("dog_up.png"), src="bot", parent=y)
check("a version of one of two roots is still two roots", not T._one_lineage(T._live_images(sess2)))
check("a single picture is not a lineage question", not T._one_lineage(T._live_images(types.SimpleNamespace(image_log=[]))))

src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(T.__file__))), "bot/tg_resolve.py"), encoding="utf-8").read()
check("the ask-which gate skips a single lineage", "tg_bot._one_lineage(live)" in src)
src2 = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(T.__file__))), "bot/tg_tasks.py"), encoding="utf-8").read()
check("a task records the picture it starts from", "ctx.source_image_id" in src2 and 'parent=_parent' in src2)
check("a fresh draw records no parent", "if _derives else" in src2)

# A FREE-TEXT edit ("надень на кота шляпу", then "убери шляпу") is not a
# button and not a reply, so ctx.source_image_id is empty; the editing tool
# itself records the picture it started from and the delivery layer uses it
# (live, 2026-09-12, journey 21: "Which picture? there are 2").
check("delivery falls back to the tool's own source", 'final.get("image_derived_from")' in src2)
import tool_image_handlers as H
_st = {"image_path": f("cat.png")}
H._note_source(_st, _st["image_path"])
check("editing handlers note their source", _st.get("image_derived_from") == f("cat.png"))
src3 = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(T.__file__))), "agent/tool_image_handlers.py"), encoding="utf-8").read()
check("every editing handler notes it (inpaint, fix_hands, fix_artifact, transfer, redraw)",
      src3.count("_note_source(state, source)") >= 5, src3.count("_note_source(state, source)"))
# end to end on the register: cat -> hat edit (no parent from the user) -> one lineage
sess3 = types.SimpleNamespace(image_log=[])
cat = T._log_image(sess3, f("cat.png"), label="нарисуй кота", src="bot", parent="")
hat_parent = (T._image_by_path(sess3, f("cat.png")) or {}).get("id", "")
hat = T._log_image(sess3, f("hat.png"), label="надень шляпу", src="bot", parent=hat_parent)
check("the edited picture is a version of the drawn one", T._one_lineage(T._live_images(sess3)))

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
