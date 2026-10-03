"""A QUESTION about a picture is answered in words, never by an image tool
(live 2026-09-17 22:31 / 2026-09-18 00:44: «Что нарисовано?» became an
enhance pass and «Сделал, но проверка показала…»)."""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")
import graph, graph_compose, tool_graph

PASSED = FAILED = 0
def check(name, cond, extra=""):
    global PASSED, FAILED
    if cond: PASSED += 1; print("PASS ", name)
    else: FAILED += 1; print("FAIL ", name, extra)

# What the message means is the model's read (agent/intent.py; phrases in
# bench/intent_live.py). Q = question about it, E = an order to change it.
import intent
Q = {"is_question": True, "about_picture": True}
E = {"needs_tool": True, "wants": ["redraw_image"]}
READ = {"Что нарисовано?": Q, "какой товар самый дорогой?": {"is_question": True},
        "Нарисуй Степана так": {"needs_tool": True, "wants": ["generate_image"]},
        "можешь сделать ярче?": {**E, "is_question": True}, "убери лужу": E,
        "круто": {"needs_tool": False}, "обсудим цитату": {"about_picture": False}}
intent.STUB = READ.get
for o, exp in [("Что нарисовано?", True), ("какой товар самый дорогой?", True),
               ("Нарисуй Степана так", False), ("можешь сделать ярче?", False),
               ("убери лужу", False), ("круто", False)]:
    check(f"is_picture_question({o!r}) == {exp}",
          graph.is_picture_question(None, {"user_input": "x", "user_input_original": o}) == exp)

st = lambda o: {"user_input": "x", "user_input_original": o}
# pointed-at picture: anything but an order is looked at
ctx = types.SimpleNamespace(last_image_path=__file__, image_pointed_at=True)
check("a question over a pointed-at picture is looked at", graph.needs_relook(ctx, st("Что нарисовано?")))
# «обсудим цитату» after ❓ is no question, yet it is about the picture (live 16:57)
check("words that are no question over it are looked at too", graph.needs_relook(ctx, st("обсудим цитату")))
check("an order to change it is not (the editing path has it)", not graph.needs_relook(ctx, st("убери лужу")))
ctx.image_pointed_at = False
check("a read about the picture looks at it without a gesture", graph.needs_relook(ctx, st("Что нарисовано?")))
check("a plain question about a generated picture does not",
      not graph.needs_relook(ctx, st("какой товар самый дорогой?")))
ctx.last_image_path = os.path.join(os.path.dirname(__file__), "_working_input_1.png")
open(ctx.last_image_path, "wb").close()
try:
    check("any question about the user's own photo does", graph.needs_relook(ctx, st("какой товар самый дорогой?")))
finally:
    os.remove(ctx.last_image_path)

# the steering block speaks of answering, not editing
class _Ctx:
    last_image_path = "x.png"
    def facts_text(self, query=""): return ""
    def memory_text(self): return ""
    facts_forgotten = False
st = {"user_input": "What is drawn here?", "picture_question": True, "image_data": b"x",
      "vision_summary": "a pencil sketch of a cylinder and a box"}
msg = graph_compose._compose_user_message(_Ctx(), st)[0]
check("the question steering replaces the edit steering",
      "asks a QUESTION about the picture" in msg and "work on THAT image" not in msg, msg[-300:])

# the control plane refuses an image tool on such a turn
gate = tool_graph.check_tool_call(_Ctx(), st, "redraw_image", {"mode": "enhance"})
check("redraw_image is refused on a picture question", gate and "QUESTION about the picture" in gate, gate)
check("inspect_image stays allowed", tool_graph.picture_question_edit(st, "inspect_image") is None)
check("no flag -> no refusal", tool_graph.picture_question_edit({"user_input": "x"}, "redraw_image") is None)
print(f"\n{PASSED} passed, {FAILED} failed"); sys.exit(1 if FAILED else 0)
