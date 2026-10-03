"""An accepted render is final for the turn: the agent may not re-edit it.

generate_image judges its output with the vision model. When that judgement
is a pass, a further edit by the agent in the SAME turn is not verification,
it is a second, unreliable critic with the power to redraw. Live, 2026-09-11:
a 10/10 render was "fixed" for a shadow that was not there, lost its subject
to the removal, and went through six more renders before the model crashed.

The user is the only critic who may send it back -- in their own words, in
their own turn, which arrives with a fresh state.

Nothing here touches a model or the GPU.

Run: venv/Scripts/python.exe tests/test_accepted_render_is_final.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import logging; logging.basicConfig(level=logging.CRITICAL)

import tool_graph as TG

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:200])


import tempfile
_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False); _tmp.write(b"PNG"); _tmp.close()
REAL = _tmp.name
ctx = types.SimpleNamespace(last_image_path=REAL)
accepted = {"image_path": REAL,
            "fresh_render": {"path": REAL, "score": 10,
                             "status": "success", "accepted": True}}
weak = {"image_path": "C:/x/render.png",
        "fresh_render": {"path": "C:/x/render.png", "score": 4,
                         "status": "success", "accepted": False}}

# ── the live case ────────────────────────────────────────────────────────────
gate = TG.check_tool_call(ctx, dict(accepted), "inpaint_image",
                          {"region": "cat-shaped object",
                           "instructions": "Remove the large black shadow covering the cat"})
check("an edit of the agent's own accepted render is refused",
      gate is not None and gate.startswith("[TOOL ERROR]"), gate)
check("and the refusal says to deliver, not to retry",
      gate and "deliver" in gate.lower(), gate)

for tool in ("redraw_image", "fix_hands", "fix_artifact"):
    check("%s is refused too" % tool,
          TG.self_edit_of_accepted_render(ctx, dict(accepted), tool, {}) is not None)

# ── what must stay allowed ───────────────────────────────────────────────────
check("inspecting is still allowed (read-only, and it may be reporting to the user)",
      TG.self_edit_of_accepted_render(ctx, dict(accepted), "inspect_image", {}) is None)
check("a render that was NOT accepted may still be worked on",
      TG.self_edit_of_accepted_render(ctx, dict(weak), "inpaint_image", {}) is None)
check("with no fresh render in the state, edits are ordinary",
      TG.self_edit_of_accepted_render(ctx, {"image_path": "C:/x/old.png"},
                                      "inpaint_image", {}) is None)
check("the pencil button (edit_intent) is the user asking, so it goes through",
      TG.self_edit_of_accepted_render(ctx, dict(accepted, edit_intent=True),
                                      "inpaint_image", {}) is None)
check("generate_image itself is never gated by this",
      TG.self_edit_of_accepted_render(ctx, dict(accepted), "generate_image", {}) is None)

# ── the next turn starts clean ───────────────────────────────────────────────
import graph as G
st = dict(accepted, user_input="убери фон")
st = G.translate_node(types.SimpleNamespace(), st) if False else st   # translate needs an LLM
# translate_node pops the key before anything else; assert on its source, since
# running it would call the translator.
import inspect
src = inspect.getsource(G.translate_node)
check("translate_node drops fresh_render at the start of every turn",
      'state.pop("fresh_render", None)' in src)

# ── the handler marks acceptance by the same threshold the gate uses ─────────
import tool_image_handlers as H
hsrc = inspect.getsource(H)
check("the handler records fresh_render", 'state["fresh_render"]' in hsrc)
check("acceptance uses tool_graph.ACCEPTED_SCORE, not a second number",
      "tool_graph.ACCEPTED_SCORE" in hsrc)
check("an accepted render no longer invites a second inspect",
      "Do NOT inspect it again" in hsrc)
check("the threshold is a pass, not perfection", 5 <= TG.ACCEPTED_SCORE <= 8, TG.ACCEPTED_SCORE)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
