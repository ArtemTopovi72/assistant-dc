"""The agent must be able to REARRANGE the layout, not just report on it.

Ideogram draws from a canvas of boxes (see `ideogram.py`), so every complaint the
agent can make about a render — wrong place, missing thing, drawn twice, the
description was too vague, the lettering had no room — has a corresponding move on
that canvas. This suite pins down that the whole vocabulary is reachable from both
places the agent forms an opinion:

  * a REVISION from the user ("put the sign top-right, drop the dog") -> `edit_layout`
  * a QUALITY JUDGEMENT on the render                                 -> `critique`

and that the operations survive the deterministic fixers that run after them, all
the way into the next render.

Run: venv/Scripts/python.exe tests/test_draw_ops.py
"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import inspect
import json

import draw_agent as D
import ideogram as G

OK = BAD = 0
def check(name, cond, detail=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {detail}")


def scene(*els):
    return G.normalize_layout({"background": "a city street", "medium": "photography",
                               "elements": [dict(e) for e in els]})


CAR = {"desc": "a dark patrol car, side view", "text": "",
       "x": 0.05, "y": 0.35, "w": 0.60, "h": 0.40}
SIGN = {"desc": "a sign board on the roof of the car", "text": "POLICE",
        "x": 0.30, "y": 0.55, "w": 0.09, "h": 0.05}
TREE = {"desc": "a bare plane tree", "text": "",
        "x": 0.70, "y": 0.10, "w": 0.25, "h": 0.45}


print("=" * 72)
print("NAMING THE BOX TO CHANGE")
print("=" * 72)

lay = scene(CAR, SIGN, TREE)
check("an element can be named by its number", D._find(lay, 2) == 1)
check("or by words from its description", D._find(lay, "the patrol car") == 0)
check("or by the string it carries", D._find(lay, "POLICE") == 1)
check("a near-miss word still resolves — the critic renames things freely",
      D._find(scene(CAR, TREE), "the plane trees") == 1,
      str(D._find(scene(CAR, TREE), "the plane trees")))
check("but something that is simply not there does not resolve to the nearest box",
      D._find(lay, "a hot air balloon") == -1)
check("and a word that merely LOOKS like one in the layout does not either",
      D._find(lay, "the barn") == -1, str(D._find(lay, "the barn")))
check("and neither does a target the model left empty", D._find(lay, "") == -1)


print()
print("=" * 72)
print("EVERY ADJUSTMENT THE AGENT MIGHT WANT")
print("=" * 72)

out, notes = D.apply_ops(lay, [{"op": "move", "target": "the sign board",
                                "x": 0.62, "y": 0.06}])
check("move puts the box where it was told", abs(out["elements"][1]["x"] - 0.62) < 1e-6
      and abs(out["elements"][1]["y"] - 0.06) < 1e-6, str(out["elements"][1]))
check("move leaves its size alone",
      (out["elements"][1]["w"], out["elements"][1]["h"]) == (SIGN["w"], SIGN["h"]))
check("a placement WORD is a move too",
      D.apply_ops(lay, [{"op": "move", "target": "the tree",
                         "where": "bottom left"}])[0]["elements"][2]["y"] > 0.5)

out, _ = D.apply_ops(lay, [{"op": "resize", "target": "the tree", "scale": 1.4}])
check("resize by a factor grows the box about its centre",
      abs(out["elements"][2]["w"] - 0.35) < 1e-6
      and abs((out["elements"][2]["x"] + out["elements"][2]["w"] / 2)
              - (TREE["x"] + TREE["w"] / 2)) < 1e-6, str(out["elements"][2]))
out, _ = D.apply_ops(lay, [{"op": "resize", "target": 3, "w": 0.4, "h": 0.5}])
check("resize to an explicit size works as well",
      (out["elements"][2]["w"], out["elements"][2]["h"]) == (0.4, 0.5))

out, notes = D.apply_ops(lay, [{"op": "add", "desc": "a wooden bench",
                                "x": 0.05, "y": 0.75, "w": 0.25, "h": 0.2}])
check("add appends a new element", len(out["elements"]) == 4
      and out["elements"][3]["desc"] == "a wooden bench")
check("and says so", any("bench" in n for n in notes), str(notes))
check("an add with no description at all is refused, not appended",
      len(D.apply_ops(lay, [{"op": "add", "x": 0.1, "y": 0.1}])[0]["elements"]) == 3)
# Measured live: this is the shape the critic actually writes when it wants a
# missing element back — the thing to add named in "target", not "desc". Three
# runs in five, every one skipped as "no description given".
out, notes = D.apply_ops(lay, [{"op": "add", "target": "a red bicycle leaning on a wall",
                                "x": 0.05, "y": 0.7, "w": 0.25, "h": 0.25}])
check("an add that names the new thing in “target” is still an add",
      len(out["elements"]) == 4 and "bicycle" in out["elements"][3]["desc"], str(notes))

out, notes = D.apply_ops(lay, [{"op": "replace", "target": "the tree",
                                "x": 0.4, "y": 0.2, "w": 0.3, "h": 0.3}])
check("a “replace” carrying only a box is a reshape, and says so",
      abs(out["elements"][2]["x"] - 0.4) < 1e-6 and any("reshaped" in n for n in notes),
      str(notes))

out, _ = D.apply_ops(lay, [{"op": "add", "desc": "a shop awning", "text": "BAKERY",
                            "where": "top left"}])
new = out["elements"][-1]
check("a new LETTERING box is sized from its string, not left a generic square",
      D.text_report({"elements": [new], "background": ""}) == [],
      f'{new["w"]:.2f}x{new["h"]:.2f}')

out, notes = D.apply_ops(lay, [{"op": "delete", "target": "the bare plane tree"}])
check("delete removes exactly that element",
      [e["desc"] for e in out["elements"]] == [CAR["desc"], SIGN["desc"]])
check("a delete that names nothing in the layout is reported, not guessed at",
      len(D.apply_ops(lay, [{"op": "delete", "target": "the helicopter"}])[0]["elements"]) == 3)

out, notes = D.apply_ops(lay, [{"op": "replace", "target": "the patrol car",
                                "desc": "a dark blue estate car, seen from the side"}])
check("replace rewrites the description in place, keeping the box",
      out["elements"][0]["desc"].startswith("a dark blue estate")
      and (out["elements"][0]["x"], out["elements"][0]["w"]) == (CAR["x"], CAR["w"]))
for alias in ("desc", "describe", "update", "set_description", "modify"):
    got, gnotes = D.apply_ops(lay, [{"op": alias, "target": "the patrol car",
                                     "desc": "an unmarked grey saloon"}])
    check(f'“{alias}” is understood as rewording a box',
          got["elements"][0]["desc"] == "an unmarked grey saloon"
          and any("replaced" in n for n in gnotes), str(gnotes))

out, _ = D.apply_ops(lay, [{"op": "text", "target": "the sign board", "text": "SHERIFF"}])
check("the lettering on a box can be changed", out["elements"][1]["text"] == "SHERIFF")
out, _ = D.apply_ops(lay, [{"op": "background", "to": "a rainy street at night"}])
check("so can the background", out["background"] == "a rainy street at night")

out, notes = D.apply_ops(lay, [{"op": "shift", "target": "the tree", "x": 0.42, "desc": "a leafy oak"}])
check("an op name nobody anticipated still applies what it asked for",
      abs(out["elements"][2]["x"] - 0.42) < 1e-6
      and out["elements"][2]["desc"] == "a leafy oak", str(notes))
out, notes = D.apply_ops(lay, [{"op": "wibble", "target": "the tree"}])
check("but an op that asks for nothing is skipped, not invented",
      out["elements"] == lay["elements"]
      and any("unknown op" in n for n in notes), str(notes))

# The lettered sign (and its host) may not be deleted on the critic's say-so
# (see tests/test_lettering_stays_on_host.py), so the tree goes instead.
out, notes = D.apply_ops(lay, [{"op": "move", "target": "the patrol car", "x": 0.1},
                               "not a dict",
                               {"op": "delete", "target": "the tree"}])
check("one malformed op costs only that op — the rest of the batch still applies",
      len(out["elements"]) == 2 and len(notes) >= 2, str(notes))
check("and the original layout is never mutated", lay["elements"][1]["text"] == "POLICE")

several, notes = D.apply_ops(lay, [
    {"op": "move", "target": "the sign board", "x": 0.55, "y": 0.60},
    {"op": "resize", "target": "the sign board", "w": 0.36, "h": 0.10},
    {"op": "replace", "target": "the patrol car", "desc": "a dark blue estate car"},
    {"op": "delete", "target": "the plane tree"},
    {"op": "add", "desc": "a wooden bench", "x": 0.05, "y": 0.78, "w": 0.25, "h": 0.18}])
check("a single answer may move, resize, reword, delete and add all at once",
      len(several["elements"]) == 3
      and several["elements"][0]["desc"] == "a dark blue estate car"
      and abs(several["elements"][1]["x"] - 0.55) < 1e-6
      and several["elements"][2]["desc"] == "a wooden bench", str(notes))


print()
print("=" * 72)
print("WHAT THE TWO PROMPTS OFFER THE MODEL")
print("=" * 72)

for op in ("move", "resize", "add", "delete", "replace"):
    check(f'the critic is told it may {op} a box', op in D._CRITIQUE_PROMPT)
for op in ("move", "resize", "add", "delete", "replace", "text"):
    check(f'the editor is told it may “{op}”', f'"{op}"' in D._EDIT_PROMPT)
for label, prompt in (("the critic", D._CRITIQUE_PROMPT), ("the editor", D._EDIT_PROMPT)):
    check(f"{label} is told the coordinate convention",
          "top-left" in prompt.lower() or "TOP-LEFT" in prompt)
check("the critic is told it may reword a description, not only swap the object",
      "reword" in D._CRITIQUE_PROMPT.lower())
check("the editor is told several ops may answer one instruction",
      "as many operations" in D._EDIT_PROMPT)
check("both are told the shape lettering needs",
      "0.6 x (number of characters)" in D._EDIT_PROMPT and "0.08" in D._EDIT_PROMPT
      and "more width and height" in D._CRITIQUE_PROMPT)
check("a correct picture still gets no ops",
      '"ops" empty when ok is true' in D._CRITIQUE_PROMPT)
# Load-bearing, and measured: the longer this prompt is, the longer the model
# deliberates before answering, and a deliberation that overruns the token budget
# is reported as "the critic could not see". A version with a worked example of
# every operation answered 1 time in 4 where this one answers 4 in 4. The full
# vocabulary lives in the editor, which the loop consults instead.
# --- the op named by its KEY, not by an "op" field -------------------------
# Live on gemma-4-12b: {"replace": {"target": 1, "desc": "a blonde man ..."}}.
# The edit inside was exactly right; we discarded it as `unknown op ''`, so
# "make the man blond" changed nothing and the picture came back as it was.
_lay, _notes = D.apply_ops(
    {"background": "s", "elements": [dict(CAR), dict(TREE)]},
    [{"replace": {"target": 1, "desc": "a blue van"}}])
check("an op named by its key is applied, not discarded",
      _lay["elements"][0]["desc"] == "a blue van", _notes)
_lay, _notes = D.apply_ops(
    {"background": "s", "elements": [dict(CAR), dict(TREE)]},
    [{"move": {"target": 2, "x": 0.11, "y": 0.06}}])
check("so is a key-named move",
      abs(_lay["elements"][1]["x"] - 0.11) < 1e-6, _notes)
_lay, _notes = D.apply_ops(
    {"background": "s", "elements": [dict(CAR)]},
    [{"replace": {"target": 1}, "something_else": 1}])
check("but a two-key object is not guessed at",
      "unknown op" in " ".join(_notes), _notes)

check("the critic's prompt is kept short — a long one silently disables it",
      len(D._CRITIQUE_PROMPT) < 1400, f"{len(D._CRITIQUE_PROMPT)} chars")
# Measured over six runs of one real render: at 1600 the critic answered once,
# at 3000 four times. The rest were cut off mid-deliberation and reported as
# "could not see" — a check that silently stops voting.
check("and it is given room for the deliberation it does do",
      "max_tokens=3000" in inspect.getsource(D.critique))


print()
print("=" * 72)
print("THE CRITIC IS TOLD WHAT THE READ-BACK ALREADY KNOWS")
print("=" * 72)


class _LLM(types.ModuleType):
    def __init__(self, reply):
        super().__init__("llm")
        self.reply, self.calls = reply, []
    def analyze_image_with_llm(self, ctx, **kw):
        self.calls.append(dict(kw, ctx_model=getattr(ctx, "model_name", None)))
        return self.reply
    def send_to_lm_studio(self, ctx, messages, **kw):
        self.calls.append(dict(kw, messages=messages))
        return {"content": self.reply}


def with_llm(reply):
    sys.modules["llm"] = _LLM(reply)
    return sys.modules["llm"]


_saved_llm = sys.modules.get("llm")
CTX = types.SimpleNamespace(is_cancelled=lambda: False, model_name="chat/model")
try:
    stub = with_llm('{"ok": false, "score": 4, "problems": ["the sign is unreadable"],'
                    ' "ops": [{"op": "resize", "target": "the sign board", "scale": 1.5}]}')
    verdict = D.critique(CTX, "x.png", lay, evidence=[
        "the lettering should read “POLICE” but the picture shows “AVCHKE”"])
    sent = stub.calls[0]["user_text"]
    check("the read-back finding reaches the critic", "AVCHKE" in sent, sent[-200:])
    check("the critic can then answer with a repair",
          verdict["ops"] and verdict["ops"][0]["op"] == "resize", str(verdict))
    check("and its verdict is still parsed", verdict["ok"] is False
          and verdict["source"] == "vision", str(verdict))

    stub = with_llm('{"ok": true, "score": 9, "problems": [], "ops": []}')
    D.critique(CTX, "x.png", lay)
    check("with nothing established, no evidence section is invented",
          "Already established" not in stub.calls[0]["user_text"])
    check("evidence is optional — the old two-argument call still works",
          D.critique(CTX, "x.png", lay)["ok"] is True)

    print()
    print("=" * 72)
    print("A REVISION FROM THE USER")
    print("=" * 72)

    with_llm('{"ops": [{"op": "move", "target": "the sign board", "x": 0.6, "y": 0.05},'
             ' {"op": "delete", "target": "the plane tree"},'
             ' {"op": "add", "desc": "a wooden bench", "x": 0.05, "y": 0.78,'
             ' "w": 0.25, "h": 0.18}]}')
    edited, notes = D.edit_layout(CTX, lay, "sign top right, lose the tree, add a bench")
    check("one instruction can move, delete and add in a single edit",
          len(edited["elements"]) == 3
          and abs(edited["elements"][1]["x"] - 0.6) < 1e-6
          and edited["elements"][2]["desc"] == "a wooden bench", str(notes))
    with_llm('{"op": "delete", "target": "the plane tree"}')
    check("a lone op answered unwrapped is still applied",
          len(D.edit_layout(CTX, lay, "lose the tree")[0]["elements"]) == 2)
    with_llm("I would rather not.")
    check("an answer with no operations leaves the layout untouched",
          D.edit_layout(CTX, lay, "do something")[0]["elements"] == lay["elements"])

    print()
    print("=" * 72)
    print("THE ADJUSTMENTS REACH THE NEXT RENDER")
    print("=" * 72)

    renders, told, mods = [], [], []

    def fake_generate(ctx, prompt, **kw):
        renders.append(G.caption_to_layout(kw["caption"]))
        return f"/tmp/render{len(renders)}.png"

    _real_gen, _real_crit = G.generate, D.critique

    def run_with(critic_ops, reads, rounds=1, start=None):
        renders.clear(); told.clear(); mods.clear()
        try:
            G.generate = fake_generate
            D.critique = lambda ctx, img, lay, evidence=None: (
                told.append(list(evidence or [])),
                # a real critic told about a finding tends to repeat it back
                {"ok": not critic_ops, "score": 5, "problems": list(evidence or []),
                 "ops": list(critic_ops), "source": "vision"})[1]
            mod = _LLM(None)
            mod.analyze_image_with_llm = lambda ctx, **kw: json.dumps(
                {"strings": [reads[min(len(renders), len(reads)) - 1]]})
            sys.modules["llm"] = mod
            mods.append(mod)      # so a test can see whether the editor was called
            return D.run(CTX, "a police car", layout=start or scene(CAR, SIGN, TREE),
                         rounds=rounds)
        finally:
            G.generate, D.critique = _real_gen, _real_crit

    # A move stays ON the host (the car): lettering is clamped to what it is
    # written on, so the move is within the car's box.
    res = run_with([{"op": "move", "target": "the sign board", "x": 0.10, "y": 0.40},
                    {"op": "replace", "target": "the patrol car",
                     "desc": "a dark blue estate car, side view"},
                    {"op": "add", "desc": "a wooden bench", "x": 0.05, "y": 0.78,
                     "w": 0.25, "h": 0.18}],
                   ["AVCHKE", "POLICE"])
    check("the picture is redrawn once the agent has adjustments to make",
          len(renders) == 2, f"{len(renders)} render(s)")
    second = renders[1] if len(renders) > 1 else {"elements": []}
    descs = [e["desc"] for e in second["elements"]]
    check("a move the agent asked for reaches the canvas",
          any(abs(e["x"] - 0.10) < 0.03 and abs(e["y"] - 0.40) < 0.03
              for e in second["elements"] if e.get("text")),
          str([(e.get("text"), round(e["x"], 2), round(e["y"], 2))
               for e in second["elements"]]))
    check("so does a reworded description", any("estate car" in d for d in descs), str(descs))
    check("and so does a whole new box", any("bench" in d for d in descs), str(descs))
    check("the enlarged lettering box survives the other fixers",
          D.text_report(second) == [], str(D.text_report(second)))
    res = run_with([], ["AVCHKE", "POLICE"])
    check("garbled lettering alone still triggers a redraw", len(renders) == 2,
          f"{len(renders)}")
    check("but it is repaired geometrically, not talked over with the editor — "
          "that repair is deterministic and owns the box",
          mods and not any("Instruction:" in str(c.get("messages"))
                           for c in mods[0].calls), str(mods[0].calls)[:200])
    first = res["history"][0]["verdict"]["problems"]
    check("the loop hands the read-back finding to the critic before it judges, so "
          "the critic can propose a repair for it",
          told and any("AVCHKE" in p for p in told[0]), str(told[:1]))
    check("and the finding still reaches the user's report",
          any("AVCHKE" in p for p in first), str(first))
    check("and the finding is reported once, not twice",
          sum("AVCHKE" in p for p in first) == 1, str(first))

    # The critic reworded the very box whose lettering came out wrong. Its wording
    # must be kept AND the letter-by-letter spelling put back — otherwise the
    # rewrite silently cancels the one repair that is known to work.
    res = run_with([{"op": "delete", "target": "the plane tree"},
                    {"op": "replace", "target": "the sign board",
                     "desc": "a large white sign board with room around it"}],
                   ["AVCHKE", "POLICE"])
    second = renders[1] if len(renders) > 1 else {"elements": []}
    sign = next((e for e in second["elements"] if e.get("text")), {})
    check("the box the agent deleted is gone from the next render",
          not any("plane tree" in e["desc"] for e in second["elements"]),
          str([e["desc"] for e in second["elements"]]))
    check("its rewording of the garbled box is kept",
          "large white sign board" in sign.get("desc", ""), sign.get("desc", ""))
    check("and the spelling is restated inside it",
          "“POLICE”" in sign.get("desc", ""), sign.get("desc", ""))
    check("re-spelling follows the STRING, not a stale index — the innocent box "
          "keeps its own description",
          all("“POLICE”" not in e["desc"] for e in second["elements"] if not e.get("text")),
          str([e["desc"] for e in second["elements"]]))

    # The critic judges through a deliberately short prompt, so it often states a
    # problem and proposes nothing usable for it — a target it names in its own
    # words, or no ops at all. That complaint must still become a rearrangement,
    # not the same picture drawn again on a new seed.
    def run_complaining(problems, critic_ops, editor_reply):
        renders.clear(); told.clear()
        try:
            G.generate = fake_generate
            D.critique = lambda ctx, img, lay, evidence=None: {
                "ok": False, "score": 3, "problems": list(problems),
                "ops": list(critic_ops), "source": "vision"}
            mod = _LLM(editor_reply)
            mod.analyze_image_with_llm = lambda ctx, **kw: json.dumps({"strings": ["POLICE"]})
            sys.modules["llm"] = mod
            return D.run(CTX, "a police car", layout=scene(CAR, SIGN, TREE), rounds=1), mod
        finally:
            G.generate, D.critique = _real_gen, _real_crit

    editor = ('{"ops": [{"op": "delete", "target": "the plane tree"},'
              ' {"op": "add", "desc": "a red bicycle", "x": 0.7, "y": 0.6,'
              ' "w": 0.25, "h": 0.3}]}')
    res, mod = run_complaining(["the bicycle is missing and the tree is in the way"],
                               [], editor)
    second = renders[1] if len(renders) > 1 else {"elements": []}
    descs = [e["desc"] for e in second["elements"]]
    check("a complaint with no operations is handed to the editor",
          any("Instruction:" in str(c.get("messages")) for c in mod.calls), str(mod.calls)[:200])
    check("and the editor's rearrangement reaches the canvas",
          any("bicycle" in d for d in descs) and not any("plane tree" in d for d in descs),
          str(descs))

    res, mod = run_complaining(["the sky is empty"],
                               [{"op": "add", "desc": "a flock of birds",
                                 "x": 0.3, "y": 0.05, "w": 0.3, "h": 0.2}], editor)
    descs = [e["desc"] for e in (renders[1] if len(renders) > 1 else {"elements": []})["elements"]]
    check("but a critic whose own op landed is not second-guessed by the editor",
          any("birds" in d for d in descs) and any("plane tree" in d for d in descs),
          str(descs))

    res, mod = run_complaining([], [], editor)
    check("and a critic with nothing to say does not summon the editor either",
          not any("Instruction:" in str(c.get("messages")) for c in mod.calls))

    res = run_with([], ["POLICE"], rounds=2)
    check("a picture the agent has no complaint about is not redrawn", len(renders) == 1,
          f"{len(renders)}")
finally:
    if _saved_llm is not None:
        sys.modules["llm"] = _saved_llm
    else:
        sys.modules.pop("llm", None)

print()
print("=" * 72)
print("A WRONG-ATTRIBUTE COMPLAINT UNDERLINES THE LAYOUT, IT DOES NOT REWRITE IT")
print("=" * 72)
# Live, 2026-09-12: "the elephant is orange/red instead of bright pink" went to
# the editor as an instruction, the element became "an orange-red elephant",
# and the critic scored the orange render 10 against the rewritten layout.
_lay = {"elements": [{"desc": "a large, vibrant bright pink elephant", "x": 0.1, "y": 0.2, "w": 0.4, "h": 0.5},
                     {"desc": "a tall palm tree", "x": 0.6, "y": 0.1, "w": 0.2, "h": 0.7}]}
_l2, _notes, _left = D.insist_on_mismatch(_lay, ["The elephant is orange/red instead of bright pink",
                                                 "The palm tree is cut off at the top"])
check("the element keeps its intended look and gains the insistence",
      _l2["elements"][0]["desc"] == "a large, vibrant bright pink elephant, bright pink -- not orange/red",
      _l2["elements"][0]["desc"])
check("the note says what was insisted on", _notes and "bright pink" in _notes[0], _notes)
check("the unrelated complaint is left for the editor", _left == ["The palm tree is cut off at the top"], _left)
_l3, _n3, _left3 = D.insist_on_mismatch(_l2, ["The elephant is brown instead of bright pink"])
check("a second insistence is added, the first is kept",
      "not orange/red" in _l3["elements"][0]["desc"] and "not brown" in _l3["elements"][0]["desc"])
_l4, _n4, _left4 = D.insist_on_mismatch(_lay, ["The sky is grey instead of blue"])
check("a complaint about nothing in the layout passes through", not _n4 and _left4 == ["The sky is grey instead of blue"])
_src = open(D.__file__, encoding="utf-8").read()
check("run() insists before it edits", _src.index("insist_on_mismatch(layout, unmet)") < _src.index("Change the layout so the NEXT render fixes them"))

print()
print("=" * 72)
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
