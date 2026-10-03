"""The LoRA trigger must condition the CHARACTER, not whatever the planner
listed first.

ensure_trigger used to glue the trigger onto elements[0]. Element order comes
from the layout planner, and for "the man robs a bank" it commonly emits the
building first -- so the trigger described a bank facade, the adapter never
saw its word on a person, and the render returned a stranger while reporting
success.

Run: venv/Scripts/python.exe tests/test_trigger_lands_on_person.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import ideogram as I

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:200])


def cap(elements, hl="A scene."):
    return {"high_level_description": hl,
            "compositional_deconstruction": {"background": "a street",
                                             "elements": elements}}


def el(desc, box=None, text=""):
    d = {"type": "obj", "desc": desc}
    if box: d["bbox"] = box
    if text: d["type"], d["text"] = "text", text
    return d


# 1. the building is listed first; the man is second
c = I.ensure_trigger(cap([el("a bank facade with columns", [0, 0, 1000, 900]),
                          el("a man in a duster coat holding a revolver",
                             [300, 200, 600, 900])]), "neurostepan")
els = c["compositional_deconstruction"]["elements"]
check("the trigger lands on the man, not the bank",
      "neurostepan" in els[1]["desc"] and "neurostepan" not in els[0]["desc"],
      els)

# 2. it still reaches the high level description
check("high_level_description carries it too",
      "neurostepan" in c["high_level_description"], c["high_level_description"])

# 3. nothing named a person -> fall back to the biggest box
c2 = I.ensure_trigger(cap([el("a small lantern", [0, 0, 100, 100]),
                           el("a tall silhouette", [200, 100, 800, 900])]), "x")
els2 = c2["compositional_deconstruction"]["elements"]
check("with no person word the largest element is used",
      "x" in els2[1]["desc"], els2)

# 4. an element that ALREADY has it is left alone
c3 = I.ensure_trigger(cap([el("a horse"), el("neurostepan, a man")]),
                      "neurostepan")
els3 = c3["compositional_deconstruction"]["elements"]
check("an existing trigger is not duplicated onto another element",
      "neurostepan" not in els3[0]["desc"], els3)

# 5. a lettering element must never be chosen, even when it says "man"
c4 = I.ensure_trigger(cap([el("", text="WANTED MAN"),
                           el("a man on a horse", [100, 100, 900, 900])]),
                      "neurostepan")
els4 = c4["compositional_deconstruction"]["elements"]
check("the trigger is not written into rendered lettering",
      "neurostepan" not in str(els4[0].get("text", "")), els4)

# 6. no elements at all -> the scene carries it (unchanged behaviour)
c5 = I.ensure_trigger(cap([]), "neurostepan")
check("an element-less caption puts it on the background",
      "neurostepan" in c5["compositional_deconstruction"]["background"])

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
