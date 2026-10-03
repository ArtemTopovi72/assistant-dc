"""A character render on the Ideogram engine must actually invoke the adapter.

Attaching a LoRA is only half of it: the adapter fires on its TRIGGER WORD, and
on this path the prompt is rewritten into a JSON caption by the planner before
the encoder ever sees it. A planner that drops the trigger leaves the adapter
attached but idle, which renders a stranger under that person's name and
reports success. That failure is invisible to every other check, so it is
pinned here.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ideogram

FAILED = []


RAN = []


def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    RAN.append(name)
    if not cond:
        FAILED.append(name)


def _caption(desc, background="a plain wall"):
    return {"compositional_deconstruction": {
        "background": background,
        "elements": [{"type": "obj", "desc": desc}]}}


def main():
    # The planner dropped the trigger -> it must be put back on the subject.
    cap = ideogram.ensure_trigger(_caption("a man in a dark coat"), "neurostepan")
    first = cap["compositional_deconstruction"]["elements"][0]["desc"]
    check("trigger restored onto the subject element",
          first.lower().startswith("neurostepan"))
    check("the original description survives", "dark coat" in first)

    # Already present -> left exactly as it was, no doubling.
    same = _caption("neurostepan, a man in a dark coat")
    out = ideogram.ensure_trigger(same, "neurostepan")
    desc = out["compositional_deconstruction"]["elements"][0]["desc"]
    check("an existing trigger is not duplicated", desc.count("neurostepan") == 1)

    # Case-insensitive: "Neurostepan" already invokes the adapter.
    out = ideogram.ensure_trigger(_caption("Neurostepan, a man"), "neurostepan")
    desc = out["compositional_deconstruction"]["elements"][0]["desc"]
    check("a differently-cased trigger counts as present",
          desc.lower().count("neurostepan") == 1)

    # No elements at all -> the scene description carries it rather than nothing.
    cap = {"compositional_deconstruction": {"background": "a street", "elements": []}}
    out = ideogram.ensure_trigger(cap, "neurostepan")
    check("with no elements the trigger lands on the background",
          "neurostepan" in out["compositional_deconstruction"]["background"])

    # The trigger must NEVER end up in a lettering element. Ideogram renders
    # `text` as legible letters on the picture, so a trigger that lands there is
    # drawn ON the character rather than conditioning them -- the word appearing
    # across a shirt, which is what it looked like live.
    cap = {"high_level_description": "a man outside a cafe",
           "compositional_deconstruction": {"background": "a street", "elements": [
               {"type": "obj", "desc": "a man"},
               {"type": "text", "desc": "a cafe sign", "text": "NEUROSTEPAN cafe"}]}}
    out = ideogram.ensure_trigger(cap, "neurostepan")
    els = out["compositional_deconstruction"]["elements"]
    check("the trigger is stripped out of lettering",
          "neurostepan" not in els[1]["text"].lower())
    check("the rest of the sign survives", els[1]["text"] == "cafe")
    check("and it still conditions the subject",
          els[0]["desc"].lower().startswith("neurostepan"))

    # A word that merely CONTAINS the trigger is not the trigger.
    cap = {"compositional_deconstruction": {"elements": [
        {"type": "obj", "desc": "a man"},
        {"type": "text", "desc": "sign", "text": "NEUROSTEPANOVICH LTD"}]}}
    out = ideogram.ensure_trigger(cap, "neurostepan")
    check("a longer word containing the trigger is left alone",
          out["compositional_deconstruction"]["elements"][1]["text"]
          == "NEUROSTEPANOVICH LTD")

    # In TRAINING the trigger sits in high_level_description as well as the
    # subject element. Rendering with it in only one of the two is a different
    # conditioning from the one the adapter was fitted on.
    cap = {"high_level_description": "A photograph of a man on a roof.",
           "compositional_deconstruction": {"elements": [{"type": "obj",
                                                          "desc": "a man"}]}}
    out = ideogram.ensure_trigger(cap, "neurostepan")
    check("the headline description carries the trigger too",
          out["high_level_description"].lower().startswith("neurostepan"))
    out2 = ideogram.ensure_trigger(dict(out), "neurostepan")
    check("and it is not doubled on a second pass",
          out2["high_level_description"].lower().count("neurostepan") == 1)

    # No trigger asked for -> untouched.
    cap = _caption("a man")
    check("no trigger is a no-op", ideogram.ensure_trigger(cap, "") is cap)
    check("None trigger is a no-op", ideogram.ensure_trigger(cap, None) is cap)

    # A character render must not go through the lettering repair loop: the
    # repair re-draws WITHOUT the adapter.
    src = (ROOT / "imaging/image_generate.py").read_text(encoding="utf-8")
    check("the lettering loop is disabled for a character render",
          "if lora_name:\n        rounds = 0" in src)
    check("a character render reaches Ideogram with its adapter",
          "lora_name=lora_name," in src and "result = _ideogram_draw(" in src)

    print(chr(10) + "%d checks, %d failed" % (len(RAN), len(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
