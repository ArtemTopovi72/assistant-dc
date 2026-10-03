"""A removal verb must name the REGION before the region is deleted.

Live, 2026-09-11: a render scored 10/10, then the agent asked
    inpaint_image(region='cat-shaped object',
                  instructions='Remove the large black shadow covering the
                                cat-shaped object ...', removal=True)
The verb fired, the CAT was deleted from the layout, and the agent spent four
more renders trying to put it back -- the last of them through 'Restore the
cat-shaped object ...', which the router read as PHOTO restoration and answered
with a 2x upscale. Seven renders of one good picture.

Whether an edit removes the region is the model's read (image_router.edit_plan
removes_region); the phrases run live in bench/removal_intent_live.py. Here:
the wiring, with the read stubbed.

Run: venv/Scripts/python.exe tests/test_removal_targets_region.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import logging; logging.basicConfig(level=logging.CRITICAL)

import image_grounding as G
import image_router as R

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:200])


READS = {("Remove the large black shadow covering the cat", ): {"kind": "object_remove", "target": "the shadow"},
         ("remove the puddle", ): {"kind": "object_remove", "target": "the puddle"},
         ("make it cleaner", ): {"kind": "object_remove", "target": ""},
         ("make the shirt blue", ): {"kind": "clothing_edit"}}
R.EDIT_STUB = lambda t: READS.get((t, ))
check("a removal kind with a named target is a removal", G._is_removal_instruction("remove the puddle"))
check("nothing named is a tidy-up, not a removal", not G._is_removal_instruction("make it cleaner"))
check("an edit is not a removal", not G._is_removal_instruction("make the shirt blue"))

R.REGION_STUB = lambda i, r: i == "remove the puddle"
check("removes_region decides _is_removal_of", G._is_removal_of("remove the puddle", "puddle") is True)
check("an edit ON the region is not its removal",
      G._is_removal_of("Remove the large black shadow covering the cat", "cat-shaped object") is False)
check("no region -> no region removal", G._is_removal_of("remove the puddle", "") is False)

import inspect
check("the region reaches the read", "removes_region(instructions, region)" in inspect.getsource(G._is_removal_of))
R.REGION_STUB = None
check("model down -> not a removal (the region is kept)", G._is_removal_of("remove the puddle", "puddle") is False)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
