"""A size/proportion fix must not go through the contained (mask) path: the new
outline never fits the old mask, QA rejects it as leaking, the picture stays
unchanged ('fix the head proportions')."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import image_router as R
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakes import deps_ctx, Over
calls = []
_IMG = Over(R._image,
            load_layout_for=lambda p: None,
            edit_image_with_firered=lambda ctx, p, instr, **k: (calls.append(("whole", instr)), "out.png")[1],
            _extract_edit_target=lambda *a, **k: ("head", "x", 0.8),
            _firered_instruction=lambda ctx, region, text, removal=False: "FR: " + text)
_EDIT = deps_ctx(edit_read=lambda t: {"kind": "subject_edit", "resizes": "proportion" in t or "меньше" in t},   # the read itself: bench/removal_intent_live.py
                 image=_IMG, _contained_edit_validated=lambda *a, **k: (calls.append(("contained",)), None)[1])

for text in ("a head with natural and realistic proportions relative to the body",
             "сделай голову меньше"):
    calls.clear()
    cat, out = R._route_edit_request(_EDIT, "in.png", text)
    assert out == "out.png" and calls and calls[0][0] == "whole", (text, cat, calls)

calls.clear()
R._route_edit_request(_EDIT, "in.png", "make his shirt red")
assert calls and calls[0][0] == "contained", calls   # ordinary edits keep the contained path
print("PASS proportion edits go whole-frame")
