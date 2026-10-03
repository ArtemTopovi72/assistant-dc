"""Deterministic tests for the stateful edit graph (no GPU).

Proves the state-preservation + subject-lock + validation-gate behaviour the user
required, by stubbing the heavy image ops and the identity metrics:

  1. a chained op (upscale) operates on the CURRENT EDITED image, not the original;
  2. a face-locked op whose identity collapses is marked FAILED and does NOT become
     the current state (so the next op still builds on the last good image);
  3. an op that legitimately changes identity (face_edit) is accepted.
"""
import sys, types
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import edit_state

fails = []
def check(c, m): (print(f"  [{'PASS' if c else 'FAIL'}] {m}")); (None if c else fails.append(m))

# --- stub the heavy modules edit_state imports lazily --------------------- #
import image as img
import identity_metrics as idm

calls = {"route_in": [], "upscale_in": []}

def fake_route(ctx, image_path, instruction, *, seed=None, timeout=1900):
    calls["route_in"].append(image_path)
    # category depends on instruction keyword
    cat = "face_edit" if "face" in instruction else "clothing_edit"
    return cat, str(_ROOT / f"edited_{len(calls['route_in'])}.png")

def fake_upscale(ctx, image_path, *, scale="4x", timeout=1900):
    calls["upscale_in"].append(image_path)
    return str(_ROOT / "upscaled.png")

img.route_edit_request = fake_route
img.upscale_image_with_comfy = fake_upscale

# identity metric stub: controllable per-call
_idstate = {"cos": 0.99, "face": 0.02, "overall": 0.2}
idm.identity_cosine = lambda a, b: _idstate["cos"]
idm.face_region_change = lambda a, b: _idstate["face"]
idm.changed_fraction = lambda a, b, **k: {"overall": _idstate["overall"]}

# make os.path.exists(edited/upscaled) return True for our fake paths
import os
_real_exists = os.path.exists
os.path.exists = lambda p: True if ("edited_" in str(p) or "upscaled" in str(p) or "source" in str(p)) else _real_exists(p)

SRC = str(_ROOT / "source.png")

# --- 1. state preservation across a chain -------------------------------- #
s = edit_state.EditSession(SRC)
_idstate.update(cos=0.98, face=0.03)           # clothing edit keeps identity
n1 = s.apply_edit("make the jacket red")        # clothing_edit
check(n1.ok, "clothing edit accepted (identity held)")
check(calls["route_in"][-1] == SRC, "edit #1 operated on the SOURCE image")

n2 = s.apply_edit("make the trousers black")   # a chained edit
check(n2.ok, "second edit accepted")
# THE KEY ASSERTION: the chained edit ran on the EDITED image, not the original
check(calls["route_in"][-1] == n1.image_path, "edit #2 operated on the EDITED state, not the original")
check(calls["route_in"][-1] != SRC, "edit #2 did NOT regenerate from source")
check(s.current.image_path == n2.image_path, "current state advanced to the second result")

# --- 2. face-locked op that loses identity is rejected -------------------- #
s2 = edit_state.EditSession(SRC)
_idstate.update(cos=0.10, face=0.8)             # clothing edit but identity collapsed
bad = s2.apply_edit("change the jacket")        # clothing_edit -> face must be kept
check(not bad.ok, "clothing edit with collapsed identity is marked FAILED")
check(s2.current_path == SRC, "failed edit does NOT become current (state preserved)")

# --- 3. face_edit legitimately changes identity, still accepted ----------- #
s3 = edit_state.EditSession(SRC)
_idstate.update(cos=0.20, face=0.6)             # identity 'change' expected for face_edit
fe = s3.apply_edit("retouch the face")          # face_edit -> expect_identity_change
check(fe.ok, "face_edit accepted despite identity change (lock policy allows it)")

# --- 4. lock policy exposes editable/locked regions ---------------------- #
check("background" in edit_state.LOCK_POLICY["clothing_edit"]["locked"],
      "clothing_edit locks the background")
check("face" in edit_state.LOCK_POLICY["clothing_edit"]["locked"],
      "clothing_edit locks the face")
check("identity" in edit_state.LOCK_POLICY["upscale"]["locked"],
      "upscale locks identity/geometry/composition")

os.path.exists = _real_exists
print(f"\nRESULT: {'ALL PASS' if not fails else str(len(fails))+' FAILED'}")
sys.exit(1 if fails else 0)
