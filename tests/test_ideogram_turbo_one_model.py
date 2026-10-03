"""Turbo at cfg 1 renders with ONE model: no unconditional branch, the 9 GB
uncond model never loaded (A/B 2026-10-02: same pictures, 4-9 s vs 7-13 s).
An explicit cfg keeps the dual-model guider."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ideogram as I
import comfy_client as C

seen = []
C._submit_and_poll = lambda ctx, wf, **k: seen.append(wf) or None
I.IDEOGRAM_CFG_TURBO = 1.0
I.IDEOGRAM_TURBO = True   # opt-in since 2026-10-02
try:
    I.generate(None, "a cat on a sofa")
except Exception:
    pass
wf = seen[-1]
assert wf["182"]["class_type"] == "CFGGuider" and wf["182"]["inputs"]["cfg"] == 1.0, wf["182"]
assert "181" not in wf and "184" not in wf, sorted(wf)
src = wf["182"]["inputs"]["model"][0]
assert src in wf and wf[src]["class_type"] == "ModelSamplingAuraFlow", src
print("ok turbo cfg 1: one-model guider, no uncond model")

try:
    I.generate(None, "a cat on a sofa", steps=20, cfg=7)
except Exception:
    pass
wf = seen[-1]
assert wf["182"]["class_type"] == "DualModelGuider" and "181" in wf, wf["182"]
print("ok an explicit cfg keeps the dual-model guider")

# Default (turbo off): the full int8 run, dual-model guider, no turbo LoRA.
I.IDEOGRAM_TURBO = False
seen.clear()
try:
    I.generate(None, "a cat on a sofa")
except Exception:
    pass
wf = seen[-1]
assert wf["182"]["class_type"] == "DualModelGuider", wf["182"]
assert not any(k.startswith("turbo_") for k in wf), "turbo LoRA attached by default"
print("ok default is the full run")
