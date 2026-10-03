"""Live ComfyUI + LM Studio test for the 2026-06-26 image-pipeline changes:
 A) region inventory + grounding (vision),
 B) absent-region short-circuit (NO wasted render),
 C) present-region real edit (full ComfyUI render, dims preserved),
 D) orientation: a scene generates LANDSCAPE (w>h) + stricter evaluator fires.

Usage: venv/Scripts/python.exe bench/live_grounding.py [image] [vision_model]
"""
import os, sys, time, glob, threading
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from PIL import Image
import image as img

VISION_MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen/qwen3-vl-8b"

class Ctx:
    """Minimal real-ish context: single multimodal model for vision + text."""
    def __init__(self, model):
        self.model_name = model
        self.no_think = True
        self.reasoning_effort = "low"
        self.api_lock = threading.Lock()
        self.last_api_call_time = 0.0
        self.api_min_interval = 0.0
        self.custom_personality_text = ""
    def is_cancelled(self): return False
    def set_stage(self, *a, **k): pass
    def remember(self, *a, **k): pass

def dims(p):
    with Image.open(p) as im: return im.size

def hr(t): print("\n" + "="*70 + f"\n{t}\n" + "="*70, flush=True)

ctx = Ctx(VISION_MODEL)
src = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(str(_ROOT/"tests"/"_asset_clothing_source_*.png")))[0]
print(f"image={os.path.basename(src)} {dims(src)}  vision_model={VISION_MODEL}", flush=True)

# ---- A: inventory + grounding -------------------------------------------------
hr("A) REGION INVENTORY + GROUNDING (live vision)")
t = time.time()
inv = img._image_region_inventory(ctx, src)
print(f"inventory ({time.time()-t:.1f}s): {inv}", flush=True)
present_region = None
for cand in ("jacket", "shirt", "dress", "coat", "top", "clothing"):
    if any(cand in it or it in cand for it in inv):
        present_region = cand; break
present_region = present_region or (inv[0] if inv else "shirt")
for r in (present_region, "umbrella"):
    t = time.time()
    g = img._ground_region_phrase(ctx, src, r)
    print(f"  ground({r!r}) -> {g!r}   [{time.time()-t:.1f}s]", flush=True)

# ---- B: absent region must short-circuit (no render) --------------------------
hr("B) ABSENT REGION SHORT-CIRCUIT (expect None, region_absent, fast)")
t = time.time()
out = img.inpaint_region_with_comfy(ctx, src, "umbrella", "make it bright red")
el = time.time()-t
reason = img._INPAINT_FAILURE.get("reason")
ok_b = (out is None) and reason == "region_absent" and el < 25
print(f"  out={out!r} reason={reason!r} elapsed={el:.1f}s  -> {'PASS' if ok_b else 'CHECK'}", flush=True)

# ---- C: present region real edit (full render, dims preserved) ----------------
hr(f"C) PRESENT-REGION EDIT (region={present_region!r}, full ComfyUI render)")
t = time.time()
out = img.inpaint_region_with_comfy(ctx, src, present_region, "change its color to bright red")
el = time.time()-t
if out and os.path.exists(out):
    iw,ih = dims(src); ow,oh = dims(out)
    ok_c = (iw,ih) == (ow,oh)
    print(f"  out={os.path.basename(out)} {ow}x{oh} (src {iw}x{ih}) dims_preserved={ok_c} [{el:.0f}s]", flush=True)
else:
    print(f"  NO OUTPUT out={out!r} reason={img._INPAINT_FAILURE.get('reason')!r} [{el:.0f}s]", flush=True)

# ---- D: orientation on a scene (expect landscape w>h) + strict eval -----------
hr("D) ORIENTATION: scene should generate LANDSCAPE (w>h)")
# keep it to a single render so the test stays bounded
img.MAX_IMAGE_REFINEMENT_ATTEMPTS = 1
t = time.time()
res = img.generate_image_with_refinement(ctx, "a giraffe standing in a philharmonic concert hall")
el = time.time()-t
p = res.get("path")
print(f"  status={res.get('status')} score={res.get('score')} reason={res.get('reason')!r}", flush=True)
if p and os.path.exists(p):
    w,h = dims(p)
    print(f"  {os.path.basename(p)} {w}x{h} -> {'LANDSCAPE OK' if w>h else 'PORTRAIT (BUG)'} [{el:.0f}s]", flush=True)
else:
    print(f"  NO IMAGE path={p!r} [{el:.0f}s]", flush=True)

print("\nDONE.", flush=True)
