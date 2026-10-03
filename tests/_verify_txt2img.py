"""Real end-to-end verification of the txt2img sampler + resolution fixes.
Calls the actual generate_image_with_comfy against the live ComfyUI server."""
import hashlib, os, sys
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import image

PROMPT = ("a red ceramic coffee mug on a wooden table, soft daylight from the left, "
          "shallow depth of field, photorealistic")

def gen(seed, w, h, steps=8, cfg=1.0):
    p = image.generate_image_with_comfy(
        ctx=None, prompt=PROMPT, negative_prompt="blurry, text, watermark",
        steps=steps, cfg=cfg, seed=seed, width=w, height=h)
    if not p or not os.path.exists(p):
        return None, None, None
    with Image.open(p) as im:
        size = im.size
    h_ = hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]
    return p, size, h_

def meandiff(a, b):
    import numpy as np
    ia = np.asarray(Image.open(a).convert("RGB"), dtype=np.float64)
    ib = np.asarray(Image.open(b).convert("RGB"), dtype=np.float64)
    if ia.shape != ib.shape:
        return 999.0
    return float(np.abs(ia - ib).mean())

print("=== PRIORITY 2: resolution fidelity ===")
res_ok = True
for (w, h) in [(512, 512), (768, 512), (1024, 576)]:
    p, size, hsh = gen(1234, w, h)
    match = size == (w, h)
    res_ok &= match
    print(f"requested {w}x{h:<4} -> delivered {size}  hash={hsh}  {'OK' if match else 'FAIL'}  {os.path.basename(p) if p else 'NONE'}")

print("\n=== PRIORITY 1: reproducibility (same seed) + sensitivity (diff seed) ===")
pA, sA, hA = gen(12345, 512, 512)
pB, sB, hB = gen(12345, 512, 512)   # identical seed
pC, sC, hC = gen(99999, 512, 512)   # different seed
print(f"A seed=12345 -> {sA} hash={hA} {os.path.basename(pA) if pA else 'NONE'}")
print(f"B seed=12345 -> {sB} hash={hB} {os.path.basename(pB) if pB else 'NONE'}")
print(f"C seed=99999 -> {sC} hash={hC} {os.path.basename(pC) if pC else 'NONE'}")
if pA and pB and pC:
    dAB = meandiff(pA, pB)
    dAC = meandiff(pA, pC)
    repro = (hA == hB) or dAB < 0.5
    sens = dAC > 2.0
    print(f"A vs B (same seed): meanpixeldiff={dAB:.4f}  -> reproducible={repro}")
    print(f"A vs C (diff seed): meanpixeldiff={dAC:.4f}  -> seed-sensitive={sens}")
else:
    repro = sens = False

print("\n=== SUMMARY ===")
print("resolution_fidelity:", "PASS" if res_ok else "FAIL")
print("reproducible_same_seed:", "PASS" if repro else "FAIL")
print("different_seed_differs:", "PASS" if sens else "FAIL")
