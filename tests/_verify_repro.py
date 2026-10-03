"""Focused reproducibility/sensitivity check with retry on transient render errors."""
import hashlib, os, sys, time
from PIL import Image
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import image

PROMPT = ("a red ceramic coffee mug on a wooden table, soft daylight from the left, "
          "shallow depth of field, photorealistic")

def gen(seed, w=512, h=512, tries=3):
    for attempt in range(tries):
        p = image.generate_image_with_comfy(
            ctx=None, prompt=PROMPT, negative_prompt="blurry, text, watermark",
            steps=8, cfg=1.0, seed=seed, width=w, height=h)
        if p and os.path.exists(p):
            with Image.open(p) as im:
                size = im.size
            h_ = hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]
            return p, size, h_
        print(f"  (retry {attempt+1}/{tries} for seed={seed} after transient None)")
        time.sleep(3)
    return None, None, None

def meandiff(a, b):
    ia = np.asarray(Image.open(a).convert("RGB"), dtype=np.float64)
    ib = np.asarray(Image.open(b).convert("RGB"), dtype=np.float64)
    if ia.shape != ib.shape:
        return 999.0
    return float(np.abs(ia - ib).mean())

print("=== reproducibility (same seed) + sensitivity (diff seed) ===")
# Order A(12345) -> C(99999) -> B(12345): the middle different-seed render
# invalidates ComfyUI's identical-prompt execution cache so B re-renders.
pA, sA, hA = gen(12345)
pC, sC, hC = gen(99999)
pB, sB, hB = gen(12345)
print(f"A seed=12345 -> {sA} hash={hA} {os.path.basename(pA) if pA else 'NONE'}")
print(f"B seed=12345 -> {sB} hash={hB} {os.path.basename(pB) if pB else 'NONE'}")
print(f"C seed=99999 -> {sC} hash={hC} {os.path.basename(pC) if pC else 'NONE'}")
if pA and pB and pC:
    dAB = meandiff(pA, pB)
    dAC = meandiff(pA, pC)
    repro = (hA == hB) or dAB < 0.5
    sens = dAC > 2.0
    print(f"A vs B (same seed): meanpixeldiff={dAB:.4f}  hash_match={hA==hB}  -> reproducible={repro}")
    print(f"A vs C (diff seed): meanpixeldiff={dAC:.4f}  -> seed-sensitive={sens}")
    print("\nreproducible_same_seed:", "PASS" if repro else "FAIL")
    print("different_seed_differs:", "PASS" if sens else "FAIL")
else:
    print("\nINCOMPLETE: a job returned None after retries")
