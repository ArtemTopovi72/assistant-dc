"""Identity + edit-containment validation: whole-frame vs contained edit.

Runs the SAME localized edit ("make the jacket bright red") two ways on a high-res
portrait and reports, per method:
  * output resolution vs source
  * identity cosine (SFace; >0.363 = same person, ~1.0 = face unchanged)
  * overall pixels changed
  * FACE-region pixels changed (containment proxy; ~0 = face untouched)

Usage: validate_identity.py [whole|contained]
"""
import os, sys, time
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from PIL import Image
import image as img
import identity_metrics as idm

SRC = str(_ROOT / "pozner_large.png")          # 1800x1200 portrait, face present
REGION = "jacket"                               # the region we intend to edit
RESULT_PROMPT = "a bright red jacket, photorealistic, same fabric folds"
INSTRUCTION = "make the jacket bright red"      # whole-frame phrasing


def dims(p):
    with Image.open(p) as im:
        return im.size


def report(name, out):
    if not out or not os.path.exists(out):
        print(f"  {name}: NO OUTPUT ({out!r})"); return
    iw, ih = dims(SRC); ow, oh = dims(out)
    cos = idm.identity_cosine(SRC, out)
    overall = idm.changed_fraction(SRC, out).get("overall")
    facechg = idm.face_region_change(SRC, out)
    print(f"  {name}: {os.path.basename(out)}")
    print(f"     resolution : {iw}x{ih} -> {ow}x{oh}")
    print(f"     identity   : cosine={cos:.3f}  ({'SAME person' if (cos or 0)>0.363 else 'DIFFERENT person'})")
    print(f"     changed    : overall={overall*100:.1f}%  face-region={facechg*100:.1f}%")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("whole", "both"):
        print("=== WHOLE-FRAME FireRed (old path) ===", flush=True)
        t = time.time()
        out = img.edit_image_with_firered(None, SRC, INSTRUCTION + ". Keep the person exactly the same.", seed=7)
        print(f"  [{time.time()-t:.0f}s]")
        report("whole_frame", out)
    if which in ("contained", "both"):
        print("=== CONTAINED edit (mask + composite-back) ===", flush=True)
        t = time.time()
        out = img.edit_region_contained_with_comfy(None, SRC, REGION, RESULT_PROMPT, denoise=0.8, seed=7)
        print(f"  [{time.time()-t:.0f}s]")
        report("contained", out)
