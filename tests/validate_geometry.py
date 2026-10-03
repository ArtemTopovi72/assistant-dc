"""Live geometry-preservation validation against the running ComfyUI.

Runs each generation-based editing pipeline on a >1MP portrait and reports, per
pipeline: input/output resolution, subject (face) bounding box, and width/height/
area drift %. Target for geometry-preserving routes: < 1% dimension drift.

Subject bbox is measured with OpenCV's Haar face detector and expressed as a
FRACTION of the frame (cx, cy, w, h) so the subject's in-frame scale/position can
be compared even if absolute resolution differed. Requires ComfyUI up + models.
"""
import os, sys, time
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import cv2
from PIL import Image
import image as img

SRC = str(_ROOT / "pozner_large.png")   # 1800x1200, 2.16 MP
_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def dims(path):
    with Image.open(path) as im:
        return im.size  # (w,h)


def face_frac(path):
    """Largest face as (cx,cy,fw,fh) fractions of the frame, or None."""
    im = cv2.imread(path)
    if im is None:
        return None
    h, w = im.shape[:2]
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = _cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    return ((x + fw / 2) / w, (y + fh / 2) / h, fw / w, fh / h)


def pct(a, b):
    return 0.0 if not a else (b - a) / a * 100.0


def run(name, fn):
    print(f"\n=== {name} ===", flush=True)
    iw, ih = dims(SRC)
    ifrac = face_frac(SRC)
    t = time.time()
    try:
        out = fn()
    except Exception as e:
        print(f"  ERROR: {e!r}")
        return
    dt = time.time() - t
    if not out or not os.path.exists(out):
        print(f"  NO OUTPUT (engine returned {out!r}) in {dt:.0f}s")
        return
    ow, oh = dims(out)
    ofrac = face_frac(out)
    print(f"  input : {iw}x{ih}  ({iw*ih/1e6:.2f} MP)")
    print(f"  output: {ow}x{oh}  ({ow*oh/1e6:.2f} MP)   [{dt:.0f}s]  {Path(out).name}")
    print(f"  drift : width {pct(iw,ow):+.2f}%  height {pct(ih,oh):+.2f}%  area {pct(iw*ih,ow*oh):+.2f}%")
    if ifrac and ofrac:
        labels = ["cx", "cy", "face_w", "face_h"]
        print("  subject bbox (fraction of frame):")
        for l, a, b in zip(labels, ifrac, ofrac):
            print(f"     {l:7} in={a:.3f}  out={b:.3f}  drift={pct(a,b):+.1f}%")
    else:
        print(f"  subject bbox: in={ifrac} out={ofrac} (face not detected in one)")


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = {
        "firered": lambda: img.edit_image_with_firered(
            None, SRC, "Change the background to a wooden bookshelf library. Keep the person exactly the same.", seed=42),
        "bg_replace": lambda: img.replace_background_with_comfy(
            None, SRC, "a sunny green park with trees", seed=42),
        "relight": lambda: img.relight_image_with_comfy(
            None, SRC, "warm golden hour sunlight from the left", seed=42),
    }
    for n, fn in jobs.items():
        if only in ("all", n):
            run(n, fn)
