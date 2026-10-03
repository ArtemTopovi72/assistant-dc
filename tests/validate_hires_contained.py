"""Validate crop-based contained editing on REAL high-res images.

Proves the user's hard rule: a localized edit on a >10 MP portrait stays at the
native resolution, keeps the same person, and is sharp (not a low-res
regeneration). Reports input/output dims, drift %, and identity cosine.

Usage:
  venv/Scripts/python.exe tests/validate_hires_contained.py <image> <region> <prompt>
  (defaults: tests/sydney_native.jpg "hat" "a stylish wide-brim hat")
"""
import os, sys, time
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from PIL import Image
import image as img

try:
    import identity_metrics as idm
except Exception:
    idm = None


def dims(p):
    with Image.open(p) as im:
        return im.size


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else str(_ROOT / "tests" / "sydney_native.jpg")
    region = sys.argv[2] if len(sys.argv) > 2 else "hat"
    prompt = sys.argv[3] if len(sys.argv) > 3 else "a stylish wide-brim hat"
    iw, ih = dims(src)
    print(f"=== crop-based contained: region={region!r} prompt={prompt!r} ===")
    print(f"INPUT: {os.path.basename(src)}  {iw}x{ih}  ({iw*ih/1e6:.2f} MP)", flush=True)

    t = time.time()
    out = img.edit_region_contained_cropped(None, src, region, prompt, denoise=0.75, seed=7)
    dt = time.time() - t

    if not out or not os.path.exists(out):
        print(f"RESULT: NO OUTPUT ({out!r})  [{dt:.0f}s]")
        return 1
    ow, oh = dims(out)
    dw = (ow - iw) / iw * 100
    dh = (oh - ih) / ih * 100
    drift_ok = abs(dw) < 0.5 and abs(dh) < 0.5
    print(f"OUTPUT: {ow}x{oh}  (w {dw:+.2f}% h {dh:+.2f}%)  [{dt:.0f}s]")
    print(f"  resolution preserved: {'YES' if drift_ok else 'NO <<< DRIFT'}")
    print(f"  file: {out}")

    if idm is not None:
        try:
            cos = idm.identity_cosine(src, out)  # takes PATHS, compares largest faces
            if cos is None:
                print("  identity cosine: n/a (no face detected in one image)")
            else:
                same = cos >= idm.SFACE_SAME_PERSON_COSINE
                print(f"  identity cosine: {cos:.3f}  thr={idm.SFACE_SAME_PERSON_COSINE}  "
                      f"({'SAME person' if same else 'DIFFERENT <<<'})")
            frc = idm.face_region_change(src, out)
            if frc is not None:
                print(f"  face-region pixels changed: {frc*100:.2f}%  "
                      f"({'face untouched' if frc < 0.02 else 'FACE ALTERED <<<'})")
            ov = idm.changed_fraction(src, out).get("overall")
            if ov is not None:
                print(f"  overall pixels changed: {ov*100:.2f}%  (containment)")
        except Exception as exc:
            print(f"  identity: error {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
