"""Prove the gray-blob fix: FireRed-contained generates REAL content in the region.

Compares, on the same image/region:
  - OLD: the old model contained inpaint (the gray-blob path)
  - NEW: FireRed-contained (crop -> FireRed -> composite masked region back)

Metrics per result:
  - resolution drift (must be 0)
  - identity cosine vs source (must stay same person)
  - edited-region texture = std-dev of pixels inside the changed area. A flat gray
    blob has near-zero std; real content (a garment) has high std. This is the
    decisive "is it gray?" number.

Usage: venv/Scripts/python.exe tests/validate_content_fidelity.py <image> <region> <instruction>
       (defaults: pozner.jpg "jacket" "change his jacket to a red leather jacket")
"""
import os, sys, time
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import numpy as np
from PIL import Image
import image as img
try: import identity_metrics as idm
except Exception: idm = None


def dims(p):
    with Image.open(p) as im: return im.size


def region_texture_std(src, out):
    a = np.asarray(Image.open(src).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(out).convert("RGB").resize(dims(src)), dtype=np.int16)
    diff = np.abs(a - b).max(axis=2)
    m = diff > 18
    if m.sum() < 50:
        return None, 0.0
    changed = b[m]                      # output pixels inside the changed region
    return float(changed.std()), float(m.mean() * 100)


def run(label, fn, src, *args):
    print(f"\n### {label}")
    t = time.time()
    out = fn(*args)
    dt = time.time() - t
    if not out or not os.path.exists(out):
        print(f"  NO OUTPUT  [{dt:.0f}s]"); return
    iw, ih = dims(src); ow, oh = dims(out)
    print(f"  {iw}x{ih} -> {ow}x{oh}  drift {((ow-iw)/iw*100):+.2f}%/{((oh-ih)/ih*100):+.2f}%  [{dt:.0f}s]")
    std, pct = region_texture_std(src, out)
    if std is not None:
        verdict = "GRAY BLOB <<<" if std < 12 else "real content"
        print(f"  region changed: {pct:.2f}%   region texture std: {std:.1f}  ({verdict})")
    if idm is not None:
        try:
            cos = idm.identity_cosine(src, out)
            print(f"  identity cosine: {cos if cos is None else round(cos,3)}")
        except Exception as exc:
            print(f"  identity err: {exc}")
    print(f"  file: {out}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else str(_ROOT / "pozner.jpg")
    region = sys.argv[2] if len(sys.argv) > 2 else "jacket"
    instr = sys.argv[3] if len(sys.argv) > 3 else "change his jacket to a red leather jacket"
    print(f"IMAGE {os.path.basename(src)} {dims(src)}  region={region!r}  instr={instr!r}")
    run("NEW FireRed-contained", img.edit_region_contained_via_firered, src,
        None, src, region, instr)
    run("OLD the old model contained (denoise 0.7)", img.edit_region_contained_with_comfy, src,
        None, src, region, instr)
