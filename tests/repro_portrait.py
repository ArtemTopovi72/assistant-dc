"""Reproduce the portrait-orientation resolution bug.

Runs the contained edit + the 'add a hat' route on portrait images (incl. a
non-16-divisible one) and prints input vs output dimensions. Usage:
  repro_portrait.py <contained|addhat> <image>
"""
import os, sys, time
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from PIL import Image
import image as img

def dims(p):
    with Image.open(p) as im: return im.size

def show(tag, src, out):
    if not out or not os.path.exists(out):
        print(f"  {tag}: NO OUTPUT ({out!r})"); return
    iw,ih=dims(src); ow,oh=dims(out)
    dw=(ow-iw)/iw*100; dh=(oh-ih)/ih*100
    flag = "  <<< RESOLUTION BUG" if (abs(dw)>0.5 or abs(dh)>0.5) else "  OK"
    print(f"  {tag}: {iw}x{ih} -> {ow}x{oh}  (w {dw:+.1f}% h {dh:+.1f}%){flag}  {os.path.basename(out)}")

if __name__=="__main__":
    mode = sys.argv[1] if len(sys.argv)>1 else "contained"
    src = sys.argv[2] if len(sys.argv)>2 else str(_ROOT/"portrait_odd.png")
    print(f"=== {mode} on {os.path.basename(src)} {dims(src)} ===", flush=True)
    t=time.time()
    if mode=="contained":
        out = img.edit_region_contained_with_comfy(None, src, "hair", "blonde hair", denoise=0.7, seed=5)
    else:
        cat,out = img.route_edit_request(None, src, "add a wide-brim hat on the head", seed=5)
        print("  route category:", cat)
    print(f"  [{time.time()-t:.0f}s]")
    show(mode, src, out)
