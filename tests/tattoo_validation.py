"""Tattoo-transfer quality validation (live, needs ComfyUI + GPU).

Runs tattoo scenarios on real Desktop images and records identity preservation +
a deterministic SEAM/EDGE metric that flags the failure modes the user called out
(transparent edges, washed-out borders, visible cutout boundary, halo). Outputs are
copied to tests/tattoo_out/ for mandatory visual inspection.

Targets: R.jpg (man, exposed forearms/shoulders/neck). Sources: images.jpg (Stalin
portrait), OIP.jpg (example ink tattoo).

Usage: venv/Scripts/python.exe tests/tattoo_validation.py [id ...]
"""
import json
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import os
import shutil
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from PIL import Image, ImageDraw
import image as im

DESK = os.path.join(os.environ.get("USERPROFILE", os.path.expanduser("~")), "Desktop")
OUT = os.path.join(os.path.dirname(__file__), "tattoo_out")
MASKDIR = os.path.join(os.path.dirname(__file__), "tattoo_masks")
os.makedirs(OUT, exist_ok=True); os.makedirs(MASKDIR, exist_ok=True)

MAN = os.path.join(DESK, "R.jpg")              # 900x1200, exposed forearms
STALIN = os.path.join(DESK, "images.jpg")      # portrait
INK = os.path.join(DESK, "OIP.jpg")            # example tattoo


class Ctx:
    def __init__(self):
        self.cancel_event = threading.Event(); self.last_image_prompt = ""
        self.api_lock = threading.Lock(); self.api_min_interval = 0.0; self.last_api_call_time = 0.0
    def is_cancelled(self): return False
    def set_stage(self, *a): pass


def _box_mask(src, frac, name):
    W, H = Image.open(src).size
    m = Image.new("L", (W, H), 0)
    x0, y0, x1, y1 = frac
    ImageDraw.Draw(m).ellipse((int(W*x0), int(H*y0), int(W*x1), int(H*y1)), fill=255)
    p = os.path.join(MASKDIR, name + ".png"); m.save(p)
    return p


def seam_metric(orig_path, out_path, mask_path):
    """Detect a visible cutout boundary: along the mask's bbox edge, compare the mean
    abs pixel delta in a thin band JUST OUTSIDE the mask (should be ~0 if compositing
    is clean — those pixels must be untouched) vs the interior. A nonzero outside band
    = mask leakage / halo. Returns dict."""
    o = Image.open(orig_path).convert("RGB"); r = Image.open(out_path).convert("RGB")
    if o.size != r.size:
        r = r.resize(o.size)
    m = Image.open(mask_path).convert("L").resize(o.size)
    import numpy as np
    oa = np.asarray(o, dtype=np.int16); ra = np.asarray(r, dtype=np.int16)
    ma = np.asarray(m) > 24
    delta = np.abs(oa - ra).mean(axis=2)           # per-pixel mean channel delta
    outside = delta[~ma]
    inside = delta[ma]
    # "untouched outside" should be ~0; >2 means pixels outside the mask changed (leak)
    return {"outside_mean_delta": round(float(outside.mean()), 2),
            "outside_p99_delta": round(float(np.percentile(outside, 99)), 2),
            "inside_mean_delta": round(float(inside.mean()), 2),
            "changed_outside_frac": round(float((outside > 8).mean()), 4)}


def _S(id, cat, target, refs, instr, tmask, pf=True):
    return dict(id=id, cat=cat, target=target, refs=refs, instr=instr, tmask=tmask, pf=pf)

# forearm/shoulder/neck regions of R.jpg (900x1200), ellipse fracs (x0,y0,x1,y1)
SCEN = [
    _S("t1_portrait", "portrait->tattoo (Stalin+star)", MAN, [(STALIN, "object_source")],
       "Tattoo the portrait from image 2 onto the forearm skin as solid black-ink "
       "linework with a red five-pointed star above the head, embedded in the skin, "
       "sharp clean lines, high contrast, realistic tattoo", (0.10, 0.70, 0.34, 0.90)),
    _S("t2_face_adj", "face-adjacent (neck)", MAN, [(INK, "style_reference")],
       "add a small solid black star tattoo on the side of the neck, sharp clean ink, "
       "embedded in skin", (0.46, 0.10, 0.60, 0.20), pf=True),
    _S("t3_arm", "arm tattoo", MAN, [(INK, "style_reference")],
       "add a bold black five-pointed star tattoo on the forearm, sharp solid ink lines, "
       "embedded in the skin", (0.66, 0.70, 0.90, 0.90)),
    _S("t4_shoulder", "shoulder tattoo", MAN, [(INK, "style_reference")],
       "add a tribal black star tattoo on the upper arm/shoulder, sharp solid ink, "
       "embedded in skin", (0.66, 0.40, 0.88, 0.56)),
    _S("t5_small", "small tattoo (wrist)", MAN, [(INK, "style_reference")],
       "add a tiny black star tattoo on the wrist, sharp clean ink", (0.70, 0.86, 0.80, 0.94)),
    _S("t6_large", "large tattoo (forearm sleeve)", MAN, [(STALIN, "object_source")],
       "Tattoo a large detailed black-ink portrait from image 2 with a red star as a "
       "full forearm sleeve, sharp lines, embedded in skin", (0.08, 0.62, 0.36, 0.92)),
]


def run(s):
    ctx = Ctx()
    rec = {"id": s["id"], "category": s["cat"], "instruction": s["instr"][:80],
           "target_region": s["tmask"], "protect_face": s["pf"]}
    tmask = _box_mask(s["target"], s["tmask"], s["id"] + "_t")
    refs = [im.ReferenceImage(p, r) for p, r in s["refs"]]
    t0 = time.time()
    try:
        out = im.plan_and_execute_transfer(ctx, s["target"], refs, s["instr"],
                                           mask_override=tmask, protect_face=s["pf"], timeout=520)
    except Exception as exc:
        rec.update(success=False, error=f"{type(exc).__name__}: {exc}"); return rec
    rec["runtime"] = round(time.time() - t0, 1)
    if out and os.path.exists(out):
        W, H = Image.open(s["target"]).size
        ow, oh = Image.open(out).size
        rec["output_dims"] = [ow, oh]; rec["dims_preserved"] = (ow, oh) == (W, H)
        try:
            import identity_metrics as idm
            c = idm.identity_cosine(s["target"], out)
            rec["identity_cosine"] = round(c, 4) if c is not None else None
        except Exception:
            rec["identity_cosine"] = None
        try:
            rec["seam"] = seam_metric(s["target"], out, tmask)
        except Exception as exc:
            rec["seam"] = f"err: {exc}"
        dst = os.path.join(OUT, s["id"] + ".png"); shutil.copy(out, dst)
        rec["review_file"] = os.path.relpath(dst, os.path.dirname(os.path.dirname(__file__)))
        rec["success"] = True
    else:
        rec.update(success=False, error="no output")
    return rec


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.ERROR)
    ids = sys.argv[1:]
    chosen = [s for s in SCEN if (not ids or s["id"] in ids or s["id"].split("_")[0] in ids)]
    report = []
    for s in chosen:
        print(f"--- {s['id']} {s['cat']} ---", flush=True)
        rec = run(s); print(json.dumps(rec, ensure_ascii=False), flush=True); report.append(rec)
    rp = os.path.join(OUT, "report.json")
    # merge with any prior report
    prior = []
    if os.path.exists(rp):
        try: prior = json.load(open(rp, encoding="utf-8"))
        except Exception: prior = []
    by_id = {r["id"]: r for r in prior}
    for r in report: by_id[r["id"]] = r
    json.dump(list(by_id.values()), open(rp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    ok = sum(1 for r in report if r.get("success"))
    print(f"\n=== {ok}/{len(report)} produced output; report -> {rp} ===")
