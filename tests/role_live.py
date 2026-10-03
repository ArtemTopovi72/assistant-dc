"""LIVE per-role transfer validation against ComfyUI/FireRed on real images.

Exercises EVERY Transfer-tab role end-to-end and measures objective quality:
  * output produced + dimensions preserved (no silent downscale);
  * the transfer actually CHANGED the image (changed_frac > 0 -> not a no-op);
  * identity_cosine target↔output (identity-preserving roles must stay high; roles
    that deliberately change the person are reported, not failed);
  * for masked/contained runs, seam_metric -> outside-the-mask leakage ~0.

Requires LM Studio's LLM UNLOADED so FireRed (Qwen-Image-Edit) fits in the 12GB GPU.

Run one:   ./venv/Scripts/python.exe tests/role_live.py clothing
Run all:   ./venv/Scripts/python.exe tests/role_live.py all
Outputs -> tests/role_out/<id>.png  + ROLE_FINDINGS.md
"""
import io
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import os
import sys
import time
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

import image as im
import identity_metrics as idm
from PIL import Image
import numpy as np

DESK = os.path.expanduser("~/Desktop")
OUT = os.path.join(os.path.dirname(__file__), "role_out")
os.makedirs(OUT, exist_ok=True)


def D(n):
    return os.path.join(DESK, n)


# real images (faces confirmed where needed)
TGT = D("photo_2026-05-20_12-40-47.jpg")     # person + face, 572x795 (target)
PERSON2 = D("photo_2026-05-20_12-47-07.jpg")  # second person/face, 1280x853
SUIT = D("OIP.jpg")                            # man in a suit (clothing/face2), 474x471
SCENE = D("i.webp")                            # landscape scene, 1280x854
ART = D("images.jpg")                          # portrait/art (style), 193x261
OBJ = D("631a23becc82b53YYfUktezstivali849.jpg.jpg")  # clean product shot (object)


def make_ctx():
    import threading
    from models import Context
    ctx = Context.__new__(Context)
    ctx.model_name = ""
    ctx.no_think = True
    ctx.last_api_call_time = 0.0
    ctx.api_min_interval = 0.0
    ctx.api_lock = threading.Lock()
    ctx.cancel_event = threading.Event()
    ctx.stage_callback = None
    return ctx


def _ellipse_mask(target, frac):
    """Write an elliptical L mask over `frac`=(x0,y0,x1,y1) of the target (manual mask)."""
    from PIL import ImageDraw
    w, h = Image.open(target).size
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).ellipse((int(frac[0]*w), int(frac[1]*h), int(frac[2]*w), int(frac[3]*h)),
                              fill=255)
    p = os.path.join(OUT, "_mask_tmp.png")
    m.save(p)
    return p


# (id, role, target, ref, mode, instruction, mask_frac_or_None, identity_preserving)
SCEN = [
    ("clothing", im.ROLE_CLOTHING, TGT, SUIT, "contained",
     "put the suit jacket from image 2 on the person", None, True),
    ("object",   im.ROLE_OBJECT,   TGT, OBJ, "contained",
     "add the object shown in image 2 onto the person, placed and scaled naturally in "
     "the masked area", (0.28, 0.45, 0.72, 0.85), True),
    ("hair",     im.ROLE_HAIR,     TGT, PERSON2, "contained",
     "give the person the hairstyle from image 2", None, True),
    ("face",     im.ROLE_FACE,     TGT, SUIT, "contained",
     "apply the facial identity from image 2", None, False),
    ("pose",     im.ROLE_POSE,     TGT, PERSON2, "whole",
     "pose the subject as shown in image 2", None, False),
    ("style",    im.ROLE_STYLE,    TGT, ART, "whole",
     "restyle image 1 using the style of image 2", None, False),
    ("scene",    im.ROLE_SCENE,    TGT, SCENE, "whole",
     "place the subject of image 1 into the scene from image 2", None, False),
    ("identity", im.ROLE_IDENTITY, TGT, SUIT, "whole",
     "render the character from image 2 into image 1", None, False),
    ("lighting", im.ROLE_LIGHTING, TGT, SCENE, "whole",
     "relight image 1 to match the lighting and mood of image 2", None, False),
    # item 10: manual mask routes a GLOBAL role (style) through the contained path,
    # with a strong, visible instruction so the masked region clearly changes.
    ("masked_style", im.ROLE_STYLE, TGT, ART, "contained",
     "render the masked region as a vivid Van Gogh oil painting with thick swirling "
     "impasto brushstrokes and bold colours", (0.20, 0.30, 0.80, 0.95), False),
]


def changed_frac(a_path, b_path):
    a = Image.open(a_path).convert("RGB"); b = Image.open(b_path).convert("RGB")
    if a.size != b.size:
        b = b.resize(a.size)
    aa = np.asarray(a, np.int16); ba = np.asarray(b, np.int16)
    return round(float((np.abs(aa - ba).mean(axis=2) > 8).mean()), 4)


def fully_opaque(path):
    """No transparency/halo: output must be fully opaque RGB."""
    im2 = Image.open(path)
    if im2.mode in ("RGBA", "LA") or (im2.mode == "P" and "transparency" in im2.info):
        al = im2.convert("RGBA").split()[-1]
        return al.getextrema()[0] == 255
    return True


def run_one(spec):
    sid, role, target, ref, mode, instr, mask_frac, ident_keep = spec
    if not os.path.exists(target) or not os.path.exists(ref):
        return {"id": sid, "role": role, "error": f"missing image (t={os.path.exists(target)} r={os.path.exists(ref)})"}
    ctx = make_ctx()
    sw, sh = Image.open(target).size
    mask = _ellipse_mask(target, mask_frac) if mask_frac else None
    t0 = time.time()
    try:
        if mode == "whole":
            out = im.transfer_with_references(ctx, target, [im.ReferenceImage(ref, role)], instr)
        else:
            refs = [im.ReferenceImage(ref, role)]
            out = im.plan_and_execute_transfer(ctx, target, refs, instr,
                                               mask_override=mask, protect_face=(role != im.ROLE_FACE))
    except Exception as exc:
        import traceback; traceback.print_exc()
        return {"id": sid, "role": role, "error": str(exc)}
    elapsed = round(time.time() - t0, 1)
    if not out or not os.path.exists(out):
        return {"id": sid, "role": role, "mode": mode, "elapsed": elapsed, "error": "no output"}
    save = os.path.join(OUT, f"{sid}.png")
    Image.open(out).convert("RGB").save(save)
    ow, oh = Image.open(out).size
    rec = {"id": sid, "role": role, "mode": mode, "elapsed": elapsed, "output": save,
           "src_dims": [sw, sh], "out_dims": [ow, oh],
           "dims_preserved": (ow, oh) == (sw, sh),
           "changed_frac": changed_frac(target, out),
           "opaque": fully_opaque(out)}
    try:
        c = idm.identity_cosine(target, out)
        rec["identity_cosine"] = round(c, 3) if c is not None else None
    except Exception:
        rec["identity_cosine"] = None
    if mask:
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            from tattoo_validation import seam_metric
            rec["seam"] = seam_metric(target, out, mask)
        except Exception as exc:
            rec["seam"] = f"err: {exc}"
    # verdict
    issues = []
    if not rec["dims_preserved"]:
        issues.append("DIMS CHANGED")
    if rec["changed_frac"] < 0.005:
        issues.append("NO-OP (image barely changed)")
    if not rec["opaque"]:
        issues.append("TRANSPARENT/halo output")
    if ident_keep and rec.get("identity_cosine") is not None and rec["identity_cosine"] < 0.80:
        issues.append(f"IDENTITY DRIFT {rec['identity_cosine']}")
    rec["issues"] = issues
    rec["ok"] = not issues
    return rec


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    specs = SCEN if which == "all" else [s for s in SCEN if s[0] == which]
    if not specs:
        print(f"unknown scenario {which!r}; options: {[s[0] for s in SCEN]}")
        return
    results = []
    for spec in specs:
        print(f"\n{'='*64}\n>>> {spec[0]} (role={spec[1]}, mode={spec[4]})\n{'='*64}", flush=True)
        rec = run_one(spec)
        results.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    # findings
    js = os.path.join(OUT, "role_results.json")
    prev = []
    if os.path.exists(js):
        try: prev = json.load(open(js, encoding="utf-8"))
        except Exception: prev = []
    by_id = {r["id"]: r for r in prev}
    for r in results:
        by_id[r["id"]] = r
    merged = list(by_id.values())
    json.dump(merged, open(js, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    md = ["# Per-role LIVE transfer validation\n",
          f"_Updated {time.strftime('%Y-%m-%d %H:%M:%S')}_\n",
          "| id | role | mode | out_dims | dims_ok | changed | opaque | identity | issues |",
          "| :-- | :-- | :-- | :-- | :-- | --: | :-- | --: | :-- |"]
    for r in sorted(merged, key=lambda x: x["id"]):
        if r.get("error"):
            md.append(f"| {r['id']} | {r['role']} | {r.get('mode','')} | — | — | — | — | — | ERROR: {r['error']} |")
            continue
        md.append(f"| {r['id']} | {r['role']} | {r['mode']} | {r['out_dims']} | "
                  f"{'✓' if r['dims_preserved'] else '✗'} | {r['changed_frac']} | "
                  f"{'✓' if r['opaque'] else '✗'} | {r.get('identity_cosine')} | "
                  f"{', '.join(r['issues']) or 'clean'} |")
    open(os.path.join(OUT, "ROLE_FINDINGS.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    ok = sum(1 for r in results if r.get("ok"))
    print(f"\n{ok}/{len(results)} scenarios clean. Wrote {OUT}/ROLE_FINDINGS.md")


if __name__ == "__main__":
    main()
