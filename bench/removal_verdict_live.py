"""Live: does the removal verdict match the picture? Real inpaint_image handler, real FireRed
removal, real vision checks, on neutral pictures. Per case: the share of pixels the edit
changed, what the handler told the agent (removed / STILL visible), what inspect_image said
when asked «is the X still there?», and an independent presence check of the THING on the
source and on the result. --old puts back the target read from before the place fix.

    venv/Scripts/python bench/removal_verdict_live.py [--old] [--only 1,3]
"""
import argparse, json, os, sys, threading, time
from pathlib import Path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [ROOT] + [os.path.join(ROOT, d) for d in ("agent", "core", "bot", "imaging", "media", "knowledge")] \
    + [p for p in sys.path if os.path.basename(p) != "bench"]
sys.stdout.reconfigure(encoding="utf-8")
os.environ.pop("F5_TEST_RUN", None)
os.chdir(ROOT)
import numpy as np
from PIL import Image
import config, image as im, image_router as R
import tool_image_handlers as H
from models import Context

SHOP = "runtime/klein_hard/ig_full/s1_bookshop.png"
STATION = "runtime/klein_hard/ig_full/s2_station.png"
CASES = [  # (source, region as the agent writes it, the thing that must go)
    (SHOP, "the counter top where the grey cat sleeps", "grey cat"),
    (SHOP, "the man's face where the glasses are", "glasses"),
    (SHOP, "the wall clock", "wall clock"),
    (STATION, "the platform floor where the pigeon stands", "pigeon"),
    (STATION, "the departure board on the right", "departure board"),
    (STATION, "the area under the roof where the round clock hangs", "round clock"),
]
OUT = Path("runtime/removal_verdict_live")


def changed(a, b):
    x = np.asarray(Image.open(a).convert("RGB"), dtype=np.int16)
    y = np.asarray(Image.open(b).convert("RGB").resize(Image.open(a).size), dtype=np.int16)
    return float((np.abs(x - y).max(axis=2) > 30).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--inspect-only", action="store_true", help="re-ask inspect_image on new.json's results")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--no-evidence", action="store_true", help="inspect without the measured change")
    a = ap.parse_args()
    if a.no_evidence:
        H._change_evidence = lambda before, after: ""
    if a.inspect_only:
        ctx = Context(models=None, transcription_cache={}, cache_file=Path("tests/_dr_cache.json"),
                      asr_lock=threading.Lock(), tts_lock=threading.Lock(),
                      model_name=config.MODEL_NAME, no_think=True, api_min_interval=0.2)
        rows = json.loads((OUT / "new.json").read_text(encoding="utf-8"))
        Image.open(SHOP).convert("RGB").save(OUT / "same_shop.jpg", quality=90)
        by = {r["case"]: r for r in rows}
        # controls, the thing still there: no change at all; another thing removed instead
        rows += [{"case": 1, "thing": "grey cat", "saved": str(OUT / "same_shop.jpg"), "ctl": "unchanged"},
                 {"case": 1, "thing": "grey cat", "saved": by[3]["saved"], "ctl": "clock removed, cat kept"},
                 {"case": 4, "thing": "pigeon", "saved": by[5]["saved"], "ctl": "board removed, pigeon kept"},
                 {"case": 2, "thing": "glasses", "saved": by[1]["saved"], "ctl": "cat removed, glasses kept"}]
        for r in rows:
            if not r.get("saved"):
                continue
            src = CASES[r["case"] - 1][0]
            if r.get("ctl"):
                r = {**r, "thing": f"{r['thing']} [CONTROL {r['ctl']}: must be STILL]"}
            for k in range(a.repeat):
                import intent
                intent._YES_CACHE.clear()
                state = {"image_path": r["saved"], "edit_before": src}
                ins = H._handle_inspect_image(ctx, state, {"check": f"Is the {r['thing'].split(' [')[0]} still there?"})
                body = ins.split("\n", 1)[-1].split("Next step:")[0]
                tail = body.upper()[-160:]
                verdict = ("STILL" if "PRESENT" in tail or "PARTIAL" in tail else
                           "GONE" if any(w in tail for w in ("MISSING", "ABSENT")) else "?")
                low = body.lower()
                same = any(w in low for w in ("images are identical", "pictures are identical",
                                              "no visible change", "identical pixel"))
                print(f"VERDICT {r['case']} {r['thing']} try{k + 1}: {verdict}{' SAID-IDENTICAL' if same else ''} | "
                      + body[:300].replace("\n", " | "), flush=True)
        return
    if a.old:
        cut = R._EDIT_SYSTEM.index("), and never where they are")
        R._EDIT_SYSTEM = R._EDIT_SYSTEM[:cut] + ")" + R._EDIT_SYSTEM[R._EDIT_SYSTEM.index("; a word that"):]
    ctx = Context(models=None, transcription_cache={}, cache_file=Path("tests/_dr_cache.json"),
                  asr_lock=threading.Lock(), tts_lock=threading.Lock(),
                  model_name=config.MODEL_NAME, no_think=True, api_min_interval=0.2)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (src, region, thing) in enumerate(CASES, 1):
        if a.only and str(i) not in a.only.split(","):
            continue
        t0 = time.time()
        state = {"image_path": src}
        ctx.last_image_path = src
        msg = H._handle_inpaint_image(ctx, state, {"region": region, "instructions": f"remove the {thing}",
                                                   "removal": True})
        out = state.get("image_path") if state.get("image_path") != src else None
        row = {"case": i, "region": region, "thing": thing, "secs": round(time.time() - t0),
               "handler": "STILL" if "STILL visible" in msg else "ERROR" if msg.startswith("[TOOL ERROR]") else "ok",
               "target": R.edit_plan(f"remove the {region}", ctx=ctx)["target"], "out": out,
               "msg": msg[:400]}
        if out:
            row["changed"] = round(changed(src, out), 4)
            row["thing_before"] = im._region_present(ctx, src, thing)
            row["thing_after"] = im._region_present(ctx, out, thing)
            ins = H._handle_inspect_image(ctx, state, {"check": f"Is the {thing} still there?"})
            row["inspect"] = ins.split("\n", 1)[-1][:300].replace("\n", " | ")
            dst = OUT / f"{'old' if a.old else 'new'}_{i}.png"
            Image.open(out).save(dst)
            row["saved"] = str(dst)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (OUT / f"{'old' if a.old else 'new'}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")


if __name__ == "__main__":
    main()
