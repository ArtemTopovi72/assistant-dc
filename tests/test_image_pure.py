"""Branch coverage for image.py's PURE-logic surface — the routing/classification/
parsing/guard functions that do NOT require live ComfyUI/GPU. The ~29 ComfyUI
render functions (generate/inpaint/redraw/upscale/restore/transfer/fix_hands/
face-detailer) need a live image server and are out of scope here.

Run: venv/Scripts/python.exe tests/test_image_pure.py
"""
import os, sys, types, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sweep_stub  # noqa: F401  the model's narrow reads, stubbed
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path

import numpy as np
from PIL import Image
import config as C
import image as I

_TMP = Path(tempfile.mkdtemp(prefix="imgpure_"))

def _png(name="p.png", w=32, h=24, col=(80, 120, 160)):
    p = _TMP / name
    Image.new("RGB", (w, h), col).save(p)
    return str(p)

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    assert cond, name + ": " + detail


def test_snap_and_resolution():
    check("snap_min", I._snap_to_8(10) == 64)
    check("snap_round", I._snap_to_8(1000) == 1000 and I._snap_to_8(1001) == 1000)
    # normalize_resolution used to return the constants (544,960)/(960,544) for
    # EVERY input, which threw away any size the user picked. It now only puts a
    # size on the rails: orientation and aspect are preserved, the min/max side
    # are enforced, and the result is a multiple of 16.
    pw, ph = I.normalize_resolution(100, 200)
    check("normalize_portrait", ph > pw and abs(ph / pw - 2.0) < 0.05,
          f"{pw}x{ph}")
    check("normalize_portrait_floor", min(pw, ph) >= C.IMAGE_MIN_SIDE, f"{pw}x{ph}")
    lw, lh = I.normalize_resolution(200, 100)
    check("normalize_landscape", lw > lh and abs(lw / lh - 2.0) < 0.05, f"{lw}x{lh}")
    check("normalize_keeps_big", I.normalize_resolution(1920, 1080)[0] >= 1900,
          str(I.normalize_resolution(1920, 1080)))
    check("normalize_snap16", all(v % 16 == 0 for v in I.normalize_resolution(1234, 999)),
          str(I.normalize_resolution(1234, 999)))


def test_detect_orientation():
    check("orient_explicit_land", I.detect_orientation_from_text("a wide landscape banner") == "landscape")
    check("orient_explicit_port", I.detect_orientation_from_text("vertical portrait phone wallpaper") == "portrait")
    check("orient_content_port", I.detect_orientation_from_text("a single person standing, headshot") == "portrait")
    check("orient_content_land", I.detect_orientation_from_text("a city street crowd panorama") == "landscape")
    check("orient_unknown", I.detect_orientation_from_text("abstract") == "unknown")
    check("orient_empty", I.detect_orientation_from_text("") == "unknown")


def test_fix_image_params():
    # portrait intent with landscape dims -> swapped then normalized
    check("fix_portrait", I.fix_image_params("", "single person headshot", 960, 544) == (544, 960))
    check("fix_landscape", I.fix_image_params("", "city panorama", 544, 960) == (960, 544))


def test_parse_generation_params():
    r = I.parse_generation_params("a red cat | steps=10 | cfg=2.5 | width=768 | height=768 | seed=42")
    check("parse_tuple_len", len(r) == 6)
    check("parse_prompt", r[0] == "a red cat")
    # resolution keyword overrides
    r2 = I.parse_generation_params("a 4k landscape")
    check("parse_res_keyword", (3840, 2160) in [(r2[3], r2[4]), (r2[4], r2[3])] or 3840 in r2)
    # explicit WxH in prompt
    r3 = I.parse_generation_params("scene 800x600")
    check("parse_wxh", 800 in r3 and 600 in r3)
    # malformed key parts ignored
    r4 = I.parse_generation_params("cat | nonsense | steps=bad")
    check("parse_malformed_ok", r4[0] == "cat")


def test_format_comfy_error():
    class Resp:
        def __init__(self, data=None, text="", raise_json=False):
            self._data = data; self.text = text; self._raise = raise_json
        def json(self):
            if self._raise: raise ValueError("no json")
            return self._data
    # node_errors path
    r = I._format_comfy_error(Resp({"node_errors": {"3": {"class_type": "KSampler",
        "errors": [{"details": "model missing"}]}}}))
    check("comfy_node_error", "KSampler" in r and "model missing" in r)
    # error dict with message
    check("comfy_err_msg", I._format_comfy_error(Resp({"error": {"message": "bad"}})) == "bad")
    # error bare string
    check("comfy_err_str", "oops" in I._format_comfy_error(Resp({"error": "oops"})))
    # json raises -> text fallback
    check("comfy_text_fallback", I._format_comfy_error(Resp(raise_json=True, text="raw error")) == "raw error")


def test_intermediate_and_deliverable():
    check("inter_true", I.is_intermediate_artifact("out/_INTERMEDIATE_tile.png"))
    check("inter_firered", I.is_intermediate_artifact("x/_firered_tile_1.png"))
    check("inter_false", not I.is_intermediate_artifact("out/final.png"))
    check("inter_none", not I.is_intermediate_artifact(None))
    # Found by the live journey run: the harness used the app's own predicate,
    # and it picked a hand-drawn source-region crop as "the picture that was
    # drawn". Both of these are working files written into OUTPUT_DIR beside
    # the real renders.
    check("inter_srcregion", I.is_intermediate_artifact("out/_srcregion_123.png"))
    check("inter_florence", I.is_intermediate_artifact("out/_florence_proxy_1.png"))
    # …and the other side of it: these ARE the finished picture at their write
    # sites (`final = ...`). Listing one as scratch makes a good render
    # undeliverable, which is a worse bug than the one above.
    for _deliverable in ("handfix_1.png", "upscaled_faceguard_1.png",
                         "contained-firered_1.png", "contained-cropped_1.png"):
        check("deliverable_" + _deliverable.split("_")[0].split("-")[0],
              not I.is_intermediate_artifact("out/" + _deliverable), _deliverable)
    # assert_deliverable: reject intermediate
    check("deliver_reject_inter", I.assert_deliverable(_png("_INTERMEDIATE_x.png"), where="t") is None)
    # reject missing
    check("deliver_reject_missing", I.assert_deliverable("C:/no/such.png", where="t") is None)
    # reject None
    check("deliver_none", I.assert_deliverable(None, where="t") is None)
    # accept a valid same-size final
    src = _png("src.png", 64, 48); out = _png("out.png", 64, 48)
    check("deliver_ok", I.assert_deliverable(out, where="t", source_path=src) == out)
    # shrink guard: output smaller than source -> rejected
    small = _png("small.png", 32, 24)
    check("deliver_shrink", I.assert_deliverable(small, where="t", source_path=src) is None)
    # non-shrink size change -> warning but accepted
    big = _png("big.png", 128, 96)
    check("deliver_grow", I.assert_deliverable(big, where="t", source_path=src) == big)


def test_log_edit_decision():
    src = _png("les.png", 40, 30); ret = _png("ler.png", 40, 30)
    # normal (persists a jsonl line; never raises)
    I.log_edit_decision(request="r", classifier="c", tool="t", workflow="w",
                        returned_file=ret, source=src, extra={"k": "v"})
    check("log_edit_ok", (I.OUTPUT_DIR / "edit_routing.jsonl").exists())
    # returned_file None, source empty -> still fine
    I.log_edit_decision(request="r2", classifier="c", tool="t", workflow="w", returned_file=None)
    check("log_edit_none_file", True)


def test_classify_edit_intent():
    # Which kind of edit a phrase is: the model's read, bench/edit_intent_live.py.
    import image_router as R
    check("classify_empty", I.classify_edit_intent("") == "subject_edit")
    R.EDIT_STUB = {"remove the car and change the sky to sunset":
                   {"kind": "multi_op", "steps": ["remove the car", "change the sky to sunset"]}}.get
    try:
        check("classify_passes_the_read", I.classify_edit_intent("remove the car and change the sky to sunset") == "multi_op")
        check("plan_carries_steps", I.edit_plan("remove the car and change the sky to sunset")["steps"]
              == ["remove the car", "change the sky to sunset"])
        check("no_read_is_the_generic_edit", I.classify_edit_intent("make her look happier") == "subject_edit")
    finally:
        R.EDIT_STUB = None
    # _strip_lead removes a leading command
    check("strip_lead", I._strip_lead("please remove the hat", r"please remove") == "the hat")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print("  ERROR in " + fn.__name__ + ": " + type(e).__name__ + ": " + str(e))
    print("\n" + str(len(fns) - failed) + "/" + str(len(fns)) + " image-pure test functions passed (" +
          str(sum(1 for _, c in RESULTS if c)) + "/" + str(len(RESULTS)) + " checks)")
    # A failed check() must reach the exit code. This used to be `1 if failed else 0`,
    # where `failed` counts EXCEPTIONS only -- so every assertion in the file could
    # print FAIL while the suite still exited 0, invisible to anything judging by
    # exit code (which is how these suites are judged).
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)
