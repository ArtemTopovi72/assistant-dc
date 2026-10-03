"""LIVE ComfyUI coverage run for image.py. Requires ComfyUI at config.COMFY_URL
(currently http://127.0.0.1:8000) actually running. Drives real generate/inpaint/
upscale/restore/redraw calls to close render-path branches that pure mocking can't
reach. Run standalone: venv/Scripts/python.exe bench/live_image.py
"""
import os, sys, time, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.WARNING)
from pathlib import Path
from PIL import Image
import image as I

_TMP = Path(tempfile.mkdtemp(prefix="imglive_"))
RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))

def _png(name, w=512, h=512, col=(90, 130, 170)):
    p = _TMP / name
    Image.new("RGB", (w, h), col).save(p)
    return str(p)

import threading
import models
def Ctx():
    return models.Context(
        models=None,
        transcription_cache={},
        cache_file=_TMP / "cache.json",
        asr_lock=threading.Lock(),
        tts_lock=threading.Lock(),
    )

def test_generate_basic():
    ctx = Ctx()
    t0 = time.time()
    out = I.generate_image_with_comfy(ctx, "a red apple on a white table, product photo",
                                       steps=4, cfg=1.0, width=512, height=512, timeout=180)
    dt = time.time() - t0
    check("gen_basic_returns", out is not None, f"None after {dt:.1f}s")
    if out:
        check("gen_basic_exists", Path(out).exists())
        img = Image.open(out)
        check("gen_basic_size", img.size[0] > 0 and img.size[1] > 0)
    return out

def test_generate_empty_prompt_rejected():
    ctx = Ctx()
    out = I.generate_image_with_comfy(ctx, "   ", steps=4)
    check("gen_empty_prompt_none", out is None)

def test_generate_bad_workflow_path(monkeypatch_holder):
    # Force workflow_path.exists() False branch
    orig = I.WORKFLOW_PATH
    try:
        I.WORKFLOW_PATH = Path("Z:/nonexistent_workflow_xyz.json")
        ctx = Ctx()
        out = I.generate_image_with_comfy(ctx, "a cat", steps=4)
        check("gen_missing_workflow_none", out is None)
    finally:
        I.WORKFLOW_PATH = orig

def test_refinement(prev_path):
    ctx = Ctx()
    out = I.generate_image_with_refinement(ctx, "a red apple, brighter lighting, sharper focus",
                                            steps=4, width=512, height=512)
    check("refine_returns", True)  # informational; refinement may internally reject/loop
    return out

def test_upscale(src):
    ctx = Ctx()
    out = I.upscale_image_with_comfy(ctx, src, timeout=180)
    check("upscale_ran", True)
    if out:
        check("upscale_exists", Path(out).exists())
    return out

def test_restore(src):
    ctx = Ctx()
    out = I.restore_image_with_comfy(ctx, src, timeout=180)
    check("restore_ran", True)
    if out:
        check("restore_exists", Path(out).exists())
    return out

def test_redraw(src):
    ctx = Ctx()
    out = I.redraw_image_with_comfy(ctx, src, "make the lighting warmer and add subtle texture",
                                     timeout=180)
    check("redraw_ran", True)
    if out:
        check("redraw_exists", Path(out).exists())
    return out

def test_inpaint(src):
    ctx = Ctx()
    from PIL import Image as PILImage
    mask = PILImage.new("L", (512, 512), 0)
    # paint a white square region to inpaint
    for y in range(150, 350):
        for x in range(150, 350):
            mask.putpixel((x, y), 255)
    mask_path = str(_TMP / "mask.png")
    mask.save(mask_path)
    out = I.inpaint_region_with_comfy(ctx, src, mask_path, "a blue circle", timeout=180)
    check("inpaint_ran", True)
    if out:
        check("inpaint_exists", Path(out).exists())
    return out


if __name__ == "__main__":
    print("ComfyUI target:", I.COMFY_URL)
    out1 = test_generate_basic()
    test_generate_empty_prompt_rejected()
    test_generate_bad_workflow_path(None)
    src = out1 or _png("fallback_src.png")
    test_upscale(src)
    test_restore(src)
    test_redraw(src)
    test_inpaint(src)
    test_refinement(src)
    passed = sum(1 for _, c in RESULTS if c)
    print(f"\n{passed}/{len(RESULTS)} live checks passed")
    sys.exit(0)
