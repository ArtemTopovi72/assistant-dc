"""Live: the 📷 Реализм preset on one picture, N seeds, outputs saved for a look. For the owner's
complaints of 2026-10-21 on an anime frame: the skin-tight suit came back as bare skin, the sleek bob
as wet clumps, the face smeared as if dirty.

    venv/Scripts/python bench/photo_restyle_live.py <picture> [--seeds 3] [--tag new]
"""
import argparse, os, sys, threading
from pathlib import Path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [ROOT] + [os.path.join(ROOT, d) for d in ("agent", "core", "bot", "imaging", "media", "knowledge", "voice")] \
    + [p for p in sys.path if os.path.basename(p) != "bench"]
sys.stdout.reconfigure(encoding="utf-8")
os.environ.pop("F5_TEST_RUN", None)
os.chdir(ROOT)
from PIL import Image
import config, image_style as S, style_presets
from models import Context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picture")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", default="new")
    a = ap.parse_args()
    ctx = Context(models=None, transcription_cache={}, cache_file=Path("tests/_dr_cache.json"),
                  asr_lock=threading.Lock(), tts_lock=threading.Lock(),
                  model_name=config.MODEL_NAME, no_think=True, api_min_interval=0.2)
    out_dir = Path("runtime/photo_restyle_live")
    out_dir.mkdir(parents=True, exist_ok=True)
    instr = style_presets.STYLE_PRESETS["photo"]["prompt"]
    who = S._who

    def logged(ctx, path):
        got = who(ctx, path)
        print("WHO", got[0], flush=True)
        return got
    S._who = logged
    for s in range(a.seeds):
        out = S.restyle(ctx, a.picture, instr, seed=1000 + s)
        if out:
            dst = out_dir / f"{a.tag}_{s}.png"
            Image.open(out).save(dst)
            print("OUT", dst, flush=True)


if __name__ == "__main__":
    main()
