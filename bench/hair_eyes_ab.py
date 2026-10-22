"""Does a hair edit keep the eyes? Klein's whole-frame hair recolour turned green eyes brown
(2026-10-22). Runs the real edit path (annotator + Klein + box lock) per seed and prints the
mean colour change (ΔE, Lab) inside each eye found by YuNet, and outside the face for scale.

    venv/Scripts/python.exe bench/hair_eyes_ab.py [image ...]
"""
import os, shutil, sys, threading
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.join(ROOT, "bench")]
os.chdir(ROOT); sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import cv2
import numpy as np
from models import Context
import image as IMG
import identity_metrics as IDM

OUT = os.path.join("tmp", "hair_eyes_ab")
SEEDS = (1234567, 42, 777)
DEFAULT = ["tg_images/tg_1792661776087_85d1a5.jpg"]


def eyes(path):
    img = cv2.imread(path)
    k = min(1.0, 1200.0 / max(img.shape[:2]))      # YuNet finds no face in a 5535 px portrait
    small = cv2.resize(img, (int(img.shape[1] * k), int(img.shape[0] * k)))
    det = cv2.FaceDetectorYN.create(IDM._YUNET, "", (small.shape[1], small.shape[0]), score_threshold=0.6)
    _, faces = det.detect(small)
    if faces is None:
        return []
    f = max(faces, key=lambda f: f[2] * f[3]) / k
    r = max(4, int(f[2] * 0.03))                    # the iris, not the lids around it
    return [(int(f[4]), int(f[5]), r), (int(f[6]), int(f[7]), r)]


def delta_e(a, b, x, y, r):
    la = cv2.cvtColor(a[y - r:y + r, x - r:x + r], cv2.COLOR_BGR2LAB).astype(float)
    lb = cv2.cvtColor(b[y - r:y + r, x - r:x + r], cv2.COLOR_BGR2LAB).astype(float)
    return float(np.sqrt(((la.mean((0, 1)) - lb.mean((0, 1))) ** 2).sum()))


def score(src, dst):
    a = cv2.imread(src)
    b = cv2.resize(cv2.imread(dst), (a.shape[1], a.shape[0]))
    return [round(delta_e(a, b, *p), 1) for p in eyes(src)]


def run(src):
    ctx = Context(models=None, transcription_cache={}, cache_file=None, asr_lock=threading.Lock(),
                  tts_lock=threading.Lock())
    ctx.set_stage = lambda *a, **k: None
    a = cv2.imread(src)
    pts = eyes(src)
    for seed in SEEDS:
        out = IMG._edit_image_with_firered_once(ctx, src, "change the hair to ginger hair", seed=seed)
        if not out:
            print(os.path.basename(src), seed, "no result"); continue
        dst = os.path.join(OUT, f"{os.path.splitext(os.path.basename(src))[0]}_{seed}.png")
        shutil.copy(out, dst)
        b = cv2.resize(cv2.imread(dst), (a.shape[1], a.shape[0]))
        print(os.path.basename(src), seed, "eye ΔE", [round(delta_e(a, b, *p), 1) for p in pts], flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for s in sys.argv[1:] or DEFAULT:
        run(s)
