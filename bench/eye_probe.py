"""One talking-head part of the Overseer, rendered with a named variant, then the eyes measured.
usage: eye_probe.py <variant>   (config knobs come in as env: VIDEO_SHORT_EDGE, VIDEO_REFINE_STEPS,
VIDEO_UPSCALE_FACTOR; prompt variants are named here). Same still, voice, seed and seconds for all,
so the only difference between two runs is the variant. Result: tmp/night/<variant>.mp4 + _sheet.jpg + .csv.
"""
import os
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "agent", "imaging", "media", "bot", "core"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.chdir(ROOT)
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
import models, video

STILL = "tmp/overseer.jpg"
VOICE = os.environ.get("PROBE_VOICE", "tmp/ov/p1.wav")
SECONDS = float(os.environ.get("PROBE_SECONDS", "6.0"))
SEED = int(os.environ.get("PROBE_SEED", "20261019"))

EYES = ("His deep-set eyes stay exactly as in <Picture 1>: dark, shadowed sockets with a faint glint, "
        "drawn in the same low-poly CGI game-render style, never photographic eyes with white sclera. ")
SQUINT = ("His heavy brows keep his eyes in deep shadow and half-lidded in a permanent squint, exactly "
          "like <Picture 1>; the eyelids never open wide and the eyes stay dark and small. ")


def prompt(name: str) -> str:
    p = open("tmp/night/prompt_base.txt", encoding="utf8").read()
    if "cgi" in name:
        p = re.sub(r"\b(a\s+)?realistic\b", "stylized 3D CGI game-render", p, flags=re.I)
    if "eyes" in name:
        p = p.replace("[Shot 1] ", "[Shot 1] " + EYES, 1)
    if "squint" in name:
        p = p.replace("[Shot 1] ", "[Shot 1] " + SQUINT, 1)
    return p


name = sys.argv[1]
out = f"tmp/night/{name}.mp4"
ctx = models.Context(models=None, transcription_cache={}, cache_file=None,
                     asr_lock=threading.Lock(), tts_lock=threading.Lock())
t0 = time.time()
r = video.generate_video(ctx, prompt(name), images=[STILL], audios=[VOICE], seconds=SECONDS,
                         seed=SEED, prepared=True)
print("RESULT", name, r.get("status"), r.get("path"), f"{time.time() - t0:.0f}s",
      r.get("width"), r.get("height"), flush=True)
if r.get("status") == "success":
    shutil.copy(r["path"], out)
    print(subprocess.run([sys.executable, "tmp/eye_track.py", out, f"tmp/night/{name}", "6"],
                         capture_output=True, text=True).stdout, flush=True)
