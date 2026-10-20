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
    if "fl3" in name:
        # the words the soundtrack really says, so the mouth has something to follow
        p = re.sub(r"<d>.*?</d>", lambda m: "<d>[Russian] " + open("tmp/night/p1_65.txt", encoding="utf8").read() + "</d>", p, flags=re.S)
    if name.startswith(("fl2", "fl3", "fl4", "fl5")):
        # keyframe path: the still IS frame 0, no reference tags; the soundtrack is anchored by AddGuide
        p = re.sub(r"<Audio 1> is the voice-timbre.*\n?", "", p)
        p = (p.replace(", referencing the timbre of <Audio 1>", "").replace(" using the voice timbre from <Audio 1>", "")
             .replace("<Picture 1>", "the first frame").replace("<Subject 1>", "the man"))
    return p


def fl2va_audio():
    """FL2VA instead of Ref2VA: the still as frame 0 (HF MiniMax-H3 discussion 91: Ref2VA+audio recast the
    identity, FL2VA kept it) and the voice clip anchored as the soundtrack through MiniMaxH3AddGuide."""
    real = video.build_workflow

    def build(prompt, **kw):
        wf = real(prompt, **kw)
        up = video._eng(None)._upload(VOICE, "audio")
        wf["90"] = {"class_type": "LoadAudio", "inputs": {"audio": up}}
        wf["91"] = {"class_type": "MiniMaxH3AddGuide", "inputs": {
            "positive": ["6", 0], "latent": ["6", 1], "audio_vae": ["5", 0], "audio": ["90", 0], "frame_idx": 0}}
        wf["8"]["inputs"]["positive"] = ["91", 0]
        return wf
    video.build_workflow = build
    video.pick_mode = lambda *a, **k: "t2va"


name = sys.argv[1]
if name.startswith(("fl2", "fl3")):
    fl2va_audio()
elif name.startswith(("fl4", "fl5")):
    video.pick_mode = lambda *a, **k: "t2va"      # FL2VA, the model speaks the line itself (voice converted after)
# fl5: the still is the first AND the last frame -- the part has to come back to the original (KeyFace /
# Lookahead Anchoring: an identity keyframe as the segment's end condition holds the face over the length)
images = [STILL, STILL] if name.startswith("fl5") else [STILL]
out = f"tmp/night/{name}.mp4"
ctx = models.Context(models=None, transcription_cache={}, cache_file=None,
                     asr_lock=threading.Lock(), tts_lock=threading.Lock())
t0 = time.time()
r = video.generate_video(ctx, prompt(name), images=images, audios=[VOICE], seconds=SECONDS,
                         seed=SEED, prepared=True)
print("RESULT", name, r.get("status"), r.get("path"), f"{time.time() - t0:.0f}s",
      r.get("width"), r.get("height"), flush=True)
if r.get("status") == "success":
    shutil.copy(r["path"], out)
    print(subprocess.run([sys.executable, "tmp/eye_track.py", out, f"tmp/night/{name}", "6"],
                         capture_output=True, text=True).stdout, flush=True)
