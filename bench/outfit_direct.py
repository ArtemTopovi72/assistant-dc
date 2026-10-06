"""Direct outfit transfer (no agent loop): target person + garment photo -> result path."""
import os, sys, threading, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "agent", "imaging", "media", "bot"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.chdir(ROOT)
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
import models, image as image_mod

target, garment = sys.argv[1], sys.argv[2]
ctx = models.Context(models=None, transcription_cache={}, cache_file=None,
                     asr_lock=threading.Lock(), tts_lock=threading.Lock())
t0 = time.time()
out = image_mod.plan_and_execute_transfer(
    ctx, target, [image_mod.ReferenceImage(garment, None)],
    "Dress the person in the target image in the clothing shown in the reference image "
    "(the garment itself: cut, colour, fabric, print). Remove their current outfit. Keep the "
    "person's face, identity, body, pose, background and framing exactly.")
print("RESULT", out, f"{time.time()-t0:.0f}s", flush=True)
