"""Can the 3090 render a long H3 clip? Text-to-video at N seconds, prints time and result."""
import os, sys, threading, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "agent", "imaging", "media", "bot", "core"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.chdir(ROOT)
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
import models, video
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
images = sys.argv[2:]          # a photo -> ref2va (the animate / continue path)
ctx = models.Context(models=None, transcription_cache={}, cache_file=None,
                     asr_lock=threading.Lock(), tts_lock=threading.Lock())
t0 = time.time()
r = video.generate_video(ctx, "A man in a kitchen picks up a mug, drinks, puts it down, walks to the "
                         "window, opens it, looks outside, turns to the camera and says «Хорошо сегодня».",
                         seconds=secs, images=images)
print("RESULT", r.get("status"), r.get("path"), r.get("seconds"), f"{time.time()-t0:.0f}s", flush=True)
