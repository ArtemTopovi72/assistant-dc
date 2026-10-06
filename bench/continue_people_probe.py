"""▶️ Continue a clip with a NEW person from a photo, end to end on the GPU:
continue_people_probe.py clip.mp4 person.jpg "what happens" """
import os, sys, threading, time, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
for d in ("", "agent", "imaging", "media", "bot", "core"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.chdir(ROOT)
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
import models, video, tg_continue
clip, person, what = sys.argv[1], sys.argv[2], sys.argv[3]
w = tempfile.mkdtemp(prefix="contppl_")
seed, tail = os.path.join(w, "seed.jpg"), os.path.join(w, "tail.mp4")
assert tg_continue.seed_frame(clip, seed) and tg_continue.cut_tail(clip, tail)
ctx = models.Context(models=None, transcription_cache={}, cache_file=None,
                     asr_lock=threading.Lock(), tts_lock=threading.Lock())
t0 = time.time()
r = video.generate_video(ctx, video.CONTINUE_PREFIX + what + video.new_people_clause(2, 1),
                         images=[seed, person], videos=[tail])
joined = video.join_continuation(clip, r["path"]) if r.get("path") else ""
print("RESULT", r.get("status"), r.get("path"), joined, f"{time.time()-t0:.0f}s", flush=True)
