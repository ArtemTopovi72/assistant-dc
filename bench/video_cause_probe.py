"""The real generate_video handler end to end on the GPU: continue a clip with a script
that needs a visible cause (a throw), split into ~6 s parts.
    video_cause_probe.py clip.mp4 "сценарий по-русски" "English script" """
import os, sys, threading, time, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
for d in ("", "agent", "imaging", "media", "bot", "core"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.chdir(ROOT)
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
import models, tg_continue, tool_image_handlers as H
clip, ru, en = sys.argv[1], sys.argv[2], sys.argv[3]
w = tempfile.mkdtemp(prefix="cause_")
seed, tail = os.path.join(w, "seed.jpg"), os.path.join(w, "tail.mp4")
assert tg_continue.seed_frame(clip, seed) and tg_continue.cut_tail(clip, tail)
ctx = models.Context(models=None, transcription_cache={}, cache_file=None,
                     asr_lock=threading.Lock(), tts_lock=threading.Lock())
ctx.continue_tail, ctx.continue_src, ctx.continue_people = tail, clip, []
ctx.video_plan_ask = False
H._current_image_paths = lambda c, s: [seed]
state = {"user_input_original": ru, "user_input": en}
t0 = time.time()
out = H._handle_generate_video(ctx, state, {"description": en})
print("RESULT", str(out)[:300], state.get("video_path"), f"{time.time()-t0:.0f}s", flush=True)
