"""Live app-path check: generate_video (int8 + turbo + Context-IR + two-stage) and text_add."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import video as V, image_router as R
t = time.time()
r = V.generate_video(None, "Старый рыжий кот сидит на подоконнике, смотрит на дождь за окном и зевает.", seconds=5)
print("video", round(time.time() - t), json.dumps(r, ensure_ascii=False, default=str)[:600], flush=True)
t = time.time()
out = R.add_text_verified(None, "runtime/_working_input_1785745490393.png", "добавь надпись «УРОЖАЙ 2026» в небо")
print("text_add", round(time.time() - t), out, flush=True)
