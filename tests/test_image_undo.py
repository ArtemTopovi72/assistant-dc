"""«нет, верни как было» hands back the previous FILE; it used to redraw a new cat."""
import os, sys, tempfile, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
from models import Context
import tool_image_handlers as h
import intent
intent.STUB = lambda t: {"undo": "верни как было" in t}   # the phrases: bench/removal_intent_live.py

d = tempfile.mkdtemp()
a, b = os.path.join(d, "a.png"), os.path.join(d, "b.png")
Image.new("RGB", (8, 8), "orange").save(a); Image.new("RGB", (8, 8), "black").save(b)
ctx = Context(models=SimpleNamespace(), transcription_cache={}, cache_file=Path(d) / "c.json",
              asr_lock=threading.Lock(), tts_lock=threading.Lock(), model_name="m")
ctx.set_stage = lambda *a, **k: None
ctx.last_image_path = a
ctx.last_image_path = b
state = {"image_path": b, "user_input": "no, put it back as it was", "user_input_original": "нет, верни как было"}
r = h._handle_inpaint_image(ctx, state, {"instructions": "a ginger cat", "region": "the black cat"})
assert r.startswith("[done]") and ctx.last_image_path == a and state["image_path"] == a, (r, ctx.last_image_path)
state2 = {"image_path": a, "user_input": "make the cat black"}
assert h._undo(ctx, state2) == "", "an ordinary edit is not an undo"
print("ok")
import copy
base = Context(models=SimpleNamespace(), transcription_cache={}, cache_file=Path(d) / "c2.json",
               asr_lock=threading.Lock(), tts_lock=threading.Lock(), model_name="m")
base.last_image_path = a; base.last_image_path = b
chat = copy.copy(base)
chat.last_image_path = a
assert base.image_undo == [a], base.image_undo
import inspect, tg_tasks
assert 'ctx.image_undo = [e["path"] for e in (getattr(sess, "image_log"' in inspect.getsource(tg_tasks)
print("ok a chat's copy does not write into the shared history; TG seeds it from image_log")
import tool_image_handlers as _H
assert '["undo"]' in inspect.getsource(_H._undo)   # the phrases: bench/removal_intent_live.py
print("ok an undo is the model's read")
