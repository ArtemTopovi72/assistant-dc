"""The model is loaded with a micro-batch big enough for a full-size picture.

Gemma 4's image attention is non-causal: all of a picture's tokens must fit
ONE llama.cpp ubatch (GGML_ASSERT n_ubatch >= n_tokens). With the default 512
anything over ~1300 px killed the model, so pictures were cut to 1024 px and
a projector on wallpaper became «телевизор» (live 2026-10-01). `lms load`
cannot set the ubatch; the REST load does.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
import lmstudio as L
import llm

sent = []
class R:
    status_code = 200
    content = b"x"
    def json(self):
        return {"status": "loaded", "load_config": {"physical_batch_size": 2048}}
L.requests.post = lambda url, json=None, timeout=None: (sent.append((url, json)), R())[1]
L._lms_unload_all = lambda: (True, "")
L._wait_served = lambda *a, **k: True
L.time.sleep = lambda *a: None
L.subprocess.run = lambda *a, **k: (_ for _ in ()).throw(AssertionError("lms load used"))
ok, msg = L.load_model_exclusive("http://x:1", "gemma4-26b")
assert ok and sent and sent[0][0].endswith("/api/v1/models/load"), (ok, msg, sent)
p = sent[0][1]
assert p["physical_batch_size"] >= 1120, p     # Gemma 4's largest image
assert p["parallel"] == L.DEFAULT_PARALLEL, p
assert llm.VISION_MAX_SIDE >= 2048, llm.VISION_MAX_SIDE
print("ok")
