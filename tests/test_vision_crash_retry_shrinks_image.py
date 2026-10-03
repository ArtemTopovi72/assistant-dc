"""A picture that crashes the model is retried smaller, and the model is revived again.

Live 2026-09-13 (journey 3): a 683x1024 photo tripped llama.cpp's
`n_ubatch >= n_tokens` assert (non-causal attention over image tokens). The
revive brought the model back in 15 s, the retry re-sent the same picture and
killed it again with "Model reloaded." -- a message the revive did not
recognise -- and the 60 s cooldown then returned the cached "ok" while every
turn died on "No models loaded". Four steps of the journey were lost.
"""
import os, sys, io, base64, json
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

from PIL import Image
import llm, lmstudio as L, comfy_client

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

def data_url(w, h):
    buf = io.BytesIO(); Image.new("RGB", (w, h), "red").save(buf, "JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def size_of(url):
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size

# --- the crash message is recognised as "model gone"
check("'Model reloaded.' counts as the model being gone", llm._MODEL_GONE.search("Model reloaded."))
check("...and as a crash", llm._MODEL_CRASHED.search("The model has crashed without additional information"))
check("a plain context error is neither", not llm._MODEL_CRASHED.search("n_keep: 9905 >= n_ctx: 8192"))

# --- the image in the payload is halved
payload = {"model": "gemma", "messages": [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": [{"type": "text", "text": "what is written?"},
                                 {"type": "image_url", "image_url": {"url": data_url(683, 1024)}}]}]}
check("an image in the payload is halved", llm._halve_payload_images(payload) is True)
check("...to half its longest side", size_of(payload["messages"][1]["content"][1]["image_url"]["url"]) == (341, 512),
      size_of(payload["messages"][1]["content"][1]["image_url"]["url"]))
check("the text part is untouched", payload["messages"][1]["content"][0] == {"type": "text", "text": "what is written?"})
check("a text-only payload reports no change", llm._halve_payload_images({"messages": [{"role": "user", "content": "hi"}]}) is False)
tiny = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(40, 40)}}]}]}
check("a tiny image is left alone (no infinite shrink)", llm._halve_payload_images(tiny) is False)

# --- the revive cooldown yields when a successfully revived model died again
served = ["gemma"]
L.loaded_model_ids = lambda b: list(served)
calls = []
L.ensure_exclusive = lambda base, model: calls.append(model) or (True, "ok")
comfy_client.card_is_exclusive = lambda: ""
llm.LLM_REVIVE_MIN_INTERVAL_S = 60.0
llm._revive_last = 0.0; llm._revive_ok = False

check("first crash: revived", llm._try_revive_model("model has crashed", {"model": "gemma"}) is True and calls == ["gemma"])
check("inside the cooldown with the model served: cached ok, no reload",
      llm._try_revive_model("No models loaded", {"model": "gemma"}) is True and calls == ["gemma"], calls)
served.clear()
check("inside the cooldown but the model is GONE again: reload",
      llm._try_revive_model("Model reloaded.", {"model": "gemma"}) is True and calls == ["gemma", "gemma"], calls)
# A model that fails to load stays rate-limited (no reload storm).
L.ensure_exclusive = lambda base, model: calls.append(model) or (False, "not being served")
llm._revive_last = 0.0
check("a failed revive is reported", llm._try_revive_model("No models loaded", {"model": "gemma"}) is False)
n = len(calls)
check("...and the next call inside the cooldown does NOT reload again",
      llm._try_revive_model("No models loaded", {"model": "gemma"}) is False and len(calls) == n, len(calls) - n)

# --- the stream path wires it: a crash inside the stream halves the image
src = open(llm.__file__, encoding="utf-8").read()
check("the stream error path shrinks images on a crash",
      "if _MODEL_CRASHED.search(str(err)):" in src and "_halve_payload_images(payload)" in src)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
