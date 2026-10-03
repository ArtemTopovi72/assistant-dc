"""Regression: a context-length rejection must be reported, not swallowed.

BUG (the root cause of "it says it cannot draw"): the model was loaded in LM
Studio with an 8192-token context, but the system prompt plus the tool schemas
needs ~9900. LM Studio rejected EVERY tool-bearing call with

    {"error": "... (n_keep: 9905 >= n_ctx: 8192) ..."}

and the code threw the body away — the non-200 branch logged only "HTTP 400", and
an in-stream error chunk has no "choices" key so it fell into a bare `continue`.
The call returned an empty message, the agent saw silent rounds, and told the user
"I cannot draw, the drawing tools are unavailable". The number needed to fix it
was in the response the whole time.

Run: venv/Scripts/python.exe tests/test_llm_context_error.py
"""
import io, json, logging, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import offline_guard
# This suite drives the context-overflow path, and the code under test answers
# that error by RELOADING the model: llm._try_heal_context -> lmstudio.heal_context
# -> `lms unload --all`. requests.post is patched below, so no HTTP left the
# process -- but subprocess was not, and every full test run silently unloaded
# the operator's real model. The next live deep-research run then got empty
# answers from a server with nothing loaded and told the reader that the pages
# it had fetched "contained no facts relevant to the topic".
offline_guard.no_model_management()

import llm

ok = bad = 0


def check(label, cond, detail=""):
    global ok, bad
    if cond:
        ok += 1
        print(f"PASS {label}")
    else:
        bad += 1
        print(f"FAIL {label}  {detail}")


ERR = ("The number of tokens to keep from the initial prompt is greater than the "
       "context length (n_keep: 9905>= n_ctx: 8192). Try to load the model with a "
       "larger context length, or provide a shorter input.")

PAYLOAD = {"model": "google/gemma-4-26b-a4b-qat",
           "tools": [{"function": {"name": f"t{i}"}} for i in range(11)]}


class _Capture:
    """Collect what assistant.llm logs."""
    def __enter__(self):
        self.records = []
        self.h = logging.Handler()
        self.h.emit = lambda r: self.records.append(r.getMessage())
        self.logger = logging.getLogger("assistant.llm")
        self.logger.addHandler(self.h)
        self._lvl = self.logger.level
        self.logger.setLevel(logging.DEBUG)
        self._disabled = logging.root.manager.disable
        logging.disable(logging.NOTSET)
        return self
    def __exit__(self, *a):
        self.logger.removeHandler(self.h)
        self.logger.setLevel(self._lvl)
        logging.disable(self._disabled)
    @property
    def text(self):
        return "\n".join(self.records)


# ── 1. the hint helper turns the error into an instruction ───────────────────
with _Capture() as cap:
    llm._log_context_hint(ERR, PAYLOAD)
check("the needed and available token counts are both reported",
      "9905" in cap.text and "8192" in cap.text, cap.text)
check("it says the model must be RELOADED with a bigger context",
      "context length of at least" in cap.text, cap.text)
check("the recommended size exceeds what the prompt needs",
      "16384" in cap.text or "11953" in cap.text, cap.text)
check("it names the model to reload",
      "google/gemma-4-26b-a4b-qat" in cap.text, cap.text)
check("it explains that NO tool can work in this state",
      "cannot use ANY tool" in cap.text, cap.text)
check("it counts the tool schemas that made the prompt too big",
      "11 tool schema" in cap.text, cap.text)

# ── 2. an unrelated error must not produce a bogus context hint ───────────────
with _Capture() as cap2:
    llm._log_context_hint("some other failure entirely", PAYLOAD)
check("an unrelated error produces no context advice", cap2.text.strip() == "",
      cap2.text)

# ── 3. a non-200 logs the BODY, not just the status code ─────────────────────
class _Resp:
    status_code = 400
    text = json.dumps({"error": ERR})
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def iter_lines(self, **kw): return iter(())
    def close(self): pass


_orig_post = llm.requests.post
llm.requests.post = lambda *a, **k: _Resp()
try:
    with _Capture() as cap3:
        out = llm._stream_chat(None, dict(PAYLOAD))
finally:
    llm.requests.post = _orig_post

# NOT None: ERR above is a context-overflow message, and _stream_chat answers
# those with the _CTX_OVERFLOW sentinel on purpose, so the caller can shrink the
# payload instead of retrying something guaranteed to be rejected again. This
# check predates that and asserted the old contract.
check("an overflow non-200 returns the shrink-me sentinel, not a bare failure",
      out is llm._CTX_OVERFLOW, repr(out))
check("the response body is logged", "n_keep" in cap3.text, cap3.text[:300])
check("and the actionable hint fires from the body too",
      "context length of at least" in cap3.text, cap3.text[:300])

# ── 4. an error INSIDE the stream is caught (no 'choices' key) ────────────────
class _StreamResp:
    status_code = 200
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def iter_lines(self, **kw):
        yield b'data: {"error": {"message": ' + json.dumps(ERR).encode() + b'}}'
        yield b"data: [DONE]"
    def close(self): pass


llm.requests.post = lambda *a, **k: _StreamResp()
try:
    with _Capture() as cap4:
        out4 = llm._stream_chat(None, dict(PAYLOAD))
finally:
    llm.requests.post = _orig_post

check("a streamed overflow error returns the sentinel, not an empty message",
      out4 is llm._CTX_OVERFLOW, repr(out4))
check("the streamed error is logged", "streamed an error" in cap4.text, cap4.text[:300])
check("the streamed error also yields the hint",
      "context length of at least" in cap4.text, cap4.text[:300])

# ── 5. a normal stream is unaffected ─────────────────────────────────────────
class _OkResp:
    status_code = 200
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def iter_lines(self, **kw):
        yield b'data: {"choices":[{"delta":{"content":"hello"}}]}'
        yield b'data: {"choices":[{"delta":{"content":" world"}}]}'
        yield b"data: [DONE]"
    def close(self): pass


llm.requests.post = lambda *a, **k: _OkResp()
try:
    out5 = llm._stream_chat(None, dict(PAYLOAD))
finally:
    llm.requests.post = _orig_post

check("a normal stream still returns its content",
      isinstance(out5, dict) and out5.get("content") == "hello world", repr(out5))


# ── 5. and an error that is NOT an overflow still fails plainly ──────────────
# The sentinel means "this can be made to fit". Anything else must stay None,
# or a caller would shrink the payload and retry a call that was refused for a
# different reason entirely.
class _OtherResp:
    status_code = 500
    text = json.dumps({"error": {"message": "the model exploded"}})
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def iter_lines(self, **kw): return iter(())
    def close(self): pass


llm.requests.post = lambda *a, **k: _OtherResp()
try:
    with _Capture():
        out5 = llm._stream_chat(None, dict(PAYLOAD))
finally:
    llm.requests.post = _orig_post
# Since 2026-09-18 a 5xx is the _SERVER_ERROR sentinel: the retry loop mends
# the payload (cut tool calls, then size) and retries; it is never the overflow
# sentinel, so nothing shrinks a call refused for another reason.
check("an unrelated 500 is the server-error sentinel, not the overflow one",
      out5 is llm._SERVER_ERROR and out5 is not llm._CTX_OVERFLOW, repr(out5))

print(f"\n{ok}/{ok + bad} checks passed")
sys.exit(1 if bad else 0)
