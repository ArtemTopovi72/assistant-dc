"""DIAGNOSTIC: prove the streaming LLM call can no longer hang when the model gets stuck.

Spins up a fake LM Studio SSE endpoint with selectable pathological behaviours and
asserts send_to_lm_studio() returns within a BOUNDED time for each (instead of the old
~32-minute / infinite hang), while a normal stream still works.

  normal   : a few content chunks then [DONE]                -> returns the text
  stall    : 200 OK then silence forever (no tokens)          -> read-stall timeout, bounded
  trickle  : 2 tokens then silence forever                    -> read-stall timeout, bounded
  runaway  : content chunks forever, never sends [DONE]       -> char/wall-clock cap, bounded
  slowdone : streams slowly but finishes before the caps      -> returns the text

Run: venv/Scripts/python.exe bench/diag_llm_stuck.py     (exit 0 = all stuck-modes bounded)
"""
import os, sys, time, json, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import llm
import config
from models import Context

# --- tighten the watchdog so the test runs in seconds, not minutes -----------
llm.LLM_STREAM_STALL_TIMEOUT = 2.0     # no token for 2s => stalled
llm.LLM_STREAM_MAX_SECONDS = 3.0       # 3s wall-clock cap per attempt
llm.LLM_STREAM_MAX_CHARS = 4000        # runaway char cap
llm.LLM_CONNECT_TIMEOUT = 5.0
llm.LLM_MAX_RETRIES = 2                # keep the bounded-total small for the test
llm.LLM_RETRY_BASE_DELAY = 0.1

_MODE = {"v": "normal"}


def _sse(wfile, obj):
    wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n"); wfile.flush()


def _delta(text=None, tool=None):
    d = {}
    if text is not None: d["content"] = text
    return {"choices": [{"index": 0, "delta": d}]}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_POST(self):
        ln = int(self.headers.get("Content-Length", 0) or 0)
        if ln: self.rfile.read(ln)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        mode = _MODE["v"]
        try:
            if mode == "normal":
                for w in ("Hello", " world", "!"):
                    _sse(self.wfile, _delta(w)); time.sleep(0.05)
                self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
            elif mode == "stall":
                self.wfile.write(b": keep-alive\n\n"); self.wfile.flush()
                time.sleep(60)                       # silence forever (test caps it)
            elif mode == "trickle":
                # A real partial answer (well past urllib3's ~512B read buffer so it
                # flushes to the client) and THEN the model freezes — the partial-keep
                # path must return what was produced instead of discarding + retrying.
                for _ in range(120):
                    _sse(self.wfile, _delta("word ")); time.sleep(0.005)
                time.sleep(60)                       # then stall
            elif mode == "runaway":
                while True:                          # never sends [DONE]
                    _sse(self.wfile, _delta("loop "))
            elif mode == "slowdone":
                for w in ("slow", " but", " fine"):
                    _sse(self.wfile, _delta(w)); time.sleep(0.4)
                self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass                                     # client aborted (the watchdog fired)


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    llm.LM_STUDIO_URL = f"http://127.0.0.1:{port}/v1/chat/completions"

    ctx = Context(models=None, transcription_cache={}, cache_file=Path("tests/_dr_cache.json"),
                  asr_lock=threading.Lock(), tts_lock=threading.Lock(),
                  model_name="qwen-test", no_think=True, api_min_interval=0.0)
    msgs = [{"role": "user", "content": "hi"}]

    # Bounded ceilings: worst case = retries * (stall or wall-clock) + backoff, with slack.
    UPPER = llm.LLM_MAX_RETRIES * (llm.LLM_STREAM_MAX_SECONDS + 1.0) + 3.0
    ok = []
    def run(mode, want_text):
        _MODE["v"] = mode
        t = time.time()
        msg = llm.send_to_lm_studio(ctx, msgs, temperature=0.0, max_tokens=64)
        dt = time.time() - t
        bounded = dt < UPPER
        got_text = bool(msg and (msg.get("content") or "").strip())
        passed = bounded and (got_text == want_text)
        ok.append(passed)
        print(f"[{mode:8s}] {dt:5.2f}s (<{UPPER:.0f}s={bounded})  "
              f"text={got_text} want={want_text}  -> {'PASS' if passed else 'FAIL'}", flush=True)

    run("normal", True)
    run("slowdone", True)
    run("stall", False)        # nothing usable -> None after bounded retries
    run("trickle", True)       # partial tokens kept; bounded by read-stall
    run("runaway", True)       # truncated loop text; bounded by char/wall cap

    srv.shutdown()
    print(f"\n{sum(ok)}/{len(ok)} stuck-modes handled within bound", flush=True)
    return 0 if ok and all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
