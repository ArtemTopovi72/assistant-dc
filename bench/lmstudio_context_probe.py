"""What is the REAL per-request context limit of the loaded model?

config.DR_MODEL_CONTEXT defaults to 4096 and, by the project's own comment, is
"the single biggest limiter on report depth". Between 4096 and the model's
262144 there is one number worth having, and it is measurable rather than
guessable: send a prompt of N filler tokens, ask for one word back, and see
where the server stops answering.

MEASURED, and it corrected an assumption worth writing down. `lms ps` reports
PARALLEL 4, and the note in memory said the context is divided between the
slots -- so a request should have died around 10k. It did not: 40439 tokens of
prompt were answered on a 40596-token instance. On this build the context is
per-request, not per-slot. Take the measurement, not the folklore.

That also means an empty reply from a large prompt is NOT automatically a
context overflow, which matters when diagnosing silent briefing calls.

Nothing here is a test: it needs a live LM Studio and it costs GPU time.

Run: venv/Scripts/python.exe bench/lmstudio_context_probe.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402

import config  # noqa: E402

URL = config.LM_STUDIO_URL
MODELS = URL.replace("/chat/completions", "/models")
# A word that tokenises to about one token, repeated. Close enough: the point is
# where the server stops answering, not an exact token count.
FILLER = "data "


def served() -> list:
    return [m["id"] for m in requests.get(MODELS, timeout=10).json()["data"]]


def answers_at(model: str, approx_tokens: int, timeout: int = 180) -> bool:
    """True when the server still produces a message at this prompt size."""
    body = FILLER * approx_tokens
    try:
        r = requests.post(URL, timeout=timeout, json={
            "model": model,
            "messages": [{"role": "user",
                          "content": body + "\n\nReply with exactly one word: OK"}],
            "max_tokens": 8, "temperature": 0.0, "stream": False,
        })
    except Exception as exc:
        print("      request failed: %s: %s" % (type(exc).__name__, exc))
        return False
    if r.status_code != 200:
        print("      HTTP %d: %s" % (r.status_code, r.text[:160]))
        return False
    try:
        msg = r.json()["choices"][0]["message"]["content"]
    except Exception:
        return False
    return bool((msg or "").strip())


def main():
    try:
        ids = served()
    except Exception as exc:
        print("LM Studio is not answering: %s" % exc)
        return 1
    if not ids:
        print("LM Studio is up but SERVING NOTHING — load a model first.")
        return 1
    model = config.MODEL_NAME if config.MODEL_NAME in ids else ids[0]
    print("model under test: %s" % model)

    reported = None
    try:
        for m in requests.get(URL.replace("/v1/chat/completions", "/api/v0/models"),
                              timeout=10).json().get("data", []):
            if m.get("id") == model:
                reported = m.get("loaded_context_length")
    except Exception:
        pass
    print("LM Studio reports loaded_context_length = %s" % reported)

    lo, hi = 256, 200_000
    if reported:
        hi = int(reported) + 1
    # First establish that the small end works at all; if it does not, nothing
    # below can be concluded and saying so is the only honest output.
    print("\nprobing...")
    t0 = time.perf_counter()
    if not answers_at(model, lo):
        print("the model does not answer even at %d tokens — it is not healthy, "
              "and no limit can be measured" % lo)
        return 1
    print("  %6d ok" % lo)

    while lo + 256 < hi:
        mid = (lo + hi) // 2
        ok = answers_at(model, mid)
        print("  %6d %s" % (mid, "ok" if ok else "EMPTY"))
        if ok:
            lo = mid
        else:
            hi = mid

    print("\nlargest prompt that still answers: ~%d tokens  (%.0fs)"
          % (lo, time.perf_counter() - t0))
    if reported and lo < int(reported) * 0.75:
        print("This is well below the reported %s -- something is taking a share of"
              % reported)
        print("  the window. Check `lms ps` for PARALLEL and for a second instance.")
    elif reported:
        print("This matches the reported window: the context is per REQUEST here,")
        print("  not divided between the PARALLEL slots.")
    safe = int(lo * 0.8) // 256 * 256
    print("\nSuggested:  DR_MODEL_CONTEXT=%d   (measured %d, less a 20%% margin "
          "for the reply)" % (safe, lo))
    print("Currently:  DR_MODEL_CONTEXT=%d" % config.DR_MODEL_CONTEXT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
