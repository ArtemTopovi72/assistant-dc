"""The live step counter was dead: ComfyUI stopped sending flat "progress".

`_ComfyProgress` listened for {"type":"progress","data":{"value":k,"max":n}}.
ComfyUI's comfy_execution/progress.py only ever emits "progress_state" — a
COMBINED message carrying every node's state at once:

    {"type": "progress_state",
     "data": {"prompt_id": "...",
              "nodes": {"8": {"value": 3, "max": 8, "state": "running"}, ...}}}

so the branch matched nothing and the counter silently went quiet for EVERY
render, images as well as video. It failed invisibly because the whole progress
socket is best-effort by design ("only the counter goes quiet").

Verified against both installed ComfyUI trees (0.28.0 and 0.30.0): neither sends
the flat form.

Run: venv/Scripts/python.exe tests/test_comfy_progress_state.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

# _ComfyProgress now lives in comfy_client.py alongside the rest of the
# ComfyUI transport it belongs to.
import comfy_client as I

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


class FakeWS:
    """Feeds a scripted list of frames to _ComfyProgress._run, then closes."""
    def __init__(self, frames):
        self.frames = list(frames)
        self.closed = False

    def recv(self):
        if not self.frames:
            raise ConnectionError("closed")
        return self.frames.pop(0)

    def close(self):
        self.closed = True


def drive(frames, prompt_id="p1"):
    """Run the real message loop over `frames`; return the (value, max) calls."""
    got = []
    prog = I._ComfyProgress("cid", lambda v, m: got.append((v, m)), {"id": prompt_id})

    fake = FakeWS([json.dumps(f) if isinstance(f, dict) else f for f in frames])
    import types
    real_ws_mod = sys.modules.get("websocket")
    stub = types.ModuleType("websocket")
    stub.create_connection = lambda *a, **k: fake
    sys.modules["websocket"] = stub
    try:
        prog._run()
    finally:
        if real_ws_mod is not None:
            sys.modules["websocket"] = real_ws_mod
        else:
            sys.modules.pop("websocket", None)
    return got


print("=" * 70)
print("1. progress_state — the format ComfyUI actually sends — is understood")
print("=" * 70)

frames = [
    {"type": "progress_state", "data": {"prompt_id": "p1", "nodes": {
        "8": {"value": 1, "max": 8, "state": "running"}}}},
    {"type": "progress_state", "data": {"prompt_id": "p1", "nodes": {
        "8": {"value": 4, "max": 8, "state": "running"}}}},
    {"type": "progress_state", "data": {"prompt_id": "p1", "nodes": {
        "8": {"value": 8, "max": 8, "state": "finished"}}}},
]
got = drive(frames)
check("every progress_state produced a callback", len(got) == 3, got)
check("the step counts come through in order", got == [(1, 8), (4, 8), (8, 8)], got)

print()
print("=" * 70)
print("2. Loader nodes (max<=1) do not yank the bar back to zero")
print("=" * 70)

got = drive([
    {"type": "progress_state", "data": {"prompt_id": "p1", "nodes": {
        "1": {"value": 0, "max": 1, "state": "running"},
        "8": {"value": 5, "max": 8, "state": "running"}}}},
])
check("the sampler wins over a 0/1 loader", got == [(5, 8)], got)

got = drive([
    {"type": "progress_state", "data": {"prompt_id": "p1", "nodes": {
        "1": {"value": 1, "max": 1, "state": "finished"}}}},
])
check("a lone max=1 node reports nothing at all", got == [], got)

print()
print("=" * 70)
print("3. A RUNNING node is preferred over a finished one")
print("=" * 70)

got = drive([
    {"type": "progress_state", "data": {"prompt_id": "p1", "nodes": {
        "8": {"value": 8, "max": 8, "state": "finished"},
        "9": {"value": 2, "max": 40, "state": "running"}}}},
])
check("the running node is the one reported", got == [(2, 40)], got)

print()
print("=" * 70)
print("4. The legacy flat 'progress' message still works")
print("=" * 70)

got = drive([{"type": "progress", "data": {"prompt_id": "p1", "value": 3, "max": 8}}])
check("old-style progress is still honoured", got == [(3, 8)], got)

print()
print("=" * 70)
print("5. Another client's job is ignored, and 'executing: done' stops the loop")
print("=" * 70)

got = drive([
    {"type": "progress_state", "data": {"prompt_id": "OTHER", "nodes": {
        "8": {"value": 7, "max": 8, "state": "running"}}}},
    {"type": "progress_state", "data": {"prompt_id": "p1", "nodes": {
        "8": {"value": 1, "max": 8, "state": "running"}}}},
])
check("a different prompt_id is filtered out", got == [(1, 8)], got)

got = drive([
    {"type": "progress_state", "data": {"prompt_id": "p1", "nodes": {
        "8": {"value": 1, "max": 8, "state": "running"}}}},
    {"type": "executing", "data": {"prompt_id": "p1", "node": None}},
    {"type": "progress_state", "data": {"prompt_id": "p1", "nodes": {
        "8": {"value": 9, "max": 9, "state": "running"}}}},
])
check("the loop stops at 'executing/node:null' and ignores later frames",
      got == [(1, 8)], got)

print()
print("=" * 70)
print("6. Malformed frames never take the render down")
print("=" * 70)

got = drive([
    "not json at all",
    {"type": "progress_state", "data": {"prompt_id": "p1", "nodes": "not a dict"}},
    {"type": "progress_state", "data": {"prompt_id": "p1", "nodes": {
        "8": {"value": "x", "max": None, "state": "running"}}}},
    {"type": "progress_state", "data": {"prompt_id": "p1", "nodes": {
        "8": {"value": 2, "max": 8, "state": "running"}}}},
])
check("garbage is skipped and the good frame still lands", got == [(2, 8)], got)

print()
print("=" * 70)
print("7. Neither installed ComfyUI actually sends the flat form")
print("=" * 70)

base = os.getenv("COMFY_BASE_DIR", os.path.expanduser(r"~\Documents\ComfyUI"))
checked = 0
for tree in ("ComfyUI-0.28.0", "ComfyUI-0.30.0"):
    p = os.path.join(base, tree, "comfy_execution", "progress.py")
    if not os.path.exists(p):
        continue
    checked += 1
    src = open(p, encoding="utf-8").read()
    check(f"{tree} emits progress_state", '"progress_state"' in src)
    check(f"{tree} does NOT emit a flat \"progress\" message",
          'send_sync(\n            "progress"' not in src
          and '"progress",' not in src.replace('"progress_state",', ''))
if not checked:
    print("  (no ComfyUI tree found to inspect — skipped)")

print()
print(f"{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
