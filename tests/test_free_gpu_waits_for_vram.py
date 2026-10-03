"""free_gpu() waits until the driver has the memory back.

`lms unload --all` returns before VRAM is released; a render that started
0.2 s later loaded beside the draining chat model and took 296 s instead of
48 s (live, 2026-09-12, journey 21). The eviction now polls nvidia-smi until
FREE_GPU_MIN_MB is free (bounded by FREE_GPU_WAIT_S), and a machine without
nvidia-smi is never held up.
"""
import os, sys, time
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import lora_training as LT

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

# a draining card: 2 GB free, then 9, then 20
readings = [2000, 9000, 20000]
LT.gpu_free_mb = lambda: readings.pop(0) if len(readings) > 1 else readings[0]
t0 = time.monotonic()
got = LT.wait_vram_free(16000, timeout=5, poll=0.01)
check("waits until the threshold is free", got == 20000, got)
check("and does not linger afterwards", time.monotonic() - t0 < 1.0)

# a card that never frees: bounded by the timeout, returns the last reading
LT.gpu_free_mb = lambda: 3000
t0 = time.monotonic()
got = LT.wait_vram_free(16000, timeout=0.2, poll=0.02)
check("gives up at the timeout", 0.15 <= time.monotonic() - t0 < 1.0 and got == 3000, got)

# no nvidia-smi: never blocks
LT.gpu_free_mb = lambda: None
t0 = time.monotonic()
check("no nvidia-smi → returns None immediately", LT.wait_vram_free(16000, timeout=5) is None and time.monotonic() - t0 < 0.5)

# free_gpu calls it after the evictions (the subprocess/urllib calls are best-effort and fail fast here)
calls = []
LT.wait_vram_free = lambda mn, timeout=0: calls.append((mn, timeout)) or 20000
LT.subprocess.run = lambda *a, **k: None
LT.urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(OSError("no comfy"))
said = []
LT.free_gpu(log=said.append)
check("free_gpu waits for the memory after evicting", calls and calls[0][0] == LT.FREE_GPU_MIN_MB, calls)
check("and reports the free reading", any("VRAM" in m for m in said), said)
check("the threshold is the chat model's size, not a token", LT.FREE_GPU_MIN_MB >= 12000)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
