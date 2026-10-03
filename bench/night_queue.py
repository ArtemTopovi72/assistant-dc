"""Run the night's GPU jobs strictly one after another.

Each step is a command line; output goes to runtime/night_queue.log. A step
that fails is logged and the queue moves on -- one broken variant must not
cost the rest of the night.
"""
import os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "venv", "Scripts", "python.exe")
LOG = os.path.join(ROOT, "runtime", "night_queue.log")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def wait_for_line(path, needle, max_wait=12 * 3600):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            # skip the queue's own "waiting for <needle>" lines, or a waiter
            # finds itself and starts at once (it did, 00:57, music ran early)
            lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
            if any(needle in ln for ln in lines if "waiting for" not in ln):
                return True
        except OSError:
            pass
        time.sleep(15)
    return False


STEPS = [a for a in sys.argv[1:]]

if __name__ == "__main__":
    if STEPS and STEPS[0].startswith("wait:"):
        path, needle = STEPS.pop(0)[5:].split("|", 1)
        log(f"waiting for {needle!r} in {path}")
        wait_for_line(os.path.join(ROOT, path), needle)
    for step in STEPS:
        log("START " + step)
        t0 = time.time()
        out = open(os.path.join(ROOT, "runtime", "night_queue_step.log"), "a", encoding="utf-8")
        rc = subprocess.call([PY] + step.split(), cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        log(f"END rc={rc} {time.time() - t0:.0f}s " + step)
