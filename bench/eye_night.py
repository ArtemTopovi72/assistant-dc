"""Night queue of eye_probe variants: waits for the GPU to be free, runs them one by one, logs
START/DONE/FAIL lines to tmp/night/LOG and a summary line per variant to tmp/night/SUMMARY.
Never kills ComfyUI: a stuck variant is /interrupt-ed and the queue moves on."""
import csv
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = os.path.join(ROOT, "venv", "Scripts", "python.exe")
COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8000")
# name -> env knobs; the name also carries the prompt variant (cgi / eyes) for eye_probe.py
QUEUE = [
    ("b576", {"VIDEO_SHORT_EDGE": "576"}),
    ("b576_cgi_eyes", {"VIDEO_SHORT_EDGE": "576"}),
    ("r4_576", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4"}),
    ("r6_576", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "6"}),
    ("b768", {"VIDEO_SHORT_EDGE": "768"}),
    ("b768_cgi_eyes", {"VIDEO_SHORT_EDGE": "768"}),
    ("r4_768_cgi_eyes", {"VIDEO_SHORT_EDGE": "768", "VIDEO_REFINE_STEPS": "4"}),
    ("up2_576_cgi_eyes", {"VIDEO_SHORT_EDGE": "576", "VIDEO_UPSCALE_FACTOR": "2.0"}),
    ("r4_576_squint", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4"}),
    ("r4_576_cgi_squint", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4"}),
    ("r4_768", {"VIDEO_SHORT_EDGE": "768", "VIDEO_REFINE_STEPS": "4"}),
    # ref2va hard-codes factor 2.0 (VIDEO_UPSCALE_FACTOR is not read there): the knob that matters is
    # one stage at full size, where the eyes are drawn at 768x576 instead of 384x288 and refined
    ("ss_576", {"VIDEO_SHORT_EDGE": "576", "VIDEO_TWO_STAGE": "0"}),
    ("ss_768", {"VIDEO_SHORT_EDGE": "768", "VIDEO_TWO_STAGE": "0"}),
    ("b576_s2", {"VIDEO_SHORT_EDGE": "576", "PROBE_SEED": "11"}),
    ("b576_s3", {"VIDEO_SHORT_EDGE": "576", "PROBE_SEED": "22"}),
    ("b768_s2", {"VIDEO_SHORT_EDGE": "768", "PROBE_SEED": "11"}),
    ("b768_s3", {"VIDEO_SHORT_EDGE": "768", "PROBE_SEED": "22"}),
    ("r4_576_s2", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "PROBE_SEED": "11"}),
    ("r4_768_s2", {"VIDEO_SHORT_EDGE": "768", "VIDEO_REFINE_STEPS": "4", "PROBE_SEED": "11"}),
    ("v6_s1", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "PROBE_VOICE": "tmp/night/ref6.wav"}),
    ("v6_s11", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "PROBE_VOICE": "tmp/night/ref6.wav", "PROBE_SEED": "11"}),
    ("v6_s22", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "PROBE_VOICE": "tmp/night/ref6.wav", "PROBE_SEED": "22"}),
    ("v15_s1", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "PROBE_VOICE": "tmp/night/ref15.wav"}),
    ("v15_s11", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "PROBE_VOICE": "tmp/night/ref15.wav", "PROBE_SEED": "11"}),
    ("v15_s22", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "PROBE_VOICE": "tmp/night/ref15.wav", "PROBE_SEED": "22"}),
    ("v9b_s1", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "PROBE_VOICE": "tmp/night/ref9b.wav"}),
    ("v9b_s11", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "PROBE_VOICE": "tmp/night/ref9b.wav", "PROBE_SEED": "11"}),
    ("v9b_s22", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "PROBE_VOICE": "tmp/night/ref9b.wav", "PROBE_SEED": "22"}),
    ("r4_576_cgi_squint_s2", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "PROBE_SEED": "11"}),
    ("r4_576_cgi_squint_s3", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "PROBE_SEED": "22"}),
    ("r4_576_squint_s2", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "PROBE_SEED": "11"}),
    ("r4_768_cgi_squint", {"VIDEO_SHORT_EDGE": "768", "VIDEO_REFINE_STEPS": "4"}),
    ("r4_768_cgi_squint_s2", {"VIDEO_SHORT_EDGE": "768", "VIDEO_REFINE_STEPS": "4", "PROBE_SEED": "11"}),
    ("fl2_576", {"VIDEO_SHORT_EDGE": "576"}),
    ("fl2_768", {"VIDEO_SHORT_EDGE": "768"}),
    ("fl2_576_s2", {"VIDEO_SHORT_EDGE": "576", "PROBE_SEED": "11"}),
    ("fl2_768_s2", {"VIDEO_SHORT_EDGE": "768", "PROBE_SEED": "11"}),
    ("fl2_768_cgi", {"VIDEO_SHORT_EDGE": "768"}),
    ("fl3_576", {"VIDEO_SHORT_EDGE": "576"}),
    ("fl3_768", {"VIDEO_SHORT_EDGE": "768"}),
    ("fl3_576_s2", {"VIDEO_SHORT_EDGE": "576", "PROBE_SEED": "11"}),
    ("fl3_768_s2", {"VIDEO_SHORT_EDGE": "768", "PROBE_SEED": "11"}),
    ("fl4_576", {"VIDEO_SHORT_EDGE": "576"}),
    ("fl4_576_s2", {"VIDEO_SHORT_EDGE": "576", "PROBE_SEED": "11"}),
    ("fl4_576_s3", {"VIDEO_SHORT_EDGE": "576", "PROBE_SEED": "22"}),
    ("fl4_768", {"VIDEO_SHORT_EDGE": "768"}),
    ("top45", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "VIDEO_REFINE_TOP": "0.45"}),
    ("top45_s2", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "VIDEO_REFINE_TOP": "0.45", "PROBE_SEED": "11"}),
    ("top30", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "VIDEO_REFINE_TOP": "0.30"}),
    ("top30_s2", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "VIDEO_REFINE_TOP": "0.30", "PROBE_SEED": "11"}),
    ("st12", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "VIDEO_STEPS_REF2VA_VOICES": "12"}),
    ("st12_s2", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "VIDEO_STEPS_REF2VA_VOICES": "12", "PROBE_SEED": "11"}),
    ("st16", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "VIDEO_STEPS_REF2VA_VOICES": "16"}),
    ("st16_s2", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "VIDEO_STEPS_REF2VA_VOICES": "16", "PROBE_SEED": "11"}),
    ("st25", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "VIDEO_STEPS_REF2VA_VOICES": "25"}),
    ("st25_s2", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "VIDEO_STEPS_REF2VA_VOICES": "25", "PROBE_SEED": "11"}),
    ("se864", {"VIDEO_SHORT_EDGE": "864", "VIDEO_REFINE_STEPS": "4"}),
    ("se864_s2", {"VIDEO_SHORT_EDGE": "864", "VIDEO_REFINE_STEPS": "4", "PROBE_SEED": "11"}),
    ("sh6", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "VIDEO_SHIFT_VIDEO": "6"}),
    ("sh6_s2", {"VIDEO_SHORT_EDGE": "576", "VIDEO_REFINE_STEPS": "4", "VIDEO_SHIFT_VIDEO": "6", "PROBE_SEED": "11"}),
]
if len(sys.argv) > 1:
    QUEUE = [q for q in QUEUE if q[0] in sys.argv[1:]]


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open("tmp/night/LOG", "a", encoding="utf8") as fh:
        fh.write(line + "\n")


def comfy_busy():
    try:
        q = urllib.request.urlopen(COMFY + "/queue", timeout=10).read().decode()
        return '"queue_running": []' not in q.replace("[ ]", "[]")
    except Exception:
        return False


def scenario_running():
    out = subprocess.run(["wmic", "process", "where", "name='python.exe'", "get", "commandline"],
                         capture_output=True, text=True).stdout
    return "live_tg_scenarios" in out


def interrupt():
    try:
        urllib.request.urlopen(urllib.request.Request(COMFY + "/interrupt", method="POST"), timeout=10)
    except Exception:
        pass


def summary(name):
    rows = list(csv.DictReader(open(f"tmp/night/{name}.csv")))
    tail = [r for r in rows if float(r["sec"]) >= 1.0 and float(r["match"]) >= 0.65]
    sh = [float(r["sharp_vs_still"]) for r in tail] or [0]
    mt = [float(r["match"]) for r in tail] or [0]
    extra = subprocess.run([PY, "tmp/clip_metrics.py", name], capture_output=True, text=True).stdout.strip().splitlines()
    return (f"{extra[-1] if extra else ''} | sharp mean {sum(sh) / len(sh):.2f} min {min(sh):.2f} | match mean {sum(mt) / len(mt):.2f} "
            f"| tracked {len(tail)}/{len(rows)}")


while scenario_running() or comfy_busy():
    time.sleep(30)
log(f"GPU free, {len(QUEUE)} variants")
for name, env in QUEUE:
    if os.path.exists(f"tmp/night/{name}.csv"):
        log(f"SKIP {name} (done)")
        continue
    log(f"START {name} {env}")
    t0 = time.time()
    try:
        p = subprocess.run([PY, "-u", "bench/eye_probe.py", name], env={**os.environ, **env},
                           capture_output=True, text=True, timeout=45 * 60)
        open(f"tmp/night/{name}.log", "w", encoding="utf8").write(p.stdout + p.stderr)
        if os.path.exists(f"tmp/night/{name}.csv"):
            line = f"DONE {name} {time.time() - t0:.0f}s {summary(name)}"
            log(line)
            open("tmp/night/SUMMARY", "a", encoding="utf8").write(line + "\n")
        else:
            log(f"FAIL {name} rc={p.returncode} see tmp/night/{name}.log")
    except subprocess.TimeoutExpired:
        interrupt()
        log(f"FAIL {name} timeout, interrupted")
        time.sleep(60)
log("QUEUE FINISHED")
