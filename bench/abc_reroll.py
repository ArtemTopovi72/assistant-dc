"""How much does re-rolling YuE2's ABC plan (yue-plan, text stage only) move the
wrong-stress count? Same lyric/style as a real song, N lm_seeds, time each."""
import json, os, sys, tempfile, time, types, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import music
from song_stress import check  # bench/song_stress.py

src = json.load(open(sys.argv[1], encoding="utf-8"))
seeds = [int(s) for s in (sys.argv[2:] or range(1, 7))]
plan_exe = os.path.join(os.path.dirname(music.YUE2_CPP_EXE), "yue-plan.exe")
env = dict(os.environ)
env["PATH"] = os.pathsep.join([os.path.dirname(music.YUE2_CPP_EXE),
                               os.path.join(os.environ.get("CUDA_PATH", ""), "bin"), env["PATH"]])
ctx = types.SimpleNamespace(set_stage=lambda *a, **k: None, cancel_event=threading.Event())
for seed in seeds:
    out = os.path.join(tempfile.gettempdir(), f"plan_{seed}.abc")
    job = {"lyrics": src["lyrics"], "style": src["style"], "cot": "full", "lm_seed": seed, "seed": seed}
    t0 = time.time()
    ok, tail = music.run_gpu_worker(ctx, "", "", job, "yue-plan", 900,
                                    cmd=[plan_exe, "--model", music.YUE2_CPP_MODEL, "--request", "{job}", "--out", out],
                                    env=env)
    dt = time.time() - t0
    if not ok or not os.path.exists(out):
        print(seed, "FAILED", tail[-300:], flush=True)
        continue
    abc = open(out, encoding="utf-8").read()
    bad, total, rows = check(src["lyrics"], abc)
    g = [r for r in rows if "гнил" in r[0].lower()]
    print(f"seed {seed}: wrong {bad}/{total}  plan {dt:.0f}s  гнилое ok={[r[3] for r in g]}", flush=True)
