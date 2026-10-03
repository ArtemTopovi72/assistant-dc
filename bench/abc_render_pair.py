"""Render the same song from two given ABC plans (best vs worst stress score)."""
import json, os, sys, tempfile, types, threading, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import music
src = json.load(open(sys.argv[1], encoding="utf-8"))
env = dict(os.environ)
env["PATH"] = os.pathsep.join([os.path.dirname(music.YUE2_CPP_EXE),
                               os.path.join(os.environ.get("CUDA_PATH", ""), "bin"), env["PATH"]])
ctx = types.SimpleNamespace(set_stage=lambda *a, **k: None, cancel_event=threading.Event())
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_abc_pair")
os.makedirs(dst, exist_ok=True)
for seed, tag in [(int(a.split(":")[0]), a.split(":")[1]) for a in sys.argv[2:]]:
    abc = open(os.path.join(tempfile.gettempdir(), f"plan_{seed}.abc"), encoding="utf-8").read()
    out = os.path.join(dst, f"{tag}_seed{seed}.mp3")
    job = {"lyrics": src["lyrics"], "style": src["style"], "cot": "full", "abc": abc,
           "lm_seed": seed, "seed": seed, "out": out}
    ok, tail = music.run_gpu_worker(ctx, "", "", job, "YuE2", 1500,
                                    cmd=[music.YUE2_CPP_EXE, "--model", music.YUE2_CPP_MODEL, "--vae", music.YUE2_CPP_VAE,
                                         "--request", "{job}", "--out", out], env=env)
    print(tag, seed, ok, os.path.exists(out), tail[-200:] if not ok else "", flush=True)
