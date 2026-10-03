"""Render one request json (lyrics+style) through yue-synth, model plans itself."""
import json, os, sys, types, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import music
src, out, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(src, encoding="utf-8"))
env = dict(os.environ)
env["PATH"] = os.pathsep.join([os.path.dirname(music.YUE2_CPP_EXE), os.path.join(os.environ.get("CUDA_PATH", ""), "bin"), env["PATH"]])
ctx = types.SimpleNamespace(set_stage=lambda *a, **k: None, cancel_event=threading.Event())
job = {"lyrics": d["lyrics"], "style": d["style"], "cot": "full", "lm_seed": seed, "seed": seed, "out": out}
ok, tail = music.run_gpu_worker(ctx, "", "", job, "YuE2", 1500, cmd=[music.YUE2_CPP_EXE, "--model", music.YUE2_CPP_MODEL,
                                "--vae", music.YUE2_CPP_VAE, "--request", "{job}", "--out", out], env=env)
print(out, ok, flush=True)
