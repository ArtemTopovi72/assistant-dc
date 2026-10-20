"""Seed-VC batch worker (venv_qwen): models load ONCE, then every job converts timbre only-ish (F0 kept) onto a target voice.
    venv_qwen/Scripts/python.exe scripts/seedvc_batch.py jobs.json     jobs = [{"source":..., "target":..., "out":...}]
A job may carry "f0": false (the speech model, 22 kHz, no F0 conditioning), "steps", "cfg".
Prints one JSON line per finished job. The cwd does not matter: it chdirs into models_ext/seed-vc."""
import argparse, json, os, shutil, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
SV = ROOT / "models_ext" / "seed-vc"
JOBS = Path(sys.argv[1]).resolve()
os.chdir(SV); sys.path.insert(0, str(SV))
import inference                                   # noqa: E402
_orig, _cache = inference.load_models, {}
inference.load_models = lambda a: _cache[a.f0_condition] if a.f0_condition in _cache else _cache.setdefault(a.f0_condition, _orig(a))
for job in json.loads(JOBS.read_text(encoding="utf-8")):
    if Path(job["out"]).exists():
        continue
    tmp = tempfile.mkdtemp()
    a = argparse.Namespace(source=job["source"], target=job["target"], output=tmp, diffusion_steps=job.get("steps", 30), length_adjust=1.0,
                           inference_cfg_rate=job.get("cfg", 0.7), f0_condition=job.get("f0", True),
                           auto_f0_adjust=job.get("f0", True), semi_tone_shift=0,
                           checkpoint=None, config=None, fp16=True)
    inference.main(a)
    w = list(Path(tmp).glob("*.wav"))
    if w:
        shutil.copy(w[0], job["out"])
    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps({"out": job["out"], "ok": bool(w)}), flush=True)
