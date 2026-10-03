"""Render one MuLaCover cover. Runs in venv_mula (its numpy/torchao pins clash with the app).

    venv_mula/Scripts/python scripts/mulacover_render.py job.json

job.json: {"ref": wav, "lyrics": str, "tags": str, "seed": int, "out": path}.
The model is CC BY-NC: personal, non-commercial use only.
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.getenv("MULACOVER_CKPT", os.path.join(ROOT, "models_ext", "mulacover_ckpt"))


def main():
    job = json.load(open(sys.argv[1], encoding="utf-8"))
    import torch
    from mulacover import MuLaCoverGenPipeline
    pipe = MuLaCoverGenPipeline.from_pretrained(
        CKPT, device=torch.device("cuda:0"),
        dtype={"mulacover": torch.bfloat16, "codec": torch.float32,
               "qwen": torch.float32, "transcriptor": torch.float32},
        lazy_load=True)
    with tempfile.TemporaryDirectory() as td:
        lp, tp = os.path.join(td, "lyrics.txt"), os.path.join(td, "tags.txt")
        open(lp, "w", encoding="utf-8").write(job["lyrics"])
        open(tp, "w", encoding="utf-8").write(job["tags"])
        torch.manual_seed(int(job["seed"]))
        pipe({"ref_audio": job["ref"], "bpm": None, "lyrics": lp, "tags": tp},
             save_path=job["out"], temperature=1.0, topk=250, cfg_scale=1.5)
    print("OK", job["out"], flush=True)


if __name__ == "__main__":
    main()
