"""Render one YuE2 song. Runs in venv_yue2 (its torch/transformers pins clash with the app).

    venv_yue2/Scripts/python scripts/yue2_render.py job.json

job.json: {"lyrics": str, "style": str, "seed": int, "out": path, ["abc": melody score, "cot": "melody"|"full"]}.
Prints "OK <path>" on success.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.getenv("YUE2_REPO", os.path.join(ROOT, "models_ext", "YuE2-3B"))
# Local VAE dir: the hub cache needs symlinks, which Windows refuses without dev mode.
VAE = os.getenv("YUE2_VAE", os.path.join(ROOT, "models_ext", "YuE2-Vae"))


def main():
    job = json.load(open(sys.argv[1], encoding="utf-8"))
    from yue2 import YuE2Pipeline
    import yue2.cuda_graph as _cg
    # The Windows torch build has the flash op's symbol but not the kernel, so
    # "auto" picks flash and dies ("USE_FLASH_ATTENTION was not enabled").
    # Masked SDPA (what "auto" fell back to) ran a slow kernel: one verse+chorus
    # took 71 s; cuDNN 41-49 s; flash-attn via yue2_fa2 24 s (bench 2026-09-27).
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import yue2_fa2
    backend = os.getenv("YUE2_ATTN") or ("flash" if yue2_fa2.install() else "cudnn")
    print("attention:", backend, flush=True)
    _init = _cg.GraphAR.__init__
    _cg.GraphAR.__init__ = lambda self, *a, **k: _init(self, *a, **{**k, "attention_backend": backend})
    # NAR attention takes the whole song as ONE query block on cuda: a long metal
    # lyric OOM'd at 22 GB (live 2026-09-25 21:15, 21:23). 1024-row blocks cap it.
    import yue2.nar as _nar
    _nar_init = _nar.CachedNAR.__init__
    _nar.CachedNAR.__init__ = lambda self, m, c, attention="sdpa", query_chunk_size=None: \
        _nar_init(self, m, c, attention, query_chunk_size or int(os.getenv("YUE2_NAR_CHUNK", "1024")))
    pipe = YuE2Pipeline.from_pretrained(REPO, vae=VAE, device="cuda", progress=False)
    # The whole song through the VAE at once: 1.2 s vs 2.0 s tiled, same audio,
    # the card has the room (the AR/NAR model is moved off before decoding).
    _decode = pipe.decode
    pipe.decode = lambda latents, **k: _decode(latents, full=True)
    try:
        # a cover: an ABC score (SheetSage2) fixes the melody, cot="melody" plans only the arrangement around it
        extra = {"abc": job["abc"], "cot": job.get("cot", "melody")} if job.get("abc") else {"cot": "full"}
        song = pipe(style=job["style"], lyrics=job["lyrics"], seed=int(job["seed"]), **extra)
        song.save(job["out"])
        if getattr(song, "truncated", False):
            print("TRUNCATED", flush=True)   # music._generate_yue2 retries shorter
    finally:
        pipe.close()
    print("OK", job["out"], flush=True)


if __name__ == "__main__":
    main()
