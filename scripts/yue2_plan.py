"""Plan (score only, no audio) or render from an edited score.

    venv_yue2/Scripts/python scripts/yue2_plan.py job.json

job.json: {"lyrics", "style", "seed", "out_abc"} -> writes the ABC plan;
          add "abc": <text> and "out" -> renders audio from that score.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yue2_render as R  # noqa: E402  (same patches, same paths)


def main():
    job = json.load(open(sys.argv[1], encoding="utf-8"))
    from yue2 import YuE2Pipeline
    import yue2.cuda_graph as _cg
    _init = _cg.GraphAR.__init__
    _cg.GraphAR.__init__ = lambda self, *a, **k: _init(self, *a, **{**k, "attention_backend": "sdpa"})
    import yue2.nar as _nar
    _nar_init = _nar.CachedNAR.__init__
    _nar.CachedNAR.__init__ = lambda self, m, c, attention="sdpa", query_chunk_size=None: \
        _nar_init(self, m, c, attention, query_chunk_size or 1024)
    pipe = YuE2Pipeline.from_pretrained(R.REPO, vae=R.VAE, device="cuda", progress=False)
    try:
        if job.get("abc"):
            song = pipe(style=job["style"], lyrics=job["lyrics"], cot="full",
                        seed=int(job["seed"]), abc=job["abc"])
            song.save(job["out"])
            print("OK", job["out"], flush=True)
        else:
            plan = pipe.plan(style=job["style"], lyrics=job["lyrics"], cot="full", seed=int(job["seed"]))
            d = job["out_abc"]
            plan.save(d)
            print("OK", d, flush=True)
    finally:
        pipe.close()


if __name__ == "__main__":
    main()
