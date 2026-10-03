"""Long-context reality check: does a big window actually work at speed?

    venv/Scripts/python bench/long_ctx_probe.py --model ID --tokens 30000

Builds a Russian haystack of ~N tokens with one needle in the middle, asks for
the needle, and reports prompt-processing time, decode tok/s and whether the
answer is right. VRAM "fitting" on Windows can hide a spill into shared system
memory that only shows up here, as a slow prompt pass.
"""
import argparse, json, subprocess, sys, time
sys.stdout.reconfigure(encoding="utf-8")
import requests

FILLER = ("В старом городе на берегу реки жили ремесленники, торговцы и рыбаки. "
          "Каждое утро они открывали лавки, чинили сети и спорили о погоде. ")
NEEDLE = "Секретный код от склада — фиолетовый барсук 4417."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=30000)
    a = ap.parse_args()
    reps = max(1, a.tokens // 40)          # ~40 tokens per filler sentence pair
    body = [FILLER] * reps
    body.insert(reps // 2, NEEDLE + " ")
    text = "".join(body)
    msgs = [{"role": "user", "content": text + "\n\nКакой секретный код от склада? Ответь одной строкой."},
            {"role": "assistant", "content": "<|channel>thought\n<channel|>"}]
    t = time.time()
    r = requests.post("http://127.0.0.1:1234/v1/chat/completions",
                      json={"model": a.model, "messages": msgs, "max_tokens": 60, "temperature": 0},
                      timeout=900).json()
    wall = time.time() - t
    u = r.get("usage", {})
    ans = (r["choices"][0]["message"].get("content") or "").strip()
    vram = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    print(json.dumps({"prompt_tokens": u.get("prompt_tokens"), "wall_s": round(wall, 1),
                      "prompt_tok_per_s": round((u.get("prompt_tokens") or 0) / max(wall, 1e-6)),
                      "found": "4417" in ans, "answer": ans[:80], "vram": vram}, ensure_ascii=False))


if __name__ == "__main__":
    main()
