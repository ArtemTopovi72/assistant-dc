"""Decode speed of the served chat model: tokens/s on a fixed long answer.

    venv/Scripts/python bench/tok_speed.py [--model ID] [--reps 3]
"""
import argparse, json, sys, time
sys.stdout.reconfigure(encoding="utf-8")
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests

PROMPT = ("Напиши на Python класс LRU-кэша с методами get/put и docstring, "
          "затем 5 тестов pytest к нему. Только код.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--url", default="http://127.0.0.1:1234/v1/chat/completions")
    a = ap.parse_args()
    import config
    model = a.model or config.MODEL_NAME
    rates = []
    for _ in range(a.reps):
        t0 = time.time()
        r = requests.post(a.url, json={"model": model, "max_tokens": 600, "temperature": 0.5,
                                       "messages": [{"role": "user", "content": PROMPT}]},
                          timeout=600).json()
        dt = time.time() - t0
        n = r["usage"]["completion_tokens"]
        rates.append(n / dt)
        print(f"{n} tok in {dt:.1f}s = {n / dt:.1f} tok/s", flush=True)
    rates.sort()
    print(json.dumps({"model": model, "median_tok_s": round(rates[len(rates) // 2], 1)}))


if __name__ == "__main__":
    main()
