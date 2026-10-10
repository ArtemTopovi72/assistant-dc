"""Russian quality + speed of whatever model LM Studio is serving.

    venv/Scripts/python bench/model_ru_probe.py --model <id> [--out file.json]

Eight everyday Russian requests through the plain chat endpoint (no tools):
time to first token, tokens/s, share of Cyrillic letters in the answer (a
model that drifts into English scores low), and the answers themselves, so a
person can read them side by side. Not a quality score -- a reading sheet.
"""
import argparse
import json
import os
import re
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import config

PROMPTS = [
    "Привет! Как дела? Ответь коротко и по-дружески.",
    "Объясни простыми словами, почему небо голубое. Три-четыре предложения.",
    "Придумай смешное поздравление с днём рождения для друга Степана, который любит рыбалку.",
    "Чем отличается электрочайник с нагревательным диском от чайника со спиралью? Что лучше купить?",
    "У меня 3 яблока, я отдал половину, потом купил ещё 5. Сколько яблок? Объясни.",
    "Перескажи в двух предложениях сюжет «Преступления и наказания».",
    "Напиши вежливый ответ соседу, который просит не шуметь после 23:00.",
    "Какие три вещи взять на пикник, если обещают дождь? Кратко.",
]


def cyr_share(text: str) -> float:
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text or "")
    return round(sum(1 for c in letters if re.match(r"[А-Яа-яЁё]", c)) / max(1, len(letters)), 3)


def ask(model: str, prompt: str, system: str = "") -> dict:
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    if "gemma" in model.lower():
        # The app's no-think lever (llm.GEMMA_NO_THINK_PREFILL).
        import llm
        msgs.append({"role": "assistant", "content": llm.GEMMA_NO_THINK_PREFILL})
    elif "qwen" in model.lower():
        import llm
        msgs.append({"role": "assistant", "content": llm.QWEN_NO_THINK_PREFILL})
    t0 = time.perf_counter()
    first = None
    content, reasoning, n = [], [], 0
    with requests.post(config.LM_STUDIO_URL, stream=True, timeout=(10, 300), json={
            "model": model, "messages": msgs, "max_tokens": 1500,
            "temperature": 0.7, "stream": True}) as r:
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            piece = delta.get("content") or ""
            think = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if (piece or think) and first is None:
                first = time.perf_counter() - t0
            if piece:
                content.append(piece); n += 1
            if think:
                reasoning.append(think); n += 1
    total = time.perf_counter() - t0
    text = re.sub(r"(?s)<think>.*?</think>|<\|channel>.*?<channel\|>", "", "".join(content))
    if "<think>" in text:        # unclosed: all of it was thinking
        reasoning.append(text); text = ""
    return {"prompt": prompt, "answer": text, "reasoning_chars": len("".join(reasoning)),
            "ttft": round(first or total, 2), "seconds": round(total, 2),
            "tok_s": round(n / max(0.01, total - (first or 0)), 1), "cyrillic": cyr_share(text)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--system", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    rows = []
    for p in PROMPTS:
        r = ask(a.model, p, a.system)
        rows.append(r)
        print(f"[{r['seconds']:5.1f}s ttft {r['ttft']:4.1f} {r['tok_s']:5.1f} tok/s "
              f"cyr {r['cyrillic']:.2f} think {r['reasoning_chars']}] {p[:40]}")
        print("   " + r["answer"].strip().replace("\n", " ")[:300])
    med = sorted(r["seconds"] for r in rows)[len(rows) // 2]
    print(f"\nmedian {med}s, mean tok/s {sum(r['tok_s'] for r in rows) / len(rows):.1f}, "
          f"min cyrillic {min(r['cyrillic'] for r in rows):.2f}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
