"""Benchmark a LOCAL LLM (LM Studio) as a Russian stress placer.

Constrained prompt: preserve text byte-for-byte, only insert '+' before the
stressed vowel of each polysyllabic word. Deterministic (temp 0). Each homograph
sentence is run REPEATS times to measure run-to-run consistency. We also measure
formatting failures and text-corruption rate (letters changed/added/dropped).

Usage: python bench/stress_llm.py <model_id> [repeats]
Prints one  RESULT {json}  line.
"""
import os, sys, json, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import requests
from bench.stress_data import (
    HOMOGRAPHS, HARD_WORDS, GENERAL,
    canonical, base, find_token, n_vowels, is_marked,
)

URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1/chat/completions")

SYSTEM = (
    "Ты расставляешь ударения в русском тексте для синтеза речи. "
    "Правила СТРОГО:\n"
    "1. Верни ИСХОДНЫЙ текст без изменений, добавив только знак '+' "
    "НЕПОСРЕДСТВЕННО ПЕРЕД ударной гласной в каждом слове из двух и более слогов.\n"
    "2. НЕ меняй, НЕ добавляй и НЕ удаляй ни одной буквы, пробела или знака препинания.\n"
    "3. В односложных словах ударение не ставь.\n"
    "4. Ровно один '+' на слово. Учитывай смысл предложения (омографы: за́мок/замо́к).\n"
    "5. Верни ТОЛЬКО размеченный текст, без пояснений, без кавычек."
)
FEWSHOT = [
    ("Старинный замок стоял на горе.", "Стар+инный з+амок сто+ял на гор+е."),
    ("Ржавый замок висел на двери.", "Рж+авый зам+ок вис+ел на двер+и."),
    ("Куриные белки очень полезны.", "Кур+иные белк+и +очень пол+езны."),
]


def call(model, text):
    msgs = [{"role": "system", "content": SYSTEM}]
    for u, a in FEWSHOT:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": text})
    body = {"model": model, "messages": msgs, "temperature": 0,
            "max_tokens": 256, "stream": False}
    r = requests.post(URL, json=body, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def strip_marks(s):
    """base letters only: drop +, U+0301, apostrophes, collapse ws."""
    s = s.replace("+", "").replace("́", "").replace("'", "").replace("’", "")
    return re.sub(r"\s+", " ", s).strip().lower().replace("ё", "е")


def main():
    model = sys.argv[1]
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    fmt_fail = corrupt = 0
    lat = []

    # cache one deterministic prediction per unique text (temp 0)
    cache = {}
    def predict(text):
        if text in cache:
            return cache[text]
        t0 = time.perf_counter()
        try:
            out = call(model, text)
        except Exception as e:
            out = f"__ERR__{e}"
        lat.append((time.perf_counter() - t0) * 1000)
        cache[text] = out
        return out

    # --- homographs (+ consistency via repeats) ---
    h_ok = 0
    consist_hits = consist_tot = 0
    for sent, target, gold in HOMOGRAPHS:
        outs = []
        for _ in range(repeats):
            # bypass cache for consistency measurement
            t0 = time.perf_counter()
            try:
                o = call(model, sent)
            except Exception as e:
                o = f"__ERR__{e}"
            lat.append((time.perf_counter() - t0) * 1000)
            outs.append(o)
        out = outs[0]
        consist_tot += 1
        if len(set(outs)) == 1:
            consist_hits += 1
        # corruption / format
        if out.startswith("__ERR__"):
            fmt_fail += 1
            continue
        if strip_marks(out) != strip_marks(sent):
            corrupt += 1
        tok = find_token(out, target)
        if tok and canonical(tok) == canonical(gold):
            h_ok += 1
    homograph_acc = h_ok / len(HOMOGRAPHS)
    consistency = consist_hits / max(1, consist_tot)

    # --- hard words ---
    w_ok = 0
    for gold in HARD_WORDS:
        word = base(gold)
        out = predict(word)
        if out.startswith("__ERR__"):
            fmt_fail += 1; continue
        tok = find_token(out, word) or out
        if canonical(tok) == canonical(gold):
            w_ok += 1
    hard_word_acc = w_ok / len(HARD_WORDS)

    # --- general ---
    g_ok = g_tot = 0
    sent_ok = 0
    for sent, golds in GENERAL:
        out = predict(sent)
        if out.startswith("__ERR__"):
            fmt_fail += 1; continue
        if strip_marks(out) != strip_marks(sent):
            corrupt += 1
        ok_all = True
        for gold in golds:
            g_tot += 1
            tok = find_token(out, gold)
            if tok and canonical(tok) == canonical(gold):
                g_ok += 1
            else:
                ok_all = False
        if ok_all:
            sent_ok += 1
    general_word_acc = g_ok / max(1, g_tot)
    sentence_acc = sent_ok / len(GENERAL)

    pooled_ok = h_ok + w_ok + g_ok
    pooled_tot = len(HOMOGRAPHS) + len(HARD_WORDS) + g_tot

    res = {
        "engine": f"LLM:{model}",
        "homograph_acc": round(homograph_acc, 4),
        "hard_word_acc": round(hard_word_acc, 4),
        "general_word_acc": round(general_word_acc, 4),
        "sentence_acc": round(sentence_acc, 4),
        "stress_acc_overall": round(pooled_ok / pooled_tot, 4),
        "consistency": round(consistency, 4),
        "format_fail": fmt_fail,
        "corrupt": corrupt,
        "latency_ms_per_sent": round(sum(lat) / max(1, len(lat)), 1),
        "n_calls": len(lat),
    }
    print("RESULT " + json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
