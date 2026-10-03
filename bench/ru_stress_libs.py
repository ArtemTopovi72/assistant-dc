"""
Comparison of Russian stress-marking libraries:
  1. ruaccent  (neural, transformer-based, HF model)
  2. russtress (tensorflow-based, dictionary + neural)

Tests on long sentences; picks and reports the best one.
Run: .\\venv\\Scripts\\python.exe bench\\ru_stress_libs.py
"""
import sys, time, unicodedata
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

# ── test sentences ────────────────────────────────────────────────────────────
SENTENCES = [
    "Государственная дума Российской Федерации приняла закон о регулировании искусственного интеллекта.",
    "Международная конференция по машинному обучению прошла в Санкт-Петербурге в конце прошлого года.",
    "Студенты университета активно участвуют в научно-исследовательской деятельности кафедры.",
    "Автоматическое проставление ударений помогает системам синтеза речи правильно произносить слова.",
    "Разработчики программного обеспечения создали инновационное приложение для обработки естественного языка.",
    "Экономические показатели страны улучшились благодаря реформированию налоговой системы.",
    "Лингвистические особенности русского языка делают его одним из самых сложных для изучения иностранцами.",
    "Правительство рассматривает возможность введения новых мер поддержки малого и среднего бизнеса.",
]

STRESS_CHAR = "́"  # combining acute accent


def has_stress(text: str) -> bool:
    return STRESS_CHAR in text or "+" in text or any(
        unicodedata.combining(c) == 230 for c in text
    )


def count_stressed(text: str) -> int:
    """Count stressed vowels.
    Handles three conventions:
      - apostrophe AFTER vowel: Госуда'рственная  (russtress / ruaccent)
      - combining acute AFTER vowel: U+0301         (combining acute)
      - plus BEFORE vowel:     Госуд+арственная   (silero_stress)
    """
    vowels = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
    count = 0
    for i, ch in enumerate(text):
        if ch in vowels:
            after  = text[i + 1] if i + 1 < len(text) else ""
            before = text[i - 1] if i > 0 else ""
            if after in ("'", "'", STRESS_CHAR):   # apostrophe/acute after
                count += 1
            elif before == "+":                     # plus before (silero_stress)
                count += 1
            elif ch in ("Ё", "ё"):                 # ё is always stressed
                count += 1
    return count


# ── Library 1: ruaccent ───────────────────────────────────────────────────────
def test_ruaccent():
    print("\n" + "=" * 60)
    print("Library 1: ruaccent (neural BiLSTM, no external deps)")
    print("=" * 60)
    try:
        from ruaccent import Accentor
        acc = Accentor()  # loads bundled model from package

        results = []
        total_time = 0.0
        for sent in SENTENCES:
            t0 = time.perf_counter()
            out = acc.put_accent(sent)
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            n = count_stressed(out)
            results.append((sent, out, n, elapsed))
            print(f"  [{elapsed:.2f}s | {n} stressed] {out[:80]}...")

        avg_time = total_time / len(SENTENCES)
        avg_stressed = sum(r[2] for r in results) / len(results)
        print(f"\n  avg time/sentence: {avg_time:.2f}s | avg stressed words: {avg_stressed:.1f}")
        return {"name": "ruaccent", "avg_time": avg_time, "avg_stressed": avg_stressed, "ok": True}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"name": "ruaccent", "ok": False, "error": str(e)}


# ── Library 2: russtress ──────────────────────────────────────────────────────
def test_russtress():
    print("\n" + "=" * 60)
    print("Library 2: russtress (TensorFlow BiLSTM + dictionary, uses tf_keras)")
    print("=" * 60)
    try:
        import os; os.environ["TF_USE_LEGACY_KERAS"] = "1"
        from russtress import Accent
        acc = Accent()

        results = []
        total_time = 0.0
        for sent in SENTENCES:
            t0 = time.perf_counter()
            out = acc.put_stress(sent)
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            n = count_stressed(out)
            results.append((sent, out, n, elapsed))
            print(f"  [{elapsed:.2f}s | {n} stressed] {out[:80]}...")

        avg_time = total_time / len(SENTENCES)
        avg_stressed = sum(r[2] for r in results) / len(results)
        print(f"\n  avg time/sentence: {avg_time:.2f}s | avg stressed words: {avg_stressed:.1f}")
        return {"name": "russtress", "avg_time": avg_time, "avg_stressed": avg_stressed, "ok": True}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"name": "russtress", "ok": False, "error": str(e)}


# ── Library 3: silero_stress (already used in models.py) ─────────────────────
def test_silero_stress():
    print("\n" + "=" * 60)
    print("Library 3: silero_stress (Silero ONNX model, already in project)")
    print("=" * 60)
    try:
        from silero_stress import load_accentor
        acc = load_accentor()

        results = []
        total_time = 0.0
        for sent in SENTENCES:
            t0 = time.perf_counter()
            out = acc(sent)
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            n = count_stressed(out)
            results.append((sent, out, n, elapsed))
            print(f"  [{elapsed:.2f}s | {n} stressed] {out[:80]}...")

        avg_time = total_time / len(SENTENCES)
        avg_stressed = sum(r[2] for r in results) / len(results)
        print(f"\n  avg time/sentence: {avg_time:.2f}s | avg stressed words: {avg_stressed:.1f}")
        return {"name": "silero_stress", "avg_time": avg_time, "avg_stressed": avg_stressed, "ok": True}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"name": "silero_stress", "ok": False, "error": str(e)}


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Russian stress-marking library benchmark")
    print(f"Testing {len(SENTENCES)} long sentences\n")

    r1 = test_ruaccent()
    r2 = test_russtress()
    r3 = test_silero_stress()

    ok = [r for r in [r1, r2, r3] if r.get("ok")]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in [r1, r2, r3]:
        status = "OK" if r.get("ok") else f"FAILED ({r.get('error', '?')})"
        if r.get("ok"):
            print(f"  {r['name']:12s}  avg {r['avg_time']:.2f}s/sent  {r['avg_stressed']:.1f} stressed words  [{status}]")
        else:
            print(f"  {r['name']:12s}  [{status}]")

    if not ok:
        print("\nNo working library found.")
        sys.exit(1)

    # Choose best: highest stressed-word coverage, then fastest
    best = max(ok, key=lambda r: (r["avg_stressed"], -r["avg_time"]))
    print(f"\n  ✓ BEST: {best['name']}  (most coverage, {'faster' if best['avg_time'] < 1.0 else 'acceptable speed'})")
    print(f"\nRecommendation: use '{best['name']}' for TTS stress pre-processing in this project.")
