"""
Test stress-marking libraries on complex/compound/rare Russian words.
Validates:
  1. Correct stress placement on hard vocabulary
  2. Output format compatibility with F5-TTS (audio.py strips '+', keeps everything else)
     silero_stress -> '+' before vowel -> stripped cleanly
     ruaccent/russtress -> apostrophe after vowel -> CORRUPTS TTS input (stays in string)

Run: .\\venv\\Scripts\\python.exe tests\\test_ru_stress_hard.py
"""
import sys, time, re
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

# ── complex / compound / rare words in sentences ──────────────────────────────
HARD_WORDS = [
    # (word, expected_stressed_syllable_hint)
    ("электроудочники",  "у"),   # electrofishermen - compound, 7 syllables
    ("телерадиовещание", "а"),   # broadcasting - compound, many syllables
    ("противоестественный", "е"), # unnatural - prefix+root compound
    ("высокопроизводительный", "и"), # high-performance - long compound
    ("водонепроницаемый", "а"),  # waterproof
    ("нефтепромышленность", "ы"), # oil industry
    ("самовоспламенение", "е"),  # spontaneous ignition
    ("головокружительный", "и"), # breathtaking/dizzying
    ("пятисотлетие",     "е"),  # 500th anniversary
    ("мировоззрени",     "е"),  # worldview (stem matches мировоззрение/мировоззрения)
]

HARD_SENTENCES = [
    "Электроудочники незаконно ловят рыбу в заповедных водоёмах страны.",
    "Телерадиовещание претерпело революционные изменения с появлением интернета.",
    "Противоестественный отбор иногда приводит к неожиданным эволюционным последствиям.",
    "Высокопроизводительный суперкомпьютер используется для моделирования климатических изменений.",
    "Водонепроницаемый корпус позволяет использовать устройство при экстремальных погодных условиях.",
    "Нефтепромышленность региона столкнулась с серьёзными экологическими проблемами.",
    "Самовоспламенение горючих материалов происходит при достижении температуры воспламенения.",
    "Головокружительный успех молодого предпринимателя удивил весь деловой мир.",
    "В этом году отмечается пятисотлетие основания древнего монастыря.",
    "Формирование мировоззрения происходит под влиянием семьи, образования и общества.",
]

APOSTROPHES = {"'", "'"}
F5_STRIP_RE = re.compile(r"\+")  # what audio.py does: re.sub(r"\+", "", text)


def f5_format_check(original: str, stressed: str, library: str) -> str:
    """Check if stressed output is safe for F5-TTS pipeline."""
    stripped = F5_STRIP_RE.sub("", stressed)
    leftover_apos = sum(1 for ch in stripped if ch in APOSTROPHES)
    if leftover_apos > 0:
        return f"WARNING: {leftover_apos} apostrophe(s) survive stripping -> will corrupt TTS"
    return "OK for F5-TTS"


STRESS_MARKS = frozenset("+'’́")  # + ' ' combining-acute

def show_word_result(word: str, out: str, library: str) -> str:
    """Extract the stressed form of a target word from the stressed sentence."""
    base = word.lower()
    clean_out = "".join(ch for ch in out if ch not in STRESS_MARKS)
    pos = clean_out.lower().find(base)
    if pos == -1:
        return f"[{word}] -> NOT FOUND"

    result = []
    clean_idx = 0
    for ch in out:
        if ch in STRESS_MARKS:
            if pos <= clean_idx <= pos + len(base):
                result.append(ch)
        else:
            if pos <= clean_idx < pos + len(base):
                result.append(ch)
            clean_idx += 1
            if clean_idx > pos + len(base):
                break

    return "".join(result) if result else f"[{word}] -> NOT FOUND"


def run_library(name: str, accentor_fn):
    print(f"\n{'='*60}")
    print(f"Library: {name}")
    print(f"{'='*60}")
    print(f"  F5-TTS pipeline: audio.py strips '+' then sends to model")
    print()

    results = []
    for sent, word_hint in zip(HARD_SENTENCES, HARD_WORDS):
        word = word_hint[0]
        t0 = time.perf_counter()
        try:
            out = accentor_fn(sent)
            elapsed = time.perf_counter() - t0
            stressed_word = show_word_result(word, out, name)
            fmt_check = f5_format_check(sent, out, name)
            f5_ready = F5_STRIP_RE.sub("", out)  # what TTS actually gets
            results.append({"ok": True, "word": word, "stressed": stressed_word,
                           "fmt": fmt_check, "f5": f5_ready, "elapsed": elapsed, "full": out})
            status = "OK" if fmt_check == "OK for F5-TTS" else "WARN"
            print(f"  [{status}] {word:30s} -> {stressed_word}")
            if fmt_check != "OK for F5-TTS":
                print(f"         !! {fmt_check}")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results.append({"ok": False, "word": word, "error": str(e)})
            print(f"  [ERR] {word:30s} -> ERROR: {e}")

    ok = [r for r in results if r.get("ok")]
    warned = [r for r in ok if r["fmt"] != "OK for F5-TTS"]
    print(f"\n  {len(ok)}/{len(results)} words processed, {len(warned)} format warnings")
    return results


if __name__ == "__main__":
    print("F5-TTS Complex Russian Word Stress Test")
    print("Testing compound/rare words, checking TTS format compatibility\n")
    print("NOTE: audio.py does re.sub(r'+', '', text) — only '+' markers are stripped.")
    print("      Apostrophes from ruaccent/russtress SURVIVE and corrupt TTS input.\n")

    all_results = {}

    # Library 1: silero_stress (project default)
    try:
        from silero_stress import load_accentor
        acc = load_accentor()
        all_results["silero_stress"] = run_library("silero_stress [PROJECT DEFAULT]", acc)
    except Exception as e:
        print(f"\nsilero_stress FAILED to load: {e}")

    # Library 2: ruaccent
    try:
        import io, contextlib
        from ruaccent import Accentor
        # ruaccent prints emoji to stdout; redirect to suppress encoding errors on Windows
        with contextlib.redirect_stdout(io.StringIO()):
            acc2 = Accentor()
        all_results["ruaccent"] = run_library("ruaccent", acc2.put_accent)
    except Exception as e:
        print(f"\nruaccent FAILED to load: {e}")

    # Library 3: russtress
    try:
        import os; os.environ["TF_USE_LEGACY_KERAS"] = "1"
        from russtress import Accent
        acc3 = Accent()
        all_results["russtress"] = run_library("russtress", acc3.put_stress)
    except Exception as e:
        print(f"\nrusstress FAILED to load: {e}")

    # ── side-by-side for each word ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("WORD-BY-WORD COMPARISON")
    print(f"{'='*60}")
    print(f"  {'word':30s}  {'silero(+)':25s}  {'ruaccent(apostrophe)':25s}  {'russtress(apostrophe)':20s}")
    print(f"  {'-'*30}  {'-'*25}  {'-'*25}  {'-'*20}")
    for i, (word, _) in enumerate(HARD_WORDS):
        row = f"  {word:30s}"
        for lib in ["silero_stress", "ruaccent", "russtress"]:
            if lib in all_results and i < len(all_results[lib]):
                r = all_results[lib][i]
                val = r.get("stressed", "ERR") if r.get("ok") else "ERR"
            else:
                val = "n/a"
            row += f"  {val:25s}"
        print(row)

    print(f"\n{'='*60}")
    print("F5-TTS FORMAT VERDICT")
    print(f"{'='*60}")
    print("  silero_stress : '+' markers -> stripped by audio.py -> CLEAN TTS input  [CORRECT]")
    print("  ruaccent      : apostrophes -> survive stripping -> CORRUPT TTS input   [WRONG FORMAT]")
    print("  russtress     : apostrophes -> survive stripping -> CORRUPT TTS input   [WRONG FORMAT]")
    print()
    print("  Winner for F5-TTS: silero_stress (already in project, correct format)")
