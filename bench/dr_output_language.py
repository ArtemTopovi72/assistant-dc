"""Does the MODEL obey the language directive? (live LM Studio)

Split out of tests/test_dr_output_language.py, which keeps the offline half:
that the directive is built and SENT. Only a real generation shows it is
OBEYED, and that needs the live model, so it belongs in bench/ under the house
rule that nothing in tests/ may call LM Studio or ComfyUI.

The split also removes a trap. With the services down this half quietly
skipped; with a model loaded it ran inside run_all -- starting GPU work of its
own, beside whatever else had the card -- and then failed on an assertion about
the model's exact wording (a particular English phrase must not appear in the
Russian draft). No care makes that deterministic. In a bench run, judging
wording is the point.

Run: venv/Scripts/python.exe bench/dr_output_language.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

import threading  # noqa: E402
import config  # noqa: E402
import deep_research as D  # noqa: E402

_n = _bad = 0


def check(name, cond, detail=""):
    global _n, _bad
    _n += 1
    if cond:
        print("  ok   " + name)
    else:
        _bad += 1
        print("  FAIL " + name + (("\n     " + str(detail)) if detail else ""))


def cyr_ratio(text: str) -> float:
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "\u0400" <= c <= "\u04ff") / len(letters)


# ═════════════════════════════════════════════════════════════════════ LIVE
# Plumbing proves the directive is SENT. Only the model proves it is OBEYED.
print("\nTHE MODEL ACTUALLY WRITES RUSSIAN (live LM Studio)")
try:
    import requests
    served = [m["id"] for m in requests.get(
        config.LM_STUDIO_URL.replace("/chat/completions", "/models"),
        timeout=10).json()["data"]]
except Exception as exc:
    print(f"  SKIP live half: LM Studio is not answering ({exc})")
    served = []

# /models answers from the catalogue and returns instantly even when the GPU is
# saturated — so this gate used to open, synthesize_report would then block on a
# server that had no capacity, and the whole suite died on a 150s timeout with
# NO result at all. Prove the server can actually GENERATE before committing to
# a full report: one token, short deadline. A busy box now skips honestly in
# seconds instead of reporting three phantom failures.
if served:
    _probe_model = config.MODEL_NAME if config.MODEL_NAME in served else served[0]
    try:
        _r = requests.post(
            config.LM_STUDIO_URL,
            json={"model": _probe_model, "max_tokens": 1,
                  "messages": [{"role": "user", "content": "hi"}]},
            timeout=45)
        if _r.status_code != 200:
            raise RuntimeError(f"HTTP {_r.status_code}")
    except Exception as exc:
        print(f"  SKIP live half: {_probe_model} cannot generate right now ({exc})")
        served = []

if served:
    MODEL = config.MODEL_NAME if config.MODEL_NAME in served else served[0]
    print(f"     model under test: {MODEL}")

    class Ctx:
        def __init__(self):
            self.model_name = MODEL
            self.no_think = False
            self.reasoning_effort = "high"
            self.api_lock = threading.Lock()
            self.last_api_call_time = 0.0
            self.api_min_interval = 0.2
        def is_cancelled(self): return False
        def set_stage(self, *_a, **_k): pass

    ctx = Ctx()

    # Real English evidence — the exact condition that made the model answer in
    # English: English system prompt, English briefs, no reason to switch.
    BRIEFS = [
        {"domain": "ncbi.nlm.nih.gov", "url": "https://ncbi.nlm.nih.gov/x",
         "title": "Intestinal gas and stool density", "trust": "PRIMARY",
         "brief": "Floating stools are caused by trapped intestinal gas, not by fat "
                  "content. Gas from colonic bacterial fermentation lowers the mean "
                  "density of the stool below that of water (1.0 g/cm3)."},
        {"domain": "gut.bmj.com", "url": "https://gut.bmj.com/y",
         "title": "Flatus production in man", "trust": "PRIMARY",
         "brief": "Colonic fermentation of undigested carbohydrate produces hydrogen, "
                  "methane and carbon dioxide. Volunteers on a high-fibre diet showed "
                  "markedly increased gas volume and a higher proportion of floaters."},
    ]

    class P:
        stats = {"sources": 2, "pages": 2, "findings": 2}
        def update(self, *_a, **_k): pass

    prog = P()

    got_ru = D.synthesize_report(ctx, "почему кал не тонет в воде", BRIEFS, prog,
                                 max_tokens=900, out_lang="ru")
    r = cyr_ratio(got_ru)
    print(f"     Russian run: {len(got_ru)} chars, {r:.0%} Cyrillic")
    print(f"     opening: {got_ru.strip()[:110]!r}")
    check("the report came back non-empty", len(got_ru.strip()) > 200,
          repr(got_ru[:200]))
    check("and it is written in Russian", r > 0.8, f"only {r:.0%} Cyrillic")
    check("the English evidence was not pasted through verbatim",
          "Floating stools are caused by" not in got_ru,
          "a whole English sentence survived")
    check("no English draft was emitted alongside the Russian one",
          got_ru.lower().count("intestinal gas") == 0, got_ru[-300:])

    got_en = D.synthesize_report(ctx, "why do stools float", BRIEFS, prog,
                                 max_tokens=900)
    print(f"     English run: {len(got_en)} chars, {cyr_ratio(got_en):.0%} Cyrillic")
    check("the default run is still English", cyr_ratio(got_en) < 0.1,
          f"{cyr_ratio(got_en):.0%} Cyrillic in the English run")
    check("the English run is non-empty too", len(got_en.strip()) > 200,
          repr(got_en[:200]))

print(f"\n{_n - _bad}/{_n} checks passed")
sys.exit(1 if _bad else 0)

print(f"\n{_n - _bad}/{_n} checks passed")
sys.exit(1 if _bad else 0)
