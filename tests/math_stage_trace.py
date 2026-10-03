"""Contiguous formula trace through the FULL Deep Research pipeline:

  SOURCE(ar5iv) -> EXTRACTION(trafilatura) -> BRIEFING(LLM) -> CLUSTERING
  -> CLAIM-MERGE -> SYNTHESIS(LLM) -> FINAL REPORT

For each canonical LaTeX token it shows presence at every stage and pinpoints the
exact stage where a token first disappears. Loss at a DETERMINISTIC stage
(extraction/clustering/claim-merge) is a real bug and fails the run; loss at an
LLM stage (briefing/synthesis) is reported (governed by the math-fidelity prompt)
but not auto-failed since the model can legitimately omit a fact. Also scans the
final report for replacement chars (U+FFFD mojibake) and runs audit_math.

Needs LM Studio's LLM loaded. Run with PYTHONIOENCODING=utf-8.
Run: ./venv/Scripts/python.exe tests/math_stage_trace.py
"""
import sys, os, io, threading
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass
import logging
logging.disable(logging.INFO)
from models import Context
import deep_research as dr
import clustering as _cl
import claim_merge as _cm
import formula_audit as _fa

MODEL = os.getenv("DR_TRACE_MODEL", "HauhauCS/Qwen9B-NO-THINK")


def make_ctx():
    ctx = Context.__new__(Context)
    ctx.model_name = MODEL
    ctx.no_think = True
    ctx.last_api_call_time = 0.0
    ctx.api_min_interval = 0.0
    ctx.api_lock = threading.Lock()
    ctx.cancel_event = threading.Event()
    ctx.stage_callback = None
    return ctx


# Real papers + canonical LaTeX tokens that must survive every stage.
PAPERS = [
    ("Attention 1706.03762", "https://arxiv.org/abs/1706.03762",
     [r"\mathrm{softmax}", r"\frac{QK^{T}}{\sqrt{d_{k}}}", r"\sqrt{d_{k}}", r"QK^{T}"]),
    ("Adam 1412.6980", "https://arxiv.org/abs/1412.6980",
     [r"\beta", r"\hat", r"\sqrt", r"\epsilon"]),
]


class _P:
    def __init__(self):
        self.stats = {}

    def update(self, *a, **k):
        pass


def presence(text, tokens):
    return {t: (t in (text or "")) for t in tokens}


def main():
    ctx = make_ctx()
    deterministic_ok = True
    mojibake_found = False
    for name, url, tokens in PAPERS:
        print("=" * 78)
        print("PAPER:", name)
        print("=" * 78)
        src_html = dr._fetch_ar5iv(url)
        src = presence(src_html, tokens)
        extracted = dr._extract_text(src_html, url) if src_html else ""
        ext = presence(extracted, tokens)
        page = {"domain": "arxiv.org", "url": url, "title": name,
                "text": extracted[:dr.DR_PDF_TEXT_CHARS]}
        raw_brief = dr.brief_source(ctx, name, page) or ""
        brief_body = dr._parse_trust(raw_brief)[1] if raw_brief else ""
        brief = presence(brief_body, tokens)
        briefs = [{"domain": "arxiv.org", "url": url, "title": name, "brief": brief_body,
                   "trust": "PRIMARY", "mode": "confirm", "cluster_size": 1,
                   "cluster_domains": ["arxiv.org"]}]
        _cl.cluster_briefs(briefs, threshold=0.15)  # indexes, must not mutate brief text
        clustered = presence(" ".join(b["brief"] for b in briefs), tokens)
        merged, _log = _cm.merge_claims(briefs)
        mtext = " ".join((m.get("brief") or m.get("text") or "") for m in merged)
        merge = presence(mtext, tokens)
        report = dr.synthesize_report(ctx, name, briefs, _P(), max_tokens=1200) or ""
        syn = presence(report, tokens)

        stages = [("SOURCE", src), ("EXTRACTION", ext), ("BRIEFING", brief),
                  ("CLUSTERING", clustered), ("CLAIM-MERGE", merge), ("SYNTHESIS", syn)]
        for tok in tokens:
            row = " ".join(f"{s}={'Y' if p[tok] else 'n'}" for s, p in stages)
            lost_at = None
            if src[tok]:
                for s, p in stages:
                    if not p[tok]:
                        lost_at = s
                        break
            flag = "" if lost_at is None else f"  <-- LOST AT {lost_at}"
            if lost_at in ("EXTRACTION", "CLUSTERING", "CLAIM-MERGE"):
                deterministic_ok = False  # deterministic-stage loss = real bug
            print(f"  {tok:30s} {row}{flag}")
        audit = dr.audit_math(report)
        ufffd = chr(0xFFFD) in report
        if ufffd:
            mojibake_found = True
        print(f"  >> final report len={len(report)} mojibake(U+FFFD)={ufffd} "
              f"audit_clean={audit['clean']} lost_backslash={audit['suspect_lost_backslash']}")

        # --- structured per-stage formula profiles + corruption pinpointing ---
        texts = [("SOURCE", src_html or ""), ("EXTRACTION", extracted),
                 ("BRIEFING", brief_body), ("CLAIM-MERGE", mtext), ("SYNTHESIS", report)]
        profiles = [(s, _fa.formula_profile(t)) for s, t in texts]
        print("  per-stage formula profile (frac/matrix/int/sum/greek/accents | "
              "ufffd/mojibake/lostbslash/clean):")
        for s, p in profiles:
            c = p["categories"]
            print(f"    {s:12s} frac={c['fractions']} mat={c['matrices']} int={c['integrals']} "
                  f"sum={c['summations']} grk={c['greek']} acc={c['accents']} | "
                  f"u={p['ufffd']} moji={p['mojibake']} lb={len(p['lost_backslash'])} clean={p['clean']}")
        # hallucination check: tokens in SYNTHESIS absent from BRIEFING/EXTRACTION
        ext_p = dict(profiles)["EXTRACTION"]
        brief_p = dict(profiles)["BRIEFING"]
        syn_p = dict(profiles)["SYNTHESIS"]
        seen_cmds = set(ext_p["cmd_counts"]) | set(brief_p["cmd_counts"])
        introduced = sorted(t for t in syn_p["cmd_counts"] if t not in seen_cmds)
        if introduced:
            print(f"  ⚠ SYNTHESIS introduced LaTeX tokens not seen upstream "
                  f"(candidate hallucination): {introduced}")
        # corruption that first APPEARS at a deterministic stage = hard fail
        for (s_prev, p_prev), (s_cur, p_cur) in zip(profiles, profiles[1:]):
            d = _fa.diff_profiles(p_prev, p_cur)
            if d["corruption_appeared"] and s_cur in ("EXTRACTION", "CLAIM-MERGE"):
                print(f"  ✗ corruption first appeared at deterministic stage {s_cur}: "
                      f"{d['corruption_appeared']}")
                deterministic_ok = False
    print("=" * 78)
    ok = deterministic_ok and not mojibake_found
    print("DETERMINISTIC-STAGE FIDELITY:", "OK" if deterministic_ok else "CORRUPTION FOUND")
    print("NO MOJIBAKE IN REPORTS:", "OK" if not mojibake_found else "U+FFFD FOUND")
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
