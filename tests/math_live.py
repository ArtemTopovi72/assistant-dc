"""LIVE math-fidelity test of the LLM stages (briefing + synthesis) with the 9B.

Feeds real, verifiable scientific-paper equations through brief_source -> synthesize
and checks the formulas survive (faithful LaTeX, no lost backslashes, balanced $).
Requires LM Studio's LLM loaded. Reports source vs brief vs report for every formula.
"""
import io
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import os
import sys
import re
import threading
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass
logging.basicConfig(level=logging.WARNING)

import deep_research as dr


def make_ctx():
    from models import Context
    ctx = Context.__new__(Context)
    ctx.model_name = ""
    ctx.no_think = True
    ctx.last_api_call_time = 0.0
    ctx.api_min_interval = 0.0
    ctx.api_lock = threading.Lock()
    ctx.cancel_event = threading.Event()
    ctx.stage_callback = None
    return ctx


# Real, well-known paper equations (verifiable), as they survive HTML/ar5iv extraction.
PAGES = [
    {"domain": "arxiv.org", "url": "https://arxiv.org/abs/1412.6980", "title": "Adam optimizer",
     "text": ("Adam (Kingma & Ba, 2014) maintains exponential moving averages of the "
              "gradient and its square: $m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t$ and "
              "$v_t = \\beta_2 v_{t-1} + (1-\\beta_2) g_t^2$. The bias-corrected estimates "
              "are $\\hat{m}_t = m_t/(1-\\beta_1^t)$ and $\\hat{v}_t = v_t/(1-\\beta_2^t)$. "
              "The update is $$\\theta_t = \\theta_{t-1} - \\frac{\\alpha\\,\\hat{m}_t}"
              "{\\sqrt{\\hat{v}_t}+\\epsilon}.$$ Default $\\beta_1=0.9$, $\\beta_2=0.999$.")},
    {"domain": "arxiv.org", "url": "https://arxiv.org/abs/1706.03762", "title": "Attention",
     "text": ("Scaled dot-product attention is defined as "
              "$$\\mathrm{Attention}(Q,K,V) = \\mathrm{softmax}\\!\\left(\\frac{QK^\\top}"
              "{\\sqrt{d_k}}\\right)V,$$ where $d_k$ is the key dimension. The softmax is "
              "$\\mathrm{softmax}(z)_i = e^{z_i}/\\sum_j e^{z_j}$.")},
]
TOPIC = "Adam optimizer and scaled dot-product attention update equations"

# canonical formula fragments that MUST survive (backslash commands + structure)
EXPECT = [
    r"\beta_1 m_{t-1}", r"(1-\beta_1) g_t", r"\sqrt{\hat{v}_t}", r"\frac{\alpha",
    r"\mathrm{softmax}", r"\frac{QK^\top}{\sqrt{d_k}}", r"\sum_j e^{z_j}",
]


def _found(text, frags):
    return {f: (f in text) for f in frags}


def main():
    ctx = make_ctx()
    print("=" * 70, "\nSTAGE 1 — per-source briefing (brief_source)\n", "=" * 70)
    briefs = []
    for pg in PAGES:
        b = dr.brief_source(ctx, TOPIC, pg)
        print(f"\n--- {pg['domain']} brief ---\n{b}\n")
        briefs.append({"domain": pg["domain"], "url": pg["url"], "title": pg["title"],
                       "brief": dr._parse_trust(b or "")[1] if b else "", "trust": "PRIMARY",
                       "cluster_size": 1, "cluster_domains": [pg["domain"]]})
    brief_text = "\n".join(b["brief"] for b in briefs)
    bfound = _found(brief_text, EXPECT)
    print("BRIEF formula survival:", bfound)
    brief_audit = dr.audit_math(brief_text)
    print("BRIEF math audit:", brief_audit)

    print("\n" + "=" * 70, "\nSTAGE 2 — synthesis (synthesize_report)\n", "=" * 70)
    class _P:
        def __init__(self): self.stats = {}
        def update(self, *a, **k): pass
    report = dr.synthesize_report(ctx, TOPIC, briefs, _P(), max_tokens=1500)
    print("\n--- REPORT ---\n", report[:4000], "\n")
    rfound = _found(report or "", EXPECT)
    raudit = dr.audit_math(report or "")
    print("REPORT formula survival:", rfound)
    print("REPORT math audit:", raudit)

    # verdict
    brief_ok = sum(bfound.values())
    report_ok = sum(rfound.values())
    print("\n" + "=" * 70)
    print(f"SOURCE fragments: {len(EXPECT)}")
    print(f"survived BRIEF:   {brief_ok}/{len(EXPECT)}  audit_clean={brief_audit['clean']}")
    print(f"survived REPORT:  {report_ok}/{len(EXPECT)}  audit_clean={raudit['clean']}")
    return brief_ok, report_ok, raudit


if __name__ == "__main__":
    main()
