"""Release-blocking scientific fidelity audit over REAL papers.

For each paper this runs the ACTUAL pipeline fetch path (_fetch_page: ar5iv ->
PDF fallback), extraction, and the deterministic equation-preservation path
(extract_equations -> brief.equations -> _build_equations_section), then profiles
formula content at SOURCE/EXTRACTION/PRESERVED/REPORT-SECTION and classifies any
loss as SOURCE / EXTRACTION / PIPELINE issue. Writes a per-paper
formula_trace.json under tests/_sci_audit/<slug>/.

This exercises the fidelity-critical DETERMINISTIC path (where the release
guarantee lives) on real network data for every paper. The LLM briefing stage is
known-lossy (paraphrases equations) and is deliberately bypassed by preservation;
its behavior is covered by the full live run + tests/math_stage_trace.py.

Run with PYTHONIOENCODING=utf-8:  ./venv/Scripts/python.exe tests/sci_fidelity_audit.py
"""
import sys, os, io, json, re
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
import deep_research as dr
import formula_audit as fa

OUT = os.path.join(os.path.dirname(__file__), "_sci_audit")

# Real papers. (arxiv_id or None). None => no canonical arXiv (SOURCE-ISSUE test).
PAPERS = [
    ("Attention Is All You Need", "1706.03762"),
    ("Adam", "1412.6980"),
    ("AdamW (Decoupled Weight Decay)", "1711.05101"),
    ("BERT", "1810.04805"),
    ("Transformer-XL", "1901.02860"),
    ("LoRA", "2106.09685"),
    ("QLoRA", "2305.14314"),
    ("Batch Normalization", "1502.03167"),
    ("RMSProp (no arXiv — source-issue case)", None),
    ("SGD (no single arXiv — source-issue case)", None),
]


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def audit_paper(name, arxiv_id):
    rec = {"paper": name, "arxiv_id": arxiv_id}
    if not arxiv_id:
        rec["route"] = "NONE"
        rec["classification"] = "SOURCE ISSUE (no arXiv/ar5iv source available)"
        rec["preserved_equations"] = 0
        return rec, None
    url = f"https://arxiv.org/abs/{arxiv_id}"
    # STAGE: DOWNLOAD + which route (ar5iv HTML vs PDF)
    ar5iv = dr._fetch_ar5iv(url)
    if ar5iv is not None:
        route, src_text = "ar5iv-HTML", ar5iv
        extracted = dr._extract_text(ar5iv, url) or ""
    else:
        html, pdf = dr._fetch_page(url)
        if pdf is not None:
            route, src_text, extracted = "PDF", pdf, pdf
        elif html is not None:
            route, src_text, extracted = "HTML/abstract", html, dr._extract_text(html, url) or ""
        else:
            rec["route"] = "FETCH-FAILED"
            rec["classification"] = "SOURCE ISSUE (fetch failed)"
            rec["preserved_equations"] = 0
            return rec, None
    rec["route"] = route
    src_p = fa.formula_profile(src_text)
    ext_p = fa.formula_profile(extracted)
    preserved = fa.extract_equations(extracted)
    # report equation section from a brief whose body dropped the LaTeX (worst case)
    briefs = [{"domain": f"arxiv.org/{arxiv_id}", "url": url, "trust": "PRIMARY",
               "brief": "prose summary (LLM dropped equations)", "equations": preserved}]
    section = dr._build_equations_section(briefs)
    sec_p = fa.formula_profile(section)

    rec["source_latex_tokens"] = src_p["latex_cmd_total"]
    rec["extraction_latex_tokens"] = ext_p["latex_cmd_total"]
    rec["preserved_equations"] = len(preserved)
    rec["report_section_latex_tokens"] = sec_p["latex_cmd_total"]
    rec["categories_extraction"] = {k: v for k, v in ext_p["categories"].items() if v}
    rec["unicode_symbols"] = ext_p["unicode_symbols"]
    rec["corruption_extraction"] = {
        "ufffd": ext_p["ufffd"], "mojibake": ext_p["mojibake"],
        "lost_backslash": ext_p["lost_backslash"], "clean": ext_p["clean"]}
    # citation integrity: every preserved eq attributed to THIS paper in the section
    attribution_ok = (not preserved) or (f"arxiv.org/{arxiv_id}" in section)
    rec["citation_attribution_ok"] = attribution_ok

    # classification
    if route == "ar5iv-HTML" and ext_p["latex_cmd_total"] > 0 and preserved:
        rec["classification"] = "OK (equations preserved verbatim, attributed)"
    elif route in ("PDF",) and ext_p["latex_cmd_total"] == 0:
        rec["classification"] = "EXTRACTION ISSUE (PDF flattens math; bare-math flag applies)"
    elif route == "HTML/abstract":
        rec["classification"] = "SOURCE ISSUE (no ar5iv render; abstract page has no body equations)"
    elif src_p["latex_cmd_total"] > 0 and ext_p["latex_cmd_total"] == 0:
        rec["classification"] = "EXTRACTION ISSUE (math in source lost during extraction)"
    elif preserved:
        rec["classification"] = "OK (equations preserved)"
    else:
        rec["classification"] = "NO-EQUATIONS (none found in fetched text)"

    # build a per-paper formula_trace.json-style artifact
    trace = fa.build_trace([
        ("EXTRACTION", extracted), ("PRESERVED", "\n".join(preserved)),
        ("REPORT_SECTION", section)])
    return rec, {"paper": name, "url": url, "route": route, "trace": trace,
                 "preserved_equations": preserved[:8]}


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = []
    for name, aid in PAPERS:
        rec, artifact = audit_paper(name, aid)
        summary.append(rec)
        if artifact:
            d = os.path.join(OUT, _slug(name))
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "formula_trace.json"), "w", encoding="utf-8") as f:
                json.dump(artifact, f, ensure_ascii=False, indent=2)
        print(f"{name[:42]:42s} route={rec.get('route','?'):12s} "
              f"src_tok={rec.get('source_latex_tokens','-')} "
              f"ext_tok={rec.get('extraction_latex_tokens','-')} "
              f"preserved={rec['preserved_equations']} "
              f"attrib={rec.get('citation_attribution_ok','-')}")
        print(f"    -> {rec['classification']}")
    with open(os.path.join(OUT, "audit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    # release-blocking aggregate
    ok = [r for r in summary if r["classification"].startswith("OK")]
    print("\n" + "=" * 70)
    print(f"papers audited: {len(summary)} | equations-preserved-OK: {len(ok)}")
    print(f"artifacts: {OUT}/<slug>/formula_trace.json + audit_summary.json")
    # any attribution failure is release-blocking
    attr_fail = [r for r in summary if r.get("citation_attribution_ok") is False]
    print("citation attribution failures:", attr_fail if attr_fail else "NONE")
    return summary


if __name__ == "__main__":
    main()
