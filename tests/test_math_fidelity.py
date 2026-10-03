"""Mathematical / LaTeX / Unicode FIDELITY through the Deep Research pipeline.

Release-blocking requirement: a formula present in a source brief must survive every
DETERMINISTIC pipeline stage byte-for-byte — no lost backslashes, no escaping damage,
no Unicode corruption, no truncation. The stages that touch source text without an LLM
are: dedupe → clustering → contradiction partition → claim-merge → graph → report
post-process → JSON state round-trip.

(The LLM stages — per-source briefing and final synthesis — cannot be byte-asserted;
their fidelity is governed by the prompts, tested separately/live. This file pins the
deterministic spine so corruption there can never silently reappear.)

Run: ./venv/Scripts/python.exe tests/test_math_fidelity.py
"""
import os
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import deep_research as dr
import clustering as _cluster
import claim_merge as _cmerge
import research_graph as _rgraph
import contradiction as _contra


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


class _MiniProg:
    """Minimal _Progress stand-in for offline _brief_pages tests."""
    def update(self, *a, **k):
        pass


# Every requested math case, as it would appear in a faithful brief.
FORMULAS = {
    "inline": r"The mass-energy relation is $E = mc^2$ per (arxiv.org).",
    "block": r"$$\int_0^\infty e^{-x^2}\,dx = \frac{\sqrt{\pi}}{2}$$",
    "multiline": ("\\begin{align}\n"
                  "\\nabla \\cdot \\mathbf{E} &= \\frac{\\rho}{\\varepsilon_0} \\\\\n"
                  "\\nabla \\times \\mathbf{B} &= \\mu_0 \\mathbf{J} + \\mu_0\\varepsilon_0 "
                  "\\frac{\\partial \\mathbf{E}}{\\partial t}\n"
                  "\\end{align}"),
    "matrix": r"$A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$",
    "fraction": r"The gradient step is $x_{t+1} = x_t - \eta \frac{\partial L}{\partial x_t}$.",
    "integral_sum": r"$\sum_{i=1}^{n} w_i x_i \le \int_a^b f(x)\,dx$",
    "greek": r"Parameters $\alpha, \beta, \gamma, \Sigma, \Omega, \theta, \lambda, \mu$ govern it.",
    "subsup": r"The tensor $T^{\mu\nu}_{\alpha\beta}$ has components $g_{ij} x^i x^j$.",
    "unicode": "Unicode math: ∑ ∫ √ π ≤ ≥ ≈ ≠ ∇ ⊗ ∂ ∞ α β γ θ λ μ Ω · × ± → ∈ ℝⁿ",
    "cite_mixed": r"Per (openalex.org), the loss $\mathcal{L}(\theta)=-\sum_k y_k\log\hat{y}_k$ is convex (claim).",
    "softmax": r"$\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$",
    "table_eq": ("| Method | Update rule |\n"
                 "| :-- | :-- |\n"
                 "| SGD | $\\theta \\leftarrow \\theta - \\eta \\nabla_\\theta L$ |\n"
                 "| Adam | $m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t$ |"),
}


def _brief(domain, text, trust="PRIMARY", mode="confirm"):
    return {"domain": domain, "url": f"https://{domain}/p{abs(hash(text))%9999}",
            "title": f"{domain} paper", "brief": text, "trust": trust, "mode": mode,
            "authority": 1.0, "cluster_size": 1, "cluster_domains": [domain]}


def _all_present(haystack, formulas):
    missing = [k for k, v in formulas.items() if v not in haystack]
    return missing


def test_shingles_do_not_mutate_source():
    """_shingles is a COMPARISON helper; it must never be written back over the brief."""
    txt = FORMULAS["block"]
    _ = dr._shingles(txt)              # compute (lowercases internally)
    _check(FORMULAS["block"] == txt, "calling _shingles mutated its input")
    print("PASS: _shingles does not mutate the source string")
    return True


def test_dedupe_preserves_formulas():
    pages = [{"url": f"https://arxiv.org/{i}", "domain": "arxiv.org",
              "title": "p", "text": v, "cluster_size": 1, "cluster_domains": ["arxiv.org"]}
             for i, v in enumerate(FORMULAS.values())]
    out = dr.dedupe_pages(pages, "science")
    kept = "\n".join(p["text"] for p in out)
    missing = _all_present(kept, FORMULAS)
    _check(not missing, f"dedupe lost/altered formulas: {missing}")
    print(f"PASS: dedupe preserves all {len(FORMULAS)} formulas byte-for-byte")
    return True


def test_clustering_preserves_formulas():
    briefs = [_brief(f"src{i}.org", v) for i, v in enumerate(FORMULAS.values())]
    cs = _cluster.cluster_briefs(briefs, threshold=0.18, major_min_size=2)
    # the cluster set indexes the ORIGINAL briefs; text must be untouched
    seen = "\n".join(b["brief"] for b in cs.briefs)
    missing = _all_present(seen, FORMULAS)
    _check(not missing, f"clustering altered formulas: {missing}")
    print("PASS: clustering keeps brief formulas intact (indices, not copies)")
    return True


def test_claim_merge_preserves_representative_formula():
    # two near-identical PRIMARY briefs with the SAME formula -> merged; the
    # representative's LaTeX must be byte-identical (not re-rendered).
    eq = FORMULAS["softmax"]
    briefs = [_brief("a.org", f"Softmax is defined as {eq} and is used widely."),
              _brief("b.org", f"Softmax is defined as {eq} and is used widely."),
              _brief("c.org", FORMULAS["block"])]
    merged, log = _cmerge.merge_claims(briefs, threshold=0.55)
    alltext = "\n".join(b["brief"] for b in merged)
    _check(eq in alltext, "claim-merge corrupted the representative formula")
    _check(FORMULAS["block"] in alltext, "claim-merge dropped a distinct formula")
    # provenance preserved, never invents math
    print(f"PASS: claim-merge keeps representative LaTeX verbatim ({len(log)} merge(s))")
    return True


def test_graph_preserves_claim_text():
    briefs = [_brief(f"s{i}.org", v) for i, v in enumerate(FORMULAS.values())]
    g = _rgraph.build_graph(briefs)
    claim_text = "\n".join(n.get("text", "") for n in g.nodes.values()
                           if n.get("type") == "claim")
    missing = _all_present(claim_text, FORMULAS)
    _check(not missing, f"graph claim nodes altered formulas: {missing}")
    print("PASS: graph stores claim text (formulas) verbatim")
    return True


def test_contradiction_partition_preserves():
    briefs = [_brief("a.org", FORMULAS["integral_sum"], mode="confirm"),
              _brief("b.org", FORMULAS["matrix"], mode="contradict")]
    _contra.tag_contradicting([b for b in briefs if b["mode"] == "contradict"])
    confirm, contra = _contra.partition_evidence(briefs)
    text = "\n".join(b["brief"] for b in confirm + contra)
    _check(FORMULAS["integral_sum"] in text and FORMULAS["matrix"] in text,
           "contradiction partition altered formulas")
    print("PASS: contradiction partition preserves confirm + contradict formulas")
    return True


def test_postprocess_report_preserves_body_math():
    """_postprocess_report only rewrites HEADER lines (for dedup); body math — incl.
    $$...$$, matrices, multi-line align, tables — must pass through byte-for-byte."""
    body = "\n\n".join(FORMULAS.values())
    md = (f"# Executive Summary\nIntro.\n\n{body}\n\n"
          f"# Detailed Analysis\nMore.\n\n# Detailed Analysis\nDUP should be dropped.\n")
    out = dr._postprocess_report(md)
    missing = _all_present(out, FORMULAS)
    _check(not missing, f"post-process mangled body math: {missing}")
    _check(out.count("# Detailed Analysis") == 1, "dedup did not drop the duplicate header")
    print("PASS: report post-process preserves ALL body math + still dedupes headers")
    return True


def test_json_state_roundtrip_preserves():
    """Briefs are persisted to state/history/graph as JSON. A round-trip must return
    backslashes and Unicode unchanged (the classic '\\f -> formfeed' / '\\sqrt lost'
    corruption must NOT happen)."""
    d = tempfile.mkdtemp()
    from pathlib import Path
    briefs = [_brief(f"s{i}.org", v) for i, v in enumerate(FORMULAS.values())]
    dr._save_state(Path(d), phase="briefing", briefs=briefs)
    loaded = json.load(open(os.path.join(d, "state.json"), encoding="utf-8"))
    text = "\n".join(b["brief"] for b in loaded["briefs"])
    missing = _all_present(text, FORMULAS)
    _check(not missing, f"JSON round-trip corrupted formulas: {missing}")
    # explicit backslash + unicode integrity
    _check(r"\frac" in text and r"\begin{pmatrix}" in text, "backslash commands lost in JSON")
    _check("∑" in text and "∫" in text and "π" in text and "ℝⁿ" in text,
           "Unicode math symbols corrupted in JSON")
    # ensure_ascii=False keeps real UTF-8 (not \uXXXX) on disk
    raw = open(os.path.join(d, "state.json"), encoding="utf-8").read()
    _check("∑" in raw, "state.json escaped Unicode instead of writing real UTF-8")
    print("PASS: JSON state/history round-trip preserves backslashes + Unicode math")
    return True


def test_full_deterministic_spine():
    """End-to-end deterministic spine: brief -> dedupe-as-pages -> cluster -> merge ->
    graph -> (assemble a report body) -> post-process. The formula set must be intact
    at the end (the same guarantee a real run relies on between briefing and synthesis)."""
    briefs = [_brief(f"s{i}.org", v, trust="PRIMARY") for i, v in enumerate(FORMULAS.values())]
    briefs += [_brief(f"s{i}.org", v, trust="PRIMARY") for i, v in enumerate(FORMULAS.values())]  # dups
    merged, _ = _cmerge.merge_claims(briefs, threshold=0.55)
    g = _rgraph.build_graph(merged)
    cs = _cluster.cluster_briefs(merged, threshold=0.18, major_min_size=2)
    body = "\n\n".join(b["brief"] for b in cs.briefs)
    report = dr._postprocess_report(f"# Detailed Analysis\n{body}\n")
    missing = _all_present(report, FORMULAS)
    _check(not missing, f"deterministic spine lost formulas end-to-end: {missing}")
    print("PASS: full deterministic spine (merge→graph→cluster→postprocess) is formula-faithful")
    return True


def test_math_audit_guard():
    """audit_math flags the corruption modes markdown/LaTeX can't survive: a clean
    formula report passes; an unbalanced $ or a stripped backslash is caught."""
    clean = ("# A\nThe loss $\\mathcal{L}=\\sum_i (y_i-\\hat{y}_i)^2$ and\n"
             "$$\\int_0^1 x\\,dx = \\frac{1}{2}$$\nare correct.\n")
    a = dr.audit_math(clean)
    _check(a["clean"], f"clean math report flagged: {a}")
    _check(a["math_spans"] >= 2, "did not detect the math spans")
    # unbalanced inline $
    bad1 = dr.audit_math("The value $E = mc^2 is unterminated.\n")
    _check(not bad1["clean"] and not bad1["inline_balanced"], "missed unbalanced $")
    # unbalanced block $$
    bad2 = dr.audit_math("$$\\frac{a}{b}\nno close\n")
    _check(not bad2["clean"] and not bad2["blocks_balanced"], "missed unbalanced $$")
    # lost backslash inside math (frac without backslash)
    bad3 = dr.audit_math("Here $x = frac{a}{b}$ has a stripped backslash.\n")
    _check(not bad3["clean"] and "frac" in bad3["suspect_lost_backslash"],
           f"missed lost-backslash: {bad3}")
    # a faithful report built from our FORMULAS set is clean
    body = "\n\n".join(FORMULAS.values())
    af = dr.audit_math(f"# R\n{body}\n")
    _check(af["clean"], f"faithful formula set flagged as broken: {af}")
    print("PASS: audit_math passes clean math, catches unbalanced $/$$ + lost backslash")
    return True


def test_ar5iv_mathml_inlining():
    """PDF text extraction has no concept of 2D formula layout: it flattens
    fractions/matrices into glyph soup with no math markup at all (verified live
    on arxiv.org/pdf/1706.03762 -> "QK T\\n(sqrt)dk", no $ delimiters, no \\frac).
    ar5iv.org renders the SAME paper as HTML where every <math> node carries the
    original LaTeX in its alttext attribute -- strictly more faithful. This pins
    the HTML rewrite that recovers it, including the failure paths."""
    _check(dr._ar5iv_url("https://arxiv.org/pdf/1706.03762") ==
           "https://ar5iv.org/abs/1706.03762", "id extraction/rewrite wrong")
    _check(dr._ar5iv_url("https://arxiv.org/abs/1706.03762v5") is not None,
           "should handle versioned ids")
    _check(dr._ar5iv_url("https://nature.com/articles/foo") is None,
           "non-arxiv url must not be rewritten")

    html = (r'<p>matrices <math alttext="K" display="inline">junk</math> and '
            r'<math alttext="\mathrm{Attention}(Q,K,V)=\mathrm{softmax}('
            r'\frac{QK^{T}}{\sqrt{d_{k}}})V" display="block">junk2</math> end</p>')
    out = dr._inline_mathml_as_tex(html)
    _check(r"$\mathrm{Attention}" in out and r"\frac{QK^{T}}{\sqrt{d_{k}}}" in out,
           f"block formula with backslashes lost: {out}")
    _check("$$" in out, "block math must use $$ delimiters")
    _check(" $K$ " in out, "inline formula not recovered")

    # a <math> node with no recoverable alttext is dropped, not left as raw MathML
    no_alt = "<p>x <math><mi>a</mi></math> y</p>"
    _check("<mi>" not in dr._inline_mathml_as_tex(no_alt), "raw MathML leaked through")

    # HTML entities inside alttext (e.g. &lt;) must be unescaped, not double-encoded
    entity = r'<math alttext="a &lt; b" display="inline">j</math>'
    _check("$a < b$" in dr._inline_mathml_as_tex(entity), "entity not unescaped")
    print("PASS: ar5iv MathML->LaTeX inlining recovers formulas pypdf flattens")
    return True


def test_bare_math_risk_flag():
    """audit_math alone can't catch a PDF that lost ALL markup (0 $ trivially
    'balances'). _looks_like_unmarked_math is the extraction-time companion check,
    confirmed live on a real NeurIPS PDF whose attention-mechanism section came
    back as plain prose with zero LaTeX (the fraction QK^T/sqrt(d_k) flattened to
    bare text). It must flag math-heavy unmarked text, and stay silent on plain
    prose or text that already carries proper $...$ markup."""
    mathy_unmarked = ("We compute the softmax of the matrix product, take the "
                       "gradient, and check the eigenvalue of the tensor ∑ term.")
    _check(dr._looks_like_unmarked_math(mathy_unmarked),
           "missed a math-heavy PDF text with zero LaTeX markup")
    plain = "This is a news article about elections and policy with no mathematics."
    _check(not dr._looks_like_unmarked_math(plain), "false positive on plain prose")
    marked = "The softmax matrix $A$ has eigenvalue $\\lambda$ and gradient $\\nabla f$."
    _check(not dr._looks_like_unmarked_math(marked),
           "false positive on text that already has proper LaTeX markup")
    print("PASS: bare-math-risk flag catches unmarked PDF math, ignores prose/marked text")
    return True


def test_formula_audit_profile_and_diff():
    """formula_audit.formula_profile categorizes math structure and detects
    corruption; diff_profiles flags hallucination (new tokens), loss, and
    corruption-appeared. REGRESSION: \\sum_i / \\int_0 (command + subscript) MUST
    be counted — the original \\b boundary failed here because '_' is a regex
    word-char (a real bug found 2026-06-20)."""
    import formula_audit as fa
    p = fa.formula_profile(
        r"$\theta_t=\theta_{t-1}-\frac{\alpha\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$ "
        r"and $$\mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})$$ with $\sum_i x_i$, "
        r"$\int_0^1 f\,dx$, matrix $\begin{pmatrix}a&b\end{pmatrix}$, and ∇∫√π.")
    c = p["categories"]
    _check(p["clean"], f"clean formula text flagged: {p}")
    _check(c["fractions"] >= 2, "fractions undercounted")
    _check(c["matrices"] == 1, "matrix not counted")
    _check(c["summations"] == 1, f"REGRESSION: \\sum_i not counted (got {c['summations']})")
    _check(c["integrals"] >= 1, f"REGRESSION: \\int_0 not counted (got {c['integrals']})")
    _check(c["accents"] >= 1 and c["greek"] >= 1, "accents/greek undercounted")

    # corruption detection
    bad = "broken � and $x = frac{a}{b}$ and mojibake Ã©Ã¨"
    pb = fa.formula_profile(bad)
    _check(not pb["clean"] and pb["ufffd"] == 1 and pb["mojibake"] >= 1
           and "frac" in pb["lost_backslash"], f"corruption not detected: {pb}")

    # diff: hallucinated tokens, lost tokens, corruption-appeared
    src = fa.formula_profile(r"$\frac{a}{b}$ and $\alpha$")
    halluc = fa.diff_profiles(src, fa.formula_profile(r"$\frac{a}{b}$ $\alpha$ NEW $\int \beta$"))
    _check(r"\int" in halluc["introduced_cmd_tokens"] and r"\beta" in halluc["introduced_cmd_tokens"],
           "hallucination (new tokens) not flagged")
    lost = fa.diff_profiles(src, fa.formula_profile(r"only $\alpha$ remains"))
    _check(r"\frac" in lost["lost_cmd_tokens"], "lost token not flagged")
    corr = fa.diff_profiles(src, fa.formula_profile("now � and $x=frac{a}{b}$"))
    _check("ufffd" in corr["corruption_appeared"] and "lost_backslash" in corr["corruption_appeared"],
           "corruption-appeared not flagged")
    print("PASS: formula_audit profile (sum_i/int_0 counted) + hallucination/loss/corruption diff")
    return True


def test_equation_preservation():
    """ROOT-CAUSE FIX: the stage trace proved display equations are present at
    SOURCE+EXTRACTION but LOST AT BRIEFING (the 9B paraphrases them away). So we
    extract raw equation blocks at extraction, carry them on each brief, and
    surface them verbatim+attributed in the report via _build_equations_section —
    guaranteeing equations reach the report even when the LLM drops them."""
    import formula_audit as fa
    src = (r"Intro text. $$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}"
           r"(\frac{QK^{T}}{\sqrt{d_{k}}})V$$ and an aligned block "
           r"\begin{align} \nabla_\theta L &= \sum_i g_i \end{align} end.")
    eqs = fa.extract_equations(src)
    _check(any("softmax" in e and r"\frac{QK^{T}}{\sqrt{d_{k}}}" in e for e in eqs),
           f"display equation not preserved verbatim: {eqs}")
    _check(any("align" in e and r"\nabla_\theta L" in e for e in eqs),
           "align environment not preserved")
    # prose-only $$ pairs are NOT captured as equations (no LaTeX/operators)
    _check(fa.extract_equations("$$just prose here$$") == [], "captured non-math $$ block")

    # brief dropped the LaTeX (prose summary) but equations carried on the brief
    # must still appear, verbatim and attributed, in the report section.
    briefs = [{"domain": "arxiv.org", "trust": "PRIMARY",
               "brief": "Attention is a weighted sum (no LaTeX in this summary).",
               "equations": eqs}]
    sec = dr._build_equations_section(briefs)
    _check("Key equations" in sec and "verbatim" in sec, "section header missing")
    _check(r"\frac{QK^{T}}{\sqrt{d_{k}}}" in sec, "equation not in report section")
    _check("arxiv.org" in sec, "source attribution missing")
    # dedupe across briefs
    sec2 = dr._build_equations_section(briefs + [dict(briefs[0])])
    _check(sec2.count(r"\mathrm{Attention}") == 1, "duplicate equations not deduped")
    # empty when nothing preserved
    _check(dr._build_equations_section([{"domain": "x", "brief": "y"}]) == "",
           "section should be empty with no equations")
    print("PASS: equation preservation (extract verbatim + carry + report section + dedupe)")
    return True


def test_claim_merge_unions_equations():
    """RELEASE-BLOCKING (found in the sci-fidelity audit 2026-06-20): when two
    near-duplicate briefs merge, the representative kept ONLY its own equations and
    silently dropped the folded brief's unique equation — equation loss at the
    claim-merge stage. Fix: union preserved equations across the whole group."""
    import claim_merge as cm
    briefs = [
        {"url": "a", "domain": "arxiv.org", "trust": "PRIMARY", "mode": "confirm",
         "brief": "Batch norm normalizes activations to reduce internal covariate shift.",
         "equations": [r"$$\hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}$$"]},
        {"url": "b", "domain": "blog.io", "trust": "COMMUNITY", "mode": "confirm",
         "brief": "Batch norm normalizes activations to reduce internal covariate shift effects.",
         "equations": [r"$$y=\gamma\hat{x}+\beta$$"]},
    ]
    merged, _log = cm.merge_claims(briefs, threshold=0.4)
    _check(len(merged) == 1, f"near-duplicates should merge to 1 (got {len(merged)})")
    eqs = merged[0].get("equations", [])
    _check(any(r"\hat{x}=\frac" in e for e in eqs), "representative's own equation lost")
    _check(any(r"y=\gamma\hat{x}+\beta" in e for e in eqs),
           f"REGRESSION: folded brief's unique equation dropped at claim-merge: {eqs}")
    # dedupe: identical equations in both briefs collapse to one
    same = [dict(b, equations=[r"$$E=mc^2$$"]) for b in briefs]
    m2, _ = cm.merge_claims(same, threshold=0.4)
    _check(m2[0]["equations"].count(r"$$E=mc^2$$") == 1, "duplicate equations not deduped on merge")
    print("PASS: claim-merge unions preserved equations across the group (no stage loss)")
    return True


def test_equations_survive_page_truncation():
    """RELEASE-BLOCKING (found in the GUI live LoRA run 2026-06-20): equation
    preservation ran on page['text'], which collect_sources truncates to
    DR_PAGE_CHARS (6000) for the LLM map step. A paper's key display equations sit
    DEEP in the body (LoRA's first $$...$$ is at char ~12.4k), so the truncated head
    had ZERO equations -> brief.equations empty -> no equations reached the report,
    even though ar5iv extraction recovered them perfectly. The sci-audit masked this
    by extracting from FULL text. Fix: extract equations from the full document at
    collection time (page['equations']) BEFORE truncation; _brief_pages prefers it."""
    import formula_audit as fa
    # full document: prose head longer than DR_PAGE_CHARS, then a deep display eq.
    deep_eq = r"$$h=W_{0}x+\Delta Wx=W_{0}x+BAx$$"
    # The head must be DERIVED from DR_PAGE_CHARS, not a hardcoded repeat count:
    # this was `* 600` (10.2k chars), which stopped exceeding the limit the moment
    # DR_PAGE_CHARS was raised from 6000 to 32000 — the setup then built a document
    # that was never truncated, so the test failed on its own premise instead of
    # exercising the regression.
    _filler = "LoRA background. "
    _reps = (dr.DR_PAGE_CHARS // len(_filler)) + 100
    full = (_filler * _reps) + "\n" + deep_eq + "\nmore text.\n"
    _check(len(full) > dr.DR_PAGE_CHARS, "test setup: head not longer than truncation")
    truncated = full[: dr.DR_PAGE_CHARS]
    # the bug: extracting from the truncated head finds nothing...
    _check(fa.extract_equations(truncated) == [], "test premise wrong: eq within head")
    # ...but the fix extracts from the full text at collection time.
    full_eqs = fa.extract_equations(full)
    _check(any(r"\Delta Wx" in e for e in full_eqs), f"full-text extract missed eq: {full_eqs}")

    # End-to-end through _brief_pages with a stubbed LLM (offline): a page whose
    # text is truncated but which carries pre-extracted equations must surface the
    # deep equation on its brief, and thus in the report section.
    saved = dr.brief_source
    dr.brief_source = lambda ctx, topic, page: "SOURCE: PRIMARY\n- LoRA adds low-rank update."
    try:
        page = {"url": "https://arxiv.org/abs/2106.09685", "domain": "arxiv.org",
                "title": "LoRA", "text": truncated, "equations": full_eqs}
        briefs, _ = dr._brief_pages(
            None, "LoRA", [page], {"category": "scholarly"}, None,
            _MiniProg(), scholarly=False, terms=[])
    finally:
        dr.brief_source = saved
    _check(briefs and any(r"\Delta Wx" in e for e in briefs[0].get("equations", [])),
           f"deep equation lost despite pre-extraction: {briefs and briefs[0].get('equations')}")
    sec = dr._build_equations_section(briefs)
    _check(r"\Delta Wx" in sec and "arxiv.org" in sec,
           "deep equation did not reach the report section")
    print("PASS: equations survive page truncation (full-text extract + brief + report)")
    return True


def test_per_stage_formula_trace():
    """Release-blocking auditability: build_trace records, per stage, token/LaTeX
    counts + Unicode + missing/newly-introduced symbols + corruption, and pinpoints
    the exact stage where a symbol first disappears or reappears."""
    import formula_audit as fa
    snaps = [
        ("EXTRACTION", r"$$\frac{QK^T}{\sqrt{d_k}}$$ and $\sum_i x_i$ with ∇"),
        ("BRIEFING", "prose summary, the LLM dropped all the math"),
        ("SYNTHESIS", "still prose"),
        ("FINAL_REPORT", r"## Key equations $$\frac{QK^T}{\sqrt{d_k}}$$"),
    ]
    tr = fa.build_trace(snaps)
    _check([r["stage"] for r in tr] == ["EXTRACTION", "BRIEFING", "SYNTHESIS", "FINAL_REPORT"],
           "stage order wrong")
    ext = tr[0]
    _check(ext["latex_token_count"] >= 2 and "∇" in ext["unicode_symbols"], "extraction profile wrong")
    _check(ext["categories"]["fractions"] >= 1 and ext["categories"]["summations"] >= 1,
           "category counts wrong at extraction")
    brief = next(r for r in tr if r["stage"] == "BRIEFING")
    _check(r"\frac" in brief["missing_symbols"] and "∇" in brief["missing_symbols"],
           f"loss at BRIEFING not pinpointed: {brief['missing_symbols']}")
    final = tr[-1]
    _check(r"\frac" in final["newly_introduced_symbols"],
           "equation reintroduction at FINAL not recorded")
    # every record carries the full required field set
    for r in tr:
        for f in ("token_count", "latex_token_count", "unicode_symbols", "categories",
                  "corruption", "missing_symbols", "newly_introduced_symbols"):
            _check(f in r, f"trace record missing field {f}")
    print("PASS: per-stage formula trace (counts + missing/introduced + corruption pinpointed)")
    return True


if __name__ == "__main__":
    tests = [
        test_math_audit_guard,
        test_ar5iv_mathml_inlining,
        test_bare_math_risk_flag,
        test_formula_audit_profile_and_diff,
        test_equation_preservation,
        test_claim_merge_unions_equations,
        test_equations_survive_page_truncation,
        test_per_stage_formula_trace,
        test_shingles_do_not_mutate_source,
        test_dedupe_preserves_formulas,
        test_clustering_preserves_formulas,
        test_claim_merge_preserves_representative_formula,
        test_graph_preserves_claim_text,
        test_contradiction_partition_preserves,
        test_postprocess_report_preserves_body_math,
        test_json_state_roundtrip_preserves,
        test_full_deterministic_spine,
    ]
    results = []
    for t in tests:
        try:
            results.append(bool(t()))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"FAIL: {t.__name__}: {e}")
            results.append(False)
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
