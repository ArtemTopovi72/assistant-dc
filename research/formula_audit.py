"""Formula-fidelity instrumentation for the Deep Research pipeline.

`formula_profile(text)` returns a structured fingerprint of the mathematical
content of a piece of text: per-category LaTeX structure counts, the multiset of
distinctive LaTeX command tokens, Unicode math symbols, and corruption
indicators (U+FFFD/mojibake/lost-backslash/unbalanced delimiters).

`diff_profiles(src, dst)` compares two stages and reports what was LOST, what was
NEWLY INTRODUCED (candidate hallucination), and any corruption that appeared at
`dst` but not `src` — i.e. it pinpoints the stage where degradation begins.

This is deterministic and dependency-free so it can be called at every pipeline
stage (including inside phase-history records) without cost or side effects.
"""
from __future__ import annotations

import re
from typing import Dict

# --- structure categories: name -> regex over the (single-backslash) text ----- #
# NB: LaTeX command tokens end at a NON-LETTER (e.g. \sum_i, \int_0, \frac{).
# A regex \b does NOT work here: '_' is a regex word-char, so \sum\b fails to
# match the extremely common \sum_i. Use (?![A-Za-z]) as the token boundary.
_B = r"(?![A-Za-z])"
_CATEGORY_PATTERNS: Dict[str, str] = {
    "fractions":     r"\\[dt]?frac" + _B + r"|\\cfrac" + _B,
    "matrices":      r"\\begin\{[bpvV]?matrix\}",
    "integrals":     r"\\i?int" + _B + r"|\\oint" + _B + r"|[∫∬∮]",
    "summations":    r"\\sum" + _B + r"|\\prod" + _B + r"|[∑∏]",
    "limits":        r"\\lim" + _B + r"|\\limsup" + _B + r"|\\liminf" + _B,
    "derivatives":   r"\\partial" + _B + r"|\\nabla" + _B + r"|[∂∇]|\\frac\{d",
    "subscripts":    r"(?<!\\)_",
    "superscripts":  r"(?<!\\)\^",
    "greek":         r"\\(?:alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|"
                     r"theta|vartheta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|"
                     r"upsilon|phi|varphi|chi|psi|omega|Gamma|Delta|Theta|Lambda|"
                     r"Xi|Pi|Sigma|Phi|Psi|Omega)" + _B + r"|[α-ωΑ-Ω]",
    "accents":       r"\\(?:hat|bar|tilde|vec|overline|widehat|widetilde|dot|ddot)" + _B,
    "vectors":       r"\\(?:vec|mathbf|boldsymbol|mathbb|mathcal)" + _B,
    "probability":   r"\\mathbb\{[PE]\}|\\Pr" + _B + r"|\\operatorname\{Var\}",
    "optimization":  r"\\arg\\?(?:min|max)" + _B + r"|\\min" + _B + r"|\\max" + _B + r"|\\nabla" + _B,
    "multiline":     r"\\begin\{(?:align|gather|split|cases|eqnarray|multline)\*?\}",
    "theorems":      r"\\begin\{(?:theorem|lemma|proof|definition|corollary|proposition)\}",
}

# distinctive LaTeX command tokens (\word) — used for lost/new-symbol diffing.
_CMD_RE = re.compile(r"\\[a-zA-Z]+")
# math-command words that, appearing WITHOUT a leading backslash inside a math
# span, indicate a lost backslash (corruption).
_MATH_CMDS = ("frac", "sum", "int", "sqrt", "alpha", "beta", "gamma", "delta",
              "theta", "lambda", "sigma", "nabla", "partial", "begin", "end",
              "hat", "bar", "vec", "mathbb", "mathcal", "left", "right", "cdot")
# common UTF-8-mis-decoded-as-cp1252 mojibake signatures.
_MOJIBAKE_RE = re.compile(r"Ã[\x80-\xbf©¨¤¶]|Â[\x80-\xbf°±·]|â€[\x9c\x9d\x99\x93\x94]|Ð[\x90-\xbf]|Ñ[\x80-\x8f]")
_UNICODE_MATH = "∑∏∫∬∮√∂∇αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ≤≥≠≈→←↔×÷±∞∈∉⊂⊆∪∩ℝℕℤℚℂ"
_DOLLAR = chr(36)


def _math_spans(text: str):
    """Yield the inside of $...$ and $$...$$ spans (where lost-backslash matters)."""
    return re.findall(r"\${1,2}(.+?)\${1,2}", text, flags=re.S)


_DISPLAY_EQ_RE = re.compile(
    r"\$\$(.+?)\$\$"                                   # $$ ... $$
    r"|\\begin\{(align\*?|equation\*?|aligned|gather\*?|multline\*?)\}(.+?)\\end\{\2\}",
    re.S)


def extract_equations(text: str, *, limit: int = 12) -> list:
    """Pull DISPLAY equations ($$...$$ and align/equation/gather environments)
    out of source text as raw, verbatim strings. Used to PRESERVE equation
    blocks separately so the pipeline can surface them in the report even if an
    LLM summary paraphrases them away — never rewriting, only carrying through.
    Deduped, order-preserving, trimmed, capped at `limit`."""
    out, seen = [], set()
    for m in _DISPLAY_EQ_RE.finditer(text or ""):
        body = (m.group(1) or m.group(3) or "").strip()
        body = re.sub(r"\s+", " ", body)
        if len(body) < 3 or body in seen:
            continue
        # require it to look like math (a LaTeX command or a math operator),
        # not just an accidental $$ pair around prose.
        if not (re.search(r"\\[a-zA-Z]+", body) or re.search(r"[=+\-*/^_∑∫√]", body)):
            continue
        seen.add(body)
        if m.group(2):  # named environment -> keep the wrapper for faithful render
            env = m.group(2)
            out.append(f"\\begin{{{env}}}{m.group(3).strip()}\\end{{{env}}}")
        else:
            out.append(f"$${body}$$")
        if len(out) >= limit:
            break
    return out


def formula_profile(text: str) -> dict:
    """Structured fingerprint of the math content of `text`."""
    text = text or ""
    # Sub/superscripts are only meaningful INSIDE math spans — counting bare '_'
    # over the whole text mis-reads markdown italics (_word_) as subscripts.
    math_only = " ".join(_math_spans(text))
    categories = {}
    for name, pat in _CATEGORY_PATTERNS.items():
        scope = math_only if name in ("subscripts", "superscripts") else text
        categories[name] = len(re.findall(pat, scope))
    cmd_tokens = _CMD_RE.findall(text)
    cmd_counts: Dict[str, int] = {}
    for t in cmd_tokens:
        cmd_counts[t] = cmd_counts.get(t, 0) + 1
    unicode_syms = sorted({c for c in text if c in _UNICODE_MATH})

    # corruption indicators
    ufffd = text.count("�")
    mojibake = len(_MOJIBAKE_RE.findall(text))
    spans = _math_spans(text)
    lost_backslash = sorted({
        w for span in spans for w in _MATH_CMDS
        if re.search(r"(?<![\\A-Za-z])" + w + r"\b", span)
    })
    n_dollar = text.count(_DOLLAR)
    # block ($$) vs inline ($) balance
    n_block = len(re.findall(r"\$\$", text))
    blocks_balanced = (n_block % 2 == 0)
    inline_balanced = (n_dollar % 2 == 0)

    return {
        "len": len(text),
        "latex_cmd_total": len(cmd_tokens),
        "latex_cmd_distinct": len(cmd_counts),
        "cmd_counts": cmd_counts,
        "categories": categories,
        "unicode_symbols": unicode_syms,
        "ufffd": ufffd,
        "mojibake": mojibake,
        "lost_backslash": lost_backslash,
        "inline_balanced": inline_balanced,
        "blocks_balanced": blocks_balanced,
        "clean": (ufffd == 0 and mojibake == 0 and not lost_backslash
                  and inline_balanced and blocks_balanced),
    }


def build_trace(snapshots: list) -> list:
    """Given an ORDERED list of (stage_name, text) snapshots through the pipeline,
    return one record per stage with every field the fidelity spec asks for:
    token_count, latex_token_count, unicode_symbols, missing_symbols (vs the
    previous stage), newly_introduced_symbols (candidate hallucination vs prev),
    corruption indicators, and the per-category structure counts. The first stage
    is the baseline; each later stage diffs against the one before it so the exact
    stage where a token first disappears / a symbol first appears is pinpointable."""
    records = []
    prev_prof = None
    for stage, text in snapshots:
        prof = formula_profile(text)
        rec = {
            "stage": stage,
            "token_count": len((text or "").split()),
            "latex_token_count": prof["latex_cmd_total"],
            "latex_distinct": prof["latex_cmd_distinct"],
            "unicode_symbols": prof["unicode_symbols"],
            "categories": prof["categories"],
            "corruption": {
                "ufffd": prof["ufffd"], "mojibake": prof["mojibake"],
                "lost_backslash": prof["lost_backslash"],
                "inline_balanced": prof["inline_balanced"],
                "blocks_balanced": prof["blocks_balanced"],
                "clean": prof["clean"],
            },
            "missing_symbols": [],
            "newly_introduced_symbols": [],
            "corruption_appeared": [],
        }
        if prev_prof is not None:
            d = diff_profiles(prev_prof, prof)
            rec["missing_symbols"] = d["lost_cmd_tokens"] + d["lost_unicode"]
            rec["newly_introduced_symbols"] = d["introduced_cmd_tokens"] + d["new_unicode"]
            rec["corruption_appeared"] = d["corruption_appeared"]
        records.append(rec)
        prev_prof = prof
    return records


def diff_profiles(src: dict, dst: dict) -> dict:
    """Compare two stage profiles. `lost` = command tokens present in src but
    gone in dst; `introduced` = tokens in dst absent from src (candidate
    hallucination if dst is a synthesis stage); `corruption_appeared` = a
    corruption indicator that is worse at dst than src."""
    s, d = src.get("cmd_counts", {}), dst.get("cmd_counts", {})
    lost = sorted(t for t in s if t not in d)
    introduced = sorted(t for t in d if t not in s)
    lost_unicode = sorted(set(src.get("unicode_symbols", [])) - set(dst.get("unicode_symbols", [])))
    new_unicode = sorted(set(dst.get("unicode_symbols", [])) - set(src.get("unicode_symbols", [])))
    corruption_appeared = []
    if dst.get("ufffd", 0) > src.get("ufffd", 0):
        corruption_appeared.append("ufffd")
    if dst.get("mojibake", 0) > src.get("mojibake", 0):
        corruption_appeared.append("mojibake")
    if set(dst.get("lost_backslash", [])) - set(src.get("lost_backslash", [])):
        corruption_appeared.append("lost_backslash")
    return {
        "lost_cmd_tokens": lost,
        "introduced_cmd_tokens": introduced,
        "lost_unicode": lost_unicode,
        "new_unicode": new_unicode,
        "corruption_appeared": corruption_appeared,
    }
