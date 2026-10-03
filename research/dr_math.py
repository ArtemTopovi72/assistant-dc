"""Mathematical notation in the finished report: repair, then audit.

Two jobs. `_repair_latex_artifacts` puts back the structure a model loses when
it writes LaTeX into prose — the dropped backslash, the split command, the
delimiter that never closed. `audit_math` then reports what is still wrong
($ and $$ balance, commands that look like they lost their backslash) so the
caller can say so instead of shipping broken notation silently.

Split out of deep_research.py. Pure: no DR_* knob, no I/O, no LLM call.
"""
import re

# --------------------------------------------------------------------------- #
# Deterministic report post-processing (guarantees, not model goodwill)
# --------------------------------------------------------------------------- #
_MATH_CMDS = ("frac", "sum", "int", "sqrt", "nabla", "alpha", "beta", "gamma",
              "sigma", "theta", "lambda", "partial", "begin", "end", "infty", "cdot")

# TeX glue spec, e.g. "3.0pt plus 1.0pt minus 1.0pt" — used to strip layout primitives.
_TEX_GLUE = r"\s*-?\d*\.?\d+\s*(?:pt|ex|em|mu|cm|in|sp|bp|dd|pc)" \
            r"(?:\s*plus\s*-?\d*\.?\d+\s*(?:pt|ex|em|mu|cm|in|sp|bp|dd|pc)?)?" \
            r"(?:\s*minus\s*-?\d*\.?\d+\s*(?:pt|ex|em|mu|cm|in|sp|bp|dd|pc)?)?"

# A domain-like token (has a dot + TLD, no spaces) — distinguishes a citation wrapped in
# \text{} from a legitimate math subscript like \text{struc} / \text{child} (no dot).
_DOMAIN_RE = r"[A-Za-z0-9][A-Za-z0-9.\-]*\.[A-Za-z]{2,}"


def _repair_latex_artifacts(text: str) -> str:
    """Deterministically fix the LaTeX-rendering bugs that survive synthesis and break
    the math renderer. THREE classes, all observed live:

      1. TeX LAYOUT PRIMITIVES leaking from ar5iv `alttext` into equations/matrices —
         e.g. a bmatrix row break came through as `\\\\ \\vskip 3.0pt plus 1.0pt minus
         1.0pt\\cr`. `\\vskip`/`\\hskip`/`\\cr`/`\\noalign`/`\\penalty`/skips are page
         layout, not math; MathJax can't parse them and shows the raw source. The real
         row separator (`\\\\`) is kept; the glue+\\cr noise is removed.
      2. WRONG MATH DELIMITERS — the model emits `\\(...\\)` / `\\[...\\]` instead of the
         `$...$` / `$$...$$` the renderer expects, so the math shows literally. Normalize.
      3. CITATIONS wrapped in math / `\\text{}` — `\\((\\text{link.springer.com})\\)` was
         a plain domain cite the model put in inline math. Unwrap domain-like `\\text{}`
         to plain `(domain)`; legit subscripts (`\\text{struc}`, no dot) are untouched.
    Safe to run on any prose/report (idempotent-ish); never alters real math content."""
    if not text:
        return text
    # (2) delimiters first so the citation/layout passes see normalized `$` spans.
    text = text.replace(r"\[", "$$").replace(r"\]", "$$")
    text = text.replace(r"\(", "$").replace(r"\)", "$")
    # (1) strip layout primitives (with their glue argument) and stray \cr.
    text = re.sub(r"\\vskip" + _TEX_GLUE, "", text)
    text = re.sub(r"\\hskip" + _TEX_GLUE, "", text)
    text = re.sub(r"\\(?:small|med|big)skip\b", "", text)
    text = re.sub(r"\\noalign\s*\{[^{}]*\}", "", text)
    text = re.sub(r"\\penalty\s*-?\d+", "", text)
    text = re.sub(r"\\cr\b", "", text)            # \cr next to \\ in a matrix is redundant noise
    # (3) citation unwrap: a math span that is essentially just a domain in \text{}.
    text = re.sub(r"\$\s*\(?\s*\\text\{(" + _DOMAIN_RE + r")\}\s*\)?\s*\$", r"(\1)", text)
    #     and any bare \text{domain} left in prose.
    text = re.sub(r"\\text\{(" + _DOMAIN_RE + r")\}", r"\1", text)
    # (4) source-paper macros MathJax doesn't know: `\boldmath` is a TEXT-mode font
    #     switch (invalid in math → renders raw/red), and operators like `\tr` are
    #     \DeclareMathOperator macros from a preamble the alttext extraction drops.
    #     Drop the bold switches; map common operators to \operatorname{...}.
    text = re.sub(r"\\(?:un)?boldmath\b\s*", "", text)
    for _op in ("trace", "argmin", "argmax", "tr", "rank", "diag", "sign", "sgn",
                "supp", "Var", "Cov"):
        text = re.sub(r"\\" + _op + r"(?![A-Za-z])", r"\\operatorname{" + _op + "}", text)
    return text


def audit_math(report: str) -> dict:
    """Lightweight math-corruption detector for the FINAL report (observability, not a
    rewrite). Flags the two damage modes markdown/LaTeX rendering can't survive:
      * unbalanced $ / $$ delimiters (a stray $ breaks all downstream math rendering);
      * a known LaTeX command word inside a math span with its backslash stripped
        (e.g. 'frac{a}{b}' instead of '\\frac{a}{b}' — the classic lost-backslash bug).
    Returns counts + a `clean` verdict; the caller logs a warning when not clean."""
    text = report or ""
    # $$ blocks first, then remaining single $.
    block_spans = text.count("$$")
    blocks_balanced = (block_spans % 2 == 0)
    stripped = text.replace("$$", "")
    # count unescaped single $
    import re as _re
    single = len(_re.findall(r"(?<!\\)\$", stripped))
    inline_balanced = (single % 2 == 0)
    # lost-backslash: a command word followed by { or _ or ^ with NO preceding backslash,
    # occurring inside any math span.
    spans = _re.findall(r"\$\$(.+?)\$\$", text, flags=_re.S) + _re.findall(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", text, flags=_re.S)
    lost = []
    for s in spans:
        for cmd in _MATH_CMDS:
            for mm in _re.finditer(rf"(?<![\\A-Za-z]){cmd}(?=[{{_^]| )", s):
                # allow plain words like "sum"/"int" only if clearly not a command;
                # require an adjacent {, _, or ^ to call it a stripped command.
                if mm.group(0) and _re.match(rf"{cmd}[{{_^]", s[mm.start():mm.start()+len(cmd)+1]):
                    lost.append(cmd)
    out = {"inline_dollars": single, "block_dollars": block_spans,
           "inline_balanced": inline_balanced, "blocks_balanced": blocks_balanced,
           "math_spans": len(spans), "suspect_lost_backslash": sorted(set(lost))}
    out["clean"] = inline_balanced and blocks_balanced and not lost
    return out
