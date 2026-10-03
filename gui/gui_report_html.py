"""Render a research report as themed HTML for the in-app viewer.

Pure string work — no Qt, no widgets, no state. It takes the report markdown
and the palette from gui_common and produces the document the report pane
displays, which is why it is the one piece of gui.py that could always have
lived anywhere.

Kept separate from the report WIDGET on purpose: the widget is about scrolling
and sizing, this is about what the document looks like, and only this half is
worth reading when the styling is wrong.
"""
from gui_common import ACCENT, ACCENT2, BG, BORDER, MUTED, PANEL, PANEL2, TEXT


# --------------------------------------------------------------------------- #
# Research report rendering — Markdown -> themed HTML with typeset math + tables
# --------------------------------------------------------------------------- #
def _render_report_html(markdown_text: str) -> str:
    """Convert a research report (Markdown) to a full dark-themed HTML page with
    real tables and LaTeX math typeset by MathJax. Qt's setMarkdown can't render
    math (and mangles backslashes), so reports are shown in a QWebEngineView using
    this. python-markdown + pymdownx.arithmatex protect the math (incl. bare
    \\begin{align} blocks) and emit \\(...\\)/\\[...\\] spans that MathJax renders."""
    import markdown as _md
    body = _md.markdown(
        markdown_text or "",
        extensions=["tables", "fenced_code", "sane_lists", "pymdownx.arithmatex"],
        extension_configs={"pymdownx.arithmatex": {"generic": True}},
    )
    # MathJax from CDN (the app already requires internet for research). If offline,
    # equations gracefully degrade to readable \[...\] source instead of mangled text.
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script>window.MathJax={{
  // tex-chtml-FULL (below) bundles EVERY TeX extension in the one static script, so
  // \\boldsymbol/\\cancel/\\color/etc. need NO runtime autoload. The default
  // tex-mml-chtml bundle autoloads those on first use, and under setHtml(base=
  // "about:blank") QWebEngine can't resolve that dynamic fetch → \\boldsymbol
  // equations rendered as raw source while base/ams math worked. -full removes the
  // autoload path entirely. macros: drop \\boldmath (text-mode, invalid in math) and
  // map author operators (\\tr etc.) that come from dropped \\DeclareMathOperator preambles.
  tex:{{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']],
  macros:{{boldmath:'',unboldmath:'',tr:'\\\\operatorname{{tr}}',rank:'\\\\operatorname{{rank}}',
    diag:'\\\\operatorname{{diag}}',sign:'\\\\operatorname{{sign}}',sgn:'\\\\operatorname{{sgn}}',
    trace:'\\\\operatorname{{trace}}',argmin:'\\\\operatorname{{arg\\\\,min}}',
    argmax:'\\\\operatorname{{arg\\\\,max}}',supp:'\\\\operatorname{{supp}}',
    Var:'\\\\operatorname{{Var}}',Cov:'\\\\operatorname{{Cov}}',vskip:'',hskip:''}}}},
svg:{{fontCache:'global'}},options:{{skipHtmlTags:['script','style']}}}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js"></script>
<style>
 html,body{{background:{BG};color:{TEXT};font-family:'Segoe UI',system-ui,sans-serif;
   font-size:15px;line-height:1.6;margin:0;padding:18px 22px;}}
 h1{{font-size:1.7em;margin:.2em 0 .5em;color:{TEXT};border-bottom:2px solid {ACCENT};padding-bottom:.2em;}}
 h2{{font-size:1.3em;margin:1.2em 0 .4em;color:{ACCENT2};border-bottom:1px solid {BORDER};padding-bottom:.15em;}}
 h3{{font-size:1.1em;margin:1em 0 .3em;color:{TEXT};}}
 a{{color:{ACCENT};text-decoration:none;}} a:hover{{text-decoration:underline;}}
 em{{color:{MUTED};}}
 code{{background:{PANEL2};padding:.1em .35em;border-radius:4px;font-size:.9em;}}
 pre{{background:{PANEL};padding:12px;border-radius:8px;overflow-x:auto;border:1px solid {BORDER};}}
 blockquote{{border-left:3px solid {ACCENT2};margin:.6em 0;padding:.2em 1em;color:{MUTED};background:{PANEL};border-radius:0 6px 6px 0;}}
 table{{border-collapse:collapse;margin:1em 0;width:auto;}}
 th,td{{border:1px solid {BORDER};padding:7px 12px;text-align:left;}}
 th{{background:{PANEL2};color:{ACCENT2};}}
 tr:nth-child(even) td{{background:{PANEL};}}
 mjx-container{{overflow-x:auto;overflow-y:hidden;max-width:100%;}}
 ul,ol{{padding-left:1.4em;}}
</style></head><body>{body}</body></html>"""
