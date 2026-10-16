"""A Deep Research report as a PDF: Markdown -> HTML -> Chromium's print (Playwright's
bundled browser, already used by the Ozon client). Cyrillic, tables and links come out as
the browser renders them; no LaTeX engine is needed."""
import html
import logging

logger = logging.getLogger("assistant.dr_pdf")

_CSS = """
@page { margin: 14mm 12mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 18pt; margin: 0 0 10pt; } h2 { font-size: 14pt; margin: 16pt 0 6pt; }
h3 { font-size: 12pt; margin: 12pt 0 4pt; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9pt; }
th, td { border: 1px solid #bbb; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { background: #f0f0f0; } tr { page-break-inside: avoid; }
a { color: #1a4fa0; overflow-wrap: anywhere; }
body.wide table { font-size: 8pt; }
body.wide td:not(:first-child) { white-space: nowrap; }
td a { overflow-wrap: normal; word-break: normal; hyphens: manual; }
code, pre { font-family: Consolas, monospace; font-size: 9pt; background: #f6f6f6; }
pre { padding: 6pt; white-space: pre-wrap; }
blockquote { border-left: 3px solid #ccc; margin: 6pt 0; padding-left: 8pt; color: #444; }
"""


_SYMBOLS = {"ge": "≥", "geq": "≥", "le": "≤", "leq": "≤", "approx": "≈", "times": "×",
            "sim": "~", "pm": "±", "to": "→", "rightarrow": "→"}


def plain_symbols(text: str) -> str:
    """«$\\ge$ 25 fps» and «$\\approx$ 1.66 Gbit/s» came out raw in a table header and a
    bullet; a lone symbol in dollars is the Unicode sign."""
    import re
    return re.sub(r"\$\s*\\(\w+)\s*\$",
                  lambda m: _SYMBOLS.get(m.group(1), m.group(0)), text or "")


def md_to_pdf(markdown_text: str, out_path: str, title: str = "") -> bool:
    """Write `markdown_text` to `out_path` as a PDF. False on any failure (the caller
    then sends the Markdown file instead)."""
    try:
        import markdown
        from playwright.sync_api import sync_playwright
        markdown_text = plain_symbols(markdown_text)
        body = markdown.markdown(markdown_text or "", extensions=["tables", "fenced_code", "sane_lists"])
        # An 11-column candidate matrix on portrait A4 broke «pvCam-Ax5G-IMX255-8.9MP-02»
        # into five lines and «8.9 MP» into three; a wide table gets a landscape page.
        cols = max((ln.count("|") - 1 for ln in (markdown_text or "").splitlines()
                    if ln.lstrip().startswith("|")), default=0)
        wide = cols >= 8
        page_html = (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}"
                     f"</title><style>{_CSS}</style></head>"
                     f"<body class='{'wide' if wide else ''}'>{body}</body></html>")
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page()
                page.set_content(page_html, wait_until="domcontentloaded")
                page.pdf(path=out_path, format="A4", landscape=wide, print_background=True)
            finally:
                browser.close()
        return True
    except Exception:
        logger.warning("report PDF build failed", exc_info=True)
        return False
