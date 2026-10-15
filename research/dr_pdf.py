"""A Deep Research report as a PDF: Markdown -> HTML -> Chromium's print (Playwright's
bundled browser, already used by the Ozon client). Cyrillic, tables and links come out as
the browser renders them; no LaTeX engine is needed."""
import html
import logging

logger = logging.getLogger("assistant.dr_pdf")

_CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 18pt; margin: 0 0 10pt; } h2 { font-size: 14pt; margin: 16pt 0 6pt; }
h3 { font-size: 12pt; margin: 12pt 0 4pt; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9pt; }
th, td { border: 1px solid #bbb; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { background: #f0f0f0; } tr { page-break-inside: avoid; }
a { color: #1a4fa0; word-break: break-all; }
code, pre { font-family: Consolas, monospace; font-size: 9pt; background: #f6f6f6; }
pre { padding: 6pt; white-space: pre-wrap; }
blockquote { border-left: 3px solid #ccc; margin: 6pt 0; padding-left: 8pt; color: #444; }
"""


def md_to_pdf(markdown_text: str, out_path: str, title: str = "") -> bool:
    """Write `markdown_text` to `out_path` as a PDF. False on any failure (the caller
    then sends the Markdown file instead)."""
    try:
        import markdown
        from playwright.sync_api import sync_playwright
        body = markdown.markdown(markdown_text or "", extensions=["tables", "fenced_code", "sane_lists"])
        page_html = (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}"
                     f"</title><style>{_CSS}</style></head><body>{body}</body></html>")
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page()
                page.set_content(page_html, wait_until="domcontentloaded")
                page.pdf(path=out_path, format="A4", print_background=True)
            finally:
                browser.close()
        return True
    except Exception:
        logger.warning("report PDF build failed", exc_info=True)
        return False
