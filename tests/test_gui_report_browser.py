import pytest
from bs4 import BeautifulSoup
from playwright.sync_api import Error, sync_playwright

from gui_report_html import _render_report_html


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        for options in ({}, {"channel": "chrome"}, {"channel": "msedge"}):
            try:
                instance = playwright.chromium.launch(headless=True, **options)
                break
            except Error:
                continue
        else:
            pytest.skip("Playwright requires installed Chromium, Chrome or Edge")
        yield instance
        instance.close()


@pytest.fixture
def page(browser):
    context = browser.new_context(viewport={"width": 480, "height": 720})
    context.route("https://cdn.jsdelivr.net/**", lambda route: route.abort())
    page = context.new_page()
    yield page
    context.close()


def test_report_content_cannot_execute_javascript(page):
    report = """# Report

<script>window.report_script_ran = true</script>
<img src="missing.png" onerror="window.report_event_ran = true">
<iframe srcdoc="<script>parent.report_frame_ran = true</script>"></iframe>
"""
    page.set_content(_render_report_html(report))
    ran = page.evaluate("() => [window.report_script_ran, window.report_event_ran, window.report_frame_ran].some(Boolean)")
    assert not ran, "Untrusted report content executed JavaScript"


@pytest.mark.parametrize("href", ["javascript:alert(1)", "java&#x73;cript:alert(1)",
                                  "java&#9;script:alert(1)", " JAVASCRIPT:alert(1)",
                                  "data:text/html,unsafe", "vbscript:unsafe", "file:///secret",
                                  "https://[", "https://example.org:invalid"])
def test_report_removes_unsafe_link_protocols(href):
    html = _render_report_html(f'<a href="{href}" onclick="alert(1)">source</a>')
    link = BeautifulSoup(html, "html.parser").body.find("a")
    assert link.get("href") is None
    assert link.get("onclick") is None


@pytest.mark.parametrize("href", ["https://example.org/report?x=1&y=2", "http://example.org",
                                  "mailto:editor@example.org", "#section", "relative/report.html"])
def test_report_keeps_safe_links(href):
    link = BeautifulSoup(_render_report_html(f'<a href="{href}">source</a>'), "html.parser").body.a
    assert link["href"] == href


@pytest.mark.parametrize("src", ["https://example.org/image.png", "http://example.org/image.png",
                                 "images/figure.png"])
def test_report_keeps_safe_markdown_images(src):
    image = BeautifulSoup(_render_report_html(f'![caption]({src} "Figure")'), "html.parser").body.img
    assert image is not None
    assert image.attrs == {"src": src, "alt": "caption", "title": "Figure"}


@pytest.mark.parametrize("src", ["javascript:alert(1)", "java&#x73;cript:alert(1)",
                                 "data:image/svg+xml,unsafe", "file:///secret", "mailto:a@example.org",
                                 "https://[", "https://example.org:invalid"])
def test_report_removes_unsafe_image_sources_and_attributes(src):
    report = f'<img src="{src}" alt="caption" title="Figure" onerror="alert(1)" srcset="unsafe" style="width:9000px" class="arithmatex">'
    image = BeautifulSoup(_render_report_html(report), "html.parser").body.img
    assert image is not None
    assert image.attrs == {"alt": "caption", "title": "Figure"}


def test_report_images_fit_viewport(page):
    page.route("https://example.org/image.svg", lambda route: route.fulfill(
        content_type="image/svg+xml", body='<svg xmlns="http://www.w3.org/2000/svg" width="2000" height="100"/>'))
    page.set_content(_render_report_html('![caption](https://example.org/image.svg)'))
    assert page.locator("img").evaluate("el => el.complete && el.naturalWidth === 2000")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")


def test_report_restricts_markup_to_report_formatting(page):
    report = '''<base href="https://attacker.example/">
<style>body {display:none}</style>
<svg><a href="javascript:alert(1)">svg</a></svg>
<form action="https://attacker.example/"><input autofocus onfocus="alert(1)"></form>
<p id="MathJax" style="background:url(javascript:alert(1))" onmouseover="alert(1)">Visible</p>

```python
print("<script>escaped code</script>")
```
'''
    page.set_content(_render_report_html(report))
    assert page.locator("body script, body style, base, svg, form, input").count() == 0
    assert page.locator("p").first.evaluate("el => [...el.attributes].map(a => a.name)") == []
    assert page.get_by_text("Visible", exact=True).is_visible()
    assert page.locator("code.language-python").inner_text() == 'print("<script>escaped code</script>")\n'


def test_mathjax_filters_links_created_after_html_sanitization(page):
    page.set_content(_render_report_html(r'$\href{javascript:alert(1)}{Click}$'))
    config = page.evaluate("() => window.MathJax")
    assert "ui/safe" in config["loader"]["load"]
    assert config["options"]["safeOptions"]["allow"]["URLs"] == "safe"
    assert config["options"]["safeOptions"]["safeProtocols"]["file"] is False


def test_report_keeps_tables_math_and_safe_links(page):
    report = r"""# Проверка

| Показатель | Значение |
|---|---:|
| Точность | 98.5% |

Формула: $\boldsymbol{v}=(1,2)$.

[Источник](https://example.org/report?id=7&lang=ru)
"""
    page.set_content(_render_report_html(report))
    assert page.get_by_role("heading", name="Проверка").count() == 1
    assert "98.5%" in page.locator("table").inner_text()
    assert page.locator(".arithmatex").count() == 1
    assert page.get_by_role("link", name="Источник").get_attribute("href") == "https://example.org/report?id=7&lang=ru"
    assert page.locator("td").last.evaluate("el => getComputedStyle(el).textAlign") == "right"
