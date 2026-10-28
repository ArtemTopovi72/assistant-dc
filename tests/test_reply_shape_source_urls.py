"""A host can supply several pages; a citation must identify the page it used."""
from types import SimpleNamespace

import tg_markup
import tg_reply_shape as shape


SOURCES = [
    {"domain": "docs.example.org", "title": "Types", "url": "https://docs.example.org/types"},
    {"domain": "docs.example.org", "title": "Release changes", "url": "https://docs.example.org/new?v=4&lang=en"},
]


def test_exact_page_citation_selects_later_result_on_the_same_host():
    reply = tg_markup._md_to_html(
        "Deferred annotations (source: https://docs.example.org/new?v=4&lang=en).")
    result = shape.shape_search_reply(reply, ["release changes"], SOURCES)
    assert "Deferred annotations [1]." in result
    assert 'href="https://docs.example.org/new?v=4&amp;lang=en"' in result
    assert "https://docs.example.org/types" not in result


def test_unread_page_is_not_replaced_with_another_page_on_its_host():
    calls = []

    def model(*a, **kw):
        calls.append(True)
        return "1"

    result = shape.shape_search_reply(
        "A claim (source: https://docs.example.org/unread).", ["q"], SOURCES,
        ctx=SimpleNamespace(text_model=model))
    assert "source:" not in result
    assert "<a href" not in result
    assert not calls


def test_ambiguous_legacy_domain_does_not_pick_the_first_page_or_guess_from_titles():
    calls = []
    ctx = SimpleNamespace(text_model=lambda *a, **kw: calls.append(True) or "1")
    result = shape.shape_search_reply("A claim (source: docs.example.org).", ["q"], SOURCES, ctx=ctx)
    assert "<a href" not in result
    assert "source:" not in result
    assert not calls


def test_legacy_domain_suffix_is_also_ambiguous_across_subdomains():
    sources = [
        {"domain": "en.example.org", "title": "First", "url": "https://en.example.org/page"},
        {"domain": "ru.example.org", "title": "Second", "url": "https://ru.example.org/page"},
    ]
    result = shape.shape_search_reply("A claim (source: example.org).", ["q"], sources)
    assert "<a href" not in result


def test_single_page_legacy_domain_still_resolves():
    result = shape.shape_search_reply("A claim (source: docs.example.org).", ["q"], SOURCES[1:])
    assert "A claim [1]." in result
    assert 'href="https://docs.example.org/new?v=4&amp;lang=en"' in result


def test_exact_page_url_can_contain_parentheses():
    url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    sources = [{"domain": "en.wikipedia.org", "title": "Python", "url": url}]
    result = shape.shape_search_reply(f"A fact (source: {url}).", ["q"], sources)
    assert "A fact [1]." in result
    assert f'href="{url}"' in result
