"""The shape of a searched answer in Telegram.

A reply that came from a web search is rendered as: what was searched, the
answer with footnote marks in place of "(source: example.com)", and the
sources listed underneath with links. This module renders that around the
model's own (already HTML) reply; the query list and the result list are
collected on the per-turn context by search.py.
"""
from __future__ import annotations

import html as _html
import re

_LABELS = {
    "ru": ("Искал", "Источники", "Прочитал"),
    "en": ("Searched", "Sources", "Read"),
}
# The scaffold tg_links puts on a turn that carries a URL (see link_context).
_LINK_NOTE_RE = re.compile(r"read for you(?: -- title: (.*?))? \((https?://[^)\s]+)\)\. Answer from it")


def links_read(user_text: str) -> list:
    """The pages tg_links attached to this turn, as source records (kind=link)."""
    from prompt_guard import user_words
    out = []
    for m in _LINK_NOTE_RE.finditer(user_words(user_text or "")):
        url = m.group(2)
        dom = re.sub(r"^https?://", "", url).split("/")[0]
        out.append({"kind": "link", "title": (m.group(1) or "").strip(), "domain": dom, "url": url})
    return out


# "(source: example.com)", "(источник: example.com)", "(sources: a.com, b.org)"
_CITE_RE = re.compile(r"\(\s*(?:source|sources|источник|источники)\s*:\s*(https?://\S+|[^)]+?)\s*\)", re.IGNORECASE)
# ...and the bare form the model also writes: "2390 ₽ (hermitage-museum.ru)".
_BARE_RE = re.compile(r"\(\s*((?:www\.)?(?:[\w-]+\.)+(?:[a-z]{2,}|xn--[a-z0-9-]+)(?:\s*[,;]\s*(?:www\.)?(?:[\w-]+\.)+(?:[a-z]{2,}|xn--[a-z0-9-]+))*)\s*\)", re.IGNORECASE)
MAX_SOURCES = 6


def _show_dom(dom: str) -> str:
    """«xn----8sbhycugqd1i.xn--p1ai» is what the reader saw for a .рф site."""
    try:
        return dom.encode("ascii").decode("idna") if "xn--" in (dom or "") else dom
    except (UnicodeError, ValueError):
        return dom


def _dom_key(dom: str) -> str:
    d = (dom or "").strip().lower()
    return d[4:] if d.startswith("www.") else d


def shape_search_reply(reply_html: str, queries: list, sources: list, lang: str = "ru", ctx=None) -> str:
    """Wrap an HTML reply: 🔎 the queries, the answer with [n] marks in place of
    "(source: domain)", and 🔗 a numbered source list with links. Sources the
    answer cites, up to MAX_SOURCES."""
    if not (queries or sources):
        return reply_html
    searched, srcs_label, read_label = _LABELS.get(lang, _LABELS["ru"])
    links = [s for s in sources or [] if s.get("kind") == "link"]
    sources = [s for s in sources or [] if s.get("kind") != "link"]
    by_dom: dict = {}
    by_url = {s.get("url", ""): s for s in sources if s.get("url")}
    for s in sources:
        by_dom.setdefault(_dom_key(s.get("domain", "")), []).append(s)

    cited: list = []          # sources in the order the answer cites them

    def _mark(m: re.Match) -> str:
        marks = []
        source = _html.unescape(m.group(1)).strip()
        is_url = source.lower().startswith(("https://", "http://"))
        for dom in ([source] if is_url else re.split(r"[,;]\s*", source)):
            k = _dom_key(dom)
            s = by_url.get(dom) if is_url else None
            if not is_url:
                # "(source: wikipedia.org)" for a ru.wikipedia.org page would stay
                # raw in the reply: match on the domain suffix.
                matches = by_dom.get(k) or [v for d, pages in by_dom.items()
                            if k and (d.endswith("." + k) or k.endswith("." + d)) for v in pages]
                pages = {v.get("url", ""): v for v in matches}
                s = next(iter(pages.values())) if len(pages) == 1 else None
            if s is None:
                continue
            if s not in cited:
                cited.append(s)
            marks.append(f"[{cited.index(s) + 1}]")
        # A domain that is not among the pages actually read is not a source:
        # dropped rather than shown as a citation nobody can follow.
        return " ".join(marks)

    # Numbers are this function's to give. A «[1]» the model wrote itself pointed at
    # whatever source happened to be listed first: «cbr.ru [1]» over a list that
    # held only checko.ru.
    reply_html = re.sub(r"[ \t]*(?<!\w)\[\d{1,2}\]", "", reply_html or "")
    body = _BARE_RE.sub(lambda m: _mark(m) or m.group(0), _CITE_RE.sub(_mark, reply_html))
    body = re.sub(r"[ \t]+([.,;:!?])", r"\1", body)       # "fact ." after a dropped mark
    body = re.sub(r"\s+(\[\d+\])", r" \1", body)          # "text [1]" not "text  [1]"
    # A title alone cannot prove a page supports an uncited answer.
    listed = cited[:MAX_SOURCES]

    head = ""
    for s in links:      # «📄 Прочитал: title — domain», one line per page read
        url = _html.escape(s.get("url", ""), quote=True)
        title = _html.escape((s.get("title") or s.get("domain") or url)[:90])
        head += f'📄 <b>{read_label}:</b> <a href="{url}">{title}</a> — {_html.escape(_show_dom(s.get("domain", "")))}\n'
    if head and not queries:
        head += "\n"
    if queries:
        qs = " · ".join("«" + _html.escape(q.strip()) + "»" for q in queries if q and q.strip())
        head += f"🔎 <b>{searched}:</b> {qs}\n\n"
    foot = ""
    if listed:
        lines = []
        for i, s in enumerate(listed, 1):
            url = _html.escape(s.get("url", ""), quote=True)
            title = _html.escape((s.get("title") or s.get("domain") or url)[:90])
            dom = _html.escape(_show_dom(s.get("domain", "")))
            lines.append(f'{i}. <a href="{url}">{title}</a> — {dom}' if url else f"{i}. {title} — {dom}")
        foot = f"\n\n🔗 <b>{srcs_label}:</b>\n" + "\n".join(lines)
    return head + body.strip() + foot
