"""The shape of a searched answer in Telegram.

A reply that came from a web search used to arrive as a bare paragraph with
"(source: example.com)" sprinkled through it. The user asked for the same
care the video retelling got: what was searched, the answer with footnote
marks, and the sources listed underneath with links. This module renders
that around the model's own (already HTML) reply; the query list and the
result list are collected on the per-turn context by search.py.
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
    out = []
    for m in _LINK_NOTE_RE.finditer(user_text or ""):
        url = m.group(2)
        dom = re.sub(r"^https?://", "", url).split("/")[0]
        out.append({"kind": "link", "title": (m.group(1) or "").strip(), "domain": dom, "url": url})
    return out


# "(source: example.com)", "(источник: example.com)", "(sources: a.com, b.org)"
_CITE_RE = re.compile(r"\(\s*(?:source|sources|источник|источники)\s*:\s*([^)]+?)\s*\)", re.IGNORECASE)
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


def shape_search_reply(reply_html: str, queries: list, sources: list, lang: str = "ru") -> str:
    """Wrap an HTML reply: 🔎 the queries, the answer with [n] marks in place of
    "(source: domain)", and 🔗 a numbered source list with links. Sources the
    answer cites come first; the rest fill up to MAX_SOURCES."""
    if not (queries or sources):
        return reply_html
    searched, srcs_label, read_label = _LABELS.get(lang, _LABELS["ru"])
    links = [s for s in sources or [] if s.get("kind") == "link"]
    sources = [s for s in sources or [] if s.get("kind") != "link"]
    by_dom: dict = {}
    for s in sources:
        by_dom.setdefault(_dom_key(s.get("domain", "")), s)

    cited: list = []          # sources in the order the answer cites them

    def _mark(m: re.Match) -> str:
        marks = []
        for dom in re.split(r"[,;]\s*", m.group(1)):
            k = _dom_key(dom)
            s = by_dom.get(k)
            if s is None:
                # "(source: wikipedia.org)" for a ru.wikipedia.org page stayed
                # raw in the reply (live 2026-09-28): match on the domain suffix.
                s = next((v for d, v in by_dom.items()
                          if k and (d.endswith("." + k) or k.endswith("." + d))), None)
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
    # held only checko.ru (live 2026-10-08).
    reply_html = re.sub(r"[ \t]*(?<!\w)\[\d{1,2}\]", "", reply_html or "")
    body = _BARE_RE.sub(lambda m: _mark(m) or m.group(0), _CITE_RE.sub(_mark, reply_html))
    body = re.sub(r"[ \t]+([.,;:!?])", r"\1", body)       # "fact ." after a dropped mark
    body = re.sub(r"\s+(\[\d+\])", r" \1", body)          # "text [1]" not "text  [1]"
    listed = list(cited)
    # With citations in the text, the list is what was cited -- padding it
    # with the other hits listed a Norfolk, Virginia museum under «Эрмитаж»
    # (live 2026-09-28).
    for s in ([] if cited else sources or []):
        if len(listed) >= MAX_SOURCES:
            break
        if s not in listed:
            listed.append(s)

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
