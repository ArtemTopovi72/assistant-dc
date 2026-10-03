"""A link in a chat message is a page to read, not a string to admire.

Live, 2026-09-12: the user sent one bare Wikipedia URL and the reply was
"This is a link to an article about omelettes on Wikipedia; you can find
detailed information there." The bot has a whole page fetcher (dr_crawl)
and never used it in chat. Now a message that carries a URL gets the page's
text attached, so "о чём это?", a summary or a question about it are
answered from the page.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("assistant.tg_links")

_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
LINK_TEXT_CHARS = 6000


def urls_in(text: str) -> list:
    return [u.rstrip(".,;:!?)»") for u in _URL_RE.findall(text or "")]


# Hosts whose links are videos to watch, not pages to read (live 10-03: a
# YouTube Shorts link got «я не умею смотреть видео по ссылкам»).
_VIDEO_RE = re.compile(
    r"^https?://(?:[\w-]+\.)*(?:youtube\.com/(?:watch|shorts/|live/)|youtu\.be/|"
    r"tiktok\.com/|vk\.com/(?:video|clip)|vkvideo\.ru/|rutube\.ru/(?:video|shorts)/|"
    r"instagram\.com/(?:reel|p)/|(?:twitter|x)\.com/\w+/status/|dzen\.ru/(?:video|shorts)/)",
    re.IGNORECASE)
VIDEO_MAX_SECONDS = 600
# url -> what was seen/heard, for follow-up questions about the same video.
# ponytail: in-memory, lost on restart; persist with the session if that bites.
WATCHED: dict = {}


def video_url(text: str) -> str:
    """The first link in `text` that is a video on a known host, or ""."""
    return next((u for u in urls_in(text) if _VIDEO_RE.match(u)), "")


def fetch_video(url: str, max_seconds: int = VIDEO_MAX_SECONDS) -> dict:
    """{"data": mp4 bytes, "seconds", "title", "description"} for a public video up to
    `max_seconds`, {"too_long": True, "seconds", "title"} past it, {} when it
    cannot be downloaded. 720p is plenty for frames and speech."""
    import os, tempfile
    try:
        import yt_dlp
        with tempfile.TemporaryDirectory(prefix="ytdl_") as d:
            opts = {"quiet": True, "no_warnings": True, "noprogress": True, "noplaylist": True,
                    # YouTube serves picture and sound as separate streams
                    "format": "bv*[height<=720]+ba/b[height<=720]/b",
                    "merge_output_format": "mp4",
                    "outtmpl": os.path.join(d, "v.%(ext)s"), "socket_timeout": 20}
            with yt_dlp.YoutubeDL(opts) as y:
                info = y.extract_info(url, download=False)
                secs = int(info.get("duration") or 0)
                title = info.get("title") or ""
                desc = (info.get("description") or "").strip()[:2000]
                if secs > max_seconds:
                    return {"too_long": True, "seconds": secs, "title": title}
                y.download([url])
            f = next((os.path.join(d, n) for n in os.listdir(d)), "")
            if f:
                return {"data": open(f, "rb").read(), "seconds": secs, "title": title,
                        "description": desc}
    except Exception:
        logger.warning("video download failed for %s", url, exc_info=True)
    return {}


def read_link(url: str) -> dict:
    """{"title", "text"} for a public page, or {} when it cannot be read.
    Goes through the research crawler's fetch+extract path (adapters for
    Wikipedia/arXiv, PDF text, the content gate)."""
    try:
        import dr_crawl
        res = dr_crawl._acquire_page(url, 0, {"title": "", "href": url}, "general", True, None)
        page = (res or {}).get("page") or {}
        if page.get("text"):
            text = page["text"]
            # The crawler's adapters return a summary (Wikipedia: 8000 chars);
            # a chat question may be about the rest (Kazan's population,
            # live). One more fetch of the whole page; _fit picks the part.
            # ponytail: second download per link; cache if links get frequent.
            try:
                if not url.lower().endswith(".pdf"):
                    html, _pdf = dr_crawl._fetch_page(url)
                    full = _pdf or (dr_crawl._extract_text(html, url) if html else "")
                    if full and len(full) > len(text):
                        text = full
            except Exception:
                pass
            return {"title": page.get("title") or "", "text": text}
        # The gate refuses landing pages and thin extractions; for a chat
        # question even a thin page beats nothing -- fall back to the raw text.
        html, pdf_text = dr_crawl._fetch_page(url)
        text = pdf_text if pdf_text else (dr_crawl._extract_text(html, url) if html else "")
        if text and text.strip():
            return {"title": dr_crawl._extract_title(html) if html else "", "text": text}
    except Exception:
        logger.debug("read_link failed for %s", url, exc_info=True)
    return {}


# ponytail: hand-listed synonyms for what chat asks about pages most; embeddings if this grows.
_SAME = [{"жител", "насел", "числе", "popul", "inhab"}, {"основ", "found", "возни"},
         {"площа", "area"}]


def _fit(body: str, question: str, limit: int) -> str:
    """The page's head plus the paragraphs that share words with the question.
    Live 2026-09-28: Kazan's population sat past the 6000-char cut, and the
    bot said the article did not mention it."""
    if len(body) <= limit:
        return body
    head = body[:limit * 2 // 3]
    stems = {w[:5] for w in re.findall(r"\w{4,}", _URL_RE.sub(" ", question or "").lower())}
    for group in _SAME:
        if stems & group:
            stems |= group
    extra = []
    room = limit - len(head)
    scored = []
    paras = re.split(r"\n|(?<=[.!?])\s+(?=[А-ЯЁA-Z])", body[len(head):])
    for i, para in enumerate(paras):
        hits = len(stems & {w[:5] for w in re.findall(r"\w{4,}", para.lower())})
        if hits and para.strip() and para not in head:
            scored.append((-hits, i, para))
            if len(para) < 80:
                # A matched heading: its section may be a bare table of
                # numbers (Kazan's population) that shares no word with anything.
                # A table runs oldest to newest: its last two rows are the answer.
                j = i + 1
                while j < len(paras) and ("|" in paras[j] or not paras[j].strip()):
                    j += 1
                scored += [(-hits - 1, k, paras[k]) for k in range(max(i + 1, j - 2), min(j + 2, len(paras)))
                           if paras[k].strip()]
    for _, i, para in sorted(scored)[:40]:
        if len(para) < room and para not in extra:
            extra.append(para)
            room -= len(para) + 1
    note = f"\n[Page cut: {limit} of {len(body)} chars shown. If the answer is not here, say you read only part of the page -- never that the page lacks it.]"
    return head + ("\n...\n" + "\n".join(extra) if extra else "") + note


def link_context(text: str, limit: int = LINK_TEXT_CHARS, url: str = "") -> str:
    """Scaffolding for the model: the text of the first readable link in
    `text`, or "" when there is none / it cannot be read."""
    for url in ([url] if url else urls_in(text)[:2]):
        page = read_link(url)
        if page:
            body = _fit(re.sub(r"\n{3,}", "\n\n", page["text"]).strip(), text, limit)
            title = (page.get("title") or "").strip()
            said = ("The user asks about a page they linked earlier" if url not in (text or "")
                    else "The user's message contains a link")
            return (f"[{said}. Here is that page, read for you"
                    f"{' -- title: ' + title if title else ''} ({url}). Answer from it; if the "
                    f"user only sent the link, tell them briefly what it is about.]\n{body}")
    return ""
