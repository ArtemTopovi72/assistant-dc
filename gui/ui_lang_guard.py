"""The one language check every outgoing Telegram message passes (user,
2026-09-27: «мучаемся каждый раз — то слова недоперевели … централизацию»).

_post_message hands every text and every inline button label here:
  1. a whole line / label that is a known stage phrase is translated
     (stages.translate — the same tables the status line uses);
  2. whatever English is still left in a Russian chat is LOGGED, once per
     distinct phrase, to the app log and runtime/untranslated.jsonl, so a
     missed string is found by the first user who meets it, not by a hunt.
Never raises and never blocks a send.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time

logger = logging.getLogger("assistant.lang_guard")

_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime", "untranslated.jsonl")
_seen: set = set()
_lock = threading.Lock()

# Names, not words: they stay Latin in a Russian sentence.
_ALLOW = {"ozon", "telegram", "youtube", "yue2", "firered", "ideogram", "minimax", "gemma",
          "whisper", "gigaam", "comfyui", "lm", "studio", "lora", "vpn", "api", "id", "ok",
          "pdf", "mp3", "mp4", "flac", "wav", "png", "jpg", "zip", "gpu", "ai", "ultra", "bpm",
          "wi", "fi", "usb", "hd", "4k", "ip", "url", "http", "https", "www", "vk", "iphone",
          "android", "windows", "google", "yandex", "wildberries", "avito", "huter", "neva",
          "chatgpt", "claude", "gpt", "rap", "rock", "pop", "metal", "hip", "hop", "edm", "lo"}
_SKIP = re.compile(r"<(code|pre|a)\b.*?</\1>|https?://\S+|\S+@\S+|`[^`]*`|\[[^\]]*\]", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_EN_RUN = re.compile(r"[A-Za-z][A-Za-z'’-]+(?:[ ,]+[A-Za-z][A-Za-z'’-]+)+")
_CYR = re.compile(r"[А-Яа-яЁё]")


def _translate_line(line: str, lang: str) -> str:
    try:
        import stages
    except Exception:
        return line
    m = re.match(r"^(\W*?)(<b>)?([A-Za-z][^<>]*?)(…|\.\.\.)?(</b>)?(…|\.\.\.)?\s*$", line)
    if not m or _CYR.search(line):
        return line
    core = m.group(3).strip()
    ru = stages.translate(core, lang)
    if ru == core:
        return line
    return line.replace(core, ru, 1)


def untranslated(text: str) -> list:
    """English phrases (2+ words, not names/links/code) left in `text`."""
    plain = _TAG.sub(" ", _SKIP.sub(" ", text or ""))
    out = []
    for m in _EN_RUN.finditer(plain):
        words = re.findall(r"[A-Za-z][A-Za-z'’-]+", m.group(0))
        if sum(w.lower() not in _ALLOW for w in words) >= 2:
            out.append(m.group(0).strip(" ,"))
    return out


def _report(where: str, phrase: str, chat_id) -> None:
    key = phrase.lower()
    with _lock:
        if key in _seen:
            return
        _seen.add(key)
    logger.warning("UNTRANSLATED in a ru chat (%s): %r", where, phrase[:160])
    try:
        os.makedirs(os.path.dirname(_LOG), exist_ok=True)
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "where": where,
                                 "chat": chat_id, "phrase": phrase[:300]}, ensure_ascii=False) + "\n")
    except Exception:
        pass


def guard_text(text: str, lang: str, chat_id=None, where: str = "text") -> str:
    try:
        if lang != "ru" or not text:
            return text
        text = "\n".join(_translate_line(l, lang) for l in text.split("\n"))
        # A reply that is wholly English is the model's language choice,
        # handled by the reply guards -- here we hunt UI leftovers in Russian.
        if _CYR.search(text) or where != "text":
            for ph in untranslated(text):
                _report(where, ph, chat_id)
        return text
    except Exception:
        return text


def guard_keyboard(kb, lang: str, chat_id=None):
    try:
        if lang != "ru" or not kb or "inline_keyboard" not in kb:
            return kb
        for row in kb["inline_keyboard"]:
            for b in row:
                if isinstance(b.get("text"), str):
                    b["text"] = guard_text(b["text"], lang, chat_id, "button")
        return kb
    except Exception:
        return kb


if __name__ == "__main__":
    assert untranslated("Нашёл на Ozon — https://ozon.ru/x смотри") == []
    assert untranslated("Готово. Looking at the image") == ["Looking at the image"]
    assert untranslated("<code>pip install foo bar</code> и всё") == []
    assert untranslated("Жанр: hip hop, 130 BPM") == []
    assert guard_text("👁 <b>Looking closer (2/4)…</b>", "ru") == "👁 <b>Рассматриваю поближе (2/4)…</b>"
    assert guard_text("Looking closer (2/4)", "en") == "Looking closer (2/4)"
    print("ok")
