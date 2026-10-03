"""Human file names for what the bot sends: «Песня — день рождения сестры.flac»,
not «song_yue2_1790516190822.flac» (user, 2026-09-27).

The name comes from the chat's current request (set per task by tg_tasks);
only machine-made names are replaced. The renamed file is a hard link (a copy
across volumes) in runtime/generated/<category>/ -- the ONE folder where
everything the bot delivered lives (user, 2026-09-27), and both the multipart
upload and the local Bot API's by-path send carry the new name.
"""
import os
import re
import shutil

TITLES: dict = {}          # chat_id -> the request the current task is about

# ComfyUI counters too: «ideogram_00115_.png» was delivered under its raw name (live 10-03)
_MACHINE = re.compile(r"\d{8,}|_\d{4,}_?$|[0-9a-f]{12,}|^(comfyui|song_|img_|image_|out|output|render|tmp|"
                      r"edit_|gen_|video_|clip_|frame|result)", re.I)
_KIND = {
    "ru": [((".flac", ".mp3", ".wav", ".ogg", ".m4a"), "Песня"), ((".png", ".jpg", ".jpeg", ".webp"), "Картинка"),
           ((".mp4", ".webm", ".mov"), "Видео"), ((".pptx", ".pdf"), "Презентация")],
    "en": [((".flac", ".mp3", ".wav", ".ogg", ".m4a"), "Song"), ((".png", ".jpg", ".jpeg", ".webp"), "Picture"),
           ((".mp4", ".webm", ".mov"), "Video"), ((".pptx", ".pdf"), "Presentation")],
}
_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime", "generated")
_CATEGORY = [((".png", ".jpg", ".jpeg", ".webp", ".gif"), "Изображения"),
             ((".mp4", ".webm", ".mov"), "Видео"),
             ((".flac", ".mp3", ".wav", ".ogg", ".m4a"), "Музыка"),
             ((".pptx", ".ppt"), "Презентации"),
             ((".pdf", ".docx", ".md", ".html", ".txt", ".epub"), "Исследования")]


def category(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return next((c for exts, c in _CATEGORY if ext in exts), "Прочее")


def _clean(title: str) -> str:
    t = re.sub(r"^(?:\[[^\]]*\]\s*)+", "", title or "")           # "[song:60] …", "[internal] …"
    t = re.sub(r"^(?:[a-z ]+(?:\([^)]*\))?:\s*)", "", t)       # "find on ozon: …" menu prefixes
    t = re.sub(r"[<>:\"/\\|?*\n\r\t]+", " ", t)
    t = " ".join(t.split()[:8])
    return t[:60].strip(" .-—")


def display_name(path: str, title: str) -> str:
    base, ext = os.path.splitext(os.path.basename(path))
    t = _clean(title)
    if not t or not _MACHINE.search(base):
        return os.path.basename(path)
    lang = "ru" if re.search("[а-яё]", t, re.I) else "en"
    kind = next((k for exts, k in _KIND[lang] if ext.lower() in exts), "")
    return (f"{kind} — {t}" if kind else t) + ext


def named_path(chat_id, path: str, who: str = "") -> str:
    """A linked copy in runtime/generated/<who>/<category>/ under a human name;
    `path` itself if anything fails. Never overwrites: a clash gets " (2)"."""
    try:
        name = display_name(path, TITLES.get(chat_id, ""))
        who = _clean(who) or str(chat_id)
        d = os.path.join(_ROOT, who, category(path))
        if os.path.abspath(os.path.dirname(path)) == os.path.abspath(d):
            return path                                     # already archived
        os.makedirs(d, exist_ok=True)
        stem, ext = os.path.splitext(name)
        dst, n = os.path.join(d, name), 2
        while os.path.exists(dst):
            if os.path.samefile(dst, path):
                return dst
            dst, n = os.path.join(d, f"{stem} ({n}){ext}"), n + 1
        try:
            os.link(path, dst)
        except OSError:
            shutil.copy2(path, dst)
        return dst
    except Exception:
        return path


if __name__ == "__main__":
    assert display_name("x/song_yue2_1790516190822.flac", "[song:60] день рождения для сестры") \
        == "Песня — день рождения для сестры.flac"
    assert display_name("x/My deck.pptx", "про котов") == "My deck.pptx"
    assert display_name("x/ComfyUI_00012_.png", "draw a red fox in snow") == "Picture — draw a red fox in snow.png"
    assert display_name("x/song_1.flac", "") == "song_1.flac"
    assert display_name("x/song_yue2_1790516190822.flac", "[song] [lyrics] Кот на балконе")         == "Песня — Кот на балконе.flac"
    assert category("a/b.MP4") == "Видео" and category("x.pptx") == "Презентации" and category("r.pdf") == "Исследования"
    print("ok")
