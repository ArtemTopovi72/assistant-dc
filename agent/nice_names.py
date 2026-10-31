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
_AI_TITLES: dict = {}      # chat_id -> (request, chosen title)

# ComfyUI counters too: «ideogram_00115_.png» was delivered under its raw name
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
    t = title or ""
    # A button payload quotes its instructions ("[style_preset] call redraw_image
    # with mode=\"redraw\" instructions=\"Turn picture 1 into …\""): the preset's
    # own words are the human name -- the raw call was delivered as
    # «Picture — call redraw_image with mode= redraw…» (2026-10-29).
    m = re.search(r'\binstructions="([^"]+)"', t)
    if m:
        t = re.sub(r"^[^:]*?\binto\s+", "", m.group(1), count=1)
    t = re.sub(r"^(?:\[[^\]]*\]\s*)+", "", t)           # "[song:60] …", "[internal] …"
    t = re.sub(r"^(?:[a-z ]+(?:\([^)]*\))?:\s*)", "", t)       # "find on ozon: …" menu prefixes
    # File names describe the result, not the command that requested it.
    t = re.sub(r"^(?:(?:сделай|нарисуй|создай|сгенерируй|изобрази)\s+)+", "", t, flags=re.I)
    t = re.sub(r"^(?:мне\s+)?(?:картинку|картину|изображение|фото|рисунок|видео|ролик|песню)\s+", "", t, flags=re.I)
    t = re.sub(r"^(?:советский\s+)?норм(?:альный|альную|альное|альные)?\s+", lambda m: "советский " if m.group(0).lower().startswith("советский") else "", t, flags=re.I)
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
    return (t if ext.lower() in (".png", ".jpg", ".jpeg", ".webp")
            else f"{kind} — {t}" if kind else t) + ext


def _suggest_title(request: str, path: str) -> str:
    """Ask the loaded chat model for a short file title; never reload it for naming."""
    if os.environ.get("F5_TEST_RUN"):
        return ""
    try:
        import comfy_client
        import llm
        import requests
        if comfy_client.card_is_exclusive() or not llm._model_served(llm.MODEL_NAME):
            return ""
        kind = category(path)
        response = requests.post(
            llm.LM_STUDIO_URL,
            json={
                "model": llm.MODEL_NAME, "temperature": 0.2, "max_tokens": 40,
                "stream": False,
                "messages": [
                    {"role": "system", "content":
                     "Назови готовый файл по запросу пользователя. Ответь только коротким "
                     "понятным названием на языке запроса: 2–5 слов, без кавычек, "
                     "расширения и слов «картинка», «видео», «песня». Не повторяй команду "
                     "и не придумывай детали, которых нет в запросе."},
                    {"role": "user", "content": f"Тип результата: {kind}\nЗапрос: {request[:1200]}"},
                    {"role": "assistant", "content": llm.GEMMA_NO_THINK_PREFILL},
                ],
            },
            timeout=(2, 8),
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]
        line = (answer or "").splitlines()[0].strip().strip('"«»`')
        line = re.sub(r"^(?:название|title)\s*:\s*", "", line, flags=re.I)
        title = _clean(line)
        if not title or len(title.split()) > 6 or len(line) > 70 or title.upper() in {"CON", "PRN", "AUX", "NUL"}:
            return ""
        return title
    except Exception:
        return ""


def named_path(chat_id, path: str, who: str = "") -> str:
    """A linked copy in runtime/generated/<who>/<category>/ under a human name;
    `path` itself if anything fails. Never overwrites: a clash gets " (2)"."""
    try:
        title = TITLES.get(chat_id, "")
        base, ext = os.path.splitext(os.path.basename(path))
        if title and _MACHINE.search(base) and ext.lower() in {
            ".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm", ".mov",
            ".flac", ".mp3", ".wav", ".ogg", ".m4a",
        }:
            cached = _AI_TITLES.get(chat_id)
            if cached and title in cached:
                title = cached[1]
            else:
                chosen = _suggest_title(title, path) or _clean(title)
                _AI_TITLES[chat_id] = (title, chosen)
                title = chosen
            TITLES[chat_id] = title
        name = display_name(path, title)
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
    assert display_name("x/ComfyUI_00012_.png", "draw a red fox in snow") == "draw a red fox in snow.png"
    assert display_name("x/klein_00001_.png", "Сделай советский норм трактор") == "советский трактор.png"
    assert display_name("x/song_1.flac", "") == "song_1.flac"
    assert display_name("x/song_yue2_1790516190822.flac", "[song] [lyrics] Кот на балконе")         == "Песня — Кот на балконе.flac"
    assert category("a/b.MP4") == "Видео" and category("x.pptx") == "Презентации" and category("r.pdf") == "Исследования"
    print("ok")
