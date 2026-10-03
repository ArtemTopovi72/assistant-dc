"""Translate live progress stages at the point they are shown to a user.

`ctx.set_stage(...)` is called from graph.py, tools.py, image.py, llm.py and
deep_research.py — the shared agent core, which the desktop app and the Telegram
bot both drive. Those strings are English by design: the pipeline reasons in
English (every non-English message is translated at entry) and the stage text
doubles as an operator-facing label in the activity log and the admin panel.

So the English string stays the canonical identity of a stage, and the
translation happens here, at the presentation boundary, for the one surface that
is actually the user's: the live status message. Nothing upstream changes, the
log stays greppable, and the icon lookup in tg_bot still keys off English.

Two shapes have to be handled:

  · fixed labels    — "Searching the web", "Drawing a picture", …
  · parameterised   — "Retrying the model call (2/3)", and the deep-research
                      progress line "Searching · 12 src / 40 pg / 7 found",
                      whose leading phase comes from deep_research.py and whose
                      units are English abbreviations.

Anything unrecognised is returned unchanged. An untranslated stage is what
happens today, so passthrough can only ever be neutral — never a crash, never a
blank status message.
"""
from __future__ import annotations

import re

DEFAULT_LANG = "en"

# ── fixed stage labels ────────────────────────────────────────────────────────
# Keys are the exact English strings passed to ctx.set_stage(). Grep for one of
# these and you find both the emitter and this table.
_STAGES: dict[str, dict[str, str]] = {
    # graph.py
    "Compacting the chat history": {"ru": "Сжимаю историю диалога"},
    "Removing the lettering":      {"ru": "Убираю надписи"},
    "Removing the lettering (logo)": {"ru": "Убираю надписи (логотип)"},
    "Looking at the image":        {"ru": "Смотрю на изображение"},
    "Looking at the video again":  {"ru": "Смотрю видео ещё раз"},
    "Choosing the sound":          {"ru": "Подбираю звучание"},
    "Writing a response":          {"ru": "Пишу ответ"},
    "Speaking":                    {"ru": "Озвучиваю"},
    # llm.py -- the card was handed to a render; this turn is waiting it out
    "Waiting for the graphics card": {"ru": "Жду видеокарту: идёт рендер"},
    "Starting the video engine": {"ru": "Запускаю видеодвижок"},
    "Restyling the video": {"ru": "Перерисовываю видео (~15 мин)"},
    "Translating":                 {"ru": "Перевожу"},
    # tools.py — search
    "Searching the web":           {"ru": "Ищу в интернете"},
    # tool_ozon_handlers.py / ozon_shopper.py
    "Searching Ozon":                {"ru": "Ищу на Озоне"},
    "Reading the Ozon card":         {"ru": "Читаю карточку товара"},
    "Reading Ozon reviews":          {"ru": "Читаю отзывы"},
    "Looking at the Ozon photos":    {"ru": "Смотрю фото товара"},
    "Choosing an Ozon pickup point": {"ru": "Выбираю пункт выдачи"},
    "Finding a photo online":      {"ru": "Ищу фото в интернете"},
    "Finding a reference photo":   {"ru": "Ищу референсное фото"},
    # tools.py — pictures
    "Drawing a picture":           {"ru": "Рисую картинку"},
    # draw_preview.py -- the plan's boxes as a sketch, checked before the render
    "Checking the layout":         {"ru": "Проверяю компоновку по эскизу"},
    "Redrawing the picture":       {"ru": "Перерисовываю картинку"},
    "Editing the picture":         {"ru": "Редактирую картинку"},
    "Removing from the picture":   {"ru": "Убираю с картинки"},
    "Inserting object":            {"ru": "Добавляю объект"},
    "Looking at the picture":      {"ru": "Рассматриваю картинку"},
    "Transferring between images": {"ru": "Переношу между изображениями"},
    "Fixing the hands":            {"ru": "Исправляю руки"},
    "Fixing the flaw":             {"ru": "Исправляю дефект"},
    "Converting the colours":      {"ru": "Меняю цвета"},
    # gui_image_fix.py — the desktop style dropdown
    "Changing style":              {"ru": "Меняю стиль"},
    # video.py — the slowest thing the assistant does; the user is watching this one
    "Generating a video":          {"ru": "Генерирую видео"},
    "Preparing the references":    {"ru": "Готовлю референсы"},
    "Encoding the video":          {"ru": "Кодирую видео"},
    # image.py
    # tool_image_handlers.py — the layout-first redraw paths
    "Redrawing from the picture's own layout":
                                   {"ru": "Перерисовываю по разметке картинки"},
    "Redrawing the photo":         {"ru": "Перерисовываю фото"},
    # music.py
    "Composing a song":            {"ru": "Сочиняю песню"},
    # tg_songs.py
    "Writing the lyrics":          {"ru": "Пишу текст песни"},
    # gui.py and the desktop tabs — the owner runs the app in Russian too
    "Working":                     {"ru": "Работаю"},
    "Cancelling":                  {"ru": "Отменяю"},
    "Switching model":             {"ru": "Переключаю модель"},
    "Ultra Search":                {"ru": "Ультра-поиск"},
    "Scanning the book":           {"ru": "Читаю книгу"},
    "Fixing artifact":             {"ru": "Исправляю артефакт"},
    "Fixing hands":                {"ru": "Исправляю руки"},
    # slides.py / create_presentation
    "Planning the presentation":   {"ru": "Планирую презентацию"},
    "Updating the presentation":   {"ru": "Обновляю презентацию"},
    # tg_bot / bot-owned
    "Deep Research":               {"ru": "Глубокое исследование"},
    "Reading your documents":      {"ru": "Читаю твои документы"},
    "Starting":                    {"ru": "Начинаю"},
    "Transcribing":                {"ru": "Расшифровываю"},
    "Searching documents":         {"ru": "Ищу по документам"},
    "Ready":                       {"ru": "Готово"},
}

# ── deep-research phase names ─────────────────────────────────────────────────
# deep_research.py passes these as the `phase` argument of its progress callback.
_PHASES: dict[str, dict[str, str]] = {
    "Searching":         {"ru": "Ищу"},
    "Crawling":          {"ru": "Читаю страницы"},
    "Deduplicating":     {"ru": "Убираю дубли"},
    "Expanding queries": {"ru": "Расширяю запросы"},
    "Verifying":         {"ru": "Проверяю"},
    "Building report":   {"ru": "Собираю отчёт"},
    "Complete":          {"ru": "Готово"},
}

# The unit abbreviations in the deep-research progress line.
_UNITS: dict[str, dict[str, str]] = {
    "src":   {"ru": "ист"},
    "pg":    {"ru": "стр"},
    "found": {"ru": "найдено"},
}

# "Searching · 12 src / 40 pg / 7 found"  (tools.py deep_research callback)
_DR_LINE = re.compile(
    r"^(?P<phase>.+?) · (?P<src>\d+) src / (?P<pg>\d+) pg / (?P<found>\d+) found$")

# "Retrying the model call (2/3)"  (llm.py)
_RETRY = re.compile(r"^Retrying the model call \((?P<n>\d+)/(?P<of>\d+)\)$")

# "Waiting for the model (2 ahead)"  (llm_gate.py)
_QUEUE = re.compile(r"^Waiting for the model \((?P<n>\d+) ahead\)$")

# "Ultra: <phase>"  (gui.py Ultra Search)
_ULTRA = re.compile(r"^Ultra: (?P<phase>.+)$")

# "Removing the lettering (2/2)"  (image_lettering_remove.py, passes > 1)
_CLOSER = re.compile(r"^Looking closer \((?P<n>\d+)/(?P<of>\d+)\)$")

_LETTERING_PASS = re.compile(r"^Removing the lettering \((?P<n>\d+)/(?P<of>\d+)\)$")


_OZON = [
    (re.compile(r"^planning the purchase$"), "планирую покупку"),
    (re.compile(r"^searching (.*)$"), "ищу {0}"),
    (re.compile(r"^choosing among (\d+) for (.*)$"), "выбираю из {0}: {1}"),
    (re.compile(r"^reading cards and reviews for (.*)$"), "читаю карточки и отзывы: {0}"),
    (re.compile(r"^looking at the photos of (.*)$"), "смотрю фото: {0}"),
    (re.compile(r"^checking what the web says about (.*)$"), "проверяю отзывы в интернете: {0}"),
]


def _pick(table: dict, key: str, lang: str) -> str | None:
    forms = table.get(key)
    if not forms:
        return None
    return forms.get(lang)


def translate(stage: str, lang: str = DEFAULT_LANG) -> str:
    """Return `stage` in `lang`, or unchanged when there is nothing to say.

    Never raises: a status message is decoration around real work, and a
    KeyError here would abort the turn that is trying to report progress.
    """
    if not stage or not isinstance(stage, str):
        return stage
    if not lang or lang == DEFAULT_LANG:
        return stage

    exact = _pick(_STAGES, stage, lang)
    if exact:
        return exact

    m = _DR_LINE.match(stage)
    if m:
        phase = _pick(_PHASES, m.group("phase"), lang) or m.group("phase")
        src = _pick(_UNITS, "src", lang) or "src"
        pg = _pick(_UNITS, "pg", lang) or "pg"
        found = _pick(_UNITS, "found", lang) or "found"
        return (f"{phase} · {m.group('src')} {src} / {m.group('pg')} {pg} / "
                f"{m.group('found')} {found}")

    m = _RETRY.match(stage)
    if m and lang == "ru":
        return f"Повторяю запрос к модели ({m.group('n')}/{m.group('of')})"

    m = _QUEUE.match(stage)
    if m and lang == "ru":
        return f"Жду своей очереди к модели (впереди: {m.group('n')})"

    m = _ULTRA.match(stage)
    if m:
        inner = translate(m.group("phase"), lang)
        return f"Ultra: {inner}"

    # "Ozon: searching 1/3 тарелки" (ozon_shopper._stage). The need is the
    # user's own words and stays as it is.
    if lang == "ru" and stage.startswith("Ozon: "):
        body = stage[6:]
        for pat, ru in _OZON:
            m = pat.match(body)
            if m:
                return "Озон: " + ru.format(*m.groups())
    m = _CLOSER.match(stage)
    if m and lang == "ru":
        return f"Рассматриваю поближе ({m.group('n')}/{m.group('of')})"

    m = _LETTERING_PASS.match(stage)
    if m and lang == "ru":
        return f"Убираю надписи ({m.group('n')}/{m.group('of')})"

    phase = _pick(_PHASES, stage, lang)
    if phase:
        return phase

    # Unknown stage — show it as-is rather than a blank or a key. This is the
    # status quo for anything not in the tables, so it can only be neutral.
    return stage


def known_stages() -> set:
    """Every English label this module can translate. Used by the test that keeps
    the tables in step with the ctx.set_stage() calls in the codebase."""
    return set(_STAGES) | set(_PHASES)
