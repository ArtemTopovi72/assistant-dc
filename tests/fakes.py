"""Stand-ins handed to the code under test, never written over a module.

`Over(base, **overrides)` answers the overrides and falls through to `base` for the rest,
so a suite gets the real `video` module with a fake renderer in one object and nothing
global changes: the next suite in the same process sees the real module.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from seams import Over  # noqa: E402,F401


def posting(post, ctx=None):
    """`ctx` (a bare namespace when None) handing the code under test an http client whose
    `post` is `post`; everything else on it is the real requests."""
    import types
    import requests
    ctx = ctx if ctx is not None else types.SimpleNamespace()
    deps = getattr(ctx, "deps", None) or types.SimpleNamespace()
    deps.requests = Over(requests, post=post)
    ctx.deps = deps
    return ctx


def model_selector(models, gb=3.0):
    """The model_selector a SettingsDialog lists models from (`modules={"model_selector": ...}`):
    the real module answering `models` with every file `gb` GB. Set `.list_models` on it to
    change the list between populates."""
    import model_selector as real
    return Over(real, list_models=lambda base: models,
                _index_model_sizes=lambda d: {}, _size_gb=lambda mid, sizes: gb)


def deps_ctx(**deps):
    """A context that hands the code under test these stand-ins (core/seams.py `dep`)."""
    import types
    return types.SimpleNamespace(deps=types.SimpleNamespace(**deps))


# How the model reads the drafts the suites script (agent/reply_read.py): what each one
# claims. The readings themselves are scored live on bench/reply_read_set; a draft not
# listed here claims nothing.
CLAIMS = {
    "Вот твоя картинка с очками!": {"delivers": True},
    "Я добавил ей очки, готово.": {"did": ["draw"]},
    "Не получилось добавить очки, правка не применилась.": {"admits_failure": True},
    "Не удалось — инструмент вернул ошибку.": {"admits_failure": True},
    "Вот твоя картинка!": {"delivers": True},
    "Вот твоя картинка.": {"delivers": True},
    "Вот твоя картинка с котом.": {"delivers": True},
    "Готово! Я добавил ей очки.": {"did": ["draw"]},
    "Готово! Я нарисовал рыжего кота.": {"did": ["draw"]},
    "Готово! Я перерисовал изображение, добавив очки.": {"did": ["draw"]},
    "Я запаковал всё обратно. Ваш обновленный архив готов.": {"did": ["file"]},
    "Презентация про Рим готова, 8 слайдов.": {"did": ["file"]},
    "Я поправил файл settings.json, теперь он парсится.": {"did": ["edited_file"]},
    "Не получилось собрать архив — упаковка не сработала.": {"admits_failure": True},
    "План видео ниже! Я прописал в сценарии максимально эпичный момент.": {"delivers": True},
    "Я удалил все файлы из рабочей директории.": {"did": ["other"]},
    "Не получилось: инструмент отказал, папка не тронута.": {"admits_failure": True},
    "Я добавил очки на фото.": {"did": ["draw"]},
    "Готово, очки добавлены.": {"did": ["draw"]},
    "Готово! Часы убраны.": {"did": ["draw"]},
    "Не вышло: на руке нарисовались новые часы.": {"admits_failure": True},
    "Проверил: очки на месте.": {},
}


def claims(extra=None):
    """A context's `reply_read` answering from CLAIMS (plus `extra`)."""
    table = {**CLAIMS, **(extra or {})}
    return reads(reply=lambda t: table.get(t))["reply_read"]


def reads(read=None, yes=None, pick=None, reply=None):
    """The model's reads a suite scripts, as the seams core/seams.py looks up on a context:
    `read(text) -> dict | None` (merged over intent.FALLBACK), `yes(question, text) -> bool`,
    `pick(question, text) -> option | None`, `reply(reply_text) -> dict | None` (what a reply
    claims, merged over reply_read.NOTHING). Unscripted: what the offline reads answer."""
    import intent
    out = {}
    if reply is not None:
        import reply_read

        def read_reply(ctx, text, user=""):
            got = reply((text or "").strip()) if (text or "").strip() else None
            return {**reply_read.NOTHING, **got, "ok": True} if got else dict(reply_read.NOTHING)
        out["reply_read"] = read_reply
    if read is not None:
        def intent_read(ctx, text, previous="", attached=""):
            got = read((text or "").strip()) if (text or "").strip() else None
            return {**intent.FALLBACK, **got, "ok": True} if got else dict(intent.FALLBACK)
        out["intent_read"] = intent_read
    if yes is not None:
        def yes_no(question, text, default=False, wait=True):
            text = (text or "").strip()
            return bool(yes(question, text)) if text else default
        out["yes_no"] = yes_no
    if pick is not None:
        def choice(question, text, options, default=""):
            text = (text or "").strip()
            return (pick(question, text) or default) if text else default
        out["choice"] = choice
    return out


# What each sentence of a shot script is (media/video.read_sentences) is the model's read,
# scored live in bench/script_read_eval.py. Suites that pin part lengths script it with
# the word rules that read it before -- here only, as the scripted answer.
import re as _re
_SR_QUOTED = _re.compile(r'«([^»]*)»|"([^"]*)"|“([^”]*)”')
_SR_BEAT = _re.compile(r"[,.;:!?—]|\b(?:then|and then|after that|afterwards|while|before|until|"
                       r"затем|потом|после этого|и снова|снова)\b", _re.I)
_SR_HOLD = _re.compile(r"\b(?:for (?:a few|several|some|\d+) seconds?|for a while|"
                       r"несколько секунд|какое-то время)\b", _re.I)
_SR_LOOK = _re.compile(
    r"(?:the\s+)?(?:soundscape|sound(?:track)?|audio|music|ambien\w*|lighting|light is|style|look|"
    r"overall|throughout|colou?r|quality|camera|(?:in\s+)?the\s+background|background|setting|"
    r"scenery|visible|in\s+view)\b|(?:the\s+)?(?:lighting|look|tone|mood|atmosphere|palette)\b"
    r".{0,40}\b(?:is|remains|stays)\b|(?:the\s+)?(?:video|clip|scene|shot)\s+"
    r"(?:continues|starts|begins|opens|picks\s+up)\b", _re.I)
_SR_ONCE = _re.compile(r"(?:the\s+)?(?:video|clip|scene|shot)\s+(?:starts|begins|opens)\b"
                       r"|(?:the\s+)?camera\s+(?:\w+ly\s+)?(?:pans|tilts|zooms|dollies|pulls|"
                       r"pushes|moves|tracks|cranes|rises|lowers|orbits|swings|cuts)\b", _re.I)
_SR_CAMERA_AS = _re.compile(r"((?:the\s+)?camera\b[^.;]*?)\s+(?:as|while|when)\s+"
                            r"(?!it\b|they\b|he\b|she\b)(\w.+)", _re.I)


def script_model(sentences):
    rows = []
    for s in sentences:
        m = _SR_CAMERA_AS.match(s.strip())
        rest = _SR_QUOTED.sub(" ", s)
        kind = "once" if _SR_ONCE.match(s.strip()) else "look" if _SR_LOOK.match(s.strip()) else "action"
        rows.append({"kind": kind,
                     "beats": len([b for b in _SR_BEAT.split(rest) if b and any(c.isalpha() for c in b)]) or 1,
                     "hold": bool(_SR_HOLD.search(rest)),
                     "camera": m.group(1) if m else "", "rest": m.group(2) if m else ""})
    return {"sentences": rows}
