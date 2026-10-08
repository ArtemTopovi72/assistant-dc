"""Song settings for the Telegram bot: genre, tempo, vocal, duration.

Shaped exactly like tg_depth.py's research-depth picker and the image-size
picker it was modelled on: a small inline menu whose keyboard STATES the
current setting (the ✅ moves), redrawn in place on every tap so the keyboard
can never lie about what is active.

Everything here resolves to a MiniMax Music 3 *explicit requirement*. Its
prompting guide defines a precedence order — explicit requirements first,
then section-tag directives, then implications of the description, then genre
defaults — so a setting the user actually chose is injected into the caption
brief as a requirement that must survive the whole caption, not as a hint
mixed in with the topic where the songwriter is free to reinterpret it.

Empty string / 0 means "not chosen", which is deliberately NOT the same as a
default value: unchosen leaves the songwriter free to pick what suits the
topic, chosen pins it. That is the same convention `sess.image_aspect` and
`sess.dr_depth` already use.
"""
from __future__ import annotations

import html as _html_mod
import logging
import re

import config as _config
import music as _music
from tg_strings import _DEFAULT_LANG, _nav_back_row, _t

logger = logging.getLogger("assistant.tg_music")

# ── the settings themselves ──────────────────────────────────────────────────
# The vocabulary (which genres exist, and the English phrase each becomes in
# the caption brief) belongs to the ENGINE, not to this surface — see the
# comment above music.GENRES. Re-exported under the old names so tg_bot's
# re-exports and every caller keep working, but there is exactly one copy: a
# genre added in music.py appears here and in the desktop tab at once, instead
# of two screens offering two different lists for the same engine.
GENRES = _music.GENRES
TEMPOS = _music.TEMPOS
VOCALS = _music.VOCALS
DURATIONS = _music.DURATIONS
_TEMPO_BPM = _music.TEMPO_BPM

PRESETS = _music.WEIGHT_PRESETS
# "quality" (Music3 weight presets + DiT steps) only while Music3 is the engine.
FIELDS: tuple = ("genre", "tempo", "vocal", "voice", "duration") + (
    ("quality",) if _music.MUSIC_ENGINE != "yue2" else ())


def voices() -> dict:
    """🎙 Own voice: {"auto": off} + the trained star voices (rvc_voice.stars()). A song
    rendered by the engine gets its lead re-sung in the chosen star's voice (10-09)."""
    out = {"auto": ""}
    try:
        import rvc_voice
        if rvc_voice.available():
            out.update(rvc_voice.stars())
    except Exception:
        logger.warning("star voices unavailable", exc_info=True)
    return out


def _table(field: str) -> dict:
    return {"genre": GENRES, "tempo": TEMPOS, "vocal": VOCALS}.get(field) or (voices() if field == "voice" else {})


# ── resolving persisted state ────────────────────────────────────────────────
def _resolve(sess, field: str) -> str:
    """The chosen value for `field`, or "" when unchosen/invalid.

    Reads user-controlled persisted state straight off a stored row, so a
    non-string sitting there — corrupted data, a hand edit, a future schema
    change — must not crash the keyboard that renders it. Same defensiveness
    as tg_depth._resolve_depth, for the same reason.
    """
    raw = getattr(sess, "music_" + field, "")
    if field == "duration":
        # "auto" = the bot picks the length from the topic; any number of
        # seconds inside DURATION_RANGE is a typed custom length.
        if isinstance(raw, str) and raw.strip().lower() == "auto":
            return "auto"
        try:
            n = int(raw or 0)
        except (TypeError, ValueError):
            return ""
        lo, hi = _music.DURATION_RANGE
        return str(n) if lo <= n <= hi else ""
    val = raw.lower() if isinstance(raw, str) else ""
    if field == "tempo" and _music.custom_bpm(val):
        return f"bpm:{_music.custom_bpm(val)}"      # a typed BPM
    if field == "quality":
        # Unlike the others there is no "no opinion" here: a render always uses
        # SOME weights, so an unchosen quality resolves to the default preset
        # rather than to Auto.
        return val if val in PRESETS else _music.DEFAULT_PRESET
    if field == "voice":
        return raw if isinstance(raw, str) and raw in voices() and raw != "auto" else ""
    table = _table(field)
    return val if val in table and val != "auto" else ""


def resolve_preset(sess) -> str:
    """Which weight preset this chat's songs render with."""
    return _resolve(sess, "quality")


STEPS_HOME = "quality"               # the screen the step count lives on

# A TYPED «сбрось настройки песни» / "reset song settings". Without this the
# phrase reached the chat model, which answered «Настройки песни сброшены.»
# and reset nothing (live, 15.09: the tempo stayed 118 BPM).
def is_reset_phrase(text: str) -> bool:
    import intent   # the model's read: song "reset"
    t = (text or "").strip()
    # the per-message read flags it; a narrow question confirms (the read also
    # took «сбрось фильтры» for one)
    return (bool(t) and len(t) < 80 and intent.read(None, t)["song"] == "reset"
            and intent.ask_choice("A user wrote: {text}\n\nWhat does the user ask to reset?", t,
                                  ("song", "music", "filters", "chat", "other", "nothing"))
            in ("song", "music"))


def reset_all(sess) -> None:
    """Every song setting back to Auto -- the one place the button and the
    typed phrase both go through."""
    for f in FIELDS:
        setattr(sess, "music_" + f, "")
    sess.music_steps = ""


def resolve_steps_for_generate(sess):
    """DiT steps to pass to generate_music: the chosen number in range, else
    None so build_workflow attaches the turbo LoRA (8 steps, cfg 1.7) --
    the same "unchosen == None, not a default" convention as every other
    field here, and the one that actually gates music.build_workflow's
    `if steps is None` turbo branch. Do not swap this for resolve_steps()
    at the generate_music call site: that one fills in a display default and
    would silently disable the turbo LoRA on every default-settings song."""
    raw = getattr(sess, "music_steps", "") or ""
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    lo, hi = _music.STEPS_RANGE
    return n if lo <= n <= hi else None


def resolve_steps(sess) -> int:
    """DiT steps for DISPLAY (menu labels, ETA quotes): the chosen number in
    range, else the turbo default (8) -- this must track whatever
    resolve_steps_for_generate(sess) being None actually renders at, or the
    menu shows a step count and ETA the render will not use."""
    n = resolve_steps_for_generate(sess)
    return n if n is not None else _config.MUSIC_STEPS_TURBO


def resolve_duration(sess) -> int:
    """The render length in seconds: the chosen one, else the config default."""
    chosen = _resolve(sess, "duration")
    if chosen == "auto":
        return 0                    # the bot decides (music.choose_auto_params)
    if chosen:
        return max(1, min(int(chosen), _config.MUSIC_MAX_SECONDS))
    return _config.MUSIC_DEFAULT_SECONDS


def auto_fields(sess) -> dict:
    """Which settings are left to the bot: {field: True} for each Auto one."""
    return {"genre": not _resolve(sess, "genre"),
            "tempo": not _resolve(sess, "tempo"),
            "vocal": not _resolve(sess, "vocal"),
            "duration": _resolve(sess, "duration") == "auto"}


def prefs_of(sess) -> dict:
    """The chosen settings as {field: english phrase} for the caption brief.

    Only fields the user actually picked appear — an unchosen field is absent
    rather than present-and-empty, so music.py can tell "no opinion" from a
    deliberate choice without re-deriving the Auto convention.
    """
    return _music.prefs_from(genre=_resolve(sess, "genre"),
                             tempo=_resolve(sess, "tempo"),
                             vocal=_resolve(sess, "vocal"))


# ── labels ───────────────────────────────────────────────────────────────────
def _value_label(field: str, value: str, lang: str) -> str:
    """The translated button text for one value of one field."""
    if field == "quality":
        # Names the cost alongside the name: the whole point of the choice is
        # the trade, and a label reading only "Max" hides that it is 6x slower.
        # ...measured, for the length the menu text names (ETA_QUOTE_SECONDS).
        key = value if value in PRESETS else _music.DEFAULT_PRESET
        return f"{_t('mq_' + value, lang)} · {_music.eta_label(_music.eta_seconds(key), lang)}"
    if field == "steps":
        return str(value)
    if field == "voice":
        return voices().get(value) or _t("m_voice_off", lang) if value else _t("m_voice_off", lang)
    if not value:
        return _t("m_auto", lang)
    if field == "duration":
        return _t("m_auto_bot", lang) if value == "auto" else _t("m_secs", lang, n=value)
    if field == "steps":
        return str(value)
    if field == "tempo" and _music.custom_bpm(value):
        return f"{_music.custom_bpm(value)} BPM"
    return _t({"genre": "mg_", "tempo": "mt_", "vocal": "mv_"}[field] + value, lang)


def _current_label(sess, field: str, lang: str) -> str:
    val = _resolve(sess, field)
    if field == "quality":
        # Menu line: «⚡ Быстро · ~5 мин · 20 шагов», the ETA at THIS step count.
        key = val if val in PRESETS else _music.DEFAULT_PRESET
        steps = resolve_steps(sess)
        return (f"{_t('mq_' + val, lang)} · "
                f"{_music.eta_label(_music.eta_seconds(key, steps=steps), lang)} · "
                f"{_t('ms_steps_short', lang, n=steps)}")
    if field == "duration" and not val:
        # Duration always HAS an effective value even unchosen, and showing
        # "Auto" for it would be a lie — the render is going to be exactly
        # MUSIC_DEFAULT_SECONDS long. Show the number, not the word.
        return _t("m_secs", lang, n=str(_config.MUSIC_DEFAULT_SECONDS))
    label = _value_label(field, val, lang)
    if field == "duration" and val == "auto":
        return _t("m_auto_bot", lang)
    if field == "tempo" and val and _TEMPO_BPM.get(val):
        return f"{label} · {_TEMPO_BPM[val]} BPM"
    return label


# ── keyboards ────────────────────────────────────────────────────────────────
def _music_menu_kb(sess, lang: str = _DEFAULT_LANG) -> dict:
    """The top level: one row per setting, each showing its current value."""
    rows = []
    for field in FIELDS:
        rows.append([{
            "text": f"{_t('ms_' + field, lang)}: {_current_label(sess, field, lang)}",
            "callback_data": "music:open:" + field,
        }])
    rows.append([{"text": _t("ms_reset", lang), "callback_data": "music:reset"}])
    rows.append(_nav_back_row(lang))
    return {"inline_keyboard": rows}


def _field_kb(sess, field: str, lang: str = _DEFAULT_LANG) -> dict:
    """The picker for one setting, ✅ on the active value, two per row.

    Every picker offers Auto first (except duration, which is always a real
    number — see _current_label) so a setting can be UNchosen again without
    hunting for a reset; ⬅ returns to the settings menu rather than out of
    the whole flow, since these are sub-screens of it.
    """
    cur = _resolve(sess, field)
    if field == "duration":
        values = ["auto"] + [str(d) for d in DURATIONS]
    elif field == "quality":
        values = list(PRESETS)          # fast / quality / max, in that order
        return _quality_kb(sess, cur, lang)
    else:
        values = list(_table(field))
        values = ["" if v == "auto" else v for v in values]
    rows, row = [], []
    per_row = 1 if field == "quality" else 2   # the cost label needs the width
    for v in values:
        label = _value_label(field, v, lang)
        if field == "tempo" and v and _TEMPO_BPM.get(v):
            label = f"{label} · {_TEMPO_BPM[v]}"
        row.append({"text": ("✅ " if cur == v else "") + label,
                    "callback_data": f"music:set:{field}:{v or 'auto'}"})
        if len(row) == per_row:
            rows.append(row); row = []
    if row:
        rows.append(row)
    if field in ("duration", "tempo"):
        # A typed value, within the engine's range. The button names the
        # current custom value when one is active, so it is not hidden.
        custom = ""
        if field == "duration" and cur and cur != "auto" and int(cur) not in DURATIONS:
            custom = _t("m_secs", lang, n=cur)
        if field == "tempo" and _music.custom_bpm(cur):
            custom = f"{_music.custom_bpm(cur)} BPM"
        rows.append([{"text": ("✅ " if custom else "") + _t("ms_custom_" + field, lang)
                      + (f" ({custom})" if custom else ""),
                      "callback_data": f"music:custom:{field}"}])
    rows.append([{"text": _t("ms_back", lang), "callback_data": "music:menu"}])
    return {"inline_keyboard": rows}


def _quality_kb(sess, cur: str, lang: str) -> dict:
    """The Quality screen: the weight presets (each with its measured time
    for a 180-second song at the CURRENT step count), then the DiT step
    count -- 20/25/30/40/50 or a typed one -- each step button quoting
    its own time on the current preset, so the trade is visible before
    the tap, not after ten minutes of waiting."""
    steps = resolve_steps(sess)
    preset = cur if cur in PRESETS else _music.DEFAULT_PRESET
    rows = []
    for v in PRESETS:
        rows.append([{"text": ("✅ " if cur == v else "")
                      + f"{_t('mq_' + v, lang)} · {_music.eta_label(_music.eta_seconds(v, steps=steps), lang)}",
                      "callback_data": f"music:set:quality:{v}"}])
    is_turbo = resolve_steps_for_generate(sess) is None
    rows.append([{"text": _t("ms_steps_head", lang, n=steps)
                  + (" " + _t("ms_turbo_on", lang) if is_turbo else ""),
                  "callback_data": "music:open:quality"}])
    row = [{"text": ("✅ " if is_turbo else "") + _t("ms_steps_auto", lang, n=_music.MUSIC_STEPS_TURBO),
            "callback_data": "music:set:steps:auto"}]
    for n in _music.STEPS_CHOICES:
        row.append({"text": ("✅ " if not is_turbo and steps == n else "")
                    + f"{n} · {_music.eta_label(_music.eta_seconds(preset, steps=n), lang)}",
                    "callback_data": f"music:set:steps:{n}"})
        if len(row) == 2:
            rows.append(row); row = []
    if row:
        rows.append(row)
    custom = str(steps) if not is_turbo and steps not in _music.STEPS_CHOICES else ""
    rows.append([{"text": ("✅ " if custom else "") + _t("ms_custom_steps", lang)
                  + (f" ({custom})" if custom else ""),
                  "callback_data": "music:custom:steps"}])
    rows.append([{"text": _t("ms_back", lang), "callback_data": "music:menu"}])
    return {"inline_keyboard": rows}


# ── a typed value (duration in seconds, tempo in BPM, steps) ─────────────────
CUSTOM_STATE = "music_custom:"       # reg_state prefix while the number is awaited


def custom_range(field: str) -> tuple:
    return {"duration": _music.DURATION_RANGE, "tempo": _music.BPM_RANGE,
            "steps": _music.STEPS_RANGE}[field]


def parse_custom(field: str, text: str):
    """The typed number for `field`, or None when it is not one in range."""
    m = re.search(r"\d{2,3}", text or "")
    if not m:
        return None
    n = int(m.group(0))
    lo, hi = custom_range(field)
    return n if lo <= n <= hi else None


def store_custom(sess, field: str, n: int) -> None:
    setattr(sess, "music_" + field, f"bpm:{n}" if field == "tempo" else str(n))


def chose_line(chosen: dict, lang: str = _DEFAULT_LANG) -> str:
    """«🎲 Бот выбрал сам: жанр — Pop · темп — 118 BPM · …» for the fields the
    bot picked (music.choose_auto_params), with its one-line reason."""
    parts = []
    if chosen.get("genre"):
        parts.append(f"{_t('ms_genre', lang).lower()} — {_t('mg_' + chosen['genre'], lang)}")
    if chosen.get("bpm"):
        parts.append(f"{_t('ms_tempo', lang).lower()} — {chosen['bpm']} BPM")
    if chosen.get("vocal"):
        parts.append(f"{_t('ms_vocal', lang).lower()} — {_t('mv_' + chosen['vocal'], lang)}")
    # YuE2 sings as long as the lyric is: a chosen length would be a promise it ignores.
    if chosen.get("seconds") and _music.MUSIC_ENGINE != "yue2":
        parts.append(f"{_t('ms_duration', lang).lower()} — {_t('m_secs', lang, n=chosen['seconds'])}")
    if not parts:
        return ""
    return _t("song_bot_chose", lang, items=_html_mod.escape(" · ".join(parts)),
              why=_html_mod.escape(chosen.get("why") or _t("song_bot_chose_why", lang)))


# ── texts ────────────────────────────────────────────────────────────────────
def _music_menu_text(sess, lang: str = _DEFAULT_LANG) -> str:
    return (_t("ms_title", lang) + "\n\n"
            + _html_mod.escape(_t("ms_hint", lang)))


def _field_text(sess, field: str, lang: str = _DEFAULT_LANG) -> str:
    body = _t("ms_pick_" + field, lang)
    return (_t("ms_field_title", lang, field=_t("ms_" + field, lang)) + "\n\n"
            + _html_mod.escape(body))


def summary(sess, lang: str = _DEFAULT_LANG) -> str:
    """One line naming every current setting — used in the song-topic prompt
    so the user sees what they are about to get BEFORE typing a topic."""
    return " · ".join(f"{_t('ms_' + f, lang)}: {_current_label(sess, f, lang)}"
                      for f in FIELDS)
