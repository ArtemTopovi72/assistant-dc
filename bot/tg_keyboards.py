"""The bot menus: every reply/inline keyboard the user can be shown.

Lifted out of tg_bot.py. These are pure builders — hand them a language (and,
for the size menu, a session) and they return a Telegram keyboard dict. They
decide nothing and send nothing.

Labels come from tg_strings via _b/_t, never from literals, so a keyboard
rendered before a language switch still matches by KEY. The lang default is
imported, not redeclared: it is bound into these signatures at def time, and a
local copy would silently answer in the wrong language.
"""
import html as _html_mod

import config as _config
from tg_strings import _DEFAULT_LANG, _b, _nav_back_row, _t


def _main_kb(voice_on: bool, is_admin: bool = False, lang: str = _DEFAULT_LANG) -> dict:
    # `is_admin` is accepted but deliberately unused: the Admin Panel lives in the
    # Settings menu (_settings_kb), not here. It is an operator tool, not one of
    # the six things a user reaches for, and a fourth row pushed the everyday
    # buttons up. The signature stays because ~40 call sites pass the flag and
    # _settings_kb still needs it. Do NOT re-add the row here.
    rows = [
        # Long labels («🎨 Творчество», «⚙️ Настройки») were cut off three to a
        # row on a phone; they get two-per-row or a row of their own.
        [_b("creativity", lang), _b("search", lang)],
        [_b("weather", lang),    _b("ozon", lang),    _b("account", lang)],
        [_b("feedback", lang),   _b("help", lang),    _b("stop", lang)],
        [_b("settings", lang)],
    ]
    return {
        "keyboard": rows,
        "resize_keyboard": True,
        "input_field_placeholder": _t("ph_main", lang),
    }


def _fwd_voice_kb(lang: str = _DEFAULT_LANG, fwd_id: str = "", board: bool = False,
                  cont: bool = False, pick: bool = False) -> dict:
    """What to do with a forwarded voice note. One definition, two callers —
    the ask and the re-ask after an unusable choice.

    `fwd_id` identifies WHICH forward this prompt is about (see _Session.
    fwd_transcript_id): a second forward arriving before this one is answered
    used to silently overwrite the single pending transcript, so tapping THIS
    (now stale) keyboard delivered the wrong voice note's content. Callers
    that don't pass one get the old bare "fwdv:<choice>" form, which the
    dispatch side still honours unconditionally — a keyboard sent before this
    id existed.
    """
    suffix = f":{fwd_id}" if fwd_id else ""
    rows = [[
        {"text": _t("fwd_voice_text", lang), "callback_data": "fwdv:text" + suffix},
        {"text": _t("fwd_voice_sum", lang),  "callback_data": "fwdv:sum" + suffix},
    ], [
        {"text": _t("fwd_voice_both", lang), "callback_data": "fwdv:both" + suffix},
    ]]
    # `board`: something was LOOKED at (a video, a кружок) -- its storyboard is
    # offered, never pushed: most of the time the words are what matters.
    if board:
        rows.append([{"text": _t("fwd_voice_board", lang), "callback_data": "fwdv:board" + suffix}])
    # one video: it can be continued; several videos/pictures: the user picks which one to work with
    # (nothing is attached to the chat by guesswork)
    if cont:
        rows.append([{"text": _t("fwd_continue", lang), "callback_data": "fwdv:cont" + suffix}])
    if pick:
        rows.append([{"text": _t("fwd_pick", lang), "callback_data": "fwdv:pick" + suffix}])
    rows.append([{"text": _t("fwd_voice_own", lang), "callback_data": "fwdv:own" + suffix}])
    return {"inline_keyboard": rows}


def _draw_kb(lang: str = _DEFAULT_LANG) -> dict:
    return {
        "keyboard": [
            [_b("gen_image", lang),  _b("edit_image", lang)],
            [_b("remove_obj", lang), _b("remove_text", lang)],
            [_b("regenerate", lang), _b("analyze", lang)],
            [_b("size", lang)],
            [_b("stop", lang),       _b("back", lang)],
        ],
        "resize_keyboard": True,
        "input_field_placeholder": _t("ph_draw", lang),
    }


def _weather_kb(lang: str = _DEFAULT_LANG) -> dict:
    """Every window reachable in ONE tap.

    Was a single "what to wear" button, which forced the whole flow through a
    24h lookup first: wanting 48h meant sitting through 24h and then pressing
    a follow-up button under the result. Each window is now its own entry
    point, and 🏙 City changes the remembered default so the other buttons
    never have to ask for it.
    """
    return {
        "keyboard": [
            [_b("wtw_now", lang), _b("wtw_24", lang), _b("wtw_48", lang)],
            [_b("wtw_date_btn", lang), _b("wtw_city_btn", lang)],
            [_b("back", lang)],
        ],
        "resize_keyboard": True,
        "input_field_placeholder": _t("ph_weather", lang),
    }


def _creativity_kb(lang: str = _DEFAULT_LANG) -> dict:
    # Grown past one screen, with labels cut to «Настройки пес…»: the images,
    # music and video tools each get a tier of their own, and a long label
    # gets a row of its own, so nothing is abbreviated.
    return _kb([[_b("cr_images", lang), _b("cr_music", lang)],
                [_b("cr_video", lang), _b("deck", lang)],
                [_b("sandbox_btn", lang)],
                [_b("back", lang)]], lang)


def _cr_images_kb(lang: str = _DEFAULT_LANG) -> dict:
    return _kb([[_b("draw", lang)],
                [_b("style_menu_btn", lang)],
                [_b("characters_btn", lang)],
                [_b("back", lang)]], lang)


def _cr_music_kb(lang: str = _DEFAULT_LANG) -> dict:
    return _kb([[_b("songs", lang), _b("cover_btn", lang)],
                [_b("song_setup", lang)],
                [_b("clone_btn", lang), _b("book_btn", lang)],
                [_b("lyr_improve_btn", lang), _b("lyr_write_btn", lang)],
                [_b("back", lang)]], lang)


def _cr_video_kb(lang: str = _DEFAULT_LANG) -> dict:
    return _kb([[_b("video_setup", lang)],
                [_b("restyle_btn", lang)],
                [_b("continue_btn", lang)],
                [_b("animate_btn", lang)],
                [_b("back", lang)]], lang)


def _kb(rows, lang):
    return {"keyboard": rows, "resize_keyboard": True,
            "input_field_placeholder": _t("ph_creativity", lang)}


def _ozon_kb(lang: str = _DEFAULT_LANG) -> dict:
    """🛒 Ozon. Every button arms a prefix for the NEXT message; plain text
    typed in this menu is a product search (see __menu_ozon__)."""
    return {
        "keyboard": [
            [_b("ozon_find", lang),    _b("ozon_photo", lang)],
            [_b("ozon_cheap", lang),   _b("ozon_best", lang)],
            [_b("ozon_fast", lang),    _b("ozon_compare", lang)],
            [_b("ozon_card", lang),    _b("ozon_reviews", lang)],
            [_b("ozon_basket", lang),  _b("ozon_cart", lang)],
            [_b("ozon_city", lang),    _b("back", lang)],
            [_b("stop", lang)],
        ],
        "resize_keyboard": True,
        "input_field_placeholder": _t("ph_ozon", lang),
    }


def _search_kb(lang: str = _DEFAULT_LANG) -> dict:
    return {
        "keyboard": [
            [_b("web_search", lang), _b("deep", lang)],
            [_b("depth", lang)],
            [_b("stop", lang),       _b("back", lang)],
        ],
        "resize_keyboard": True,
        "input_field_placeholder": _t("ph_search", lang),
    }


def _settings_kb(reply_mode, is_admin: bool = False,
                 lang: str = _DEFAULT_LANG) -> dict:
    if not isinstance(reply_mode, str):
        reply_mode = "both" if reply_mode else "text"
    # A fixed name, not the current value: a button labelled with the state
    # reads as "press to get this" and each press cycled to something else.
    # It opens an inline picker (_reply_mode_kb) that marks the current mode.
    rows = [
        [_b("reply_fmt", lang), _b("notif", lang)],
        [_b("remember", lang), _b("clear", lang)],
        [_b("library", lang),  _b("facts", lang)],
        [_b("lang", lang),     _b("status", lang)],
        [_b("think", lang),    _b("photo_file", lang)],
        [_b("my_voice", lang)],
        [_b("back", lang)],
    ]
    if is_admin:
        rows.insert(-1, [_b("admin", lang)])   # operator tool: last, above Back
    return {
        "keyboard": rows,
        "resize_keyboard": True,
        "input_field_placeholder": _t("ph_settings", lang),
    }


def _reply_mode_kb(reply_mode: str, lang: str = _DEFAULT_LANG) -> dict:
    return {"inline_keyboard": [[{"text": ("✅ " if m == reply_mode else "") + _b("reply_" + m, lang),
                                  "callback_data": "reply:" + m}]
                                for m in ("text", "both", "voice")] + [_nav_back_row(lang)]}


def _library_kb(use_docs: bool, lang: str = _DEFAULT_LANG) -> dict:
    return {
        "keyboard": [
            [_b("lib_list", lang)],
            [_b("lib_on", lang) if not use_docs else _b("lib_off", lang)],
            [_b("lib_clear", lang), _b("back", lang)],
        ],
        "resize_keyboard": True,
        "input_field_placeholder": _t("ph_library", lang),
    }


# ── image size picker ─────────────────────────────────────────────────────────
# Ratios are digits, so they read the same in every language; the glyph shows the
# shape at a glance. "auto" is a first-class choice, not the absence of one — it
# means "let the request decide the orientation" (image.fix_image_params).
_ASPECT_GLYPH: dict = {
    "1:1": "◻", "4:3": "▭", "3:4": "▯", "3:2": "▭",
    "2:3": "▯", "16:9": "▬", "9:16": "▮",
}
_ASPECT_ROWS = (("1:1", "4:3", "3:4"), ("3:2", "2:3"), ("16:9", "9:16"))
_QUALITY_ORDER = ("draft", "standard", "high", "ultra")


def _size_of(sess) -> tuple:
    return _config.resolve_image_size(getattr(sess, "image_aspect", "") or "",
                                      getattr(sess, "image_quality", "") or "")


def _size_summary(sess, lang: str, key: str = "size_set") -> str:
    w, h = _size_of(sess)
    return _t(key, lang, w=w, h=h, mp=f"{w * h / 1_000_000:.1f}")


def _size_menu_kb(sess, lang: str = _DEFAULT_LANG) -> dict:
    """Inline picker. The ✅ sits on the ACTIVE choice, so the keyboard always
    states the current setting — a picker that shows no state is a picker the
    user has to press to find out what it does."""
    cur_a = getattr(sess, "image_aspect", "") or ""
    cur_q = getattr(sess, "image_quality", "") or _config.DEFAULT_IMAGE_QUALITY
    rows = [[{"text": ("✅ " if not cur_a else "") + _t("size_auto", lang),
              "callback_data": "size:ar:auto"}]]
    for row in _ASPECT_ROWS:
        rows.append([
            {"text": ("✅ " if cur_a == ar else "") + f"{_ASPECT_GLYPH.get(ar, '')} {ar}",
             "callback_data": "size:ar:" + ar.replace(":", "x")}
            for ar in row
        ])
    rows.append([
        {"text": ("✅ " if cur_q == q else "") + _t("q_" + q, lang),
         "callback_data": "size:q:" + q}
        for q in _QUALITY_ORDER
    ])
    rows.append(_nav_back_row(lang))
    return {"inline_keyboard": rows}


def _size_menu_text(sess, lang: str = _DEFAULT_LANG) -> str:
    return (_size_summary(sess, lang, "size_title") + "\n\n"
            + _html_mod.escape(_t("size_hint", lang)))
