# -*- coding: utf-8 -*-
"""Mutation harness for the image-size feature.

Each mutant reintroduces one plausible defect. A SURVIVOR means the suite would
not have caught that defect, so the assertion behind it is decoration.

Run: venv/Scripts/python.exe tests/_mut_image_size.py
"""
import io
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE = os.path.join(ROOT, "tests", "test_image_size_choice.py")
PY = sys.executable

# (label, file, old, new)
MUTANTS = [
    # ── config.resolve_image_size ────────────────────────────────────────────
    ("config: max-side clamp removed", "core/config.py",
     "    over = max(w, h) / IMAGE_MAX_SIDE\n    if over > 1.0:\n        w, h = w / over, h / over",
     "    over = max(w, h) / IMAGE_MAX_SIDE"),
    ("config: aspect applied transposed", "core/config.py",
     "    scale = ((mp * 1_000_000) / float(rw * rh)) ** 0.5\n    w, h = rw * scale, rh * scale",
     "    scale = ((mp * 1_000_000) / float(rw * rh)) ** 0.5\n    w, h = rh * scale, rw * scale"),
    ("config: unknown aspect no longer falls back", "core/config.py",
     '    rw, rh = IMAGE_ASPECTS.get(aspect or "") or IMAGE_ASPECTS.get(\n        DEFAULT_IMAGE_ASPECT) or (16, 9)',
     '    rw, rh = IMAGE_ASPECTS.get(aspect or "") or (1, 1)'),
    ("config: unknown quality no longer falls back", "core/config.py",
     '    mp = IMAGE_QUALITIES.get(quality or "") or IMAGE_QUALITIES.get(\n        DEFAULT_IMAGE_QUALITY) or 1.6',
     '    mp = IMAGE_QUALITIES.get(quality or "") or 0.52'),
    ("config: snaps to 8 instead of 16", "core/config.py",
     "        return max(IMAGE_MIN_SIDE, int(round(n / 16.0)) * 16)",
     "        return max(IMAGE_MIN_SIDE, int(round(n / 8.0)) * 8)"),
    ("config: tiers collapse to one size", "core/config.py",
     '    "high":     1.6,',
     '    "high":     1.0,'),

    # ── image.normalize_resolution / fix_image_params ────────────────────────
    ("image: normalize crushes to the old constant (the original bug)", "imaging/image.py",
     "    over = max(w, h) / float(max_side)",
     "    return (544, 960) if h > w else (960, 544)\n    over = max(w, h) / float(max_side)"),
    ("image: min-side floor removed", "imaging/image.py",
     "    under = min_side / float(min(w, h))\n    if under > 1.0:\n        w, h = w * under, h * under",
     "    under = min_side / float(min(w, h))"),
    ("image: max-side clamp removed", "imaging/image.py",
     "    over = max(w, h) / float(max_side)\n    if over > 1.0:\n        w, h = w / over, h / over",
     "    over = max(w, h) / float(max_side)"),
    ("image: junk size no longer defaults", "imaging/image.py",
     "    if w <= 0 or h <= 0:\n        return _config.DEFAULT_WIDTH, _config.DEFAULT_HEIGHT",
     "    if w < 0 or h < 0:\n        return _config.DEFAULT_WIDTH, _config.DEFAULT_HEIGHT"),
    ("image: pinned aspect ignored", "imaging/image.py",
     "    if pinned:\n        return normalize_resolution(width, height)",
     "    if False:\n        return normalize_resolution(width, height)"),
    ("image: everything treated as pinned", "imaging/image.py",
     "    if pinned:\n        return normalize_resolution(width, height)",
     "    if True:\n        return normalize_resolution(width, height)"),

    # ── image: the ctx plumbing ──────────────────────────────────────────────
    ("image: session size ignores the ctx choice", "imaging/image.py",
     '    return _config.resolve_image_size(getattr(ctx, "image_aspect", "") or "",\n'
     '                                      getattr(ctx, "image_quality", "") or "")',
     "    return _config.DEFAULT_WIDTH, _config.DEFAULT_HEIGHT"),
    ("image: quality alone counts as a pin", "imaging/image.py",
     '    return bool(getattr(ctx, "image_aspect", ""))',
     '    return bool(getattr(ctx, "image_aspect", "") or getattr(ctx, "image_quality", ""))'),
    ("image: parse ignores the caller default", "imaging/image.py",
     "    width, height = defaults or (DEFAULT_WIDTH, DEFAULT_HEIGHT)",
     "    width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT"),
    ("image: an explicit width= no longer wins over the default", "imaging/image.py",
     '            elif key == "width":\n                v = int(val)\n                if 64 <= v <= 8192:\n                    width = v',
     '            elif key == "width":\n                v = int(val)\n                if 64 <= v <= 8192:\n                    pass'),
    ("image: build_prompt does not seed from the session", "imaging/image.py",
     "    return parse_generation_params(body, defaults=(dw, dh),\n"
     "                                   pinned=session_size_pinned(ctx))",
     "    return parse_generation_params(body)"),
    ("image: a pinned aspect no longer beats the model's orientation", "imaging/image.py",
     "                if _aspect and not pinned:",
     "                if _aspect:"),
    ("image: empty-LLM fallback loses the session size", "imaging/image.py",
     "    dw, dh = session_image_size(ctx)\n    if not response:",
     "    dw, dh = DEFAULT_WIDTH, DEFAULT_HEIGHT\n    if not response:"),

    # ── tools: the schema clamp ──────────────────────────────────────────────
    ("tools: schema clamp back to the old 1920 ceiling", "agent/tools.py",
     '        return min(getattr(config, "IMAGE_MAX_SIDE", 2048),\n'
     '                   max(getattr(config, "IMAGE_MIN_SIDE", 256), round(v / 8) * 8))',
     "        return min(1920, max(256, round(v / 8) * 8))"),

    # ── tg_bot: the picker ───────────────────────────────────────────────────
    ("tg: the tick never marks the chosen aspect", "bot/tg_bot.py",
     '            {"text": ("✅ " if cur_a == ar else "") + f"{_ASPECT_GLYPH.get(ar, \'\')} {ar}",',
     '            {"text": f"{_ASPECT_GLYPH.get(ar, \'\')} {ar}",'),
    ("tg: the tick never marks the chosen quality", "bot/tg_bot.py",
     '        {"text": ("✅ " if cur_q == q else "") + _t("q_" + q, lang),',
     '        {"text": _t("q_" + q, lang),'),
    ("tg: quality tick defaults to nothing when unset", "bot/tg_bot.py",
     '    cur_q = getattr(sess, "image_quality", "") or _config.DEFAULT_IMAGE_QUALITY',
     '    cur_q = getattr(sess, "image_quality", "")'),
    ("tg: one aspect dropped from the picker", "bot/tg_bot.py",
     '_ASPECT_ROWS = (("1:1", "4:3", "3:4"), ("3:2", "2:3"), ("16:9", "9:16"))',
     '_ASPECT_ROWS = (("1:1", "4:3", "3:4"), ("3:2", "2:3"), ("16:9",))'),
    ("tg: header shows the default instead of the chosen size", "bot/tg_bot.py",
     '    w, h = _size_of(sess)\n    return _t(key, lang, w=w, h=h, mp=f"{w * h / 1_000_000:.1f}")',
     '    w, h = _config.DEFAULT_WIDTH, _config.DEFAULT_HEIGHT\n'
     '    return _t(key, lang, w=w, h=h, mp=f"{w * h / 1_000_000:.1f}")'),
    ("tg: the picker header is not translated", "bot/tg_bot.py",
     '    return (_size_summary(sess, lang, "size_title") + "\\n\\n"\n'
     '            + _html_mod.escape(_t("size_hint", lang)))',
     '    return (_size_summary(sess, _DEFAULT_LANG, "size_title") + "\\n\\n"\n'
     '            + _html_mod.escape(_t("size_hint", _DEFAULT_LANG)))'),

    # ── tg_bot: the callback ─────────────────────────────────────────────────
    ("tg: the tap is never persisted", "bot/tg_bot.py",
     '                self._store.put(sess)\n                # Redraw in place',
     '                # Redraw in place'),
    # Both halves in ONE mutant: the validation below re-clears an "auto" that
    # leaked through, so removing either alone is an equivalent mutant and the
    # pair would mask each other forever.
    ("tg: Auto does not clear the pin (guard removed too)", "bot/tg_bot.py",
     '                    sess.image_aspect = "" if value == "auto" else value.replace("x", ":")\n'
     "                    if sess.image_aspect and sess.image_aspect not in _config.IMAGE_ASPECTS:\n"
     '                        logger.warning("size callback: unknown aspect %r", value)\n'
     '                        sess.image_aspect = ""',
     '                    sess.image_aspect = value.replace("x", ":")'),
    ("tg: an unknown aspect is accepted", "bot/tg_bot.py",
     "                    if sess.image_aspect and sess.image_aspect not in _config.IMAGE_ASPECTS:\n"
     "                        logger.warning(\"size callback: unknown aspect %r\", value)\n"
     '                        sess.image_aspect = ""',
     "                    pass"),
    ("tg: an unknown quality is accepted", "bot/tg_bot.py",
     "                    if value not in _config.IMAGE_QUALITIES:\n"
     '                        logger.warning("size callback: unknown quality %r", value)\n'
     "                        return\n                    sess.image_quality = value",
     "                    sess.image_quality = value"),
    ("tg: the picker is re-sent instead of redrawn", "bot/tg_bot.py",
     "                if msg_id:\n                    self._edit_text(chat_id, msg_id, _size_menu_text(sess, lang),",
     "                if False:\n                    self._edit_text(chat_id, msg_id, _size_menu_text(sess, lang),"),
    ("tg: nothing is sent when there is no message to edit", "bot/tg_bot.py",
     "                else:\n                    self._send_text(chat_id, _size_summary(sess, lang),\n"
     '                                    parse_mode="HTML")',
     "                else:\n                    pass"),
    ("tg: the choice is dropped from to_dict (lost on restart)", "bot/tg_bot.py",
     '                "image_aspect":      self.image_aspect,',
     '                "image_aspect":      "",'),
    ("tg: a reloaded session forgets the quality", "bot/tg_bot.py",
     '        self.image_quality: str = d.get("image_quality", "")',
     '        self.image_quality: str = ""'),
    ("tg: the Size button does nothing", "bot/tg_bot.py",
     '                if direct == "__image_size__":\n'
     "                    self._send_text(chat_id, _size_menu_text(sess, lang),",
     '                if False:\n'
     "                    self._send_text(chat_id, _size_menu_text(sess, lang),"),
    ("tg: the Size button is missing from the Draw menu", "bot/tg_bot.py",
     '            [_b("size", lang)],\n',
     ""),
    ("tg: the Size label is not translated", "bot/tg_bot.py",
     '    "size":       {"en": "📐 Size",            "ru": "📐 Размер"},',
     '    "size":       {"en": "📐 Size",            "ru": "📐 Size"},'),
    ("tg: /size is not advertised", "bot/tg_bot.py",
     '        ("size",        "Pick image size and aspect ratio",\n'
     '                        "Размер и пропорции картинки"),\n',
     ""),

    # ── tg_bot: the ctx swap ─────────────────────────────────────────────────
    ("tg: the session size never reaches ctx", "bot/tg_bot.py",
     "        ctx.image_aspect  = sess.image_aspect\n        ctx.image_quality = sess.image_quality",
     "        pass"),
    # REMOVED: "the ctx swap is never restored (leaks to the next caller)".
    # It tested a save/restore dance that no longer exists — every task gets its
    # OWN Context now, so there is nothing global to hand back. The mutant above
    # ("the size swap ignores the session") covers what still has to hold.
]


def run_suite():
    r = subprocess.run([PY, SUITE], cwd=ROOT, capture_output=True,
                       encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return r.returncode == 0, (r.stdout or "") + (r.stderr or "")


def main():
    ok, out = run_suite()
    if not ok:
        print("BASELINE IS RED — fix the suite before mutating.")
        print(out[-3000:])
        return 1
    print("baseline green\n")

    survivors = []
    for i, (label, fname, old, new) in enumerate(MUTANTS, 1):
        path = os.path.join(ROOT, fname)
        src = io.open(path, encoding="utf-8").read()
        if old not in src:
            print(f"{i:2}. SKIP (anchor missing) {label}")
            survivors.append(label + "  [ANCHOR MISSING — treat as a failure]")
            continue
        io.open(path, "w", encoding="utf-8").write(src.replace(old, new, 1))
        try:
            passed, _ = run_suite()
        finally:
            io.open(path, "w", encoding="utf-8").write(src)
        if passed:
            print(f"{i:2}. SURVIVED  {label}")
            survivors.append(label)
        else:
            print(f"{i:2}. killed    {label}")

    print()
    print(f"{len(MUTANTS) - len(survivors)}/{len(MUTANTS)} mutants killed")
    for s in survivors:
        print("  SURVIVOR: " + s)
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
