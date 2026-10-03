"""Mutation harness for tests/test_tg_localization.py.

Each mutant restores one of the pre-fix behaviours (a hardcoded English message,
a missing translation, a bad placeholder). A survivor means the suite would not
have caught that regression.

Run: PYTHONIOENCODING=utf-8 venv/Scripts/python.exe tests/_mut_localization.py
"""
import os, sys, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

SUITE = "tests/test_tg_localization.py"

MUTANTS = [
    ("approval message hardcoded English again",
     '_t("acct_approved", lang) + _help_text(lang),',
     '"Your account has been approved!" + _help_text(lang),'),

    ("logout message hardcoded English again",
     '_t("logged_out", lang, name=_html_mod.escape(user.name)),',
     '"Logged out. Send your password to log back in.",'),

    ("subscribe confirmation hardcoded English again",
     '                self._send_text(chat_id, _t("subscribed", lang),',
     '                self._send_text(chat_id, "Subscribed to notifications.",'),

    ("Draw submenu header hardcoded English again",
     '_t("menu_draw_title", lang),',
     '"\U0001f3a8 <b>Draw &amp; Edit</b>",'),

    ("a message loses its Russian translation",
     '"pass_updated":  {"en": "✅ Password updated.", "ru": "✅ Пароль обновлён."},',
     '"pass_updated":  {"en": "✅ Password updated.", "ru": "✅ Password updated."},'),

    ("a Russian translation loses a placeholder",
     '"ru": "✅ Имя изменено на <b>{name}</b>."},',
     '"ru": "✅ Имя изменено."},'),

    ("a template gains a placeholder its callers never pass",
     '"pass_updated":  {"en": "✅ Password updated.", "ru": "✅ Пароль обновлён."},',
     '"pass_updated":  {"en": "✅ Password updated {who}.", '
     '"ru": "✅ Пароль обновлён {who}."},'),

    ("the account menu falls back to the default language",
     '                    _t("ask_new_name", lang, name=_html_mod.escape(user.name)),',
     '                    _t("ask_new_name", "en", name=_html_mod.escape(user.name)),'),

    # ── the surfaces the first detector could not see ─────────────────────────
    ("image action buttons go back to a hardcoded English constant",
     '{"text": _t("img_upscale", lang), "callback_data": "upscale"},',
     '{"text": "🔍 Upscale 4×", "callback_data": "upscale"},'),

    ("the image keyboard ignores the user's language",
     "        ok = self._send_photo(chat_id, out_img, keyboard=_image_kb(lang))",
     '        ok = self._send_photo(chat_id, out_img, keyboard=_image_kb("en"))'),

    ("the account page header goes back to English",
     '            _t("acct_title", lang) + "\\n"',
     '            "👤 <b>Your Account</b>\\n\\n"'),

    ("an account button label goes back to English",
     '[{"text": _t("acct_logout_btn", lang),',
     '[{"text": "🚪 Log out / switch profile",'),

    ("the status page header goes back to English",
     '            _t("status_title", lang),',
     '            "📊 <b>Status</b>\\n",'),

    ("the status page prints the raw quota key again",
     'lines.append(f"• {_kind_label(k, lang)}: <b>{used.get(k, 0)}{cap}</b>")',
     'lines.append(f"• {k.replace(chr(95), chr(32))}: '
     '<b>{used.get(k, 0)}{cap}</b>")'),

    ("the admin panel header goes back to English",
     '            _t("adm_title", lang),',
     '            "🔐 <b>Admin Panel</b>\\n",'),

    ("the usage stats header goes back to English",
     '        lines = [_t("adm_stats_title", lang)]',
     '        lines = ["📊 <b>Usage — last 7 days</b>\\n"]'),

    ("a keyboard placeholder goes back to a hardcoded conditional",
     '        "input_field_placeholder": _t("ph_draw", lang),',
     '        "input_field_placeholder": "Draw or edit…" if lang == "en" '
     'else "Рисовать…",'),

    ("the library keyboard loses its placeholder again",
     '        "input_field_placeholder": _t("ph_library", lang),\n',
     ''),

    ("the / command menu is registered in one language only",
     '        self._api_post("setMyCommands", {\n            "language_code": "ru",',
     '        return\n        self._api_post("setMyCommands", {\n'
     '            "language_code": "ru",'),

    ("a command description loses its Russian form",
     '("docs",        "List your indexed documents",\n'
     '                        "Список твоих документов"),',
     '("docs",        "List your indexed documents",\n'
     '                        "List your indexed documents"),'),

    ("the account status prints the raw DB value again",
     '                 state=_t(f"st_{user.status}", lang)) + "\\n"',
     '                 state=user.status) + "\\n"'),
]

env = dict(os.environ, PYTHONIOENCODING="utf-8")
path = os.path.join(ROOT, "bot/tg_bot.py")
with open(path, encoding="utf-8") as fh:
    ORIGINAL = fh.read()

caught = survived = skipped = 0
for label, find, repl in MUTANTS:
    if ORIGINAL.count(find) != 1:
        print(f"SKIP (anchor x{ORIGINAL.count(find)})  {label}")
        skipped += 1
        continue
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(ORIGINAL.replace(find, repl, 1))
        r = subprocess.run([sys.executable, SUITE], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=300)
    finally:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(ORIGINAL)
    if r.returncode != 0:
        caught += 1
        fails = [l for l in (r.stdout or "").splitlines() if l.startswith("FAIL")]
        print(f"CAUGHT    {label}\n            by: "
              f"{(fails[0][6:].strip() if fails else '(crashed)')[:100]}")
    else:
        survived += 1
        print(f"SURVIVED  {label}   <-- not caught")

print()
print(f"{caught} caught, {survived} survived, {skipped} skipped ({len(MUTANTS)} mutants)")
sys.exit(1 if survived or skipped else 0)
