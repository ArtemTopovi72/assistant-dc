"""The bot is bilingual — all of it, not just the buttons.

Task "Localization RU/EN" translated the keyboards and the error strings, but 38
user-facing messages stayed hardcoded English, and they sat on the paths that
matter most: the approval message a new user reads first, the whole account menu
(name, password, replace profile, logout), feedback, subscribe/unsubscribe, and
the Draw/Search submenu headers. `approve_user` even computed `lang` and passed it
to `_help_text()`, so a Russian user got an English paragraph followed by Russian
help in the same message.

Three layers of protection here:

  1. an AST invariant — no _send_text/_edit_text call may pass a bare English
     string literal. This is what actually stops the regression coming back;
  2. table integrity — every key has both languages, and the {placeholders} match.
     `_t` swallows KeyError and returns the raw template, so a placeholder typo in
     one language ships "{name}" to the user instead of raising;
  3. behaviour — a Russian user walked through the account flow gets Cyrillic.

Run: venv/Scripts/python.exe tests/test_tg_localization.py
"""
import sys, os, re, ast, types, tempfile, glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_loc_")
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


# TelegramBot lives across tg_bot.py plus its mixin modules. Scanning only
# tg_bot.py after the split would make every AST check below silently vacuous
# instead of red, so read the whole family.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Derived, not hand-listed. A hardcoded list fails in the dangerous direction:
# splitting a module out of tg_bot silently NARROWS the scan instead of turning
# this suite red, so newly-moved strings stop being checked and nobody is told.
# Every tg_*.py in the repo root is a send surface until proven otherwise.
SRC_FILES = sorted(os.path.relpath(p, _ROOT).replace("\\", "/") for p in glob.glob(os.path.join(_ROOT, "bot", "tg_*.py")))
assert "bot/tg_bot.py" in SRC_FILES and len(SRC_FILES) >= 9, SRC_FILES
SRC_PARTS = {f: open(os.path.join(_ROOT, f), encoding="utf-8").read()
             for f in SRC_FILES}
SRC_PATH = os.path.join(_ROOT, "bot/tg_bot.py")
SRC = chr(10).join(SRC_PARTS.values())


def _walk_src():
    """Every AST node across tg_bot.py and its mixin modules."""
    import ast as _a
    for _f, _text in SRC_PARTS.items():
        for _node in _a.walk(_a.parse(_text)):
            yield _node


def _called_name(node):
    """Name of a Call's callee, seeing through the `tg_bot.` qualifier the
    mixin modules use to read helpers back through their defining module."""
    f = getattr(node, "func", None)
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None

print("=" * 70)
print("INVARIANT: NO HARDCODED ENGLISH REACHES A USER")
print("=" * 70)

SENDERS = {"_send_text", "_edit_text", "_send_get_id"}
LATIN = re.compile(r"[A-Za-z]{3,}")

def _uses_translator(node):
    return any(isinstance(n, ast.Call) and _called_name(n) in ("_t", "_b")
               for n in ast.walk(node))

def _string_parts(node):
    return " ".join(n.value for n in ast.walk(node)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str))

offenders = []
for n in _walk_src():
    if not isinstance(n, ast.Call):
        continue
    f = n.func
    fname = f.attr if isinstance(f, ast.Attribute) else \
            (f.id if isinstance(f, ast.Name) else "")
    if fname not in SENDERS or len(n.args) < 2 or _uses_translator(n.args[1]):
        continue
    text = _string_parts(n.args[1])
    if text and LATIN.search(re.sub(r"<[^>]+>", "", text)):
        offenders.append((n.lineno, text[:70].replace("\n", " ")))

check("no _send_text/_edit_text passes a bare English literal",
      not offenders,
      "; ".join(f"line {ln}: {t}" for ln, t in offenders[:4]))

# The check above is necessary but nowhere near sufficient: it only sees a LITERAL
# passed directly as the second argument. Everything below was invisible to it and
# was, in fact, still English —
#
#   · bodies built into a local ("text = ...", "lines = [...]") and sent later:
#     the whole Account page, the Admin panel, the Status page, usage stats;
#   · inline-keyboard button labels: the six actions under every generated image,
#     and every button on the Account page;
#   · keyboard input_field_placeholders;
#   · caption= on photos and documents.
#
# So instead of trusting the call shape, walk the FUNCTIONS that talk to users and
# reject English prose anywhere inside them.
USER_FACING_FUNCS = {
    "_send_account_menu", "_send_admin_panel", "_send_admin_stats", "_send_status",
    "_send_facts", "_send_library_list", "_send_lang_menu", "_send_report_file",
    "_logout", "_confirm_broadcast", "_index_document", "_broadcast_startup",
    "_broadcast_shutdown", "_recover_inflight", "_send_admin_panel",
    "_image_kb", "_main_kb", "_draw_kb", "_search_kb", "_settings_kb", "_library_kb",
    "_help_text", "_settings_header",
}
# Strings that are structure, not prose: Telegram API field names, callback ids,
# session/DB keys, config names, format scaffolding.
STRUCTURAL = re.compile(
    r"^(inline_keyboard|callback_data|resize_keyboard|input_field_placeholder|"
    r"reply_markup|language_code|parse_mode|keyboard|text|HTML|"
    r"[a-z_]+:|__[a-z_]+__|[A-Z][A-Z0-9_]+|acct_[a-z_]+|admin_[a-z_]+|bcast:[a-z]+|"
    r"[a-z_]+\.(json|jsonl|txt|db))$")

def _prose(v):
    """English prose = two or more Latin words, separated by real whitespace.

    The whitespace rule is what distinguishes a sentence from an identifier:
    "img_upscale" and "acct_toggle_startup" are translation keys and callback ids,
    not text anybody reads.
    """
    if STRUCTURAL.match(v.strip()) or not re.search(r"\s", v.strip()):
        return False
    plain = re.sub(r"<[^>]+>", "", v)
    return len(re.findall(r"[A-Za-z]{3,}", plain)) >= 2

# Operator-facing text stays English on purpose: the log file and the desktop
# activity feed are read by whoever runs the bot, not by its users.
OPERATOR_SINKS = {"debug", "info", "warning", "error", "exception", "critical", "log"}

def _operator_lines(node):
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else \
                   (f.id if isinstance(f, ast.Name) else "")
            if name in OPERATOR_SINKS:
                out.update(x.lineno for x in ast.walk(n) if hasattr(x, "lineno"))
    return out

deep = []
for fn in _walk_src():
    if not isinstance(fn, ast.FunctionDef) or fn.name not in USER_FACING_FUNCS:
        continue
    body = fn.body[1:] if (fn.body and isinstance(fn.body[0], ast.Expr)
                           and isinstance(fn.body[0].value, ast.Constant)) else fn.body
    skip = _operator_lines(fn)
    for stmt in body:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) \
                    and n.lineno not in skip and _prose(n.value):
                deep.append((n.lineno, fn.name, n.value[:60].replace("\n", " ")))

check("no user-facing builder contains English prose", not deep,
      "; ".join(f"{f}():{ln} {t}" for ln, f, t in deep[:5]))

# Telegram draws the "/" command menu itself, so it never passes through a sender —
# it has to be registered per language or Russian users get an English menu.
cmds = getattr(T.TelegramBot, "_COMMANDS", None)
check("the / command menu is defined once per language",
      isinstance(cmds, (list, tuple)) and cmds
      and all(len(row) == 3 and all(isinstance(x, str) and x for x in row)
              for row in cmds),
      repr(cmds)[:80])
if cmds:
    CYR_ = re.compile(r"[А-Яа-яЁё]")
    untranslated = [c for c, en, ru in cmds if not CYR_.search(ru)]
    check("every command description has a Russian form",
          not untranslated, ", ".join(untranslated))

    registered = []
    bot_r = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {},
                          silent_mode=True)
    bot_r._api_post = lambda m, p=None, **k: (registered.append((m, p)), {})[1]
    bot_r._register_commands()
    langs = [(p or {}).get("language_code") for m, p in registered
             if m == "setMyCommands"]
    check("setMyCommands is called for the default locale and for ru",
          None in langs and "ru" in langs, str(langs))


print()
print("=" * 70)
print("TABLE INTEGRITY")
print("=" * 70)

# Names may carry a format spec — "{elapsed:.0f}" is one placeholder named elapsed.
PLACEHOLDER = re.compile(r"\{(\w+)(?::[^}]*)?\}")

# Terms Russian genuinely writes in Latin script too. Russian music writing
# uses "EDM", "R&B" and "Lo-fi" as-is; a Cyrillic transliteration here would
# be an invented word no reader uses, so identical is the CORRECT translation
# rather than a missing one. Deliberately a short, named list: anything not on
# it is still required to differ between the two languages.
_SAME_IN_BOTH = {"mg_edm", "mg_rnb", "mg_lofi"}

missing_ru, mismatched, identical = [], [], []
for key, forms in T._MSG.items():
    if "en" not in forms:
        missing_ru.append(f"{key}: no en")
        continue
    if "ru" not in forms:
        missing_ru.append(f"{key}: no ru")
        continue
    en, ru = forms["en"], forms["ru"]
    if set(PLACEHOLDER.findall(en)) != set(PLACEHOLDER.findall(ru)):
        mismatched.append(f"{key}: {PLACEHOLDER.findall(en)} vs {PLACEHOLDER.findall(ru)}")
    # A translation that is byte-identical is either untranslated or emoji-only.
    # Placeholder NAMES are stripped before the Latin check alongside the HTML
    # tags: "{field}" is substituted with already-translated text at render
    # time, so a format-only template like "🎛 <b>{field}</b>" is correctly
    # identical in both languages and its Latin characters are never seen by
    # anyone. Without this the detector flags every such wrapper as untranslated.
    _visible = PLACEHOLDER.sub("", re.sub(r"<[^>]+>", "", en))
    if en == ru and LATIN.search(_visible) and key not in _SAME_IN_BOTH:
        identical.append(key)

check("every message has both languages", not missing_ru, "; ".join(missing_ru[:5]))
check("placeholders match between en and ru", not mismatched,
      "; ".join(mismatched[:5]))
check("no message is left as untranslated English in the ru slot",
      not identical, ", ".join(identical[:8]))

# _t swallows formatting errors, so prove every template renders with its own keys.
broken = []
for key, forms in T._MSG.items():
    names = set(PLACEHOLDER.findall(forms.get("en", "")))
    # 1.0 satisfies both plain "{n}" and numeric specs like "{elapsed:.0f}".
    kw = {n: 1.0 for n in names}
    for lang in ("en", "ru"):
        rendered = T._t(key, lang, **kw)
        if PLACEHOLDER.search(rendered):
            broken.append(f"{key}/{lang}: {rendered[:50]}")
check("every template renders with no placeholder left behind",
      not broken, "; ".join(broken[:5]))

# The check above only proves a template renders when you HAND it every name it
# asks for. The failure that actually reaches a user is the other one: a template
# that grew a placeholder no caller passes. `_t` catches the KeyError and returns
# the raw text, so "{who}" is shipped verbatim. Check the call sites.
_TSIG = {}
for n in _walk_src():
    if not (isinstance(n, ast.Call) and _called_name(n) == "_t" and n.args):
        continue
    key_node = n.args[0]
    if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
        continue                       # computed key — nothing to verify statically
    if any(kw.arg is None for kw in n.keywords):
        continue                       # **kwargs — opaque
    _TSIG.setdefault(key_node.value, []).append(
        (n.lineno, {kw.arg for kw in n.keywords}))

starved = []
for key, calls in _TSIG.items():
    needed = set(PLACEHOLDER.findall((T._MSG.get(key) or {}).get("en", "")))
    for lineno, passed in calls:
        if needed - passed:
            starved.append(f"line {lineno} _t({key!r}) misses {sorted(needed - passed)}")
check("every _t() call passes the placeholders its template needs",
      not starved, "; ".join(starved[:5]))

check("an unknown key degrades to the key, not a crash",
      T._t("no_such_key_at_all", "ru") == "no_such_key_at_all")
check("an unknown language falls back to the house default, not to the raw key",
      T._t("cancelled", "de") == T._t("cancelled", T._DEFAULT_LANG)
      and T._t("cancelled", "de") != "cancelled",
      f"{T._DEFAULT_LANG}: {T._t('cancelled', 'de')!r}")
check("Russian is the house default", T._DEFAULT_LANG == "ru", T._DEFAULT_LANG)
check("and English is still selectable", "en" in T._LANGS, T._LANGS)


print()
print("=" * 70)
print("BEHAVIOUR: A RUSSIAN USER SEES RUSSIAN")
print("=" * 70)

CYR = re.compile(r"[А-Яа-яЁё]")
CID = 999701


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None,
                        lambda: {}, silent_mode=True)
    bot.sent = []
    bot._send_text = lambda cid, text, **kw: (bot.sent.append((cid, text)), 1)[1]
    bot._send_get_id = lambda cid, text, **kw: (bot.sent.append((cid, text)), 1)[1]
    bot._activity.log = lambda *a, **k: None
    bot._delete = lambda *a, **k: None
    bot._backend = types.SimpleNamespace(push=lambda t: None, depth=lambda: 0,
                                         name=lambda: "stub", close=lambda: None,
                                         drop_chat=lambda c: 0)
    return bot


bot = make_bot()
user = T._User(chat_id=CID, name="Артём", status="approved",
               password_hash=T._hash_password("secret", CID))
bot._user_store.put(user)
sess = bot._get_session(CID)
sess.lang = "ru"
bot._store.put(sess)


def last(bot):
    return bot.sent[-1][1] if bot.sent else ""


def russian(name, fn, *a, **kw):
    bot.sent.clear()
    fn(*a, **kw)
    txt = " ".join(t for _c, t in bot.sent)
    check(name, bool(txt) and bool(CYR.search(txt)) and not re.search(
              r"\b(Send|Cancelled|Password|Name|Welcome|Subscribed|Unsubscribed|"
              r"account|approved|declined)\b", txt),
          repr(txt[:90]))


def msg(text):
    return {"message_id": 1, "chat": {"id": CID}, "text": text,
            "from": {"id": CID, "username": "art"}}


russian("approval message", bot.approve_user, CID)

# Account edit prompts and their outcomes.
sess = bot._get_session(CID); sess.reg_state = "change_name"; bot._store.put(sess)
russian("name too short", bot._user_gate, CID, msg("A"))

sess = bot._get_session(CID); sess.reg_state = "change_name"; bot._store.put(sess)
russian("name updated", bot._user_gate, CID, msg("Иваний"))

sess = bot._get_session(CID); sess.reg_state = "change_password"; bot._store.put(sess)
russian("password updated", bot._user_gate, CID, msg("newsecret"))

sess = bot._get_session(CID); sess.reg_state = "change_name"; bot._store.put(sess)
russian("/cancel out of an edit", bot._user_gate, CID, msg("/cancel"))

sess = bot._get_session(CID); sess.reg_state = "feedback"; bot._store.put(sess)
russian("feedback accepted", bot._user_gate, CID, msg("всё отлично"))

russian("/subscribe", bot._handle_command, CID, "/subscribe")
russian("/unsubscribe", bot._handle_command, CID, "/unsubscribe")
russian("/setname prompt", bot._handle_command, CID, "/setname")
russian("/setpassword prompt", bot._handle_command, CID, "/setpassword")
russian("logout", bot._logout, CID, bot._user_store.get(CID), bot._get_session(CID))

# Submenu headers and the notifications page come through the resolve path.
def push_label(key):
    label = T._b(key, "ru")
    bot._resolve_and_push(CID, [{"type": "text", "text": label}])

for key, name in (("draw", "Draw submenu header"),
                  ("search", "Search submenu header"),
                  ("notif", "Notifications page")):
    russian(name, push_label, key)

# The suspension / rejection notices go out after the user is gone from the menu.
bot._user_store.put(T._User(chat_id=CID, name="Артём", status="approved"))
russian("account rejected notice",
        lambda cid: bot.reject_user(cid, allow_from_any_status=True), CID)
bot._user_store.put(T._User(chat_id=CID, name="Артём", status="approved"))
russian("account banned notice", bot.ban_user, CID)
bot._user_store.put(T._User(chat_id=CID, name="Артём", status="approved"))
russian("admin granted notice", bot.make_admin, CID)
russian("admin revoked notice", bot.revoke_admin, CID)


print()
print("=" * 70)
print("THE PAGES ASSEMBLED FROM PARTS")
print("=" * 70)

# These build their body into a local and send the variable, which is exactly why
# the first detector never saw them and they stayed English the longest.
bot._user_store.put(T._User(chat_id=CID, name="Артём", status="approved",
                            tg_username="art", subscriptions=["startup"]))
bot._backend = types.SimpleNamespace(push=lambda t: None, depth=lambda: 0,
                                     name=lambda: "InMemoryBackend",
                                     close=lambda: None, drop_chat=lambda c: 0)
bot._start_time = 0.0

russian("account page", bot._send_account_menu, CID,
        bot._user_store.get(CID), bot._get_session(CID))

# The status page probes two HTTP services; keep it offline and instant.
import requests as _rq
_real_get = _rq.get
_rq.get = lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
try:
    russian("status page", bot._send_status, CID, bot._get_session(CID))
finally:
    _rq.get = _real_get

adm = T._User(chat_id=CID, name="Артём", status="approved", is_admin=True)
bot._user_store.put(adm)
russian("admin panel", bot._send_admin_panel, CID)

bot._user_store.bump_usage(CID, T.KIND_TASK, 3)
russian("usage stats", bot._send_admin_stats, CID)

# "Contains Cyrillic" is too weak for a page that is mostly Cyrillic already: one
# English header survives it unnoticed. Assert the specific translated pieces.
def contains(name, fn, args, keys):
    bot.sent.clear()
    fn(*args)
    txt = " ".join(t for _c, t in bot.sent)
    missing = [k for k in keys if k not in txt]
    check(name, not missing, f"missing {missing} in {txt[:100]!r}")

_rq.get = lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
try:
    contains("status page uses the translated header and quota names",
             bot._send_status, (CID, bot._get_session(CID)),
             [T._t("status_title", "ru").strip(),
              T._t("status_usage", "ru"),
              T._kind_label(T.KIND_RESEARCH, "ru"),
              T._kind_label(T.KIND_IMAGE, "ru")])
finally:
    _rq.get = _real_get

contains("admin panel uses the translated header and sections",
         bot._send_admin_panel, (CID,),
         [T._t("adm_title", "ru").strip(), T._t("adm_bcast_tip", "ru").strip()])

contains("account page uses the translated header, status and buttons",
         bot._send_account_menu,
         (CID, bot._user_store.get(CID), bot._get_session(CID)),
         [T._t("acct_title", "ru").strip(), T._t("st_approved", "ru"),
          T._t("acct_ask", "ru").strip()])

contains("usage stats use the translated header and quota names",
         bot._send_admin_stats, (CID,),
         [T._t("adm_stats_title", "ru").strip(),
          T._kind_label(T.KIND_TASK, "ru")])

# The raw DB/quota keys must never reach a user in either language.
bot.sent.clear()
_rq.get = lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
try:
    bot._send_status(CID, bot._get_session(CID))
finally:
    _rq.get = _real_get
bot._send_account_menu(CID, bot._user_store.get(CID), bot._get_session(CID))
page = " ".join(t for _c, t in bot.sent)
leaked = [k for k in ("deep_research", "deep research", "approved", "pending")
          if k in page]
check("no raw quota or status key leaks into a Russian page",
      not leaked, f"{leaked} in {page[:120]!r}")

bot.sent.clear(); bot._send_facts(CID, bot._get_session(CID))
check("facts page is Russian",
      bool(CYR.search(last(bot))), repr(last(bot)[:80]))

bot.sent.clear(); bot._send_library_list(CID, bot._get_session(CID))
check("empty-library notice is Russian",
      bool(CYR.search(last(bot))), repr(last(bot)[:80]))


print()
print("=" * 70)
print("KEYBOARDS AND BUTTON LABELS")
print("=" * 70)

def kb_texts(kb):
    rows = (kb or {}).get("inline_keyboard") or []
    return [b.get("text", "") for row in rows for b in row]

for lang, want_cyr in (("ru", True), ("en", False)):
    labels = kb_texts(T._image_kb(lang))
    # 8: ask, change_clothes, remove_object, remove_text, regenerate, edit, style, animate
    # (2026-09-23: upscale, enhance, outpaint and restore went with their
    # engines; 2026-09-24: 🧽 remove_object added; 2026-09-25: 🔤 remove_text).
    check(f"the eight image-action buttons exist ({lang})", len(labels) == 8,
          str(labels))
    check(f"image-action buttons are {'Russian' if want_cyr else 'English'}",
          all(bool(CYR.search(t)) == want_cyr for t in labels), str(labels))

# Building the keyboard correctly is worthless if the delivery path hands it a
# constant. Every _image_kb() call must pass a variable, never a literal.
img_calls = [n for n in _walk_src()
             if isinstance(n, ast.Call) and _called_name(n) == "_image_kb"]
hardcoded = [n.lineno for n in img_calls
             if n.args and isinstance(n.args[0], ast.Constant)]
check("the image keyboard is always built for the caller's language",
      img_calls and not hardcoded, f"literal lang at lines {hardcoded}")

check("image buttons keep their callback_data across languages",
      [b["callback_data"] for row in T._image_kb("ru")["inline_keyboard"] for b in row]
      == [b["callback_data"] for row in T._image_kb("en")["inline_keyboard"] for b in row])

for builder, name in ((T._main_kb, "main"), (T._draw_kb, "draw"),
                      (T._search_kb, "search"), (T._settings_kb, "settings"),
                      (T._library_kb, "library")):
    # _main_kb/_settings_kb/_library_kb all take a state flag before `lang`.
    stateful = name in ("main", "settings", "library")
    kb_ru = builder(True, lang="ru") if stateful else builder("ru")
    kb_en = builder(True, lang="en") if stateful else builder("en")
    ph_ru = kb_ru.get("input_field_placeholder", "")
    ph_en = kb_en.get("input_field_placeholder", "")
    check(f"the {name} keyboard has a placeholder in both languages",
          bool(ph_ru) and bool(ph_en), f"{ph_ru!r} / {ph_en!r}")
    check(f"the {name} placeholder is actually translated",
          bool(CYR.search(ph_ru)) and not CYR.search(ph_en),
          f"{ph_ru!r} / {ph_en!r}")

# Every account-page button must be translated, and its callback ids must not move.
def acct_buttons(lang):
    s2 = bot._get_session(CID); s2.lang = lang; bot._store.put(s2)
    bot.sent.clear()
    seen = {}
    real = bot._send_text
    bot._send_text = lambda cid, text, **kw: (seen.update(kw), real(cid, text, **kw))
    try:
        bot._send_account_menu(CID, bot._user_store.get(CID), bot._get_session(CID))
    finally:
        bot._send_text = real
    return seen.get("keyboard") or {}

kb_ru_a, kb_en_a = acct_buttons("ru"), acct_buttons("en")
check("account page offers the same actions in both languages",
      [b["callback_data"] for r in kb_ru_a["inline_keyboard"] for b in r]
      == [b["callback_data"] for r in kb_en_a["inline_keyboard"] for b in r],
      str(kb_ru_a))
check("every account button label is Russian for a Russian user",
      all(CYR.search(t) for t in kb_texts(kb_ru_a)), str(kb_texts(kb_ru_a)))
# The language switcher is bilingual in BOTH forms on purpose — someone stuck in
# the wrong language has to be able to recognise the way out.
check("and English for an English user",
      not any(CYR.search(t) for t in kb_texts(kb_en_a) if "Язык" not in t),
      str(kb_texts(kb_en_a)))
check("the language button is recognisable in either language",
      all("Язык" in t and "Lang" in t
          for kb in (kb_ru_a, kb_en_a)
          for t in kb_texts(kb) if "Язык" in t or "Lang" in t),
      str(kb_texts(kb_ru_a)))

# Reset the session language for the checks that follow.
s2 = bot._get_session(CID); s2.lang = "ru"; bot._store.put(s2)


print()
print("=" * 70)
print("AND AN ENGLISH USER STILL SEES ENGLISH")
print("=" * 70)

EID = 999702
bot._user_store.put(T._User(chat_id=EID, name="Sam", status="approved"))
es = bot._get_session(EID); es.lang = "en"; bot._store.put(es)

bot.sent.clear(); bot.approve_user(EID)
check("approval message is English for an English user",
      "approved" in last(bot).lower() and not CYR.search(last(bot)),
      repr(last(bot)[:80]))

bot.sent.clear(); bot._handle_command(EID, "/subscribe")
check("subscribe confirmation is English",
      "subscrib" in last(bot).lower() and not CYR.search(last(bot)),
      repr(last(bot)[:80]))

print()
print("=" * 70)
print("EVERY _t()/_b() CALL SITE NAMES A KEY THAT EXISTS")
print("=" * 70)
# `_t` ends in `or key`, so a key that is not in _MSG does not raise — it ships
# the RAW KEY into the user's chat ("dr_no_topic" instead of a sentence). The
# table-shaped checks above only validate keys that DO exist, which cannot see a
# call site naming one that does not. This walks the AST instead: every literal
# first argument to _t/_b in tg_bot.py must resolve.
import ast as _ast, io

_src = io.open(T.__file__, encoding="utf-8").read()
_missing = []
_sites = 0
for _n in _walk_src():
    if not (isinstance(_n, _ast.Call) and _called_name(_n) in ("_t", "_b")
            and _n.args
            and isinstance(_n.args[0], _ast.Constant)
            and isinstance(_n.args[0].value, str)):
        continue
    _sites += 1
    _key = _n.args[0].value
    _table = T._MSG if _called_name(_n) == "_t" else T._BTN
    if _key not in _table:
        _missing.append(f"line {_n.lineno}: {_called_name(_n)}({_key!r})")

check("the AST scan found the call sites at all", _sites > 100, _sites)
check("no call site names a missing key", not _missing, "; ".join(_missing[:8]))

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
