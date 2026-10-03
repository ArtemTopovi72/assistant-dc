"""The Telegram bot's own UI: keyboards, menu navigation and menu STATE.

Everything else about the bot is tested through its delivery layer. This suite is
about what the user actually touches — the reply keyboards — and the invariant
that binds them to the session:

  the keyboard on screen and `pending_prefix` must never disagree.

`pending_prefix` silently rewrites the next free-text message ("generate an image
of: …"). It is submenu state. Presenting the MAIN keyboard while it is still set
means the user sees the top-level menu but their next sentence gets turned into an
image prompt — which is exactly what Stop, Help, /start and the transcribe-failure
paths used to do.

Run: venv/Scripts/python.exe tests/test_tg_menu_ui.py
"""
import sys, types, tempfile, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import tg_bot as T

# Before any bot exists — otherwise fixture users land in the LIVE store.
_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_menu_")
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


CID = 999501


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: object(), lambda: object(),
                        lambda: {}, silent_mode=True)
    bot.kbs = []          # (text, keyboard) of everything the bot sent
    bot.pushed = []

    def _send(cid, text, **kw):
        bot.kbs.append((text, kw.get("keyboard")))
        return 1
    bot._send_text = _send
    bot._send_get_id = lambda cid, text, **kw: (_send(cid, text, **kw), 1)[1]
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._backend = types.SimpleNamespace(
        push=lambda task: bot.pushed.append(task),
        depth=lambda: 0, name=lambda: "stub", drop_chat=lambda cid: 0,
        close=lambda: None)
    bot._activity.log = lambda *a, **k: None
    bot._dl_bytes = lambda fid: b""
    return bot


def labels(kb):
    if not isinstance(kb, dict) or "keyboard" not in kb:
        return []
    return [b for row in kb["keyboard"] for b in row]


def last_kb(bot):
    for _text, kb in reversed(bot.kbs):
        if isinstance(kb, dict) and "keyboard" in kb:
            return kb
    return None


def sess(bot):
    return bot._get_session(CID)


def reset(bot, lang="en"):
    bot.kbs.clear(); bot.pushed.clear()
    s = bot._get_session(CID)
    s.pending_prefix = ""
    s.pending_photo = ""
    s.menu = ""
    s.lang = lang
    s.voice_on = False
    s.clear_history()
    bot._store.put(s)


def tap(bot, key, lang="en"):
    """Press the button `key` exactly as Telegram delivers it: as its label."""
    bot._resolve_and_push(CID, [{"type": "text", "text": T._b(key, lang)}])


def kb_is(kb, builder_labels):
    return set(labels(kb)) == set(builder_labels)


bot = make_bot()

print("=" * 66)
print("INVARIANT: main keyboard on screen  =>  no pending prefix")
print("=" * 66)

# Everything that drops the user back to the top level, from inside a submenu.
for entry, escape in (("draw", "help"), ("draw", "clear"),
                      ("search", "help"), ("search", "clear")):
    reset(bot)
    tap(bot, entry)
    check(f"{entry}: prefix armed on entry", bool(sess(bot).pending_prefix),
          repr(sess(bot).pending_prefix))
    tap(bot, escape)
    kb = last_kb(bot)
    shows_main = T._b("draw", "en") in labels(kb) and T._b("account", "en") in labels(kb)
    if shows_main:
        check(f"{entry} -> {escape}: main keyboard clears the prefix",
              not sess(bot).pending_prefix,
              f"keyboard is the main menu but prefix is {sess(bot).pending_prefix!r}")
    else:
        check(f"{entry} -> {escape}: stayed in a submenu (prefix may persist)", True)

# ⛔ Stop is intercepted before the debounce queue, so drive it through _dispatch.
reset(bot)
tap(bot, "draw")
check("stop: prefix armed before Stop", bool(sess(bot).pending_prefix))
bot._user_gate = lambda cid, msg: True
bot._request_stop = lambda cid: 0
bot._on_stage = lambda *a, **k: None
bot._dispatch({"message": {"chat": {"id": CID}, "from": {"id": CID},
                           "message_id": 1, "text": T._b("stop", "en")}})
check("Stop shows the main keyboard AND clears the prefix",
      not sess(bot).pending_prefix,
      f"prefix survived Stop: {sess(bot).pending_prefix!r}")

# A failed voice transcription also drops to the main keyboard.
reset(bot)
tap(bot, "draw")
bot._transcribe = lambda ctx, fid, media="voice", **kw: ""
bot._resolve_and_push(CID, [{"type": "voice", "file_id": "x"}])
kb = last_kb(bot)
if kb and T._b("account", "en") in labels(kb):
    check("failed transcription: main keyboard clears the prefix",
          not sess(bot).pending_prefix, repr(sess(bot).pending_prefix))
else:
    check("failed transcription kept the submenu", True)


print()
print("=" * 66)
print("SUBMENU BUTTONS KEEP THEIR MENU OPEN")
print("=" * 66)

DRAW = lambda lang="en": labels(T._draw_kb(lang))
SEARCH = lambda lang="en": labels(T._search_kb(lang))

for key, expect, name in (("gen_image", DRAW, "draw"),
                          ("edit_image", DRAW, "draw"),
                          ("web_search", SEARCH, "search"),
                          ("deep", SEARCH, "search")):
    reset(bot)
    tap(bot, name)                      # open the submenu
    bot.kbs.clear()
    tap(bot, key)                       # tap a button inside it
    kb = last_kb(bot)
    check(f"{key} stays in the {name} menu", kb_is(kb, expect()),
          f"got {labels(kb)}")

# 📌 Remember lives in Settings and must not eject the user either.
reset(bot)
tap(bot, "settings")
bot.kbs.clear()
tap(bot, "remember")
check("remember stays in the settings menu",
      kb_is(last_kb(bot), labels(T._settings_kb(False, False, "en"))),
      f"got {labels(last_kb(bot))}")


print()
print("=" * 66)
print("BACK GOES UP ONE LEVEL, NOT ALL THE WAY OUT")
print("=" * 66)

reset(bot)
tap(bot, "settings")
tap(bot, "library")
check("documents menu opened",
      kb_is(last_kb(bot), labels(T._library_kb(False, "en"))),
      f"got {labels(last_kb(bot))}")
check("menu state records library", sess(bot).menu == "library", sess(bot).menu)
bot.kbs.clear()
tap(bot, "back")
check("back from Documents returns to Settings (not the main menu)",
      kb_is(last_kb(bot), labels(T._settings_kb(False, False, "en"))),
      f"got {labels(last_kb(bot))}")
bot.kbs.clear()
tap(bot, "back")
check("back from Settings returns to the main menu",
      T._b("account", "en") in labels(last_kb(bot)),
      f"got {labels(last_kb(bot))}")
check("menu state cleared at the top level", sess(bot).menu == "", sess(bot).menu)

for name in ("draw", "search"):
    reset(bot)
    tap(bot, name)
    bot.kbs.clear()
    tap(bot, "back")
    # Draw lives under Creativity > Images: Back goes up one tier.
    up = T._b("style_menu_btn", "en") if name == "draw" else T._b("account", "en")
    check(f"back from {name} goes up one level",
          up in labels(last_kb(bot)),
          f"got {labels(last_kb(bot))}")
    check(f"back from {name} clears the prefix", not sess(bot).pending_prefix)


print()
print("=" * 66)
print("MENU TEXT IS SELF-EXPLANATORY")
print("=" * 66)

# The Settings header used to be "⚙️ Настройки\n• 🎙 <b>ВЫКЛ ❌</b>": a one-item
# bullet list whose item had no name. On screen that is a stray dot, a mic and a
# red cross with nothing saying WHAT is off.
for _lang in ("en", "ru"):
    reset(bot, lang=_lang)
    tap(bot, "settings", lang=_lang)
    _txt = bot.kbs[-1][0]
    check(f"[{_lang}] settings header has no orphan bullet",
          "•" not in _txt, repr(_txt))
    check(f"[{_lang}] the reply setting is named, not just an icon",
          T._b("reply_text", _lang) in _txt, repr(_txt))

# /settings lists several settings, so bullets are right there — but each line
# must still say what it controls.
reset(bot)
bot._handle_command(CID, "/settings")
_txt = bot.kbs[-1][0]
for _line in [l for l in _txt.splitlines() if l.startswith("•")]:
    check("every /settings line names its setting", ":" in _line, repr(_line))


print()
print("=" * 66)
print("BUTTONS THAT NEED AN IMAGE WAIT FOR ONE")
print("=" * 66)

# 📷 Analyze Photo is a DIRECT action: it used to fire "describe and analyze this
# image in detail" as a full turn the moment it was pressed. Pressed in the natural
# order — tell the bot what you want, then hand it the photo — that ran against
# nothing and the agent answered "there is no image".
reset(bot)
sess(bot).last_image_path = ""
bot._store.put(sess(bot))
tap(bot, "analyze")
check("analyze with no image does not run a turn", not bot.pushed,
      f"pushed {bot.pushed}")
check("analyze with no image arms the next photo",
      sess(bot).pending_photo == T._DIRECT_KB["analyze"],
      repr(sess(bot).pending_photo))
check("analyze with no image says what it is waiting for",
      any("photo" in t.lower() or "фото" in t.lower() for t, _ in bot.kbs),
      str([t for t, _ in bot.kbs]))

bot.pushed.clear()
bot._dl_bytes = lambda fid: b"\x89PNG-not-real"
bot._resolve_and_push(CID, [{"type": "photo", "file_id": "f1"}])
check("the next photo runs the armed instruction", bool(bot.pushed),
      "the photo did not start a turn")
if bot.pushed:
    _txt = str(bot.pushed[-1])
    check("the turn carries the analyze instruction",
          "describe and analyze" in _txt, _txt[:200])
check("the armed instruction is consumed exactly once",
      not sess(bot).pending_photo, repr(sess(bot).pending_photo))

# A caption is an explicit instruction and must win over the armed one.
reset(bot)
sess(bot).pending_photo = T._DIRECT_KB["analyze"]
bot._store.put(sess(bot))
bot._resolve_and_push(CID, [{"type": "photo", "file_id": "f2",
                             "caption": "what breed is this dog"}])
_txt = str(bot.pushed[-1]) if bot.pushed else ""
check("a caption wins over the armed instruction",
      "what breed is this dog" in _txt and "describe and analyze" not in _txt,
      _txt[:200])

# With an image already in hand the button must still work immediately.
reset(bot)
sess(bot).last_image_path = "C:/tmp/whatever.png"
bot._store.put(sess(bot))
tap(bot, "analyze")
check("analyze with an image in hand runs straight away", bool(bot.pushed),
      "nothing was queued")
# item 11: the button gave no way to hand over a DIFFERENT photo when one
# was already in the chat. It still runs immediately on the current image,
# but now also arms pending_photo so a fresh photo sent right after is
# analyzed instead of just captioned.
check("analyze with an image ALSO arms pending_photo, for a fresh photo",
      sess(bot).pending_photo == T._DIRECT_KB["analyze"],
      repr(sess(bot).pending_photo))

# Armed with no image, then an image arrives some other way (a generation) and a
# direct action runs: the arm is satisfied and must not survive to ambush the next
# caption-less photo the user sends for something else entirely.
reset(bot)
sess(bot).pending_photo = T._DIRECT_KB["analyze"]
sess(bot).last_image_path = "C:/tmp/generated.png"
bot._store.put(sess(bot))
tap(bot, "regenerate")
check("a direct action clears a stale armed instruction",
      not sess(bot).pending_photo, repr(sess(bot).pending_photo))

# Leaving to the main menu disarms it — same rule as pending_prefix.
reset(bot)
sess(bot).last_image_path = ""
bot._store.put(sess(bot))
tap(bot, "analyze")
tap(bot, "clear")
check("going back to the main menu disarms the photo wait",
      not sess(bot).pending_photo, repr(sess(bot).pending_photo))


print()
print("=" * 66)
print("MENU STATE MATCHES THE KEYBOARD ON SCREEN")
print("=" * 66)

# Every handler that answers with the Settings keyboard must also say we are in
# Settings, or ⬅ Back is decided by whatever menu we happened to be in before.
for _key in ("status", "facts", "lang", "notif"):
    if _key not in T._BTN:
        continue
    reset(bot)
    tap(bot, "settings")
    tap(bot, "library")                 # get menu into a DIFFERENT state first
    tap(bot, _key)
    check(f"{_key} leaves the session in the settings menu",
          sess(bot).menu == "settings", repr(sess(bot).menu))

# 🔔 Notifications answered with silence when no user record backed the session.
reset(bot)
_missing = 999520
bot._get_session(_missing).menu = ""
bot.kbs.clear()
bot._resolve_and_push(_missing, [{"type": "text", "text": T._b("notif", "en")}])
check("Notifications answers even with no account behind the session",
      bool(bot.kbs), "the bot said nothing at all")

# Uploading a document puts the Documents keyboard up; the session has to agree.
reset(bot)
bot._library_stats = lambda cid: {"documents": 1, "chunks": 3, "docs": []}
class _Lib:
    # _open_library now returns a knowledge_api.KnowledgeClient, and tg_library
    # calls its boundary method ingest(). `build` stays as the legacy alias the
    # real client also still exposes, so this fake matches the shipped contract.
    def ingest(self, paths, **kw): return {"documents_indexed": 1, "errors": []}
    def build(self, paths, **kw): return self.ingest(paths, **kw)
    def documents(self): return [{"title": "t", "chunks": 3}]
    def close(self): pass
bot._open_library = lambda cid: _Lib()
import tempfile as _tf, pathlib as _pl
_f = _pl.Path(_tf.mkdtemp()) / "book.txt"
_f.write_text("hello", encoding="utf-8")
bot._index_document(CID, sess(bot), str(_f), "book.txt")
check("indexing never opens the library menu by itself (owner 10-03)",
      sess(bot).menu != "library", repr(sess(bot).menu))

# Wiping the context must not leave Back pointing into a menu we walked out of.
reset(bot)
tap(bot, "settings"); tap(bot, "library")
sess(bot).clear_context()
check("clearing the context clears the menu state",
      sess(bot).menu == "", repr(sess(bot).menu))


print()
print("=" * 66)
print("INLINE BUTTONS ARE GATED ON WHO PRESSED THEM")
print("=" * 66)

# Inline buttons never expire. A demoted admin's old admin-panel message still has
# live ✅/❌ buttons, and in a group any member can press a prompt meant for the
# admin standing next to them. The old code trusted the button, not the presser.
ADMIN, PLAIN, VICTIM = 999510, 999511, 999512
for _cid, _adm in ((ADMIN, True), (PLAIN, False)):
    _u = T._User(chat_id=_cid, name=f"u{_cid}", status="approved", is_admin=_adm)
    bot._user_store.put(_u)
bot._user_store.put(T._User(chat_id=VICTIM, name="victim", status="pending"))

def press(data, presser):
    bot.kbs.clear()
    bot._dispatch({"callback_query": {"id": "1", "data": data,
                                      "from": {"id": presser},
                                      "message": {"chat": {"id": presser}}}})

press(f"admin_approve:{VICTIM}", PLAIN)
check("a non-admin cannot approve a user with the approve button",
      bot._user_store.get(VICTIM).status != "approved",
      f"status is now {bot._user_store.get(VICTIM).status!r}")
check("the non-admin is told why",
      any(any(T._t("admins_only", lg) in t for lg in T._LANGS) for t, _ in bot.kbs),
      str([t for t, _ in bot.kbs]))

press(f"admin_approve:{VICTIM}", ADMIN)
check("a real admin can still approve",
      bot._user_store.get(VICTIM).status == "approved",
      f"status is {bot._user_store.get(VICTIM).status!r}")

# Demotion must take effect immediately, even for buttons already on screen.
_ex = bot._user_store.get(ADMIN); _ex.is_admin = False; bot._user_store.put(_ex)
bot._user_store.put(T._User(chat_id=999513, name="v2", status="pending"))
press("admin_approve:999513", ADMIN)
check("a demoted admin's old button stops working",
      bot._user_store.get(999513).status != "approved")

# Malformed payload must not take the poll loop down.
_ex.is_admin = True; bot._user_store.put(_ex)
try:
    press("admin_approve:not-a-number", ADMIN)
    check("a malformed target id is rejected quietly", True)
except Exception as e:
    check("a malformed target id is rejected quietly", False, repr(e))


print()
print("=" * 66)
print("KEYBOARD TABLE INTEGRITY")
print("=" * 66)

# _LABEL2KEY maps LABEL -> key, so two keys sharing a label silently overwrite
# each other and one button stops working. Nothing else guards this.
_seen = {}
_dupes = []
for _key, _forms in T._BTN.items():
    for _lbl in _forms.values():
        if _lbl in _seen:
            _dupes.append((_lbl, _seen[_lbl], _key))
        _seen[_lbl] = _key
check("no two buttons share a label", not _dupes, str(_dupes))

# Every button rendered by every keyboard must resolve back to a known action.
_all_kbs = [T._main_kb(True, True, "en"), T._main_kb(True, True, "ru"),
            T._draw_kb("en"), T._draw_kb("ru"),
            T._search_kb("en"), T._search_kb("ru"),
            T._settings_kb(True, True, "en"), T._settings_kb(True, True, "ru"),
            T._library_kb(False, "en"), T._library_kb(True, "ru")]
_unrouted = []
for _kb in _all_kbs:
    for _lbl in labels(_kb):
        _k = T._LABEL2KEY.get(_lbl)
        if not _k or not (_k in T._DIRECT_KB or _k in T._PROMPT_KB):
            _unrouted.append(_lbl)
check("every rendered button routes to an action", not _unrouted, str(set(_unrouted)))

# Both languages must offer the same set of actions — a button that exists only
# in English strands Russian users (and vice versa).
for _builder, _bname in ((lambda l: T._main_kb(True, True, l), "main"),
                         (T._draw_kb, "draw"),
                         (T._search_kb, "search"),
                         (lambda l: T._settings_kb(True, True, l), "settings"),
                         (lambda l: T._library_kb(False, l), "library")):
    _en = {T._LABEL2KEY.get(x) for x in labels(_builder("en"))}
    _ru = {T._LABEL2KEY.get(x) for x in labels(_builder("ru"))}
    check(f"{_bname} keyboard offers the same actions in EN and RU",
          _en == _ru, f"en-only={_en - _ru} ru-only={_ru - _en}")

# A label rendered before a language switch must still route afterwards.
reset(bot, lang="ru")
tap(bot, "draw", lang="en")          # stale English label, Russian session
check("stale-language label still routes",
      sess(bot).pending_prefix == "generate an image of: ",
      repr(sess(bot).pending_prefix))

# Reply format: the Settings button is a fixed name that opens an inline picker
# (✅ on the current mode). It used to carry the current value and cycle on each
# press, so "Replies: text+voice" pressed gave voice-only, and the reply keyboard
# was re-sent, which folded the menu away.
reset(bot)
tap(bot, "settings")
_set = [b for row in last_kb(bot)["keyboard"] for b in row]
check("settings shows 'Reply format', not the current value",
      T._b("reply_fmt", "en") in _set and T._b("reply_text", "en") not in _set, repr(_set))
bot.kbs.clear()
tap(bot, "reply_fmt")
_t, _kb = bot.kbs[-1]
_opts = [r[0]["text"] for r in (_kb or {}).get("inline_keyboard", [])]
check("the press opens an inline picker, current mode ticked, reply keyboard untouched",
      _opts and _opts[0].startswith("✅") and not any(o.startswith("✅") for o in _opts[1:])
      and "keyboard" not in _kb and sess(bot).reply_mode == "text", repr((_t, _opts)))
_edits = []
bot._user_store.put(T._User(chat_id=CID, name="me", status="approved"))
bot._edit_text = lambda cid, mid, text, **kw: _edits.append(kw.get("keyboard"))
bot._dispatch({"callback_query": {"id": "1", "data": "reply:voice", "from": {"id": CID},
                                  "message": {"message_id": 7, "chat": {"id": CID}}}})
check("picking a mode sets exactly it and moves the tick in place",
      sess(bot).reply_mode == "voice" and _edits
      and _edits[-1]["inline_keyboard"][2][0]["text"].startswith("✅"), repr(_edits))
_adm = T._settings_kb("text", True, "en")["keyboard"]
check("admin panel is the last row above Back, not the top",
      _adm[-2] == [T._b("admin", "en")] and _adm[-1] == [T._b("back", "en")], repr(_adm))

_posted = []
bot._api_post = lambda m, p: _posted.append(p) or {"ok": True}
T.TelegramBot._post_message(bot, "sendMessage", {"chat_id": CID, "text": "x"}, None,
                            T._settings_kb("text", False, "en"))
check("reply keyboards are sent persistent, so a press does not fold the menu away",
      '"is_persistent": true' in _posted[-1].get("reply_markup", ""), repr(_posted))


# No trap menus: an inline menu with no exit gets ↩ Назад; a step flow gets ✖️ Отмена.
_s = sess(bot); _s.drop_waiting(); bot._store.put(_s)
_kb = T.TelegramBot._with_wait_cancel(bot, CID, {"inline_keyboard": [[{"text": "✏️ Name",
                                                                       "callback_data": "acct_setname"}]]})
check("an inline menu without an exit gains Back",
      _kb["inline_keyboard"][-1][0]["callback_data"] == "nav:close", repr(_kb))
_kb2 = T.TelegramBot._with_wait_cancel(bot, CID, {"inline_keyboard": [[{"text": "x", "callback_data": "wait:cancel"}]]})
check("a menu that already has an exit is left alone", len(_kb2["inline_keyboard"]) == 1, repr(_kb2))
_s.mashup_state = "want1"; bot._store.put(_s)
_kb3 = T.TelegramBot._with_wait_cancel(bot, CID, None)
check("a mashup step waiting for a track carries Cancel",
      _kb3 and _kb3["inline_keyboard"][-1][0]["callback_data"] == "wait:cancel", repr(_kb3))
_s.drop_waiting(); bot._store.put(_s)
check("Cancel ends the mashup step", sess(bot).mashup_state == "")

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
