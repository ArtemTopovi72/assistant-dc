"""Search depth is chooseable in Telegram, and it says how long it will take.

A deep-research run on this machine takes ~35 minutes. Before this there was one
speed, no way to ask for a faster one, and no number quoted anywhere — a user who
pressed 🔬 Deep Research got silence for half an hour and no way to know that was
normal, or to ask for less.

What is verified here:
  · the picker exists in the Search menu, in both languages, and states the
    CURRENT setting (the ✅) together with its cost
  · the choice persists on the session, survives a round-trip through storage,
    and actually reaches run_deep_research as `depth=`
  · the estimate is MEASURED — deep_research records completed runs and the quote
    is their median — and it degrades to a labelled guess, never to a lie
  · a cancelled or failed run does NOT poison the estimate

Run: venv/Scripts/python.exe tests/test_tg_research_depth.py
"""
import sys, os, types, tempfile, json, inspect, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_depth_")
import tg_bot as T

# TelegramBot is split across a dozen mixin modules, and the split keeps
# moving -- this list was already several modules out of date. Source-level
# checks must read ALL of them or moving a method makes the assertion vacuous
# instead of red, so read the family off disk rather than naming its members.
import glob as _glob
def _bot_source():
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return chr(10).join(open(_p, encoding="utf-8").read()
                        for _p in sorted(_glob.glob(os.path.join(_root, "bot", "tg_*.py"))))
T.redirect_data_dir(_DATA_DIR)          # before any bot exists
import deep_research as D
# Patch the timings store where it lives (dr_timing), not on deep_research:
# deep_research does not re-export it, precisely so a stale patch site cannot
# quietly write the user's real history.
import dr_timing as DT

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


CID = 999801


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: object(), lambda: object(),
                        lambda: {}, silent_mode=True)
    bot.kbs = []
    bot.edits = []

    def _send(cid, text, **kw):
        bot.kbs.append((text, kw.get("keyboard")))
        return 1
    bot._send_text = _send
    bot._send_get_id = lambda cid, text, **kw: (_send(cid, text, **kw), 1)[1]
    bot._edit_text = lambda cid, mid, text, **kw: bot.edits.append(
        (text, kw.get("keyboard")))
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._backend = types.SimpleNamespace(
        push=lambda t: None, depth=lambda: 0, name=lambda: "stub",
        drop_chat=lambda cid: 0, close=lambda: None)
    bot._activity.log = lambda *a, **k: None
    # The callback gate added later this session refuses any non-exempt
    # callback (including depth:) for a chat with no approved user record —
    # this suite drives depth: presses via _dispatch, so it needs one.
    u = T._User(chat_id=CID, name="U", tg_username="u", password_hash="x",
               status="approved")
    bot._user_store.put(u)
    return bot


def texts(bot):
    return " ".join(t for t, _ in bot.kbs)


def inline_labels(kb):
    if not isinstance(kb, dict) or "inline_keyboard" not in kb:
        return []
    return [b["text"] for row in kb["inline_keyboard"] for b in row if b.get("callback_data") != "nav:back"]


def inline_data(kb):
    return [b["callback_data"] for row in kb["inline_keyboard"] for b in row if b["callback_data"] != "nav:back"]


bot = make_bot()

# ═══════════════════════════════════════════════════════ the estimate itself
print("\n" + "=" * 66)
print("THE ESTIMATE IS MEASURED, NOT INVENTED")
print("=" * 66)

# Isolate the timing store: the real one is the user's own history.
_TIMING_BACKUP = DT._TIMING_FILE
DT._TIMING_FILE = __import__("pathlib").Path(_DATA_DIR) / "_timings.json"

for d in D.DEPTHS:
    secs, n = D.estimate_duration(d)
    check(f"{d} has an estimate before anything has run", secs > 0, f"{secs}")
    check(f"{d} admits it is a guess (0 samples)", n == 0, f"n={n}")

check("deeper costs more than quicker",
      D.estimate_duration("quick")[0] < D.estimate_duration("standard")[0]
      < D.estimate_duration("deep")[0],
      [D.estimate_duration(d)[0] for d in D.DEPTHS])

D.record_run_duration("quick", 600)
D.record_run_duration("quick", 400)
D.record_run_duration("quick", 500)
secs, n = D.estimate_duration("quick")
check("a real run replaces the seed", secs == 500 and n == 3, f"{secs}s over {n}")
check("the median is used, not the mean — one freak run cannot skew it",
      (D.record_run_duration("quick", 99999) or D.estimate_duration("quick")[0]) == 550,
      D.estimate_duration("quick"))

check("a zero or negative duration is refused",
      (D.record_run_duration("standard", 0),
       D.record_run_duration("standard", -5),
       D.estimate_duration("standard")[1])[2] == 0,
      D.estimate_duration("standard"))
check("an unknown depth is refused, not stored",
      (D.record_run_duration("nonsense", 100),
       "nonsense" not in json.loads(DT._TIMING_FILE.read_text(encoding="utf-8")))[1])
check("an unknown depth still gets an estimate rather than a crash",
      D.estimate_duration("nonsense")[0] == D.estimate_duration("standard")[0])
# Above the plausibility floor on purpose. These used to be 100..117s, which
# the floor now rejects as stub runs, so the store kept 0 of 18 and this looked
# like a retention bug. What is under test is the cap, not the floor.
for i in range(D._TIMING_KEEP + 6):
    D.record_run_duration("deep", 1000 + i)
check("only the last N runs count — the store cannot grow forever",
      D.estimate_duration("deep")[1] == D._TIMING_KEEP,
      f"kept {D.estimate_duration('deep')[1]} of {D._TIMING_KEEP + 6}")

# A cancelled run took as long as the user's patience, not as long as the work.
fin_src = inspect.getsource(D._finish)
check("a cancelled run is NOT recorded",
      "if not cancelled and report:" in fin_src, fin_src[:300])
check("_finish is told which depth it was running",
      "depth" in inspect.signature(D._finish).parameters)
check("run_deep_research passes the depth to _finish",
      "depth=depth" in inspect.getsource(D.run_deep_research))

# ═════════════════════════════════════════════════════════════ the picker
print("\n" + "=" * 66)
print("THE PICKER")
print("=" * 66)

for lang in ("ru", "en"):
    s = bot._get_session(CID)
    s.lang = lang; s.dr_depth = ""
    bot._store.put(s)
    kb = T._depth_menu_kb(s, lang)
    lbls, datas = inline_labels(kb), inline_data(kb)
    check(f"[{lang}] all three depths are offered", len(lbls) == 3, lbls)
    check(f"[{lang}] each option carries its own time estimate",
          all("~" in l for l in lbls), lbls)
    check(f"[{lang}] exactly one option is marked as current",
          sum(1 for l in lbls if l.startswith("✅")) == 1, lbls)
    check(f"[{lang}] the callbacks are the three depths",
          datas == ["depth:quick", "depth:standard", "depth:deep"], datas)
    check(f"[{lang}] the labels are localized",
          all(T._t("d_" + d, lang).split()[-1] in " ".join(lbls) for d in D.DEPTHS),
          lbls)
    body = T._depth_menu_text(s, lang)
    check(f"[{lang}] the menu text names the current depth and its wait",
          T._t("d_standard", lang).split()[-1] in body and "~" not in body[:0] and
          any(u in body for u in ("min", "мин", "h", "ч", "s", "с")), body[:120])

check("the estimate's provenance is stated when it is only a guess",
      T._t("depth_eta_guess", "en").lower().startswith("times are rough"),
      T._t("depth_eta_guess", "en"))
check("and when it is measured, it says how many runs it is based on",
      "{n}" in T._MSG["depth_eta_measured"]["en"])

# The picker must be REACHABLE — a menu nobody can open is not a feature.
check("🎚 Depth sits in the Search menu",
      T._b("depth", "en") in [b for row in T._search_kb("en")["keyboard"] for b in row],
      T._search_kb("en")["keyboard"])
check("and in Russian too",
      T._b("depth", "ru") in [b for row in T._search_kb("ru")["keyboard"] for b in row],
      T._search_kb("ru")["keyboard"])
check("both labels route to the same key",
      T._LABEL2KEY[T._b("depth", "ru")] == T._LABEL2KEY[T._b("depth", "en")] == "depth")
check("the button is wired to an action, not left dangling",
      T._DIRECT_KB.get("depth") == "__research_depth__", T._DIRECT_KB.get("depth"))

# ═══════════════════════════════════════════════ pressing it does something
print("\n" + "=" * 66)
print("PRESSING IT")
print("=" * 66)

bot.kbs.clear()
bot._resolve_and_push(CID, [{"type": "text", "text": T._b("depth", "ru")}])
check("tapping the button opens the picker",
      any(isinstance(kb, dict) and "inline_keyboard" in kb for _t_, kb in bot.kbs),
      [k for _t_, k in bot.kbs])


def press(data):
    bot.kbs.clear(); bot.edits.clear()
    bot._dispatch({"callback_query": {"id": "1", "data": data, "from": {"id": CID},
                                      "message": {"chat": {"id": CID, "type": "private"},
                                                  "message_id": 77}}})


press("depth:deep")
check("the choice is stored on the session", bot._get_session(CID).dr_depth == "deep",
      bot._get_session(CID).dr_depth)
check("the menu is redrawn in place, not sent again",
      bot.edits and not bot.kbs, f"edits={len(bot.edits)} sends={len(bot.kbs)}")
check("and the ✅ moved to the new choice",
      any(l.startswith("✅") and "🔬" in l for l in inline_labels(bot.edits[-1][1])),
      inline_labels(bot.edits[-1][1]) if bot.edits else None)

press("depth:quick")
check("switching again works", bot._get_session(CID).dr_depth == "quick")
press("depth:nonsense")
check("a junk depth is ignored, and does not clobber the setting",
      bot._get_session(CID).dr_depth == "quick", bot._get_session(CID).dr_depth)

# It has to survive being written to disk and read back, or it resets every restart.
raw = bot._get_session(CID).to_dict()
check("the depth is persisted", raw.get("dr_depth") == "quick", raw.get("dr_depth"))
check("and comes back on reload",
      T._Session(CID, raw).dr_depth == "quick")
check("a session that never chose one resolves to the configured default",
      T._resolve_depth(T._Session(CID, {})) == T._config_default_depth())

# /depth, for people who type rather than tap.
s = bot._get_session(CID); s.lang = "en"; bot._store.put(s)
bot.kbs.clear()
bot._handle_command(CID, "/depth deep")
check("/depth deep sets it outright", bot._get_session(CID).dr_depth == "deep",
      bot._get_session(CID).dr_depth)
bot.kbs.clear()
bot._handle_command(CID, "/depth")
check("a bare /depth opens the picker",
      any(isinstance(kb, dict) and "inline_keyboard" in kb for _t_, kb in bot.kbs))
bot.kbs.clear()
bot._handle_command(CID, "/depth sideways")
check("/depth with junk opens the picker instead of setting junk",
      bot._get_session(CID).dr_depth == "deep"
      and any(isinstance(kb, dict) and "inline_keyboard" in kb for _t_, kb in bot.kbs),
      bot._get_session(CID).dr_depth)
check("/depth is advertised in the command menu",
      any(c == "depth" for c, _e, _r in T.TelegramBot._COMMANDS),
      [c for c, _e, _r in T.TelegramBot._COMMANDS])

# ═════════════════════════════════════════ the choice reaches the pipeline
print("\n" + "=" * 66)
print("THE CHOICE REACHES THE RESEARCH RUN")
print("=" * 66)

bot_src = _bot_source()
m = re.search(r"_dr\.run_deep_research\((.{0,160})", bot_src, re.S)
check("the bot passes the chosen depth to run_deep_research",
      m is not None and "depth=depth" in m.group(1),
      m.group(1) if m else "call not found")
check("the depth is resolved from the session, not hardcoded",
      re.search(r"depth = (tg_bot\.)?_resolve_depth\(sess\)", bot_src) is not None)
check("the user is told the expected wait when the run STARTS",
      "dr_started" in bot_src and "{eta}" in T._MSG["dr_started"]["ru"],
      T._MSG["dr_started"]["ru"])

# The queue's own wait estimate must use the same measured number, not a stale
# constant — it used to quote a flat 900s for a run that takes 2100s.
eta_src = inspect.getsource(T._fmt_eta)
check("the queue ETA uses the measured research duration too",
      "_depth_eta_seconds()" in eta_src, eta_src)
check("an explicit TG_ETA_RESEARCH_SEC still overrides it",
      'TG_ETA_RESEARCH_SEC' in eta_src)

# Formatting: a wait of hours must not be reported as "7200 s".
check("seconds are shown for short waits", T._fmt_secs(45, "en") == "45 s")
check("minutes for medium ones", T._fmt_secs(2100, "en") == "35 min")
check("hours for long ones", T._fmt_secs(7200, "en") == "2.0 h")
check("and in Russian", T._fmt_secs(2100, "ru") == "35 мин"
      and T._fmt_secs(45, "ru") == "45 с", T._fmt_secs(2100, "ru"))
check("a zero duration does not render as negative or blank",
      T._fmt_secs(0, "en") == "0 s")

# If deep_research cannot be consulted at all, the UI must not invent a number.
_saved = D.estimate_duration
try:
    D.estimate_duration = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))
    check("an unavailable estimator shows '?', never a made-up time",
          T._depth_eta_text("standard", "en") == "?", T._depth_eta_text("standard", "en"))
    check("and the picker still renders instead of raising",
          len(inline_labels(T._depth_menu_kb(bot._get_session(CID), "en"))) == 3)
finally:
    D.estimate_duration = _saved
    DT._TIMING_FILE = _TIMING_BACKUP

print(f"\n{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
