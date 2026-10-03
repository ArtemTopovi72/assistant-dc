"""The depth/ETA UI was found to structurally lie about research cost.

A real-chat audit (2026-08-03) proved: a Quick research run took 92 minutes
while the picker advertised ~15. Root cause chain, all fixed here:

  1. `_get_queue_eta`'s `_cfg_int("TG_ETA_RESEARCH_SEC", 0) or int(_depth_eta_seconds())`
     could never reach the measured branch — config.py bakes a NONZERO default
     (900) for TG_ETA_RESEARCH_SEC whether or not the user set it, so the "0
     means unset" sentinel never fired. Fixed to read the raw env var directly.
  2. `dr_started` (the message sent the instant a research run begins — the one
     moment the user commits to the wait) quoted only the ETA text, discarding
     whether it was measured or a 0-sample seed.
  3. A stale/hand-crafted `depth:` tap with an unknown value used to `return`
     silently, leaving a keyboard whose ✅ lied about the real setting.
  4. A non-string `dr_depth` already in a stored session crashed `_resolve_depth`
     on `.lower()`.

Run: venv/Scripts/python.exe tests/test_depth_eta_fixes.py
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_depth_")
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
    sent = []
    def rec(method, payload=None, **kw):
        sent.append((method, payload))
        return {"ok": True, "result": {"message_id": len(sent)}}
    bot._api_post = rec
    return bot, sent


print("=" * 70)
print("1. THE QUEUE ETA CAN NOW REACH THE MEASURED VALUE")
print("=" * 70)

# TG_ETA_RESEARCH_SEC is never set in the environment here, so the measured
# branch must be reachable — the old `_cfg_int(..., 0) or ...` could NEVER
# reach it because config.py bakes a nonzero default.
os.environ.pop("TG_ETA_RESEARCH_SEC", None)
import importlib
import config as _config_mod
importlib.reload(_config_mod)

class _FakeTask:
    def __init__(self, text): self.user_text = text

# A wildly different value than the 900s floor proves the measured path, not
# the fallback, was used.
_orig_estimate = T._depth_eta_seconds
T._depth_eta_seconds = lambda depth="standard": 12345.0
try:
    eta_text = T._fmt_eta([_FakeTask("do a deep research on: x")], "en")
finally:
    T._depth_eta_seconds = _orig_estimate

check("the queue ETA used the measured 12345s (3.4h), not the flat 900s (15min)",
      "3.4" in eta_text or "h" in eta_text, eta_text)

# An explicit env override still wins.
os.environ["TG_ETA_RESEARCH_SEC"] = "42"
T._depth_eta_seconds = lambda depth="standard": 12345.0
try:
    eta_text2 = T._fmt_eta([_FakeTask("do a deep research on: x")], "en")
finally:
    T._depth_eta_seconds = _orig_estimate
    os.environ.pop("TG_ETA_RESEARCH_SEC", None)
check("an explicit env override (42s) still wins over the measured value",
      "42" in eta_text2 or "s" in eta_text2, eta_text2)

print()
print("=" * 70)
print("2. dr_started NOW DISCLOSES WHETHER THE ETA IS MEASURED OR A SEED")
print("=" * 70)

# This is exactly the message construction added at the deep-research bypass
# site in tg_bot.py (search "_dr_provenance") — reproduced here directly rather
# than driving the whole async debounce/consumer/graph pipeline, which needs a
# real model/graph to complete a turn.
import deep_research as D
_orig_dur = D.estimate_duration

D.estimate_duration = lambda depth="standard": (900.0, 0)   # force a 0-sample seed
try:
    depth = "standard"
    lang = "en"
    _dr_secs, _dr_n = T._depth_eta(depth)
    _dr_provenance = (T._t("depth_eta_measured", lang, n=_dr_n) if _dr_n
                      else T._t("depth_eta_guess", lang))
    msg = (T._t("dr_started", lang, name=T._depth_name(depth, lang),
                eta=T._depth_eta_text(depth, lang))
           + "\n" + _dr_provenance)
finally:
    D.estimate_duration = _orig_dur

check("the 0-sample case discloses it's a rough estimate, not a fact",
      "rough" in msg.lower() or "estimate" in msg.lower(), msg)

D.estimate_duration = lambda depth="standard": (1800.0, 7)   # a real measurement
try:
    _dr_secs, _dr_n = T._depth_eta(depth)
    _dr_provenance2 = (T._t("depth_eta_measured", lang, n=_dr_n) if _dr_n
                       else T._t("depth_eta_guess", lang))
finally:
    D.estimate_duration = _orig_dur
check("a real 7-sample measurement is disclosed as measured, with the count",
      "7" in _dr_provenance2 and "median" in _dr_provenance2.lower(),
      _dr_provenance2)

print()
print("=" * 70)
print("3. A STALE/UNKNOWN depth: TAP REDRAWS AN HONEST KEYBOARD, NOT SILENCE")
print("=" * 70)

bot, sent = make_bot()
CID = 6002
u = T._User(chat_id=CID, name="U", tg_username="u", password_hash="x", status="approved")
bot._user_store.put(u)
sess = bot._get_session(CID)
sess.lang = "en"; sess.dr_depth = "standard"
bot._store.put(sess)
sent.clear()
bot._dispatch({"callback_query": {"id": "x", "data": "depth:garbage_value",
              "message": {"chat": {"id": CID}, "message_id": 42}}})
check("the depth was NOT changed to the garbage value",
      T._resolve_depth(bot._get_session(CID)) == "standard",
      bot._get_session(CID).dr_depth)
check("the keyboard was redrawn (edited) rather than left silently stale",
      any(m == "editMessageText" for m, _ in sent), sent)

print()
print("=" * 70)
print("4. A NON-STRING dr_depth DOES NOT CRASH _resolve_depth")
print("=" * 70)

sess2 = bot._get_session(6003)
sess2.dr_depth = 5   # corrupted/garbage persisted state
try:
    resolved = T._resolve_depth(sess2)
    check("a non-string dr_depth resolves to the configured default instead of crashing",
          resolved in T._DEPTHS, resolved)
except Exception as exc:
    check("a non-string dr_depth resolves to the configured default instead of crashing",
          False, repr(exc))

# ── the ETA must not be dragged to nothing by runs that never happened ───────
# Measured 2026-08-29 from the live history: quick held
# [5492, 120, 120, 93, 117, 90] and standard [119, 244, 120, 126]. One of those
# is a research run; the rest are stubs and aborted runs from test and bench
# passes that still reached the "completed" call. The median came out at ~2
# minutes and the bot quoted "2 min" to a user about to wait half an hour.
import json as _json
import tempfile as _tempfile
import dr_timing as _DT


def _with_history(history):
    d = _tempfile.mkdtemp(prefix="drtiming_")
    _DT.redirect_timings(d)
    payload = dict(history)
    payload["_profile"] = _DT._profile_signature()
    with open(_DT._TIMING_FILE, "w", encoding="utf-8") as fh:
        _json.dump(payload, fh)


def test_stub_runs_are_not_mistaken_for_research():
    _with_history({"quick": [5492.6, 120.1, 93.4, 90.2],
                   "standard": [], "deep": []})
    secs, n = _DT.estimate_duration("quick")
    check("stub_runs_discarded", n == 1 and secs > 3000, (secs, n))


def test_with_nothing_left_the_seed_is_used_and_says_so():
    _with_history({"quick": [90.0, 110.0], "standard": [], "deep": []})
    secs, n = _DT.estimate_duration("quick")
    check("falls_back_to_the_seed", n == 0 and secs == float(_DT._TIMING_SEED["quick"]),
          (secs, n))


def test_real_runs_still_count():
    _with_history({"quick": [], "standard": [1700.0, 1900.0], "deep": []})
    secs, n = _DT.estimate_duration("standard")
    check("real_runs_kept", n == 2 and 1700 <= secs <= 1900, (secs, n))


def test_redirect_moves_the_file_off_the_operators_history():
    real = _DT._TIMING_FILE
    _with_history({"quick": [], "standard": [], "deep": []})
    check("timing_file_redirected", _DT._TIMING_FILE != real, _DT._TIMING_FILE)


test_stub_runs_are_not_mistaken_for_research()
test_with_nothing_left_the_seed_is_used_and_says_so()
test_real_runs_still_count()
test_redirect_moves_the_file_off_the_operators_history()

print()
print(f"{OK}/{OK + BAD} checks passed")


sys.exit(1 if BAD else 0)