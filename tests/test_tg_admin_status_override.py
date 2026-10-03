"""Verify the desktop GUI's Approve/Reject admin buttons still work after
tg_bot.approve_user/reject_user gained the `allow_from_any_status` status
guard (added because the Telegram callback path must NOT let a stale inline
button un-ban/un-reject/un-approve a user).

Contract to protect:
  1. gui.TelegramTab._approve_selected calls approve_user(cid,
     allow_from_any_status=True) so the explicit desktop "Approve" action can
     override a banned/rejected status (unlike the Telegram callback path,
     which must stay guarded).
  2. gui.TelegramTab._reject_selected calls reject_user(cid,
     allow_from_any_status=True) for the same reason, mirroring approve_user.
     reject_user used to have NO status guard at all, which meant a stale
     admin_reject: inline button -- left over on an old admin panel message
     after a DIFFERENT admin (or a later re-approval) already moved that chat
     to 'approved' -- could silently downgrade a live, actively-in-use
     account back to 'rejected'. It is now guarded to only fire from
     'pending' on the Telegram callback path; the GUI's explicit Reject
     action opts out of that guard via allow_from_any_status=True, same
     pattern as approve_user.
  3. Both slots ignore the string return value ("ok"/"not_found"/
     "not_pending") entirely -- calling them must never raise (no crash from
     treating the new str return as None, no unpacking).

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_tg_admin_status_override.py
"""
import os, sys, tempfile, inspect

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("USE_GUI", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

from PyQt5.QtWidgets import QApplication
import gui
import tg_bot as _T

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_adminstatus_")
gui.redirect_settings(_DATA_DIR)
_T.redirect_data_dir(_DATA_DIR)

_app = QApplication.instance() or QApplication(sys.argv)

OK = BAD = 0
def check(name, cond, detail=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {detail}")


class _Host:
    def __init__(self):
        import types
        self.ctx = types.SimpleNamespace(cancel_event=None, throttle=0)
        self.graph = types.SimpleNamespace() if False else None
        self.base_state = {}
        self.tabs = None


class _FakeBot:
    """Records calls and returns whatever outcome string the test wants,
    mimicking the real approve_user/reject_user return contract."""
    def __init__(self, approve_outcome="ok", reject_outcome="ok", ban_outcome="ok"):
        self.approve_calls = []
        self.reject_calls = []
        self.ban_calls = []
        self._approve_outcome = approve_outcome
        self._reject_outcome = reject_outcome
        self._ban_outcome = ban_outcome

    def approve_user(self, chat_id, *, allow_from_any_status=False):
        self.approve_calls.append((chat_id, allow_from_any_status))
        return self._approve_outcome

    def reject_user(self, chat_id, *, allow_from_any_status=False):
        self.reject_calls.append((chat_id, allow_from_any_status))
        return self._reject_outcome

    def ban_user(self, chat_id, *, allow_from_any_status=False):
        self.ban_calls.append((chat_id, allow_from_any_status))
        return self._ban_outcome

    def get_users(self):
        return []


def _tab_with_one_row(bot):
    host = _Host()
    tab = gui.TelegramTab(host)
    tab._bot = bot
    tab.user_table.setRowCount(0)
    tab.user_table.insertRow(0)
    from PyQt5.QtWidgets import QTableWidgetItem
    tab.user_table.setItem(0, 0, QTableWidgetItem("42"))
    tab.user_table.selectRow(0)
    return tab


print("=" * 70)
print("SIGNATURE CHECKS (real tg_bot.py, read-only)")
print("=" * 70)

sig_approve = inspect.signature(_T.TelegramBot.approve_user)
check("approve_user has allow_from_any_status kwarg",
      "allow_from_any_status" in sig_approve.parameters)
check("allow_from_any_status defaults to False (callback path stays guarded)",
      sig_approve.parameters["allow_from_any_status"].default is False)

sig_reject = inspect.signature(_T.TelegramBot.reject_user)
check("reject_user has allow_from_any_status kwarg",
      "allow_from_any_status" in sig_reject.parameters)
check("allow_from_any_status defaults to False (callback path stays guarded)",
      sig_reject.parameters["allow_from_any_status"].default is False)

sig_ban = inspect.signature(_T.TelegramBot.ban_user)
check("ban_user has allow_from_any_status kwarg (brought in line with "
      "approve_user/reject_user's contract)",
      "allow_from_any_status" in sig_ban.parameters)
check("ban_user's allow_from_any_status defaults to False",
      sig_ban.parameters["allow_from_any_status"].default is False)

print("=" * 70)
print("GUI CALL SITES")
print("=" * 70)

for outcome in ("ok", "not_found", "not_pending"):
    bot = _FakeBot(approve_outcome=outcome)
    tab = _tab_with_one_row(bot)
    try:
        tab._approve_selected()
        crashed = False
    except Exception as e:
        crashed = True
        detail = repr(e)
    check(f"_approve_selected does not crash on approve_user -> {outcome!r}",
          not crashed, detail if crashed else "")
    check(f"_approve_selected passes allow_from_any_status=True (outcome {outcome!r})",
          bot.approve_calls == [(42, True)], bot.approve_calls)

for outcome in ("ok", "not_found", "not_pending"):
    bot = _FakeBot(reject_outcome=outcome)
    tab = _tab_with_one_row(bot)
    try:
        tab._reject_selected()
        crashed = False
    except Exception as e:
        crashed = True
        detail = repr(e)
    check(f"_reject_selected does not crash on reject_user -> {outcome!r}",
          not crashed, detail if crashed else "")
    check(f"_reject_selected passes allow_from_any_status=True (outcome {outcome!r})",
          bot.reject_calls == [(42, True)], bot.reject_calls)

# _ban_selected shows a confirmation QMessageBox.question dialog first
# (approve/reject act immediately) -- auto-answer Yes so this test exercises
# the actual ban_user call instead of hanging/crashing on a real modal in
# offscreen mode.
from PyQt5.QtWidgets import QMessageBox as _QMB
_orig_question = _QMB.question
_QMB.question = staticmethod(lambda *a, **k: _QMB.Yes)
try:
    for outcome in ("ok", "not_found", "not_banned_already"):
        bot = _FakeBot(ban_outcome=outcome)
        tab = _tab_with_one_row(bot)
        try:
            tab._ban_selected()
            crashed = False
        except Exception as e:
            crashed = True
            detail = repr(e)
        check(f"_ban_selected does not crash on ban_user -> {outcome!r}",
              not crashed, detail if crashed else "")
        check(f"_ban_selected passes allow_from_any_status=True (outcome {outcome!r})",
              bot.ban_calls == [(42, True)], bot.ban_calls)
finally:
    _QMB.question = _orig_question

print("=" * 70)
print("REAL BOT: override actually lifts a banned/rejected status")
print("=" * 70)

for start_status in ("banned", "rejected"):
    bot = _T.TelegramBot.__new__(_T.TelegramBot)
    # Minimal real-bot wiring: reuse the real UserStore/session machinery via
    # a throwaway instance is out of scope for a read-only-tg_bot.py check;
    # instead verify the guarded method directly against a stand-in user
    # object shaped like tg_bot's User record.
    import types as _types
    class _StubUser:
        def __init__(self, status):
            self.status = status
            self.is_admin = False
            self.name = "x"
    class _StubStore:
        def __init__(self, user): self._u = user
        def get(self, cid): return self._u
        def put(self, u): self._u = u
    u = _StubUser(start_status)
    bot._user_store = _StubStore(u)
    bot._get_session = lambda cid: _types.SimpleNamespace(is_admin=False)
    bot._store = _StubStore(None)
    bot._lang = lambda sess: "en"
    bot._send_text = lambda *a, **k: None
    bot._main_menu_kb = lambda *a, **k: None
    bot._activity = _types.SimpleNamespace(log=lambda *a, **k: None)
    bot._on_user_change = lambda *a, **k: None

    outcome_blocked = bot.approve_user(1)  # default: guarded
    check(f"unguarded approve_user refuses to move '{start_status}' -> approved",
          outcome_blocked == "not_pending" and u.status == start_status,
          (outcome_blocked, u.status))

    outcome_override = bot.approve_user(1, allow_from_any_status=True)
    check(f"allow_from_any_status=True DOES move '{start_status}' -> approved",
          outcome_override == "ok" and u.status == "approved",
          (outcome_override, u.status))

print("=" * 70)
print("REAL BOT: reject_user guard -- stale reject after another action "
      "already approved the user")
print("=" * 70)


def _wire_minimal_bot(status):
    import types as _types
    bot = _T.TelegramBot.__new__(_T.TelegramBot)
    class _StubUser:
        def __init__(self, status):
            self.status = status
            self.is_admin = False
            self.name = "x"
    class _StubStore:
        def __init__(self, user): self._u = user
        def get(self, cid): return self._u
        def put(self, u): self._u = u
    u = _StubUser(status)
    bot._user_store = _StubStore(u)
    bot._get_session = lambda cid: _types.SimpleNamespace(is_admin=False)
    bot._store = _StubStore(None)
    bot._lang = lambda sess: "en"
    bot._send_text = lambda *a, **k: None
    bot._main_menu_kb = lambda *a, **k: None
    bot._activity = _types.SimpleNamespace(log=lambda *a, **k: None)
    bot._on_user_change = lambda *a, **k: None
    return bot, u


# The scenario this guard exists for: chat_id 1 was pending, an admin
# approved it (status is now 'approved', account actively in use), and a
# STALE admin_reject: button from the original pending-applicant panel
# message -- still tappable, inline buttons never expire -- gets pressed.
bot, u = _wire_minimal_bot("approved")
outcome = bot.reject_user(1)   # default: guarded, Telegram callback path
check("a stale reject on an ALREADY-APPROVED user is refused, not applied",
      outcome == "not_pending" and u.status == "approved",
      (outcome, u.status))

bot, u = _wire_minimal_bot("pending")
outcome = bot.reject_user(1)   # the intended, normal use
check("reject_user still works normally from 'pending' (no over-refusal)",
      outcome == "ok" and u.status == "rejected",
      (outcome, u.status))

bot, u = _wire_minimal_bot("approved")
outcome = bot.reject_user(1, allow_from_any_status=True)
check("allow_from_any_status=True lets the desktop GUI reject an approved "
      "user explicitly, same override pattern as approve_user",
      outcome == "ok" and u.status == "rejected",
      (outcome, u.status))

print("=" * 70)
print("REAL BOT: ban_user guard -- brought in line with approve_user/"
      "reject_user's contract")
print("=" * 70)

# ban_user used to have no status guard at all AND silently returned None on
# a missing user row -- the caller could not distinguish "banned" from
# "there was no such user". Both fixed together.
bot, u = _wire_minimal_bot("pending")
outcome = bot.ban_user(1)   # normal use: ban a live user
check("ban_user still works normally on a non-banned user (no over-refusal)",
      outcome == "ok" and u.status == "banned",
      (outcome, u.status))

bot, u = _wire_minimal_bot("banned")
outcome = bot.ban_user(1)   # default: guarded, refuses a no-op re-ban
check("ban_user on an ALREADY-banned user is refused as a no-op by default",
      outcome == "not_banned_already" and u.status == "banned",
      (outcome, u.status))

bot, u = _wire_minimal_bot("banned")
outcome = bot.ban_user(1, allow_from_any_status=True)
check("allow_from_any_status=True still lets the desktop GUI explicitly "
      "re-ban (no crash, no special-casing needed in the caller)",
      outcome == "ok" and u.status == "banned",
      (outcome, u.status))

bot2 = _T.TelegramBot.__new__(_T.TelegramBot)
class _EmptyStore:
    def get(self, cid): return None
    def put(self, u): pass
bot2._user_store = _EmptyStore()
outcome_missing = bot2.ban_user(999999)
check("ban_user on a chat_id with no user row returns 'not_found' "
      "(used to silently return None, same class of bug already fixed on "
      "approve_user/reject_user)",
      outcome_missing == "not_found", outcome_missing)

print("=" * 70)
print(f"RESULT: {OK} passed, {BAD} failed")
print("=" * 70)
sys.exit(1 if BAD else 0)
