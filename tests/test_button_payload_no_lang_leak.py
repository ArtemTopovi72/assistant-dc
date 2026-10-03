"""A plain-prose button payload must never look like the user typed English.

Live, 2026-09-19: pressing "Сменить одежду" (change_clothes) in a Russian
chat answered in English and skipped asking what outfit. tg_tasks._is_button_
payload only recognised _DIRECT_KB/_PROMPT_KB phrases as machine text; the
three _CB_CMDS buttons whose payload has no [bracket] tag (change_clothes,
regenerate, outpaint) fell through, so their English prose was read as the
USER writing in English and flipped sess.turn_lang for the whole session --
the exact bug this file's sibling, test_reply_language_follows_user.py,
already covers for a different code path (live, 2026-09-18, the "📷 Analyze
Photo" bare press).

Run: venv/Scripts/python.exe tests/test_button_payload_no_lang_leak.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T
import tg_tasks as TT

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:200])


print("=" * 70)
print("Every bracket-less _CB_CMDS payload is recognised as machine text")
print("=" * 70)

for verb, payload in T._CB_CMDS.items():
    check(f"{verb!r} payload is a recognised button payload",
          TT._is_button_payload(payload), payload)

print()
print("=" * 70)
print("The three bracket-less payloads WOULD have flipped turn_lang")
print("=" * 70)
print("(a bracket-tagged payload is separately safe: _message_script ignores")
print(" anything starting with '[' on its own, regardless of this guard)")
print()

for verb in ("change_clothes", "regenerate"):
    payload = T._CB_CMDS[verb]
    # The real call site (tg_tasks.py, around 'is_internal = _is_button_payload')
    # never even asks _message_script when is_internal is True -- this proves
    # these three specifically WOULD have looked English-scripted if the fix
    # above were reverted, which is exactly what made this bug real.
    check(f"{verb!r} payload reads as English by alphabet alone (the trap)",
          TT._message_script(payload) == "en", payload)

print()
print(f"{OK}/{OK + BAD} checks passed")
if BAD:
    sys.exit(1)
