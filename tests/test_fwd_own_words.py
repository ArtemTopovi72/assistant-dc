"""The user forwarded their own bug notes («Artem: ...») and the bot answered as if a friend had written them
(23:17). A forwarded author who is the forwarding user is labelled as the user.

Run: venv/Scripts/python.exe tests/test_fwd_own_words.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_dispatch as D

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


me = {"first_name": "Artem", "last_name": "T"}
mine = D._fwd_author_of({"from": me, "forward_origin": {"sender_user": me}})
other = D._fwd_author_of({"from": me, "forward_origin": {"sender_user": {"first_name": "Ivan"}}})
check("the forwarding user's own line is labelled as theirs", mine == "Artem T (the user themself)", mine)
check("someone else keeps the plain name", other == "Ivan", other)
check("the bot's own words keep their verified label",
      D._fwd_author_of({"from": me, "_fwd_own": True}) == D.tg_bot._OWN_AUTHOR)

print(f"\n{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
