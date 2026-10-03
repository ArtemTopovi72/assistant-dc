"""The reply language is the USER's, not the house's.

Live, 2026-09-12: a Telegram user whose session language is English asked
"what's the capital of Australia?" and got "Столицей Австралии является город
Канберра" -- the [Language] block hard-coded Russian for everyone. Pure.

Run: venv/Scripts/python.exe tests/test_reply_language_follows_user.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
import logging; logging.basicConfig(level=logging.CRITICAL)

import prompts as P

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:200])


ru, en = P.build_system_prompt(), P.build_system_prompt(reply_lang="en")
check("default is Russian", "reply to the user in Russian" in ru and "in English. No exceptions" not in ru)
check("an English session gets English", "reply to the user in English" in en and "in Russian. No exceptions" not in en)
check("no placeholder leaks", "{language_rules}" not in ru + en)
check("unknown codes fall back to Russian", "reply to the user in Russian" in P.build_system_prompt(reply_lang="xx"))
check("'en-US' counts as English", "reply to the user in English" in P.build_system_prompt(reply_lang="en-US"))
lru, len_ = P.build_system_prompt_lite(), P.build_system_prompt_lite(reply_lang="en")
check("lite: default Russian", "Reply in Russian only" in lru)
check("lite: English session", "Reply in English only" in len_ and "in Russian" not in len_.replace("In Russian you always speak", ""))
check("lite: no placeholder", "{lite_language}" not in lru + len_)
check("the legacy alias is the Russian prompt", "reply to the user in Russian" in P.SYSTEM_PROMPT_PERSONALITY)

# "а теперь расскажи её по-русски" on an English session came back in English
# (journey 26): an explicit request, or a whole message in another language,
# overrides the session default for that turn — in both prompts.
for _p in (ru, en, lru, len_):
    check("an explicit language request overrides the session language",
          "explicitly asks for another language" in _p and "whole message" in _p)
_fp = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(P.__file__))), "agent/graph_fastpath.py"), encoding="utf-8").read()
check("the fast path runs the language-drift guard too",
      "_match_reply_language(ctx, fast_content, original_input)" in _fp)

for f, needle in (("agent/graph_compose.py", 'reply_lang=getattr(ctx, "reply_lang", "ru")'),
                  ("agent/graph_fastpath.py", 'reply_lang=getattr(ctx, "reply_lang", "ru")'),
                  ("bot/tg_tasks.py", "ctx.reply_lang = lang")):
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(P.__file__))), f), encoding="utf-8").read()
    check(f"{f} passes the user's language", needle in src)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
