"""A searched answer has a shape: 🔎 what was searched, the answer with [n]
footnotes, 🔗 the sources with links underneath (tg_reply_shape). Real stack,
fake wire; the app must be stopped.

  venv/Scripts/python.exe -u bench/torture_search.py > runtime/torture_search.log
"""
from __future__ import annotations
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import live_journeys as LJ
from live_journeys import say, has, no_file, says, silent_on
import live_tg_drive as D

CHAT = LJ.BASE_CHAT + 97


def steps():
    return [
        say("найди, какая самая высокогорная дорога в Европе и на какой она высоте",
            expect=[no_file(), says(r"🔎 Искал:"), says(r"\[1\]"), says(r"🔗 Источники:"),
                    says(r"^1\. .+ — [a-z0-9.-]+\.[a-z]{2,}", re.M), says(r"\d{3,4}\s?м|метр")]),
        say("а сколько стоит билет в Эрмитаж в 2026 году?",
            expect=[no_file(), says(r"🔎 Искал:"), says(r"🔗 Источники:"), says(r"руб|₽|\d{3}")]),
        say("о чём это? https://ru.wikipedia.org/wiki/Омлет",
            expect=[no_file(), says(r"📄 Прочитал:"), says(r"ru[.]wikipedia[.]org"), says(r"омлет|яиц"),
                    silent_on(r"Искал:")]),
        # A plain chat turn must NOT carry the search shape.
        say("спасибо! а как тебя зовут?", expect=[silent_on(r"Искал:|Источники:")]),
    ]


def main():
    bot, ctx = D.build()
    LJ._CTX = ctx
    try:
        LJ.run_journey(97, "TORTURE: a searched answer has a shape", CHAT, "Ivan", steps(), bot, ctx)
    finally:
        try: bot.stop()
        except Exception: pass
        r = LJ.REPORT[-1]
        print("\n" + "=" * 72)
        for s in r["steps"]:
            print(f"  {'✓' if s.get('ok') else '✗'} step {s['i']} {str(s.get('label',''))[:60]!r}"
                  + ("" if s.get("ok") else "  -> " + "; ".join(s.get("problems") or [s.get("why", "")])[:300]))
        print(f"#97 {'OK' if r['ok'] else 'BAD'} search")


if __name__ == "__main__":
    main()
