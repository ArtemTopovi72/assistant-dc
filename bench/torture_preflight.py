"""The pre-render layout check, live: a scene whose relations a planner gets
wrong often («кот НА диване», «лампа СЛЕВА», «на стене вывеска») is drawn
through the real bot; the log must show the sketch check ran before the
render, and the picture must show the relations. App must be stopped.

  venv/Scripts/python.exe -u bench/torture_preflight.py > runtime/torture_preflight.log
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import live_journeys as LJ
from live_journeys import say, has, picture
import live_tg_drive as D

CHAT = LJ.BASE_CHAT + 97


def steps():
    return [
        say("нарисуй чёрного кота, спящего на зелёном диване, слева от дивана торшер, на стене вывеска «HOME»",
            expect=[has("sendPhoto"),
                    picture("Is the black cat ON the green sofa (not beside it), and is there a floor lamp to the LEFT of the sofa?")]),
        say("нарисуй девочку с красным зонтом и её собаку на дождливой улице",
            expect=[has("sendPhoto"), picture("Are there BOTH a girl with a red umbrella AND a dog?")]),
    ]


def main():
    bot, ctx = D.build()
    LJ._CTX = ctx
    try:
        LJ.run_journey(97, "TORTURE: layout preflight — relations kept, nothing missing", CHAT, "Ivan", steps(), bot, ctx)
    finally:
        try: bot.stop()
        except Exception: pass
        r = LJ.REPORT[-1]
        print("\n" + "=" * 72)
        for s in r["steps"]:
            print(f"  {'✓' if s.get('ok') else '✗'} step {s['i']} {str(s.get('label',''))[:60]!r}"
                  + ("" if s.get("ok") else "  -> " + "; ".join(s.get("problems") or [s.get("why", "")])[:300]))
        print(f"#97 {'OK' if r['ok'] else 'BAD'} layout preflight")


if __name__ == "__main__":
    main()
