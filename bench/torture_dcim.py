"""Torture the real bot with the real DCIM.zip under REAL-CHAT conditions.

The user's verdict of 2026-09-14 ("твои 30 тестов это мусор"): journeys 31-33
went green while the real chat kept failing on the same archive. The bench
chats were pristine -- a fresh user, no history, the archive as the first
message, sent as text/plain. The real chat had a long history with old
pictures in it, a drawing, a photo the user asked about, /reset_sandbox
right before, the request self-forwarded from Saved Messages with the
archive attached, and the same message sent twice when the first attempt
failed. Each of those broke a different thing (image register, forward
framing, "which picture?", "already done").

This run recreates that chat. It is NOT one of the thirty. Run it alone
(desktop app stopped):

  venv/Scripts/python.exe -u bench/torture_dcim.py > runtime/torture.log
"""
from __future__ import annotations
import os, shutil, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import live_journeys as LJ
from live_journeys import (say, photo, doc, has, no_file, says, silent_on, picture,
                           sandbox_has, sandbox_lacks, collage_is_deduped, A_DCIM, A_CAFE, A_RECEIPT)
import live_tg_drive as D

CHAT = LJ.BASE_CHAT + 99
REQUEST = "найди все мосты, убери дубликаты (из копий самый резкий) и собери коллаж"
_GRID = "Is this a collage (grid) of several photos showing bridges?"
_NOT_DONE = (r"какую картинк|посмотри /files|спроси ещё раз|распаковал архив в рабочую|"
             r"начинаю|приступаю|сначала я|будет готов|применю|проведу|уже выполнил|уже сделал")

STEPS = [
    # ── a day of ordinary chat first: old pictures live in the register ──
    say("привет", expect=[no_file()]),
    say("нарисуй рыжего кота на велосипеде", expect=[has("sendPhoto")]),
    photo(A_CAFE, "что на этой фотке?", expect=[no_file(), says(r"\w{4,}")]),
    photo(A_RECEIPT, "сколько тут итого?", expect=[says(r"1\s?088")]),
    say("расскажи короткий анекдот про программиста", expect=[no_file()]),
    say("сделай кота из первой картинки чёрным", expect=[has("sendPhoto")]),
    # ── the real sequence ──
    say("/reset_sandbox", code=True, expect=[says(r"пуст|очищ|удал|готов|empty|reset")]),
    doc(A_DCIM, REQUEST, self_forward=True, code=True, timeout=1500,
        expect=[has("sendPhoto"), collage_is_deduped(), silent_on(_NOT_DONE),
                picture(_GRID)]),
    # the user resends the very same message when he doubts the result
    # Run #4: «заново» redrew the CAT and the sendPhoto passed -- the picture
    # itself must be a grid, every time.
    doc(A_DCIM, REQUEST, self_forward=True, code=True, timeout=1500,
        expect=[has("sendPhoto"), collage_is_deduped(), silent_on(_NOT_DONE), picture(_GRID)]),
    say("заново", code=True, timeout=1500,
        expect=[has("sendPhoto"), collage_is_deduped(), silent_on(_NOT_DONE), picture(_GRID)]),
    say("а сколько мостов нашёл и какие дубликаты выкинул?", expect=[says(r"\d")]),
    say("не захламляй: удали архив и распакованную папку, а коллаж и скрипт оставь", code=True,
        expect=[sandbox_lacks("DCIM*.zip", "*unpacked*"), sandbox_has("*collage*.*"),
                says(r"удал|убрал|очист|почист")]),
    say("удали все файлы", code=True,
        expect=[sandbox_lacks("*"), says(r"удал|очист|пуст")]),
]


def main():
    assert os.path.exists(A_DCIM), A_DCIM
    box = os.path.join(os.path.dirname(HERE), "runtime", "sandboxes", str(CHAT))
    shutil.rmtree(box, ignore_errors=True)
    bot, ctx = D.build()
    LJ._CTX = ctx
    try:
        LJ.run_journey(99, "TORTURE: DCIM.zip in a lived-in chat", CHAT, "Ivan", STEPS, bot, ctx)
    finally:
        try: bot.stop()
        except Exception: pass
        r = LJ.REPORT[-1]
        print("\n" + "=" * 72)
        for s in r["steps"]:
            print(f"  {'✓' if s.get('ok') else '✗'} step {s['i']} {str(s.get('label',''))[:60]!r}"
                  + ("" if s.get("ok") else "  -> " + "; ".join(s.get("problems") or [s.get("why", "")])[:300]))
        print(f"#99 {'OK' if r['ok'] else 'BAD'} torture")


if __name__ == "__main__":
    main()
