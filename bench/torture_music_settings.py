"""Song settings: a typed length and BPM within the engine's range, «🎲 На
усмотрение бота» for the length, refusal of an out-of-range number. Real
stack, fake wire; the app must be stopped. No song is rendered here (a render
is ~10 min) -- the engine path is covered by tests/test_music_custom_and_auto.py.

  venv/Scripts/python.exe -u bench/torture_music_settings.py > runtime/torture_music_settings.log
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import live_journeys as LJ
from live_journeys import say, has, says, raw_says, silent_on
import live_tg_drive as D

CHAT = LJ.BASE_CHAT + 98


def steps():
    return [
        say("🎨 Творчество", expect=[has("sendMessage")]),
        say("🎛 Настройки песни", expect=[raw_says(r"Настройки песни"), raw_says(r"Длина: 60 с")]),
        dict(kind="tap", verb="Длина: 60 с", expect=[raw_says(r"🎲 На усмотрение бота"), raw_says(r"✏️ Своя длина")]),
        dict(kind="tap", verb="✏️ Своя длина", expect=[says(r"Сколько секунд\? Число от 20 до 180")]),
        say("999", expect=[says(r"от 20 до 180"), silent_on(r"Установлено")]),
        say("100 секунд", expect=[raw_says(r"✅ Установлено: 100 с"), raw_says(r"Длина: 100 с")]),
        dict(kind="tap", verb="Длина: 100 с", expect=[raw_says(r"✅ ✏️ Своя длина \(100 с\)")]),
        # picking a value keeps the field screen open with the pick marked ✅
        dict(kind="tap", verb="🎲 На усмотрение бота", expect=[raw_says(r"✅ 🎲 На усмотрение бота")]),
        dict(kind="tap", verb="⬅ Назад", expect=[raw_says(r"Длина: 🎲 На усмотрение бота")]),
        dict(kind="tap", verb="Темп: ✨ Авто", expect=[raw_says(r"✏️ Свой BPM")]),
        dict(kind="tap", verb="✏️ Свой BPM", expect=[says(r"BPM\? Число от 50 до 200")]),
        say("118", expect=[raw_says(r"✅ Установлено: 118 BPM"), raw_says(r"Темп: 118 BPM")]),
        # a capture that is abandoned: the next plain message is a normal turn
        dict(kind="tap", verb="Темп: 118 BPM", expect=[raw_says(r"✏️ Свой BPM")]),
        dict(kind="tap", verb="✏️ Свой BPM", expect=[says(r"от 50 до 200")]),
        say("/cancel", expect=[has("sendMessage")]),
        # steps live on the Quality screen: presets with a time each, 20 ticked
        say("🎛 Настройки песни", expect=[raw_says(r"Качество: ⚡ Быстро · ~\d+ мин · 20 шагов")]),
        dict(kind="tap", verb="Качество: ⚡ Быстро · ~", expect=[
            raw_says(r"✅ ⚡ Быстро · ~\d+ мин"), raw_says(r"🐘 Максимум · ~\d+ мин"),
            raw_says(r"🔬 Шагов: 20"), raw_says(r"✅ 20 · ~\d+ мин"), raw_says(r"50 · ~\d+ мин"),
            raw_says(r"✏️ Другое число шагов"), says(r"Шаги — проходы сэмплера")]),
        dict(kind="tap", verb="50 · ~", expect=[raw_says(r"✅ 50 · ~\d+ мин"), raw_says(r"🔬 Шагов: 50")]),
        dict(kind="tap", verb="✏️ Другое число шагов", expect=[says(r"Сколько шагов сэмплера\? Число от 20 до 50")]),
        say("60", expect=[says(r"от 20 до 50"), silent_on(r"Установлено")]),
        say("35", expect=[raw_says(r"✅ Установлено: .*35 шагов"), raw_says(r"✅ ✏️ Другое число шагов \(35\)"),
                          raw_says(r"🔬 Шагов: 35")]),
        dict(kind="tap", verb="⬅ Назад", expect=[raw_says(r"Качество: ⚡ Быстро · ~\d+ мин · 35 шагов")]),
        say("♻️ сбрось настройки песни", expect=[says(r"сброшены на «Авто»"), raw_says(r"Темп: ✨ Авто")]),
        say("🎛 Настройки песни", expect=[raw_says(r"Темп: ✨ Авто"), raw_says(r"Качество: ⚡ Быстро · ~\d+ мин · 20 шагов")]),
    ]


def main():
    bot, ctx = D.build()
    LJ._CTX = ctx
    try:
        LJ.run_journey(98, "TORTURE: song settings — custom length/BPM, bot's choice", CHAT, "Ivan", steps(), bot, ctx)
    finally:
        try: bot.stop()
        except Exception: pass
        r = LJ.REPORT[-1]
        print("\n" + "=" * 72)
        for s in r["steps"]:
            print(f"  {'✓' if s.get('ok') else '✗'} step {s['i']} {str(s.get('label',''))[:60]!r}"
                  + ("" if s.get("ok") else "  -> " + "; ".join(s.get("problems") or [s.get("why", "")])[:300]))
        print(f"#98 {'OK' if r['ok'] else 'BAD'} music settings")


if __name__ == "__main__":
    main()
