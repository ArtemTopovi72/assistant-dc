"""Synthetic diarization test set with a KNOWN answer: scripted Russian dialogues voiced by F5 with the
project's own voice samples, turns glued with known gaps. Writes bench/diar_set/clipN.wav + clipN.json
({"turns": [{"speaker": "DC", "text": ..., "start": s, "end": e}]}).

    unset F5_TEST_RUN; venv/Scripts/python.exe -u bench/diar_make_set.py      (the app must be stopped)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("", "bench", "voice", "agent", "core", "media", "bot"):
    sys.path.insert(0, os.path.join(ROOT, sub))
OUT = os.path.join(ROOT, "bench", "diar_set")

VOICES = {"DC": "DC_short_ref12.wav", "San": "San_short_ref12.wav", "Stepan": "Stepan_short_ref12.wav",
          "Leha": "Leha_short.wav", "Prig": "Prig_short.wav", "Dh": "Dh_short.wav", "Egr": "Egr_short.wav"}

# (speakers, [(who index, line)]) -- everyday talk, a few words to a sentence per turn
DIALOGUES = [
    (["DC", "San"], [(0, "Слушай, ты вчера видел, что творилось у подъезда?"), (1, "Нет, я рано лёг спать. А что там было?"),
                     (0, "Приехали сразу две машины, и всю ночь мигали огни."), (1, "Странно. Надо будет спросить у соседей."),
                     (0, "Давай вечером зайдём к Петровичу, он наверняка всё знает."), (1, "Хорошо, во сколько встретимся?"),
                     (0, "Часов в семь, после работы.")]),
    (["Stepan", "Leha"], [(0, "Ты сделал отчёт, который я просил?"), (1, "Почти, осталось добавить таблицу с расходами."),
                          (0, "Отлично, тогда пришли мне его до обеда."), (1, "Договорились, скину в течение часа."),
                          (0, "И не забудь про графики."), (1, "Помню, они уже готовы."), (0, "Тогда спасибо.")]),
    (["Prig", "Dh"], [(0, "Как думаешь, завтра будет дождь?"), (1, "По прогнозу ливень с самого утра."),
                      (0, "Тогда зонт точно нужен."), (1, "И сапоги, дороги будут как болото."),
                      (0, "Может, останемся дома?"), (1, "Нет уж, мы ждали эту поездку целый месяц."),
                      (0, "Ладно, поедем, но возьмём термос с чаем.")]),
    (["Egr", "DC", "San"], [(0, "Ребята, у нас завтра собрание в десять утра."), (1, "Опять без предупреждения?"),
                            (2, "Мне нужно будет уйти пораньше, у меня врач."), (0, "Хорошо, мы начнём с твоего вопроса."),
                            (1, "Тогда я подготовлю бумаги заранее."), (2, "Спасибо, я это очень ценю."),
                            (0, "Всё, значит, до завтра."), (1, "До завтра.")]),
    (["Leha", "Stepan", "Prig"], [(0, "Кто сегодня идёт за хлебом?"), (1, "Я вчера ходил, теперь очередь Прохора."),
                                  (2, "Ну почему опять я? У меня и так куча дел."), (0, "Зато погода хорошая, прогуляешься."),
                                  (1, "И заодно купи молока."), (2, "Ладно, уговорили, только деньги дайте."),
                                  (0, "Держи, тут хватит на всё.")]),
    (["Dh", "Egr"], [(0, "Ты помнишь, где мы оставили ключи?"), (1, "Кажется, на полке в коридоре."),
                     (0, "Там их нет, я уже смотрел."), (1, "Посмотри в кармане куртки."),
                     (0, "Нашёл, они были там всё это время."), (1, "Вот видишь, а ты волновался.")]),
]


def main():
    import numpy as np
    import soundfile as sf
    import live_tg_drive as D
    import voice_clone
    os.makedirs(OUT, exist_ok=True)
    print("loading the stack...")
    _bot, ctx = D.build()
    refs = {}
    for name, f in VOICES.items():
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            refs[name] = voice_clone.prepare_reference(ctx, p, os.path.join(OUT, "_refs", name), lang="ru")
            print("ref", name, "->", refs[name][1][:60])
    gaps = [0.35, 0.6, 0.9, 0.45]
    for n, (who, lines) in enumerate(DIALOGUES, start=1):
        if any(w not in refs for w in who):
            print("skip clip", n, "(voice missing)")
            continue
        sr, parts, turns, t = 24000, [], [], 0.0
        for k, (wi, text) in enumerate(lines):
            name = who[wi]
            wav = voice_clone.speak(ctx, refs[name][0], refs[name][1], text, os.path.join(OUT, "_tmp"))
            x, sr_ = sf.read(wav, dtype="float32")
            x = x.mean(axis=1) if x.ndim > 1 else x
            assert sr_ == sr, sr_
            gap = gaps[k % len(gaps)]
            turns.append({"speaker": name, "text": text, "start": round(t, 2), "end": round(t + len(x) / sr, 2)})
            parts += [x, np.zeros(int(gap * sr), dtype="float32")]
            t += len(x) / sr + gap
        sf.write(os.path.join(OUT, f"clip{n}.wav"), np.concatenate(parts[:-1]), sr)
        json.dump({"speakers": who, "turns": turns}, open(os.path.join(OUT, f"clip{n}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"clip{n}: {len(who)} speakers, {t:.1f} s")


if __name__ == "__main__":
    main()
