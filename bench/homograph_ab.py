"""Homograph stress: RUAccent alone vs RUAccent + chat model on the homographs
(user, 2026-09-28). Each line has one homograph with a known stress; the
score is how many come out right. Run with LM Studio up and the card free."""
import os, sys, threading, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import music

A = "́"
CASES = [  # (line, the right stressed form)
    ("Пока вы там в затылок мне не дышите!", "дыши" + A + "те"),
    ("Вы тяжело и часто дышите, как после бега", "ды" + A + "шите"),
    ("Старинный замок на скале стоит", "за" + A + "мок"),
    ("Повесил на ворота ржавый замок", "замо" + A + "к"),
    ("Мешок муки принёс я с мельницы", "муки" + A),
    ("Какие адские муки я терпел", "му" + A + "ки"),
    ("Сколько стоит твоя любовь, скажи", "сто" + A + "ит"),
    ("Он стоит у окна и молчит", "стои" + A + "т"),
    ("Эти слова я запомню навек", "слова" + A),
    ("У этого слова нет конца", "сло" + A + "ва"),
    ("Подними руки выше к небу", "ру" + A + "ки"),
    ("Не дождёшься ты моей руки", "руки" + A),
    ("Мне уже не страшно ничего", "уже" + A),
    ("Эта дорога всё уже и уже", "у" + A + "же"),
    ("На полки ставлю книги в ряд", "по" + A + "лки"),
    ("Шли полки солдат по площади", "полки" + A),
    ("Вокруг меня темно, и кругом голова", "круго" + A + "м"),
    ("Я плачу за всё наличными", "плачу" + A),
    ("Я плачу по ночам в подушку", "пла" + A + "чу"),
    ("Ты меня не узнаёшь, а я тебя узнаю из тысячи", "узна" + A + "ю"),
    ("Пропасть бездонная под ногами", "про" + A + "пасть"),
    ("Не дай мне пропасть в этой тьме", "пропа" + A + "сть"),
    ("Белки на ветках грызут орехи", "бе" + A + "лки"),
    ("Отделил белки от желтков", "белки" + A),
    ("Стрелки часов бегут вперёд", "стре" + A + "лки"),
    ("Мы вышли в поле, словно стрелки на охоте", "стрелки" + A),
    ("Воды холодной принеси", "воды" + A),
    ("Тихие воды глубоки", "во" + A + "ды"),
    ("Цены растут, а зарплата стоит", "це" + A + "ны"),
    ("Эти духи пахнут летом", "духи" + A),
    ("Лесные духи бродят в ночи", "ду" + A + "хи"),
    ("Грабли в сарае, косить пора", "коси" + A + "ть"),
]


def score(outs):
    ok = [c[1] in o.lower() for c, o in zip(CASES, outs)]
    return sum(ok), ok


if __name__ == "__main__":
    ctx = types.SimpleNamespace(set_stage=lambda *a, **k: None, cancel_event=threading.Event())
    ru = [music.stress_lyrics(c[0]) for c in CASES]
    # the whole set as one song: the real path asks once per song
    song = "\n".join(c[0] for c in CASES)
    hy = music.stress_lyrics(song, ctx=ctx).splitlines()
    (a, oka), (b, okb) = score(ru), score(hy)
    for c, r, h, x, y in zip(CASES, ru, hy, oka, okb):
        if not (x and y):
            print(f"{'+' if x else '-'}RU {'+' if y else '-'}HY  want {c[1]:12} | RU: {r} | HY: {h}")
    print(f"RUAccent {a}/{len(CASES)}   RUAccent+LLM {b}/{len(CASES)}")
