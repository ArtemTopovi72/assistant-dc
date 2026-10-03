"""Shared gold test set + scoring for the Russian stress benchmark.

Gold is encoded as '+'-before-the-stressed-vowel. All engines' outputs are
normalised to that canonical form (handles '+'-before, apostrophe-after, the
U+0301 combining acute, and ё auto-stress) so heterogeneous engines compare on an
equal footing. ё is treated as inherently stressed; comparison is ё-folded.
"""
import re
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

ACUTE = "́"
_VOWELS = "аеёиоуыэюя"
_WORD_RE = re.compile(r"[А-Яа-яЁё'’ˈ+́]+")


def to_plus(word: str) -> str:
    w = re.sub(r"(.)" + ACUTE, r"+\1", word)          # combining acute after vowel
    w = re.sub(r"(.)['’ˈ]", r"+\1", w)                 # apostrophe/IPA stress after vowel
    if "+" not in w and ("ё" in w.lower()):            # ё is always stressed
        w = re.sub(r"(?i)(ё)", r"+\1", w, count=1)
    return w


def canonical(word: str) -> str:
    """Lowercased, ё-folded, single-'+'-before-stressed-vowel canonical form."""
    w = to_plus(word).lower()
    w = w.replace(ACUTE, "").replace("'", "").replace("’", "").replace("ˈ", "")
    parts = w.split("+")
    if len(parts) > 2:                                  # keep only the first marker
        w = parts[0] + "+" + "".join(parts[1:])
    return w.replace("ё", "е")


def base(word: str) -> str:
    return canonical(word).replace("+", "")


def n_vowels(word: str) -> int:
    return sum(1 for c in base(word) if c in _VOWELS)


def find_token(text: str, target: str):
    """Return the model's token in `text` matching `target` (by ё-folded base)."""
    tb = base(target)
    for tok in _WORD_RE.findall(text):
        if base(tok) == tb:
            return tok
    return None


def is_marked(canon_word: str) -> bool:
    return "+" in canon_word


# ---------------------------------------------------------------------------
# GOLD SETS
# ---------------------------------------------------------------------------
# Homographs: (sentence, target_surface_form, gold_plus_form). Two senses each.
HOMOGRAPHS = [
    ("Старинный замок стоял на горе.", "замок", "з+амок"),            # castle
    ("Ржавый замок висел на двери.", "замок", "зам+ок"),             # lock
    ("Невыносимая мука терзала его.", "мука", "м+ука"),             # torment
    ("Белая мука рассыпалась по столу.", "мука", "мук+а"),          # flour
    ("Новый атлас лежал на полке.", "атлас", "+атлас"),             # atlas
    ("Блестящий атлас струился по плечам.", "атлас", "атл+ас"),      # satin
    ("Главный орган человека — сердце.", "орган", "+орган"),         # organ
    ("Старинный орган звучал в соборе.", "орган", "орг+ан"),         # pipe organ
    ("Рыжие белки прыгали по веткам.", "белки", "б+елки"),          # squirrels
    ("Куриные белки очень полезны.", "белки", "белк+и"),            # proteins
    ("Длинная дорога вела в город.", "дорога", "дор+ога"),          # road
    ("Ты мне очень дорога.", "дорога", "дорог+а"),                  # dear (f)
    ("Мягкий хлопок приятен телу.", "хлопок", "хл+опок"),           # cotton
    ("Громкий хлопок разорвал тишину.", "хлопок", "хлоп+ок"),       # clap
    ("Перед нами зияла пропасть.", "пропасть", "пр+опасть"),        # abyss
    ("Он боялся пропасть без вести.", "пропасть", "проп+асть"),      # to vanish
    ("Стрелки часов показывали полночь.", "стрелки", "стр+елки"),    # hands
    ("Меткие стрелки попали в цель.", "стрелки", "стрелк+и"),        # shooters
    ("Свежие вести пришли утром.", "вести", "в+ести"),              # news
    ("Нужно вести машину осторожно.", "вести", "вест+и"),           # to lead
    ("Красивый ирис расцвёл в саду.", "ирис", "+ирис"),             # iris
    ("Сладкий ирис прилип к зубам.", "ирис", "ир+ис"),             # toffee
    ("Французские духи пахли розой.", "духи", "дух+и"),             # perfume
    ("Лесные духи охраняли поляну.", "духи", "д+ухи"),             # spirits
    ("Я плачу горькими слезами.", "плачу", "пл+ачу"),              # I cry
    ("Я плачу за обед наличными.", "плачу", "плач+у"),             # I pay
    ("Уже стемнело на улице.", "уже", "уж+е"),                     # already
    ("Эта тропа намного уже.", "уже", "+уже"),                     # narrower
    ("Его вина была доказана.", "вина", "вин+а"),                  # guilt
    ("Дорогие вина хранились в подвале.", "вина", "в+ина"),         # wines (nom. pl.)
    ("Острые гвоздики торчали из доски.", "гвоздики", "гв+оздики"),  # small nails
    ("Букет гвоздики благоухал.", "гвоздики", "гвозд+ики"),         # carnations
    ("Чайные кружки стояли в ряд.", "кружки", "кр+ужки"),           # mugs
    ("Дети ходили в кружки рисования.", "кружки", "кружк+и"),       # clubs
    ("Сделаю это потом.", "потом", "пот+ом"),                      # later
    ("Лоб покрылся потом.", "потом", "п+отом"),                    # sweat
]

# Hard / commonly-misstressed words (dictionary stress known; a proxy for OOV —
# many fall outside smaller models' explicit dictionaries).
HARD_WORDS = [
    "обесп+ечение", "феном+ен", "щав+ель", "жалюз+и", "катал+ог", "догов+ор",
    "кварт+ал", "крас+ивее", "балов+ать", "звон+ит", "т+орты", "б+анты",
    "ш+арфы", "асимметр+ия", "газопров+од", "христиан+ин", "к+ухонный",
    "сл+ивовый", "ход+атайство", "г+енезис", "диспанс+ер", "эксп+ерт",
    "мусоропров+од", "стол+яр", "цем+ент", "пул+овер", "марк+етинг",
    "новорождённый", "облегч+ить", "вероисповед+ание",
]

# General sentences with every scorable (>=2-vowel, content) word gold-annotated.
GENERAL = [
    ("Мама мыла раму.", ["м+ама", "м+ыла", "р+аму"]),
    ("Дети играют в парке.", ["д+ети", "игр+ают", "п+арке"]),
    ("Солнце светит ярко сегодня.", ["с+олнце", "св+етит", "+ярко", "сег+одня"]),
    ("Книга лежит на столе.", ["кн+ига", "леж+ит"]),
    ("Музыка звучала весь вечер.", ["м+узыка", "звуч+ала", "в+ечер"]),
    ("Студенты сдавали экзамен.", ["студ+енты", "сдав+али", "экз+амен"]),
    ("Город готовится к празднику.", ["г+ород", "гот+овится", "пр+азднику"]),
    ("Работа была очень трудной.", ["раб+ота", "был+а", "+очень", "тр+удной"]),
    ("Учёные сделали важное открытие.", ["учёные", "сд+елали", "в+ажное", "откр+ытие"]),
    ("Машина быстро ехала по шоссе.", ["маш+ина", "б+ыстро", "+ехала", "шосс+е"]),
]


def score_model(predict, *, latency_text=None):
    """Run `predict(text)->str` over all gold sets. Returns a metrics dict.

    Counts: homograph accuracy, hard-word accuracy, general word + sentence
    accuracy, pooled stress accuracy, unmarked-polysyllable rate, and errors.
    """
    import time
    res = {"errors": 0, "unmarked": 0}

    # --- homographs ---
    h_ok = 0
    for sent, target, gold in HOMOGRAPHS:
        try:
            out = predict(sent)
        except Exception:
            res["errors"] += 1
            continue
        tok = find_token(out, target)
        if tok is None:
            continue
        if n_vowels(tok) >= 2 and not is_marked(canonical(tok)):
            res["unmarked"] += 1
        if canonical(tok) == canonical(gold):
            h_ok += 1
    res["homograph_acc"] = h_ok / len(HOMOGRAPHS)

    # --- hard words (fed standalone) ---
    w_ok = 0
    for gold in HARD_WORDS:
        word = base(gold)
        try:
            out = predict(word)
        except Exception:
            res["errors"] += 1
            continue
        tok = find_token(out, word) or out
        if n_vowels(tok) >= 2 and not is_marked(canonical(tok)):
            res["unmarked"] += 1
        if canonical(tok) == canonical(gold):
            w_ok += 1
    res["hard_word_acc"] = w_ok / len(HARD_WORDS)

    # --- general sentences ---
    g_word_ok = g_word_tot = 0
    g_sent_ok = 0
    for sent, golds in GENERAL:
        try:
            out = predict(sent)
        except Exception:
            res["errors"] += 1
            continue
        all_ok = True
        for gold in golds:
            g_word_tot += 1
            tok = find_token(out, gold)
            if tok and canonical(tok) == canonical(gold):
                g_word_ok += 1
            else:
                all_ok = False
        if all_ok:
            g_sent_ok += 1
    res["general_word_acc"] = g_word_ok / max(1, g_word_tot)
    res["sentence_acc"] = g_sent_ok / len(GENERAL)

    # --- pooled stress accuracy (all single-target items) ---
    pooled_ok = h_ok + w_ok + g_word_ok
    pooled_tot = len(HOMOGRAPHS) + len(HARD_WORDS) + g_word_tot
    res["stress_acc_overall"] = pooled_ok / pooled_tot

    # --- latency: avg ms over homograph sentences ---
    texts = latency_text or [s for s, _, _ in HOMOGRAPHS]
    t0 = time.perf_counter()
    for s in texts:
        try:
            predict(s)
        except Exception:
            pass
    res["latency_ms_per_sent"] = (time.perf_counter() - t0) * 1000 / len(texts)
    return res
