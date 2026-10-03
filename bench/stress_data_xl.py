"""Expanded gold set for the Russian stress benchmark + bootstrap CIs.

Reuses the canonicalisation/scoring helpers from bench_stress_data. All golds are
hand-verified against standard orthoepic norms. This is a CURATED expansion (not
an auto-generated "several hundred") to keep gold correctness high — the binomial
CIs reported make the remaining sample-size uncertainty explicit rather than
hiding it. Categories: homographs, hard/OOV words, plus mixed registers
(punctuation, dialogue, literary, technical, conversational).
"""
import random
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from bench.stress_data import (
    HOMOGRAPHS as _H0, HARD_WORDS as _W0, GENERAL as _G0,
    canonical, base, find_token, n_vowels, is_marked,
)

# --- extra homograph pairs (clear disambiguating context, dictionary stress) ---
_H_EXTRA = [
    ("Высокие книжные полки стояли вдоль стен.", "полки", "п+олки"),     # shelves
    ("Вражеские полки отступили за реку.", "полки", "полк+и"),          # regiments
    ("Этот билет стоит слишком дорого.", "стоит", "ст+оит"),            # costs
    ("Хрустальная ваза стоит на столе.", "стоит", "сто+ит"),           # stands
    ("Я ношу домой тяжёлую сумку.", "ношу", "нош+у"),                  # I carry
    ("Носильщик взвалил на спину ношу.", "ношу", "н+ошу"),             # a load
    ("Монаху явилось небесное видение.", "видение", "вид+ение"),       # apparition
    ("Своё видение проблемы он изложил кратко.", "видение", "в+идение"), # faculty of sight
    ("Танковая броня выдержала прямой удар.", "броня", "брон+я"),       # armour
    ("Броня на билеты уже закончилась.", "броня", "бр+оня"),           # reservation
    ("Он трусит перед каждым экзаменом.", "трусит", "тр+усит"),         # is a coward
    ("По утрам он трусит вдоль набережной.", "трусит", "трус+ит"),      # jogs
    ("Передний привод лучше держит дорогу.", "привод", "пр+ивод"),      # drive
    ("Нарушителю грозит привод в полицию.", "привод", "прив+од"),       # being hauled in
    ("Положительный отзыв обрадовал автора.", "отзыв", "+отзыв"),       # review
    ("Отзыв посла означал дипломатический кризис.", "отзыв", "отз+ыв"), # recall
    ("Банк наконец одобрил мне кредит.", "кредит", "кред+ит"),          # loan
    ("Эта сумма пошла в кредит счёта.", "кредит", "кр+едит"),          # accounting credit
    ("Орёл умел парить под облаками.", "парить", "пар+ить"),            # to soar
    ("Полезно парить ноги при простуде.", "парить", "п+арить"),         # to steam
    ("Как чудно пахнут здесь цветы!", "чудно", "ч+удно"),               # wonderful
    ("Чудно, что он опять опоздал.", "чудно", "чудн+о"),               # strange
    ("Горные козлы легко скакали по скалам.", "козлы", "козл+ы"),       # goats
    ("Козлы для распилки дров стояли во дворе.", "козлы", "к+озлы"),    # trestle
]

HOMOGRAPHS = _H0 + _H_EXTRA

# --- extra hard / OOV words (standard orthoepic stress) ---
_W_EXTRA = [
    "прида+ное", "б+армен", "деф+ис", "зак+упорить", "+искра", "исч+ерпать",
    "килом+етр", "кокл+юш", "крем+ень", "лом+ота", "нам+ерение", "нед+уг",
    "некрол+ог", "+отрочество", "парт+ер", "премиров+ать", "прин+удить",
    "св+ёкла", "ср+едства", "сосредот+очение", "т+аинство", "танц+овщица",
    "т+отчас", "углуб+ить", "факс+имиле", "апостр+оф", "бал+ованный",
    "+издавна", "испов+едание", "н+арост", "осв+едомиться", "отк+упорить",
    "повтор+ённый", "п+охороны", "прозорл+ивый", "рассредот+очение",
    "гражд+анство", "ди+оптрия", "заржав+еть", "м+усоропровод",
]
HARD_WORDS = _W0 + _W_EXTRA

# --- mixed-register sentences (every >=2-vowel content word gold-annotated) ---
GENERAL = _G0 + [
    # punctuation-heavy
    ("Подожди, пожалуйста, я сейчас приду!", ["подожд+и", "пож+алуйста", "сейч+ас", "прид+у"]),
    ("Что случилось? Почему так темно?", ["случ+илось", "почем+у", "темн+о"]),
    # dialogue
    ("«Куда ты идёшь?» — тихо спросила она.", ["куд+а", "ид+ёшь", "т+ихо", "спрос+ила", "он+а"]),
    ("— Я вернусь завтра, — пообещал отец.", ["верн+усь", "з+автра", "пообещ+ал", "от+ец"]),
    # literary
    ("Над озером клубился утренний туман.", ["+озером", "клуб+ился", "+утренний", "тум+ан"]),
    ("Берёзы тихо шелестели на ветру.", ["берёзы", "т+ихо", "шелест+ели", "ветр+у"]),
    # technical
    ("Алгоритм обрабатывает входные данные параллельно.", ["алгор+итм", "обраб+атывает", "вх+одные", "д+анные", "паралл+ельно"]),
    ("Сервер вернул ошибку при подключении.", ["с+ервер", "верн+ул", "ош+ибку", "подключ+ении"]),
    # conversational
    ("Слушай, давай просто закажем пиццу.", ["сл+ушай", "дав+ай", "пр+осто", "зак+ажем", "п+иццу"]),
    ("Honestly, мне сегодня совсем не хочется идти.", ["сег+одня", "совс+ем", "х+очется", "идт+и"]),
]


# ---------------------------------------------------------------------------
def _ci95_binom(k, n):
    """Wilson 95% CI for k/n. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    import math
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def score_model_xl(predict):
    """Like score_model but on the XL set, with Wilson 95% CIs on the three
    accuracy buckets. Returns metrics + (k, n) counts so callers can pool."""
    res = {"errors": 0, "unmarked": 0}

    h_ok = 0
    for sent, target, gold in HOMOGRAPHS:
        try:
            out = predict(sent)
        except Exception:
            res["errors"] += 1; continue
        tok = find_token(out, target)
        if tok is None:
            continue
        if n_vowels(tok) >= 2 and not is_marked(canonical(tok)):
            res["unmarked"] += 1
        if canonical(tok) == canonical(gold):
            h_ok += 1
    nH = len(HOMOGRAPHS)
    res["homograph_acc"] = h_ok / nH
    res["homograph_ci"] = _ci95_binom(h_ok, nH)
    res["homograph_k_n"] = (h_ok, nH)

    w_ok = 0
    for gold in HARD_WORDS:
        word = base(gold)
        try:
            out = predict(word)
        except Exception:
            res["errors"] += 1; continue
        tok = find_token(out, word) or out
        if n_vowels(tok) >= 2 and not is_marked(canonical(tok)):
            res["unmarked"] += 1
        if canonical(tok) == canonical(gold):
            w_ok += 1
    nW = len(HARD_WORDS)
    res["hard_word_acc"] = w_ok / nW
    res["hard_word_ci"] = _ci95_binom(w_ok, nW)
    res["hard_word_k_n"] = (w_ok, nW)

    g_ok = g_tot = 0
    s_ok = 0
    for sent, golds in GENERAL:
        try:
            out = predict(sent)
        except Exception:
            res["errors"] += 1; continue
        all_ok = True
        for gold in golds:
            g_tot += 1
            tok = find_token(out, gold)
            if tok and canonical(tok) == canonical(gold):
                g_ok += 1
            else:
                all_ok = False
        if all_ok:
            s_ok += 1
    res["general_word_acc"] = g_ok / max(1, g_tot)
    res["general_word_ci"] = _ci95_binom(g_ok, g_tot)
    res["general_word_k_n"] = (g_ok, g_tot)
    res["sentence_acc"] = s_ok / len(GENERAL)

    pooled_ok = h_ok + w_ok + g_ok
    pooled_tot = nH + nW + g_tot
    res["stress_acc_overall"] = pooled_ok / pooled_tot
    res["overall_ci"] = _ci95_binom(pooled_ok, pooled_tot)
    res["overall_k_n"] = (pooled_ok, pooled_tot)
    return res
