"""✨ Улучшить текст / ✍️ Сочинить текст (owner 10-03).

The lyric is checked by rules -- a real rhyme on the stressed vowel, no lazy
pairs (свет/рассвет, любовь/кровь), the rhythm, the line lengths -- and by the
model for sense, then revised until it passes; the best version wins. The bot
answers in two messages: what changed, then the lyric alone to copy.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")

import lyrics_craft as L  # noqa: E402

GOOD = """[verse]
Над городом гаснет закат
Неон загорается вновь
И окна, как свечи, горят
Сжигая чужую любовь"""

LAZY = """[verse]
Я вышел ночью в тихий свет
И ждал, когда придёт рассвет
Луна смотрела на меня
А я искал кусок огня"""


def _kinds(text, **kw):
    return [i["kind"] for i in L.analyse(text)["issues"]]


def test_real_rhymes_pass_lazy_ones_are_named():
    assert _kinds(GOOD) == ["worn"]                     # вновь/любовь is worn out
    kinds = _kinds(LAZY)
    assert "same_root" in kinds and "worn" in kinds     # свет/рассвет, меня/огня


def test_a_non_rhyme_is_named_and_a_true_one_is_not():
    bad = "Мой полковник\nСмотрит в ИИ\nБелый подоконник\nПесни мои"
    issues = L.analyse(bad)["issues"]
    assert [i["kind"] for i in issues] == ["no_rhyme"] and issues[0]["words"] == ("ИИ", "мои")


def test_stress_decides_the_rhyme_and_the_rhythm(monkeypatch):
    """With stress marks (the voice's RUAccent + silero) «за́мок / зама́нок» is
    no rhyme, and a line whose stresses fall off the beat is named."""
    marks = {
        "Тихо спит старинный замок": "Т+ихо сп+ит стар+инный з+амок",
        "Ветер гладит пыль дорог": "В+етер гл+адит п+ыль дор+ог",
        "Ждёт хозяйку за туманом": "Жд+ёт хоз+яйку з+а тум+аном",
        "Одинокий старый рог": "Один+окий ст+арый р+ог",
    }
    monkeypatch.setattr(L, "ACCENT_STUB", lambda line: marks[line])
    res = L.analyse("\n".join(marks))
    kinds = [i["kind"] for i in res["issues"]]
    assert res["stressed"] and "no_rhyme" in kinds      # за́мок / тума́ном
    assert not any(i["kind"] == "no_rhyme" and 4 in i["lines"] for i in res["issues"])  # дорог/рог


def test_polish_revises_until_clean_and_keeps_the_best(monkeypatch):
    calls = []

    def llm(role, system, user):
        calls.append(role)
        if role == "critic":
            return '{"score": 9, "problems": []}'
        if role == "revise":
            if calls.count("revise") == 1:            # the first pass sees the lazy rhyme, numbered
                assert "3: И ждал, когда придёт рассвет" in user and "one word inside the other" in user
            return GOOD.replace("вновь", "снова").replace("любовь", "основу")  # still imperfect
        return ""
    monkeypatch.setattr(L, "LLM_STUB", llm)
    res = L.polish(None, LAZY, rounds=2)
    assert res["original"] == LAZY.strip()
    assert res["text"] != LAZY.strip() and res["rounds"] >= 1
    assert calls.count("revise") >= 1


def test_a_worse_or_broken_rewrite_never_replaces_the_lyric(monkeypatch):
    def llm(role, system, user):
        if role == "critic":
            return '{"score": 8, "problems": ["line 2 is vague"]}'
        return "ок"                                   # a broken rewrite
    monkeypatch.setattr(L, "LLM_STUB", llm)
    res = L.polish(None, GOOD, rounds=3)
    assert res["text"] == GOOD.strip()


# ── the bot ──────────────────────────────────────────────────────────────────
def _bot():
    import tg_bot
    tg_bot.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_lyr_"))
    b = tg_bot.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
    b.sent = []
    b._send_text = lambda cid, text, parse_mode=None, keyboard=None, **kw: b.sent.append(
        (text, parse_mode, keyboard)) or 1
    b._run_busy = lambda cid, fn, *a: fn(*a)
    b.queued = []
    b._enqueue_item = lambda cid, item: b.queued.append(item)
    return b


def test_the_music_menu_has_both_buttons_in_one_new_row():
    import tg_keyboards as K
    rows = [[b for b in r] for r in K._cr_music_kb("ru")["keyboard"]]
    assert rows[-2] == ["✨ Улучшить текст", "✍️ Сочинить текст"]
    assert rows[-1] == [K._b("back", "ru")]
    assert ["🎤 Кавер", "🎙 Клонировать голос"] in rows          # the old rows stay


def test_improve_answers_in_two_messages_and_can_be_sung(monkeypatch):
    def llm(role, system, user):
        if role == "critic":
            return '{"score": 9, "problems": []}'
        if role == "revise":
            return GOOD
        if role == "notes":
            return "• строка 3: свет/рассвет → закат/горят — однокоренная рифма"
        return ""
    monkeypatch.setattr(L, "LLM_STUB", llm)
    bot = _bot()
    sess = bot._get_session(5); sess.lang = "ru"; bot._store.put(sess)
    bot._start_lyrics_flow(5, bot._get_session(5), "ru", "improve")
    assert bot._get_session(5).lyrics_state == "improve"
    assert bot._lyrics_take_text(5, bot._get_session(5), "ru", LAZY)
    notes, lyric = bot.sent[-2], bot.sent[-1]
    assert notes[0].startswith("✨ <b>Ваш улучшенный текст</b>") and "свет/рассвет" in notes[0]
    assert lyric[0] == GOOD.strip() and lyric[1] is None           # plain: copies whole
    cbs = [b["callback_data"] for r in lyric[2]["inline_keyboard"] for b in r]
    assert cbs == ["lyr:sing", "lyr:again"]
    assert bot._get_session(5).lyrics_state == ""                  # one text per press
    bot._cb_lyrics(5, "lyr:sing")
    assert bot.queued and "[lyrics]" in bot.queued[-1]["text"] and "закат" in bot.queued[-1]["text"]


def test_write_takes_a_theme(monkeypatch):
    def llm(role, system, user):
        if role == "draft":
            assert "Theme: город ночью" in user and "Russian" in system
            return GOOD
        if role == "critic":
            return '{"score": 9, "problems": []}'
        return GOOD
    monkeypatch.setattr(L, "LLM_STUB", llm)
    bot = _bot()
    bot._start_lyrics_flow(6, bot._get_session(6), "ru", "write")
    assert bot._lyrics_take_text(6, bot._get_session(6), "ru", "город ночью")
    assert bot.sent[-2][0].startswith("✍️ <b>Текст готов</b>") and bot.sent[-1][0] == GOOD.strip()


def test_text_goes_to_the_lyrics_flow_through_the_resolver(monkeypatch):
    seen = []
    bot = _bot()
    monkeypatch.setattr(bot, "_lyrics_run", lambda cid, lang, mode, text: seen.append((mode, text)))
    s = bot._get_session(7); s.lyrics_state = "improve"; s.reg_state = ""; bot._store.put(s)
    bot._resolve_and_push(7, [{"type": "text", "text": LAZY}])
    assert seen == [("improve", LAZY.strip())]


def test_numbering_copied_back_is_removed():
    assert L._clean("1: [verse]\n2: Раз два\n3: Три четыре") == "[verse]\nРаз два\nТри четыре"
