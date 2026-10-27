"""✨ Улучшить текст / ✍️ Сочинить текст.

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
import types  # noqa: E402

import pytest  # noqa: E402

# What a test hands the lyric code: `lyrics_llm(role, system, user)`, `accent(line)`,
# `DRAFTS`, `text_model`, `run_web_search` -- on the context of every call (_C).
_F = {}


@pytest.fixture(autouse=True)
def _fresh():
    _F.clear()
    yield
    _F.clear()


class _Ctx:
    def is_cancelled(self):
        ev = getattr(self, "cancel_event", None)
        return bool(ev is not None and ev.is_set())


def _C(base=None):
    ctx = base if base is not None else _Ctx()
    if "text_model" in _F:
        ctx.text_model = _F["text_model"]
    ctx.deps = types.SimpleNamespace(**{k: v for k, v in _F.items() if k != "text_model"})
    return ctx

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
    return [i["kind"] for i in L.analyse(text, _C())["issues"]]


def test_real_rhymes_pass_lazy_ones_are_named():
    assert _kinds(GOOD) == ["worn"]                     # вновь/любовь is worn out
    kinds = _kinds(LAZY)
    assert "same_root" in kinds and "worn" in kinds     # свет/рассвет, меня/огня


def test_a_non_rhyme_is_named_and_a_true_one_is_not():
    bad = "Мой полковник\nСмотрит в ИИ\nБелый подоконник\nПесни мои"
    issues = L.analyse(bad, _C())["issues"]
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
    _F["accent"] = lambda line: marks[line]
    res = L.analyse("\n".join(marks), _C())
    kinds = [i["kind"] for i in res["issues"]]
    assert res["stressed"] and "no_rhyme" in kinds      # за́мок / тума́ном
    assert not any(i["kind"] == "no_rhyme" and 4 in i["lines"] for i in res["issues"])  # дорог/рог


def test_polish_revises_until_clean_and_keeps_the_best(monkeypatch):
    calls = []

    def llm(role, system, user):
        calls.append(role)
        if role == "critic":
            return '{"problems": [], "score": 9}'
        if role == "revise":
            if calls.count("revise") == 1:            # the first pass sees the lazy rhyme, numbered
                assert "section [verse]" in user and "one word inside the other" in user
            return GOOD.replace("вновь", "снова").replace("любовь", "основу")  # still imperfect
        return ""
    _F["lyrics_llm"] = llm
    res = L.polish(_C(), LAZY, rounds=2)
    assert res["original"] == LAZY.strip()
    assert res["text"] != LAZY.strip() and res["rounds"] >= 1
    assert calls.count("revise") >= 1


def test_a_worse_or_broken_rewrite_never_replaces_the_lyric(monkeypatch):
    def llm(role, system, user):
        if role == "critic":
            return '{"problems": [{"kind": "style", "lines": [], "problem": "line 2 is vague"}], "score": 8}'
        return "ок"                                   # a broken rewrite
    _F["lyrics_llm"] = llm
    res = L.polish(_C(), GOOD, rounds=3)
    assert res["text"] == GOOD.strip()


def test_improving_polish_uses_all_requested_rounds():
    scores = iter([3, 5, 7, 9])
    edits = iter(['рекою', 'полем', 'домом'])
    calls = []
    def llm(role, system, user):
        calls.append(role)
        if role == 'critic':
            return '{"problems": [{"kind": "style", "lines": [], "problem": "line 2: make the image concrete"}], "score": %d}' % next(scores)
        if role == 'revise':
            return GOOD.replace('городом', next(edits))
        return ''
    _F['lyrics_llm'] = llm
    res = L.polish(_C(), GOOD, rounds=3)
    assert calls.count('revise') == 3
    assert 'домом' in res['text']


def test_long_story_ending_reaches_editor_and_rewriter():
    topic = 'Степан ловил рыбу. ' * 100 + 'Три желания касаются сапога; золотой вечный сапог уплывает, мудрец шутит.'
    seen = []
    def llm(role, system, user):
        seen.append(system)
        return '{"problems": [], "score": 5}' if role == 'critic' else GOOD
    _F['lyrics_llm'] = llm
    L.critique(_C(), GOOD, topic)
    L.revise(_C(), GOOD, ['restore the ending'], 'Russian', topic=topic)
    assert all(topic in prompt for prompt in seen)


def test_pretty_rhymes_cannot_outscore_a_coherent_story():
    assert L._total({'score': 10}, 2) < L._total({'score': 6}, 8)
    assert L._total({'score': 10}, 2) <= 3


def test_truncated_critic_is_retried_instead_of_silently_passing():
    replies = iter(['{"faithful": true, "coherent": true, "score": 3, "problems": ["unfinished', '{"problems": [{"kind": "style", "lines": [], "problem": "restore the ending"}], "score": 4}'])
    _F['lyrics_llm'] = lambda *args: next(replies)
    assert L.critique(_C(), GOOD, 'полная история') == (4, ['restore the ending'])


def test_unavailable_critic_never_reports_a_good_verified_score():
    _F['lyrics_llm'] = lambda *args: ''
    score, problems = L.critique(_C(), GOOD)
    assert score == 0 and problems


def test_unclassified_editor_problem_is_retried():
    replies = iter(['{"problems": [{"kind": "unknown", "lines": [], "problem": "lost ending"}], "score": 9}', '{"problems": [{"kind": "source", "lines": [], "problem": "lost ending"}], "score": 4}'])
    _F['lyrics_llm'] = lambda *args: next(replies)
    assert L.critique(_C(), GOOD, 'полный сюжет') == (4, ['lost ending'])


def test_fixed_song_length_is_in_the_tuner_brief():
    seen = []
    def llm(role, system, user):
        seen.append(system + user)
        if role == 'critic':
            return '{"problems": [], "score": 8}'
        return GOOD
    _F['lyrics_llm'] = llm
    _F['DRAFTS'] = 1
    L.write(_C(), 'сюжет', 'Russian', max_lines=10)
    assert any('AT MOST 10 sung lines' in text for text in seen)


def test_source_fidelity_caps_score_even_when_model_praises_rhymes():
    _F['lyrics_llm'] = lambda *args: '{"problems": [{"kind": "source", "lines": [], "problem": "restore the third wish"}], "score": 9}'
    assert L.critique(_C(), GOOD, 'три желания') == (4, ['restore the third wish'])


def test_nonsensical_rhyme_cannot_receive_a_good_semantic_score():
    _F['lyrics_llm'] = lambda *args: '{"problems": [{"kind": "meaning", "lines": [], "problem": "nonsense for rhyme"}], "score": 9}'
    assert L.critique(_C(), GOOD)[0] == 4


def test_structured_editor_notes_keep_line_numbers_without_prescribing_a_bad_rhyme():
    _F['lyrics_llm'] = lambda *args: '{"problems": [{"kind": "style", "lines": [2, 3], "problem": "forced rhyme"}], "score": 6}'
    assert L.critique(_C(), GOOD)[1] == ['Строки 2,3: forced rhyme']


def test_a_bad_middle_rewrite_is_retried_from_the_best_text():
    scores = iter([5, 3, 9])
    rewritten = []
    first = GOOD.replace('городом', 'рекою')
    second = GOOD.replace('городом', 'полем')
    def llm(role, system, user):
        if role == 'critic':
            return '{"problems": [{"kind": "style", "lines": [], "problem": "make the scene clear"}], "score": %d}' % next(scores)
        if role == 'revise':
            rewritten.append(user)
            return first if len(rewritten) == 1 else second
        return ''
    _F['lyrics_llm'] = llm
    result = L.polish(_C(), GOOD, rounds=2)
    assert len(rewritten) == 2 and 'городом' in rewritten[1] and 'рекою' not in rewritten[1]
    assert 'полем' in result['text']


def test_meaning_improvement_wins_even_before_its_rhythm_is_polished():
    scores = iter([4, 8])
    coherent = '[verse]\nТеперь здесь понятная история, даже если длиннее строка\nБез случайных образов и чужих событий'
    def llm(role, system, user):
        if role == 'critic':
            return '{"problems": [{"kind": "style", "lines": [], "problem": "repair wording"}], "score": %d}' % next(scores)
        return coherent if role == 'revise' else ''
    _F['lyrics_llm'] = llm
    res = L.polish(_C(), GOOD, rounds=1)
    assert res['text'] == coherent and res['sense'] == 8


def test_fewer_critical_defects_win_when_both_scores_are_capped():
    import json
    issues = iter([[{'kind': 'source', 'lines': [], 'problem': 'lost cause'},
                    {'kind': 'meaning', 'lines': [], 'problem': 'unclear phrase'}],
                   [{'kind': 'meaning', 'lines': [], 'problem': 'unclear phrase'}]])
    better = '[verse]\nТеперь здесь восстановлено условие\nОстался только неровный размер и одна слабая фраза'
    def llm(role, system, user):
        if role == 'critic':
            return json.dumps({'score': 9, 'problems': next(issues)})
        return better if role == 'revise' else ''
    _F['lyrics_llm'] = llm
    result = L.polish(_C(), GOOD, rounds=1)
    assert result['text'] == better


def test_semantic_stanza_repair_works_without_blank_lines_and_keeps_other_verses():
    text = '[verse 1]\nИстория началась\nГерой у воды\n[verse 2]\nБессмысленная строка\nЕщё одна\n[chorus]\nНаш припев\nОстаётся здесь'
    seen = []
    _F['lyrics_llm'] = lambda role, system, user: seen.append(user) or '[verse 2]\nЖеланья только для сапога\nНе для самого старика'
    out = L.revise(_C(), text, ['Lines 5: restore the condition'], 'Russian', meaning_only=True)
    assert 'Бессмысленная' not in out and 'Желанья только' in out
    assert '[verse 1]\nИстория началась\nГерой у воды' in out
    assert '[chorus]\nНаш припев\nОстаётся здесь' in out
    assert len(seen) == 1 and 'История началась' not in seen[0]


def test_numeric_critic_line_references_use_stanza_only_instructions():
    text = '[verse]\nНачало\nСмысл\n\n[chorus]\nЛишние слова\nШум'
    seen = []
    def llm(role, system, user):
        seen.append(system)
        return '[chorus]\nЯсная мысль\nПонятный итог'
    _F['lyrics_llm'] = llm
    out = L.revise(_C(), text, ['6-7 (Chorus): replace nonsense'], 'Russian', meaning_only=True)
    assert 'Лишние слова' not in out and 'Начало\nСмысл' in out
    assert 'NOT the whole song' in seen[0]
    assert 'Output only the full lyric' not in seen[0]


def test_untagged_lyric_keeps_its_first_line_in_the_full_rewrite():
    text = 'A fisherman named Stepan arrived.\nA broken phrase.\nHe cast a line.\nHe waited for a fish.'
    seen = []
    _F['lyrics_llm'] = lambda role, system, user: seen.append((system, user)) or text
    L.revise(_C(), text, ['Lines 2: repair wording'], 'English', meaning_only=True)
    assert 'Output only the full lyric' in seen[0][0]
    assert '1: A fisherman named Stepan arrived.' in seen[0][1]


def test_stanza_repair_preserves_unflagged_cause_and_excludes_all_bad_lines():
    text = '[bridge]\nПервое желание — золото\nЛишняя бессмыслица\nТретье желание — скорость\nЕщё бессмыслица'
    seen = []
    _F['lyrics_llm'] = lambda role, system, user: seen.append(user) or '[bridge]\nЗолото и вечность\nЗатем скорость\nСапог умчался\nДед остался'
    L.revise(_C(), text, ['Строки 3,5: restore the second wish and consequence'], 'Russian', meaning_only=True)
    assert 'Первое желание — золото' in seen[0] and 'Третье желание — скорость' in seen[0]
    assert 'Лишняя бессмыслица' not in seen[0] and 'Ещё бессмыслица' not in seen[0]


def test_rhyme_repairs_resume_once_meaning_is_sound():
    calls = []
    def llm(role, system, user):
        calls.append(role)
        if role == 'critic':
            return '{"problems": [{"kind": "style", "lines": [], "problem": "a minor style issue"}], "score": 8}'
        return ''
    _F['lyrics_llm'] = llm
    L.polish(_C(), LAZY, rounds=1)
    assert 'lines' in calls


def test_the_singer_gender_reaches_the_lyric_prompts(monkeypatch):
    assert "WOMAN" in L.singer_rule("a FEMALE lead vocal") and "MAN" in L.singer_rule("a MALE lead vocal")
    assert "WOMAN" not in L.singer_rule("a MALE lead vocal")
    assert "duet" in L.singer_rule("duet") and L.singer_rule("") == ""
    seen = []
    _F["lyrics_llm"] = lambda role, system, user: (seen.append(system), "")[1]
    L.draft(_C(), "дорога домой", "Russian", singer="a MALE lead vocal")
    assert any("The singer is a MAN" in s for s in seen)


def test_the_theme_reaches_the_critic_and_the_rewrite(monkeypatch):
    # a comic «два еврея и монета» came back a lament: the critic never saw what was asked
    seen = {}

    def llm(role, system, user):
        seen.setdefault(role, []).append(system)
        if role == "critic":
            return '{"problems": [{"kind": "style", "lines": [], "problem": "line 2 is off the theme"}], "score": 4}'
        return "" if role == "revise" else GOOD
    _F["lyrics_llm"] = llm
    _F["DRAFTS"] = 1
    L.write(_C(), "два еврея и монета", "Russian")
    assert all("два еврея и монета" in s for s in seen["critic"] + seen["revise"])
    assert "TONE" in seen["plan"][0]


# ── the bot ──────────────────────────────────────────────────────────────────
def _bot():
    import tg_bot
    tg_bot.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_lyr_"))
    b = tg_bot.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
    b.sent = []
    b._send_text = lambda cid, text, parse_mode=None, keyboard=None, **kw: b.sent.append(
        (text, parse_mode, keyboard)) or 1
    b._run_busy = lambda cid, fn, *a: fn(*a)
    b._send_get_id = lambda cid, text, parse_mode=None, keyboard=None: b.sent.append(
        (text, parse_mode, keyboard)) or 1
    b._api_post = lambda *a, **k: {}
    b._get_ctx = _C
    b.queued = []
    b._enqueue_item = lambda cid, item: b.queued.append(item)
    return b


def test_the_music_menu_has_both_buttons_in_one_new_row():
    import tg_keyboards as K
    rows = [[b for b in r] for r in K._cr_music_kb("ru")["keyboard"]]
    assert rows[-2] == ["✨ Улучшить текст", "✍️ Сочинить текст"]
    assert rows[-1] == [K._b("back", "ru")]
    assert ["🎙 Клонировать голос", "📚 Аудиокнига"] in rows and ["🎵 Песни", "🎚 Кавер / Мэшап"] in rows


def test_improve_answers_in_two_messages_and_can_be_sung(monkeypatch):
    def llm(role, system, user):
        if role == "critic":
            return '{"problems": [], "score": 9}'
        if role == "revise":
            return GOOD
        if role == "notes":
            return "• строка 3: свет/рассвет → закат/горят — однокоренная рифма"
        return ""
    _F["lyrics_llm"] = llm
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
            return '{"problems": [], "score": 9}'
        return GOOD
    _F["lyrics_llm"] = llm
    bot = _bot()
    bot._start_lyrics_flow(6, bot._get_session(6), "ru", "write")
    assert bot._lyrics_take_text(6, bot._get_session(6), "ru", "город ночью")
    assert bot.sent[-2][0].startswith("✍️ <b>Текст готов</b>") and bot.sent[-1][0] == GOOD.strip()
    # «✨ Ещё улучшить» keeps the theme; a pass that changes nothing is not «улучшенный»
    critic_seen = []
    _F["lyrics_llm"] = lambda role, system, user: (
        critic_seen.append(system) or '{"problems": [], "score": 9}') if role == "critic" else ""
    bot._cb_lyrics(6, "lyr:again")
    assert critic_seen and all("город ночью" in s for s in critic_seen)
    assert bot.sent[-2][0].startswith("✨ <b>Текст</b>"), bot.sent[-2][0]


def test_text_goes_to_the_lyrics_flow_through_the_resolver(monkeypatch):
    seen = []
    bot = _bot()
    bot._lyrics_run = (lambda ctx, cid, lang, mode, text: seen.append((mode, text)))
    s = bot._get_session(7); s.lyrics_state = "improve"; s.reg_state = ""; bot._store.put(s)
    bot._resolve_and_push(7, [{"type": "text", "text": LAZY}])
    assert seen == [("improve", LAZY.strip())]


def test_numbering_copied_back_is_removed():
    assert L._clean("1: [verse]\n2: Раз два\n3: Три четыре") == "[verse]\nРаз два\nТри четыре"


def test_what_is_left_is_shown_in_the_users_language(monkeypatch):
    def llm(role, system, user):
        if role == "critic":
            assert "Write the problems in Russian" in system
            return '{"problems": [{"kind": "style", "lines": [], "problem": "строка 2 непонятна"}], "score": 6}'
        return "ок"                                    # no usable rewrite: the lazy rhyme stays
    _F["lyrics_llm"] = llm
    bot = _bot()
    assert bot._lyrics_take_text(8, _armed(bot, 8), "ru", LAZY)
    notes = bot.sent[-2][0]
    assert "Что ещё можно доработать" in notes and "одно слово внутри другого" in notes
    assert "строка 2 непонятна" in notes and "lines " not in notes


def _armed(bot, cid):
    s = bot._get_session(cid); s.lyrics_state = "improve"; bot._store.put(s)
    return bot._get_session(cid)


def test_song_form_is_checked():
    song = """[verse 1]
Над городом гаснет закат
И окна, как свечи, горят
[chorus]
Мы будем гореть до утра
Пока не наступит пора
[verse 2]
Ночная дорога ведёт нас сквозь самый тёмный закат
И где-то вдали фонари над рекой незаметно горят
[chorus]
Мы будем гореть до конца
Пока не наступит пора"""
    kinds = [i["kind"] for i in L.analyse(song, _C())["issues"]]
    assert "verse_shape" in kinds            # verse 2 is far longer than verse 1
    assert "chorus_drift" in kinds           # «до утра» became «до конца»
    assert "end_repeat" in kinds             # «закат» closes lines in both verses
    long_hook = "[chorus]\n" + "\n".join(["раз два три четыре пять шесть семь восемь девять десять одиннадцать"] * 2)
    assert "hook_long" in [i["kind"] for i in L.analyse(long_hook, _C())["issues"]]


def test_write_plans_then_keeps_the_best_of_several_drafts(monkeypatch):
    drafts = iter([LAZY, GOOD, LAZY])
    seen = []

    def llm(role, system, user):
        seen.append(role)
        if role == "plan":
            return "HOOK: огни над городом"
        if role == "draft":
            assert "Plan:\nHOOK: огни над городом" in user
            return next(drafts)
        if role == "critic":
            return '{"problems": [], "score": 8}'
        return ""
    _F["lyrics_llm"] = llm
    _F["DRAFTS"] = 3
    res = L.write(_C(), "город ночью", "Russian")
    assert seen[0] == "plan" and seen.count("draft") == 3
    assert res["original"] == GOOD.strip()   # the best draft is the one polished


def test_a_long_lyric_cut_into_pieces_is_one_text(monkeypatch):
    seen = []
    bot = _bot()
    bot._lyrics_run = (lambda ctx, cid, lang, mode, text: seen.append(text))
    s = bot._get_session(12); s.lyrics_state = "improve"; s.reg_state = ""; bot._store.put(s)
    bot._resolve_and_push(12, [{"type": "text", "text": "первая часть"},
                               {"type": "text", "text": "вторая часть"}])
    assert seen == ["первая часть\nвторая часть"]


def test_song_words_the_bot_wrote_are_polished_before_singing(monkeypatch):
    """Wishes in 🎵 Songs -> good lyrics «под капотом», automatically."""
    import tg_songs

    class Ctx:
        stages = []

        def set_stage(self, s):
            self.stages.append(s)

        def is_cancelled(self):
            return False
    _F["lyrics_llm"] = (lambda role, s, u: '{"problems": [], "score": 9}' if role == "critic"
                        else "[verse]\n" + GOOD.split("\n", 1)[1])
    c = Ctx()
    out = tg_songs._polished(_C(c), LAZY)
    assert "закат" in out and "Polishing the lyrics" in c.stages
    _F["lyrics_llm"] = (lambda role, s, u: '{"problems": [], "score": 9}' if role == "critic"
                        else "без тегов вообще\nи всё")
    assert tg_songs._polished(_C(c), LAZY) == LAZY            # lost its [tags]: the draft is sung


def test_ready_words_polish_then_sing(monkeypatch):
    def llm(role, system, user):
        if role == "critic":
            return '{"problems": [], "score": 9}'
        if role == "notes":
            return "• строка 3: свет/рассвет → закат/горят"
        return GOOD
    _F["lyrics_llm"] = llm
    bot = _bot()
    s = bot._get_session(13); s.song_draft = LAZY; s.lang = "ru"; bot._store.put(s)
    bot._cb_song_lyrics(13, "song_lyr:polish")
    texts = [t for t, _, _ in bot.sent]
    assert any("Что поправил" in t for t in texts) and GOOD.strip() in texts
    assert bot.queued and "[lyrics]" in bot.queued[-1]["text"] and "закат" in bot.queued[-1]["text"]


def test_the_ready_words_question_offers_keep_or_polish():
    import tg_strings
    assert tg_strings._MSG["song_polish_btn"]["ru"] == "✨ Доработать и спеть"
    src = open(os.path.join(ROOT, "bot", "tg_registration.py"), encoding="utf-8").read()
    assert '"song_lyr:polish"' in src and '"song_lyr:keep"' in src


# ── rich rhyme, one line = one phrase, pairwise pick, line fixes
def test_a_rich_rhyme_scores_above_a_plain_one():
    plain = "Ночью вышла луна\nНа пороге стена\nОгонёк у окна\nТишина и весна"
    rich = "Ночью вышла луна\nЗазвенела струна\nНа пороге стена\nОпустела страна"
    a, b = L.analyse(plain, _C()), L.analyse(rich, _C())
    assert b["rich"] > a["rich"] and b["score"] >= a["score"]


def test_a_line_too_long_to_sing_is_named():
    text = ("[verse]\nМы шли по улицам ночного города и говорили обо всём на свете\n"
            "Огни горели\nИ ветер пел\nНо мы не знали")
    issues = L.analyse(text, _C())["issues"]
    long = [i for i in issues if i["kind"] == "long_line"]
    assert long and long[0]["lines"] == (2,)
    assert "одна строка — одна фраза" in L.describe(long[0], "ru")


def test_two_close_drafts_meet_head_to_head_asked_both_ways(monkeypatch):
    seen = []

    def llm(role, system, user):
        seen.append(role)
        if role == "judge":
            # GOOD wins in both orders
            return "B" if "B:\n" + GOOD in user else "A"
        return ""
    _F["lyrics_llm"] = llm
    assert L.compare(_C(), "т", LAZY, GOOD) == GOOD
    assert seen.count("judge") == 2

    # a judge that just likes position A: the checks' pick stays
    _F["lyrics_llm"] = lambda role, s, u: "A"
    assert L.compare(_C(), "т", LAZY, GOOD) == LAZY


def test_a_non_rhyming_line_is_mended_alone_from_versions(monkeypatch):
    text = "Над рекой туман\nТихо спит вода\nВетер был сильный\nНе уйду туда"
    asked = []

    def llm(role, system, user):
        if role == "lines":
            asked.append((system, user))
            return "1. Сердце в огне\n2. Ветер был сильный\n3. Вновь пришёл обман"  # only 3 rhymes
        return ""
    _F["lyrics_llm"] = llm
    out = L.fix_rhymes(_C(), text, L.analyse(text, _C()), "Russian")
    lines = out.splitlines()
    assert asked and "«туман»" in asked[0][0] and ">>> Ветер был сильный" in asked[0][1]
    assert lines[2] == "Вновь пришёл обман"
    assert lines[:2] + lines[3:] == text.splitlines()[:2] + text.splitlines()[3:]  # the rest stays
    assert not L.analyse(out, _C())["issues"]


def test_creative_calls_ask_for_min_p(monkeypatch):
    got = {}
    monkeypatch.delenv("F5_TEST_RUN", raising=False)
    _F["text_model"] = lambda *a, **k: got.update(k) or "x"
    L._call(_C(), "draft", "s", "u", temperature=0.9, max_tokens=10)
    assert got.get("sampling") == {"min_p": L.MIN_P}
    got.clear()
    L._call(_C(), "critic", "s", "u", temperature=0.0, max_tokens=10)
    assert "sampling" not in got


def test_a_two_syllable_rhyme_counts_more():
    one = "Мне не нужна дорога\nЯ помню всё до слова\nСтою я у порога\nИ жду тебя я снова"
    two = "Мне не нужна дорога\nЯ помню всё до слова\nВ глазах твоих тревога\nИ жду тебя я снова"
    a, b = L.analyse(one, _C()), L.analyse(two, _C())
    assert L._deep(L._row(_C(), 1, "дорога"), L._row(_C(), 3, "порога"))
    assert a["rich"] >= b["rich"]


def test_yo_words_rhyme_with_the_accentor_on(monkeypatch):
    """RUAccent leaves ё unmarked: «огнём / куём» was read as no rhyme."""
    marks = {"Это сбруя — контроль над безумным огнём": "Это сбр+уя контр+оль над без+умным огнём",
             "Мы структуру и рамки для мысли куём": "Мы стр+уктуру и р+амки для м+ысли куём"}
    _F["accent"] = lambda line: marks[line]
    res = L.analyse("\n".join(marks), _C())
    assert res["stressed"] and not [i for i in res["issues"] if i["kind"] == "no_rhyme"]


def test_a_latin_word_in_a_russian_song_is_named():
    t = "Чтобы хаос в агента послушного стал\nHarness держит её, чтоб поток не пропал"
    issues = [i for i in L.analyse(t, _C())["issues"] if i["kind"] == "foreign"]
    assert issues and issues[0]["lines"] == (2,) and "Harness" in L.describe(issues[0], "ru")


def test_an_unchanged_lyric_with_problems_left_is_not_called_good(monkeypatch):
    bot = _bot()
    res = {"text": LAZY.strip(), "original": LAZY.strip(), "score": 4.6, "rounds": 3,
           "left_rules": L.analyse(LAZY, _C())["issues"], "left_sense": []}
    bot._lyrics_show(5, "ru", "improve", res, [])
    head = bot.sent[0][0]
    assert "Править нечего" not in head and "не вышла лучше" in head


def test_back_under_a_result_keeps_it_and_removes_only_the_buttons():
    """↩ Назад under «🎵 Спеть / ✨ Ещё» deleted the song text."""
    import tg_bot
    tg_bot.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_close_"))
    b = tg_bot.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
    calls = []
    b._api_post = lambda method, p, *a, **k: calls.append((method, p)) or {}
    b._close_menu(5, {"message_id": 7, "text": GOOD * 3})
    assert calls[-1][0] == "editMessageReplyMarkup"
    b._close_menu(5, {"message_id": 8, "photo": [{}], "caption": "x"})
    assert calls[-1][0] == "editMessageReplyMarkup"
    b._close_menu(5, {"message_id": 9, "text": "Короткий куплет"})
    assert not [c for c in calls if c[0] == "deleteMessage"]


# ── ё and repeated-chorus handling ───────────────────────────────────────────
def test_a_restored_yo_keeps_the_lines_stress(monkeypatch):
    """RUAccent restores ё («все» -> «всё»): the strict word match threw the
    whole line's stress away."""
    _F["accent"] = lambda line: "Мы вс+ё отд+али"
    assert L._accent_line(_C(), "Мы все отдали", ["Мы", "все", "отдали"]) == ["Мы", "вс+ё", "отд+али"]


CHORUS3 = """[chorus]
Это сбруя на диком огне
Мы куём её снова и вновь

[chorus]
Это сбруя на диком огне
Мы куём её снова и вновь"""


def test_a_repeated_chorus_is_judged_once():
    issues = [i for i in L.analyse(CHORUS3, _C())["issues"] if i["kind"] == "no_rhyme"]
    assert len(issues) == 1 and issues[0]["lines"] == (2, 3)


def test_a_mended_chorus_line_is_mended_in_every_repeat(monkeypatch):
    _F["lyrics_llm"] = lambda role, s, u: "Мы куём её в каждом окне" if role == "lines" else ""
    out = L.fix_rhymes(_C(), CHORUS3, L.analyse(CHORUS3, _C()), "Russian")
    assert out.count("Мы куём её в каждом окне") == 2 and "снова и вновь" not in out


def test_a_cyrillic_verdict_is_read(monkeypatch):
    _F["lyrics_llm"] = lambda role, s, u: "В"
    assert L._verdict(_C(), "т", "a", "b") == "B"


def test_the_person_lookup_does_not_show_up_as_a_search_in_the_reply(monkeypatch):
    import person_look as P
    monkeypatch.delenv("F5_TEST_RUN", raising=False)
    ctx = types.SimpleNamespace(web_search_enabled=True, turn_queries=["кто такой Ленин"], turn_sources=[])

    def run(c, q):
        c.turn_queries.append(q); c.turn_sources.append({"url": "x"})
        return "bald, goatee"
    _F["run_web_search"] = run
    assert P._search(_C(ctx), "Lenin appearance") == "bald, goatee"
    assert ctx.turn_queries == ["кто такой Ленин"] and ctx.turn_sources == []



def test_long_lyrics_work_has_a_cancel_and_stops_on_it(monkeypatch):
    """«✍️ Сочиняю…» ran for minutes with no way to stop it. One
    central runner puts ⛔ under the note; ⛔ stops the work and nothing is shown."""
    bot = _bot()
    seen = {}

    def llm(role, system, user):
        if role == "draft":
            seen["job"].set()                         # ⛔ pressed while drafting
        return GOOD if role == "draft" else '{"problems": [], "score": 9}'
    _F["lyrics_llm"] = llm
    _F["DRAFTS"] = 3
    real = bot._run_busy

    def run_busy(cid, fn, *a):
        seen["job"] = next(iter(bot._busy_jobs.values()))[1]
        return real(cid, fn, *a)
    bot._run_busy = run_busy
    s = bot._get_session(21); s.lyrics_state = "write"; s.lang = "ru"; bot._store.put(s)
    assert bot._lyrics_take_text(21, s, "ru", "город ночью")
    note, _, kb = bot.sent[0]
    assert "Сочиняю" in note and kb["inline_keyboard"][0][0]["callback_data"].startswith("busy:cancel:")
    assert len(bot.sent) == 1                         # cancelled: no lyric, no «failed»
    assert not bot._busy_jobs                         # the job is gone, its button removed
