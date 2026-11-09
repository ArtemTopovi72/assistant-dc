"""Video plans keep the original speaker attribution and dialogue intact."""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "core", "agent", "media", "bot"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ.setdefault("F5_TEST_RUN", "1")

import tg_video_plan as P  # noqa: E402

PARTS = ['A man waves.', 'A woman nods.']


def _owner(*replies):
    asked = []

    def model(ctx, system, user, **kw):
        asked.append(kw.get("max_tokens"))
        return replies[len(asked) - 1]
    return types.SimpleNamespace(text_model=model), asked


def test_a_broken_reply_is_asked_again_before_the_prompts_show():
    owner, asked = _owner('{"lines": ["Мужчина машет"', '{"lines": ["1. Мужчина машет", "2. Женщина кивает"]}')
    assert P._summaries(PARTS, "ru", owner) == ['Мужчина машет', 'Женщина кивает']
    assert len(asked) == 2


def test_a_reply_that_corrects_itself_gives_its_whole_list():
    """One line, then «Wait… let me do all of them» and the full list (live 2026-10-20)."""
    owner, asked = _owner('{"lines": ["Мужчина машет"]}\nWait, both shots:\n'
                          '```json\n{"lines": ["Мужчина машет", "Женщина кивает"]}\n```')
    assert P._summaries(PARTS, "ru", owner) == ['Мужчина машет', 'Женщина кивает']
    assert len(asked) == 1


def test_one_quote_with_two_people_preserves_original_speaker():
    owner, asked = _owner('{"lines": ["Алиса говорит: \\\"Привет\\\""]}')
    assert P._summaries(['Alice watches Bob. Bob says "Привет".'], 'ru', owner) == ['']
    assert not asked


def test_spoken_camera_cuts_phrase_is_still_preserved():
    owner, asked = _owner('{"lines": ["Мужчина говорит"]}')
    assert P._summaries(['The man says "The camera cuts to the chase".'], 'ru', owner) == ['']
    assert not asked


def test_unquoted_parts_still_receive_translations():
    owner, asked = _owner('{"lines": ["Женщина кивает"]}')
    assert P._summaries(['Bob says "Привет".', 'A woman nods.'], 'ru', owner) == ['', 'Женщина кивает']
    assert len(asked) == 1


def test_plan_shows_timing_and_spacing_inside_dialogue_unchanged():
    sess = types.SimpleNamespace(video_plan=[{'text': 'Bob says "Start  [3.2 s] now".', 'sec': 5}])
    text = P.VideoPlanMixin()._video_plan_script(sess, 'ru')
    assert 'Start  [3.2 s] now' in text


def test_multiple_quotes_keep_original_plan_without_misattributing_speakers():
    owner, asked = _owner('{"lines": ["Алиса: \\\"Нет\\\". Боб: \\\"Да\\\""]}')
    assert P._summaries(['Alice says "Да". Bob says "Нет".'], 'ru', owner) == ['']
    assert not asked


def test_repeated_quotes_do_not_get_collapsed():
    owner, asked = _owner('{"lines": ["Он: \\\"Да\\\""]}')
    assert P._summaries(['He says "Да", then repeats "Да".'], 'ru', owner) == ['']
    assert not asked


def test_the_budget_grows_with_long_descriptions():
    long = ['The scene shows ' + "details " * 120] * 6
    owner, asked = _owner('{"lines": ["x", "x", "x", "x", "x", "x"]}')
    P._summaries(long, "ru", owner)
    assert asked[0] > 200 + 90 * 6 + sum(len(t) for t in long) // 3


def test_plan_translation_reserves_reasoning_and_uses_structured_output():
    calls = []
    def model(ctx, system, user, **kw):
        calls.append(kw)
        return '{"lines": ["Первый кадр", "Второй кадр"]}'
    assert len(P._summaries(PARTS, 'ru', types.SimpleNamespace(text_model=model))) == 2
    assert calls[0]['max_tokens'] >= 4096
    assert calls[0]['json_schema']['required'] == ['lines']


def test_two_failures_fall_back_to_the_prompts():
    owner, asked = _owner("no json", "still none")
    assert P._summaries(PARTS, "ru", owner) == ["", ""]
    assert len(asked) == 2


def test_plan_warns_when_it_cannot_reach_requested_duration():
    sess = types.SimpleNamespace(video_plan=[{'text': 'first', 'sec': 10.1},
                                             {'text': 'second', 'sec': 10.1}],
                                 video_plan_requested_seconds=25)
    text, _ = P.VideoPlanMixin()._video_plan_card(sess, 'ru')
    assert '25' in text and ('меньше' in text or 'не хватает' in text)


def test_plan_does_not_hide_end_of_part_before_approval():
    text = 'Мужчина говорит: ' + 'слово ' * 25 + 'важный конец фразы'
    sess = types.SimpleNamespace(video_plan=[{'text': text, 'sec': 6.2}])
    assert 'важный конец фразы' in P.VideoPlanMixin()._video_plan_script(sess, 'ru')


def test_long_plan_is_sent_in_full_across_messages():
    plan = P.VideoPlanMixin()
    sent = []
    plan._send_text = lambda chat_id, text, **kw: sent.append(text)
    plan._state_kb = lambda sess, lang: None
    plan._video_plan_card = lambda sess, lang: ('карточка', {})
    sess = types.SimpleNamespace(video_plan=[{'text': 'слово ' * 1000 + 'ФИНАЛ', 'sec': 6.2}])
    plan._send_video_plan(1, sess, 'ru')
    assert len(sent) > 2 and all(len(s) <= 4096 for s in sent)
    assert 'ФИНАЛ' in ''.join(sent[:-1])
