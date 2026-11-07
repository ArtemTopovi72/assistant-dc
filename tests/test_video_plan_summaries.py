"""The video plan shows each part in the chat's language, never the English prompts.

A six-part Russian monologue came back unusable once and the chat saw «The elderly man …
looks directly into the came…» for every part (live 2026-10-20): 90 tokens a part did not
hold the spoken lines copied whole, and the fallback was silent.
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "core", "agent", "media", "bot"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ.setdefault("F5_TEST_RUN", "1")

import tg_video_plan as P  # noqa: E402

PARTS = ['The man says: "Ах, вот и ты."', 'He goes on: "Нам нужен твой морозильник."']


def _owner(*replies):
    asked = []

    def model(ctx, system, user, **kw):
        asked.append(kw.get("max_tokens"))
        return replies[len(asked) - 1]
    return types.SimpleNamespace(text_model=model), asked


def test_a_broken_reply_is_asked_again_before_the_prompts_show():
    owner, asked = _owner('{"lines": ["Мужчина: \\"Ах, вот и ты.\\""', '{"lines": ["1. Мужчина: \\"Ах, вот и ты.\\"", '
                          '"2. Он: \\"Нам нужен твой морозильник.\\""]}')
    assert P._summaries(PARTS, "ru", owner) == ['Мужчина: "Ах, вот и ты."', 'Он: "Нам нужен твой морозильник."']
    assert len(asked) == 2


def test_a_reply_that_corrects_itself_gives_its_whole_list():
    """One line, then «Wait… let me do all of them» and the full list (live 2026-10-20)."""
    owner, asked = _owner('{"lines": ["Мужчина: \\"Ах, вот и ты.\\""]}\nWait, both shots:\n'
                          '```json\n{"lines": ["Мужчина: \\"Ах, вот и ты.\\"", "Он: \\"Нам нужен твой морозильник.\\""]}\n```')
    assert P._summaries(PARTS, "ru", owner) == ['Мужчина: "Ах, вот и ты."', 'Он: "Нам нужен твой морозильник."']
    assert len(asked) == 1


def test_the_budget_grows_with_the_spoken_lines():
    long = ['He says: "' + "Слово " * 120 + '"'] * 6
    owner, asked = _owner('{"lines": ["x", "x", "x", "x", "x", "x"]}')
    P._summaries(long, "ru", owner)
    assert asked[0] > 200 + 90 * 6 + sum(len(t) for t in long) // 3


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
