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


def test_the_budget_grows_with_the_spoken_lines():
    long = ['He says: "' + "Слово " * 120 + '"'] * 6
    owner, asked = _owner('{"lines": ["x", "x", "x", "x", "x", "x"]}')
    P._summaries(long, "ru", owner)
    assert asked[0] > 200 + 90 * 6 + sum(len(t) for t in long) // 3


def test_two_failures_fall_back_to_the_prompts():
    owner, asked = _owner("no json", "still none")
    assert P._summaries(PARTS, "ru", owner) == ["", ""]
    assert len(asked) == 2
