import random

import pytest

import randomizer as R
from types import SimpleNamespace


class Fixed:
    def __init__(self, values):
        self.values = iter(values)

    def choice(self, values):
        value = next(self.values)
        assert value in values
        return value


def test_slots_win_only_when_all_three_match():
    assert R.jackpot(R.spin(Fixed(['🍒'] * 3)))
    assert not R.jackpot(R.spin(Fixed(['🍒', '🍒', '💎'])))


@pytest.mark.parametrize('text,expected', [
    ('1 100', (1, 100)), ('от -10 до 10', (-10, 10)),
    ('1..100', (1, 100)), ('1-100', (1, 100)),
    ('-10--1', (-10, -1)), ('-10 -1', (-10, -1)), ('7 7', (7, 7)),
])
def test_range_formats(text, expected):
    assert R.parse_range(text) == expected


@pytest.mark.parametrize('text', ['100 1', '1.5 10', 'one ten', '1 2 3', '1'])
def test_invalid_range_is_rejected(text):
    with pytest.raises(ValueError):
        R.parse_range(text)


def test_number_includes_both_endpoints_and_supports_negative_values():
    rng = random.Random(12)
    assert {R.number(-1, 1, rng) for _ in range(100)} == {-1, 0, 1}
    assert R.number(7, 7, rng) == 7
    assert {R.binary(rng) for _ in range(100)} == {True, False}
    with pytest.raises(ValueError):
        R.number(2, 1, rng)


@pytest.mark.parametrize('period', ['day', 'week', 'month'])
def test_prediction_uses_the_selected_period_and_real_random_numbers(period):
    calls = []
    ctx = SimpleNamespace(text_model=lambda c, s, u, **kw: calls.append((s, u)) or 'Впереди приятная встреча.')
    text = R.prediction(ctx, period, 'ru', random.Random(9))
    assert 'Впереди приятная встреча.' in text and 'Важные числа:' in text
    assert period in calls[0][1]
    assert len(text.split('Важные числа:')[1].strip().split(', ')) == 3


def test_invalid_prediction_period_does_not_call_the_model():
    ctx = SimpleNamespace(text_model=lambda *a, **kw: pytest.fail('invalid period reached model'))
    with pytest.raises(ValueError):
        R.prediction(ctx, 'year')


def test_prediction_varies_the_creative_brief_not_just_a_number():
    calls = []
    ctx = SimpleNamespace(text_model=lambda c, s, u, **kw: calls.append((s, u)) or 'Ответ')
    for seed in range(12):
        R.prediction(ctx, 'week', 'ru', random.Random(seed))
    assert len({user for _, user in calls}) == 12
    assert len({user.split('Variation token:')[0] for _, user in calls}) > 4
