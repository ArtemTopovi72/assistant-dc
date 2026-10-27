"""Local random draws shared by the desktop and Telegram interfaces."""
import random
import re
from seams import text_model

SYMBOLS = ('🍒', '🍋', '🍇', '🔔', '⭐', '💎')
_RNG = random.SystemRandom()
_RANGE = re.compile(r'(?:от\s*)?([+-]?\d{1,64})(?:\s+до\s+|\s*\.\.\s*|\s*[…,:;]\s*|\s+|\s*[-–—]\s*)([+-]?\d{1,64})', re.I)


def spin(rng=None):
    return tuple((rng or _RNG).choice(SYMBOLS) for _ in range(3))


def jackpot(symbols):
    return len(symbols) == 3 and len(set(symbols)) == 1


def number(low, high, rng=None):
    if low > high:
        raise ValueError('Lower bound must not exceed upper bound.')
    return (rng or _RNG).randint(low, high)


def binary(rng=None):
    return (rng or _RNG).choice((True, False))


def parse_range(text):
    match = _RANGE.fullmatch(text.strip()) if len(text) <= 150 else None
    if not match:
        raise ValueError('Enter two integers, for example: -10 100.')
    low, high = map(int, match.groups())
    if low > high:
        raise ValueError('Lower bound must not exceed upper bound.')
    return low, high


def prediction(ctx, period, lang='ru', rng=None):
    labels = {'day': ('день', 'day'), 'week': ('неделю', 'week'), 'month': ('месяц', 'month')}
    if period not in labels:
        raise ValueError('Unknown prediction period.')
    rng = rng or _RNG
    numbers = sorted(rng.sample(range(1, 100), 3))
    language = 'Russian' if lang == 'ru' else 'English'
    system = (f'Write a playful mini horoscope in {language}, 70–100 words. '
              'Give four short points: what may happen; who is easy to work with; '
              'who to be cautious with; one small suggestion. Describe people by '
              'behaviour (keeping promises, listening, rushing), never by real names '
              'or assumed hidden intentions. Everyday themes: mood, conversations, '
              'small projects and rest. This is imaginative entertainment, not a '
              'factual forecast. No guarantees, scary threats, introduction, disclaimer, '
              'HTML or Markdown. Do not add lucky numbers; those are attached separately.')
    user = f'Period: {period}. Variation token: {rng.randrange(1_000_000)} (do not print this token).'
    body = text_model(ctx)(ctx, system, user, temperature=0.9, max_tokens=450)
    if not isinstance(body, str) or not body.strip():
        raise RuntimeError('Prediction could not be generated. Try again.')
    heading = f'🔮 Мини-предсказание на {labels[period][0]}' if lang == 'ru' else f'🔮 Mini prediction for a {labels[period][1]}'
    caption = 'Важные числа' if lang == 'ru' else 'Important numbers'
    return heading + '\n\n' + body.strip() + '\n\n' + caption + ': ' + ', '.join(map(str, numbers))
