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
    # The scale must be FELT: a day is one tiny concrete moment, a month is a
    # trend with a decision. One shared 4-point shape made all three periods
    # read as the same horoscope soup (2026-10-28).
    shapes = {
        'day': ('ONE ordinary day, and it is small. 2-3 sentences, about 40 words: one little '
                'thing that will go suspiciously well today (a find, a snack, green lights all '
                'the way); one tiny trap (something forgotten, a stubborn zipper); one '
                'micro-move that saves the evening. A friend winking, not a fortune teller.'),
        'week': ('ONE week -- a small story arc. 3-4 sentences, about 60 words: the week\'s '
                 'theme in three words; one meeting or message that bends the plan for the '
                 'better; one thing to say "no" to so the weekend stays yours; end with a tiny '
                 'challenge for the reader. Playful, slightly cheeky.'),
        'month': ('A WHOLE month -- big strokes and decisions, not daily weather. 5-6 '
                  'sentences, about 90 words: the month\'s trend as a headline; one door '
                  'opening (a skill, a trip, an idea worth starting); one door closing that '
                  'turns out fine; a decision the reader will be proud of by month\'s end; a '
                  'bold, warm send-off. Epic but cosy.'),
    }
    banned = ('BANNED phrases, never use them or their calques: «вспышки вдохновения» / bursts '
              'of inspiration, «душевное равновесие» / inner peace, people who rush or '
              'interrupt, «пойдут как по маслу», any generic horoscope filler. ')
    system = (f'Write a playful mini prediction in {language}. Period and required shape: '
              f'{shapes[period]} {banned}'
              'Describe people by behaviour only, never real names or claimed hidden '
              'intentions. This is imaginative entertainment, not a factual forecast. No '
              'guarantees, no scary threats, no introduction, disclaimer, HTML or Markdown. '
              'Do not add lucky numbers; those are attached separately.')
    user = f'Period: {period}. Variation token: {rng.randrange(1_000_000)} (do not print this token).'
    body = text_model(ctx)(ctx, system, user, temperature=0.9, max_tokens=450)
    if not isinstance(body, str) or not body.strip():
        raise RuntimeError('Prediction could not be generated. Try again.')
    heading = f'🔮 Мини-предсказание на {labels[period][0]}' if lang == 'ru' else f'🔮 Mini prediction for a {labels[period][1]}'
    caption = 'Важные числа' if lang == 'ru' else 'Important numbers'
    return heading + '\n\n' + body.strip() + '\n\n' + caption + ': ' + ', '.join(map(str, numbers))
