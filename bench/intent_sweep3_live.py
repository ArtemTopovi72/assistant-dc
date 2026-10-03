"""Live: the reads that replaced inline keyword checks (10-02 sweep 3): the
picture's look, a memory card's kind, echo asks, a picture on every slide,
lettering removal, outermost swaps, rotate/mirror/crop, caption placement,
«пришли файлом», the city in a saved fact. Run with LM Studio up."""
import os, sys
os.environ["INTENT_LIVE"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "agent", "imaging", "research", "core", "bot", "media", "voice", "gui", "knowledge"):
    sys.path.insert(0, os.path.join(ROOT, d))
import intent
from ideogram_layout import named_style, wants_realism
from context_v2 import _kind_of
from graph_language import _asks_to_echo
from draw_agent import _instruction_removes, _swaps_outermost
from text_overlay import placement
from tg_dispatch import _asks_as_file
from tg_weather import _city_of_fact
from image_sizing import detect_orientation_from_text as orient
from ideogram import requested_strings

EVERY = "A user asked for a presentation: {text}. Does the user ask for a picture on EVERY slide?"
TRANSFORM = ("A user asked to transform a picture: {text}. Which? crop = crop to the face; "
             "mirror = flip left-right; flip = turn upside down; rotate_left = rotate "
             "counter-clockwise; rotate_right = rotate clockwise.")


def every(t):
    return intent.ask_yes(EVERY, t)


def transform(t):
    return intent.ask_choice(TRANSFORM, t, ("crop", "mirror", "flip", "rotate_left", "rotate_right"), "rotate_right")


CASES = [
    (named_style, "нарисуй кота в стиле аниме", "anime"),
    (named_style, "карикатура на политика", "caricature"),
    (named_style, "комната, на стене висит картина с морем", ""),
    (named_style, "draw a painting of a harbour", "painting"),
    (named_style, "кот на диване", ""),
    (wants_realism, "реалистичное фото кота на диване", True),
    (wants_realism, "мультяшный кот", False),
    (wants_realism, "кот на диване", False),
    (_kind_of, "поменяй фон на картинке на синий", "image"),
    (_kind_of, "исправь баг в main.py", "code"),
    (_kind_of, "найди исследования про микропластик", "research"),
    (_kind_of, "как дела?", "chat"),
    (_asks_to_echo, "повтори за мной: я следую общим принципам", True),
    (_asks_to_echo, "исправь ошибки в тексте: мама мыла раму", True),
    (_asks_to_echo, "мой калькулятор показывает 418, это правильно?", False),
    (every, "презентация про динозавров, с картинкой на каждом слайде", True),
    (every, "презентация про динозавров", False),
    (lambda t: _instruction_removes(t, "Вика", ""), "убери Вику", True),
    (lambda t: _instruction_removes(t, "Text for step two", "label of step two"), "убери второй шаг", True),
    (lambda t: _instruction_removes(t, "SALE", "sign"), "добавь кота", False),
    (lambda t: _instruction_removes(t, "SALE", "sign"), "убери все надписи", True),
    (_swaps_outermost, "поменяй местами крайних пингвинов", True),
    (_swaps_outermost, "поменяй местами кота и собаку", False),
    (transform, "поверни на 90 градусов влево", "rotate_left"),
    (transform, "отрази зеркально", "mirror"),
    (transform, "переверни вверх ногами", "flip"),
    (transform, "обрежь по лицу", "crop"),
    (placement, "добавь снизу подпись «Привет»", "bottom"),
    (placement, "напиши по центру «SALE»", "middle"),
    (placement, "напиши в небе «Привет»", "top"),
    (_asks_as_file, "пришли файлом", True),
    (_asks_as_file, "без сжатия", True),
    (_asks_as_file, "вот документ по налогам", False),
    (_asks_as_file, "нарисуй кота", False),
    (_city_of_fact, "Артём переехал в Самару", "Самара"),
    (_city_of_fact, "user lives in Berlin", "Berlin"),
    (_city_of_fact, "у меня собака Жужа", ""),
    (orient, "a wide landscape banner", "landscape"),
    (orient, "vertical portrait phone wallpaper", "portrait"),
    (orient, "a single person standing, headshot", "portrait"),
    (orient, "a city street crowd panorama", "landscape"),
    (orient, "landscape photo of a man", "landscape"),
    (requested_strings, 'a neon sign reading "CAFE ROSA"', ["CAFE ROSA"]),
    (requested_strings, "нарисуй вывеску с надписью «ОТКРЫТО»", ["ОТКРЫТО"]),
    (requested_strings, 'draw a "cat" on a sofa', []),
]
bad = 0
for fn, text, want in CASES:
    got = fn(text)
    ok = got == want
    bad += not ok
    print("ok  " if ok else "BAD ", getattr(fn, "__name__", "fn"), repr(text), "->", repr(got))
print("ALL OK" if not bad else f"{bad} wrong")
