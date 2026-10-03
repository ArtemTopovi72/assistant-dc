"""Live: the reads that replaced the last user-text regexes (10-02 sweep):
preferences, deck edits, face-changing edits, research kind, news queries,
placement on the body, whole-face regions. Run with LM Studio up."""
import os, sys
os.environ["INTENT_LIVE"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "agent", "imaging", "research", "core", "bot", "media"):
    sys.path.insert(0, os.path.join(ROOT, d))
from context_v2 import _is_preference
from tools import _edits_deck
from image_identity import _changes_face
from dr_policy import _ask_kind
from search import _is_news
from image_transfer import _placement_region_from_instruction
from image_maskqa import _is_whole_face
from search import wanted_count
from graph_finalize import _fact_of_remember_request

CASES = [
    (_is_preference, "всегда отвечай коротко", True),
    (_is_preference, "не присылай мне голосовые никогда", True),
    (_is_preference, "I prefer metric units", True),
    (_is_preference, "какая завтра погода?", False),
    (_is_preference, "нарисуй кота", False),
    (_edits_deck, "добавь слайд про цены", True),
    (_edits_deck, "убери третий слайд", True),
    (_edits_deck, "rename the deck to Q3 results", True),
    (_edits_deck, "презентация про историю Рима", False),
    (_changes_face, "make her smile", True),
    (_changes_face, "сделай его моложе", True),
    (_changes_face, "add red lipstick", True),
    (_changes_face, "change the jacket to red", False),
    (_changes_face, "make the background a beach", False),
    (_changes_face, "сделай фото чётче", False),
    (_ask_kind, "чем iPhone 16 отличается от iPhone 15", "compare"),
    (_ask_kind, "где купить билеты в Эрмитаж и сколько стоят", "practical"),
    (_ask_kind, "влияние микропластика на здоровье", "research"),
    (_is_news, "новости Зеленоград взрыв", True),
    (_is_news, "latest headlines Ukraine", True),
    (_is_news, "рецепт борща", False),
    (_is_news, "история Рима", False),
    (_placement_region_from_instruction, "put the tattoo on his upper arm", "upper arm"),
    (_placement_region_from_instruction, "place the logo across her chest", "chest"),
    (_placement_region_from_instruction, "put this hat on him", None),
    (_is_whole_face, "his face", True),
    (_is_whole_face, "the man's body and face", False),
    (wanted_count, "5 анекдотов про Чапаева", 5),
    (wanted_count, "три рецепта борща", 3),
    (wanted_count, "iphone 5 price", 0),
    (_fact_of_remember_request, "запомни: меня зовут Артём", "меня зовут Артём"),
    (_fact_of_remember_request, "remember that my dog is Zhuzha", "my dog is Zhuzha"),
]
bad = 0
for fn, text, want in CASES:
    got = fn(text)
    ok = got == want
    bad += not ok
    print("ok  " if ok else "BAD ", fn.__name__, repr(text), "->", got)
print("ALL OK" if not bad else f"{bad} wrong")
