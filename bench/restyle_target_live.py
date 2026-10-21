"""Live: image_style._expand's target -- a style of this picture (drawn / photo) vs a new scene
(other, edited as written). «A cinematic, highly detailed scene of Lenin in a cybernetic suit in
hell fighting demons» read as a drawn style, got «keep the same composition», and the Wikipedia
screenshot came back as itself tinted with lava (live 2026-10-21).

    venv/Scripts/python bench/restyle_target_live.py
"""
import os, sys, types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [ROOT] + [os.path.join(ROOT, d) for d in ("agent", "core", "bot", "imaging", "media", "knowledge")] \
    + [p for p in sys.path if os.path.basename(p) != "bench"]
sys.stdout.reconfigure(encoding="utf-8")
os.environ.pop("F5_TEST_RUN", None)
import image_style as S

CASES = [
    ("A cinematic, highly detailed scene of Vladimir Lenin wearing a futuristic cybernetic suit, falling into a "
     "fiery hellscape filled with flowing lava, fighting demons with his fists.", "other"),
    ("put him on a beach at sunset, he is surfing a huge wave", "other"),
    ("an epic fantasy painting of the cat riding a dragon over a burning castle", "other"),
    ("dress her in a red ball gown at a royal palace ball", "other"),
    ("make it look like a Japanese anime frame", "drawn"),
    ("turn it into an oil painting in the style of Van Gogh", "drawn"),
    ("make it a pencil sketch", "drawn"),
    ("make it a realistic photo", "photo"),
    ("ultra-realistic DSLR photograph of this", "photo"),
    ("cyberpunk neon illustration style", "drawn"),
    ("steampunk engraving look", "drawn"),
    ("film noir black and white style", "photo"),
    ("make it look like a watercolor painting", "drawn"),
    ("the same dog, but now chasing a ball in a snowy park, in pixar style", "other"),
]
bad = 0
for words, want in CASES:
    got = S._expand(types.SimpleNamespace(), words)[0]
    bad += got != want
    print(("PASS " if got == want else "FAIL ") + f"{want:5} {words[:70]!r} -> {got}")
print(f"\n{bad} failed")
