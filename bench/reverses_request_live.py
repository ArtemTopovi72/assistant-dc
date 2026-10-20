"""Live: the edit guard's «does the edit do the OPPOSITE of the request?» read
(graph_personality._reverses_request). It blocked a removal written as the look of the result
(«remove from the image: …» -> instructions describing the uncovered state, live 2026-10-21
01:02). Neutral pairs: (user request, edit args, is it the opposite?).

    venv/Scripts/python bench/reverses_request_live.py
"""
import os, sys, types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [ROOT] + [os.path.join(ROOT, d) for d in ("agent", "core", "bot", "imaging", "media", "knowledge")] \
    + [p for p in sys.path if os.path.basename(p) != "bench"]
sys.stdout.reconfigure(encoding="utf-8")
os.environ.pop("F5_TEST_RUN", None)
import graph_personality as gp

CASES = [
    ("remove from the image: убери с него шапку",
     {"region": "the knitted hat on his head", "instructions": "bare head, natural short hair, no hat, removing the hat completely"}, False),
    ("remove from the image: убери очки",
     {"region": "the glasses on his face", "instructions": "bare eyes and eyebrows, natural skin, removing the glasses entirely"}, False),
    ("убери чашку со стола",
     {"region": "the cup on the table", "instructions": "empty wooden table surface, removing the cup"}, False),
    ("сними с неё куртку, пусть будет в футболке",
     {"region": "the red jacket", "instructions": "a plain white t-shirt, the jacket removed"}, False),
    ("убери машину с улицы",
     {"region": "the red car", "instructions": "empty asphalt road, removing the car completely, realistic texture"}, False),
    ("убери фонарный столб",
     {"region": "the lamp post", "instructions": "plain brick wall and pavement where the post stood, the post removed"}, False),
    ("убери вывеску над дверью",
     {"region": "the sign above the door", "instructions": "a clean plain wall above the door, no sign"}, False),
    ("убери книгу из его рук",
     {"region": "the book in his hands", "instructions": "his empty hands resting on the counter, the book removed"}, False),
    ("убери фонарный столб", {"region": "the pavement", "instructions": "a tall iron lamp post on the pavement"}, True),
    ("сделай небо голубым",
     {"region": "the sky", "instructions": "clear bright blue sky"}, False),
    ("убери шапку", {"region": "his head", "instructions": "a thick wool hat covering his head"}, True),
    ("убери очки", {"region": "his face", "instructions": "large black sunglasses on his eyes"}, True),
    ("сделай фото светлее", {"region": "the whole picture", "instructions": "darker, night-time, dim lighting"}, True),
    ("добавь собаку рядом с ним", {"region": "the dog", "instructions": "remove the dog, empty grass"}, True),
    ("сними с него куртку", {"region": "his shirt", "instructions": "a heavy winter jacket zipped up over his shirt"}, True),
]
bad = 0
ctx = types.SimpleNamespace()
for req, args, want in CASES:
    got = gp._reverses_request(ctx, req, {**args, "removal": True})
    bad += got is not want
    print(("PASS " if got is want else "FAIL ") + f"{'opposite' if want else 'same aim'}: {req!r} / {args['instructions'][:60]!r} -> {got}")
print(f"\n{bad} failed")
