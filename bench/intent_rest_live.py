"""Live check of the last word lists turned into model reads: song reset,
no-vocals / own form, camera motion, side questions, complaints, collage and
dedupe asks, add-only and centre edits, lettering removal.

    venv/Scripts/python bench/intent_rest_live.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [ROOT] + [os.path.join(ROOT, d) for d in ("agent", "core", "bot", "imaging", "media", "knowledge")] \
    + [p for p in sys.path if os.path.basename(p) != "bench"]
sys.stdout.reconfigure(encoding="utf-8")
os.environ.pop("F5_TEST_RUN", None)
import intent
import tg_music, music, video, steer, turn_audit, draw_agent
import image_lettering_remove as L

bad = 0
def check(name, ok):
    global bad
    bad += not ok
    print(("PASS " if ok else "FAIL ") + name)

def table(name, fn, rows):
    for t, want in rows:
        check(f"{name} {'yes' if want else 'no'}: {t}", bool(fn(t)) is want)

table("reset", tg_music.is_reset_phrase, [("сбрось настройки песни", True), ("reset song settings", True),
      ("обнули параметры музыки", True), ("сочини песню про сброс", False), ("сбрось фильтры", False)])
table("no vocals", music._no_vocals, [("джингл без вокала", True), ("эмбиент без слов", True),
      ("instrumental lofi beat", True), ("песня про кота", False), ("рэп без мата", False)])
table("own form", music._own_form, [("ровно три куплета, без припева", True), ("4 lines only", True),
      ("песня про кота", False), ("весёлая песня на минуту", False)])
table("camera", lambda d: d != video._lock_framing_unless_requested(d) and False or
      video._lock_framing_unless_requested(d) == d,
      [("slow zoom in on the cat's face", True), ("the camera pans left across the street", True),
       ("a cat jumps on the table", False), ("a woman waves and smiles", False)])
table("side question", lambda t: not steer.eligible(t, False, True),
      [("а сколько сейчас времени?", True), ("который час", True), ("сделай короче", False)])
table("complaint", turn_audit.is_complaint, [("не то", True), ("опять не так", True), ("где файл?", True),
      ("я же просил три картинки", True), ("that's wrong", True), ("нет, спасибо", False),
      ("спасибо, отлично", False), ("а теперь нарисуй собаку", False)])
r = lambda t: intent.read(None, t)
table("collage", lambda t: r(t)["collage"], [("найди все мосты и собери коллаж", True),
      ("сделай сетку из этих фото", True), ("найди фото мостов", False)])
table("dedupe", lambda t: r(t)["dedupe"], [("убери дубли и собери коллаж", True),
      ("оставь по одному кадру из повторов", True), ("собери коллаж из мостов", False)])
table("add only", draw_agent._additive_only, [("добавь рыжего кота", True), ("верни коробку обратно", True),
      ("замени кота на собаку", False), ("убери кота", False), ("добавь кота вместо собаки", False)])
table("lettering", lambda t: L.is_lettering_removal(t), [("надпись", True), ("the watermark", True),
      ("logo in the corner", True), ("the cat", False), ("лужа", False)])
print("ALL OK" if not bad else f"{bad} failed")
sys.exit(1 if bad else 0)
