"""Live check of intent.redo / asks_for_file and the new-picture read (wants)
on the house model. Phrases from the word lists they replaced
(graph_finalize._REDO_REQUEST_RE, _ASKS_FOR_FILE_RE, graph_compose._FRESH_DRAW_RE).

    venv/Scripts/python bench/intent_redo_file_live.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "agent"), os.path.join(ROOT, "core")]
sys.stdout.reconfigure(encoding="utf-8")
os.environ.pop("F5_TEST_RUN", None)
import intent
from graph_fastpath import _IMAGE_ACTION_TOOLS

bad = 0
def check(name, ok, r):
    global bad
    bad += not ok
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  {r}"))

for t, want in [("заново", "same"), ("сделай ещё раз", "same"), ("again please", "same"),
                ("пересобери коллаж, удалив дубликаты", "changed"), ("найди мост", ""),
                ("а теперь коллаж", ""), ("привет", "")]:
    r = intent.read(None, t)
    check(f"redo {t!r} -> {want!r}", r["redo"] == want, r["redo"])
for t, want in [("сделай по этому презентацию", True), ("запакуй всё обратно в архив", True),
                ("пришли отчёт в pdf", True), ("что думаешь?", False), ("нарисуй кота", False)]:
    r = intent.read(None, t)
    check(f"file {t!r} -> {want}", r["asks_for_file"] == want, r["asks_for_file"])
for t, new in [("нарисуй набережную Сочи в такую погоду", True), ("нарисуй серого кота на подоконнике", True),
               ("сгенерируй логотип пекарни", True), ("нарисуй ей шляпу", False),
               ("нарисуй на этой картинке солнце", False), ("сделай слона розовым", False),
               ("нарисуй такую же, но ночью", False), ("дорисуй фон", False)]:
    r = intent.read(None, t, "", "a photo")
    w = set(r["wants"])
    got = "generate_image" in w and not (w & _IMAGE_ACTION_TOOLS - {"generate_image"})
    check(f"new picture {t!r} -> {new}", got == new, r["wants"])
print(f"{'ALL OK' if not bad else str(bad) + ' failed'}")
sys.exit(1 if bad else 0)
