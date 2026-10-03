"""Live check of image_router.edit_plan (the edit category + multi-op steps) on
the house model. The phrases are the ones the 17-regex table was tested on
(test_text_edit_routing, test_colour_convert_routing, test_router_sharpen_extend,
test_text_add_verified, test_removal_targets_region, test_image_pure) plus the
live misses each regex was tuned against.

    venv/Scripts/python bench/edit_intent_live.py
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT] + [os.path.join(ROOT, d) for d in ("agent", "core", "imaging", "bot", "media")]
sys.stdout.reconfigure(encoding="utf-8")
os.environ.pop("F5_TEST_RUN", None)
import image_router as R

ANY = None
CASES = [
    # lettering
    ("исправь надписи на корректные", {"text_edit"}), ("надпись неправильная, поправь", {"text_edit"}),
    ("на вывеске опечатка", {"text_edit"}), ("поменяй текст на вывеске", {"text_edit"}),
    ("исправь текст", {"text_edit"}), ("fix the lettering on the sign", {"text_edit"}),
    ("the text is garbled, correct it", {"text_edit"}), ("there's a typo in the caption", {"text_edit"}),
    ("change the text on the sign to OPEN", {"text_edit"}), ("the sign says the wrong thing", {"text_edit"}),
    ("fix the spelling", {"text_edit"}), ("убери надпись", {"text_edit", "object_remove"}),
    ("добавь надпись СТОП", {"text_add", "text_edit"}),
    ("добавь надпись «УРОЖАЙ 2026» в небо", {"text_add"}), ('add a title "SALE 50%" at the top', {"text_add"}),
    ("add the text SALE on the window", {"text_add"}), ("исправь надпись на вывеске", {"text_edit"}),
    # ordinary edits
    ("убери фон", {"background_remove"}), ("удали человека справа", {"person_remove"}),
    ("сделай рубашку зелёной", {"clothing_edit"}), ("добавь шляпу", {"object_insert"}),
    ("add a hat to the dog", {"object_insert"}), ("добавь кота", {"object_insert"}),
    ("сделай в стиле аниме", {"style_transfer"}), ("make it anime style", {"style_transfer"}),
    ("восстанови лицо", {"restore"}), ("remove the trash can", {"object_remove"}),
    ("change the background to a beach", {"background_replace"}),
    ("make the background blurry", {"background_replace"}), ("make the eyes blue", {"face_edit"}),
    ("make her look happier", {"subject_edit", "face_edit"}),
    ("add Russian prison-style tattoos on both forearms", {"subject_edit", "object_insert"}),
    ("Restore the cat-shaped object so that its entire body is visible", {"subject_edit", "object_insert"}),
    ("restore the old photo", {"restore"}), ("repair the scratches", {"restore"}),
    ("fix the damaged corner", {"restore"}), ("upscale and restore the face", {"restore"}),
    # hair colour = the face, never a whole-frame redraw
    ("пусть мужик будет блондин", {"face_edit"}), ("сделай его блондином", {"face_edit"}),
    ("сделай её брюнеткой", {"face_edit"}), ("make the man blond", {"face_edit"}),
    ("turn him into a redhead", {"face_edit"}), ("make him bald", {"face_edit"}),
    ("добавь рыжего кота", {"object_insert"}), ("нарисуй блондинку", {"object_insert"}),
    # colour conversion vs a colour
    ("make this picture black and white", {"colour_convert"}), ("convert it to grayscale", {"colour_convert"}),
    ("b&w version", {"colour_convert"}), ("desaturate the photo", {"colour_convert"}),
    ("sepia tone it", {"colour_convert"}), ("сделай эту картинку чёрно-белой", {"colour_convert"}),
    ("переведи в чб", {"colour_convert"}), ("обесцветь картинку", {"colour_convert"}),
    ("turn it black and white", {"colour_convert"}),
    ("make the car black", {"subject_edit", "product_edit"}), ("покрась стену в белый", {"subject_edit"}),
    ("remove the white background", {"background_remove"}),
    ("make it look like an oil painting", {"style_transfer"}), ("upscale it", {"upscale"}),
    # sharpen / extend / transform
    ("make the photo sharper", {"upscale"}), ("сделай фото чётче", {"upscale"}),
    ("extend the picture to the right", {"outpaint"}), ("make it square", {"outpaint"}),
    ("расширь картинку вправо", {"outpaint"}), ("сделай квадратной", {"outpaint"}),
    ("увеличь разрешение", {"upscale"}), ("расширь кадр", {"outpaint"}),
    ("rotate the image 90 degrees", {"transform"}), ("flip it horizontally", {"transform"}),
    ("crop to the face", {"transform"}), ("поверни картинку на 90 градусов влево", {"transform"}),
    ("turn his head to the left", {"subject_edit", "face_edit"}),
    ("mirror the text on the sign", {"text_edit", "subject_edit"}),
    ("mirror the entire image horizontally", {"transform"}),
    # one edit vs several
    ("remove the background, make it white", {"background_remove"}),
    ("remove the background and make it black", {"background_remove"}),
    ("remove the car and change the sky to sunset", {"multi_op"}),
    ("remove the cat and the dog and add a tree", {"multi_op"}),
]
bad, ts = 0, []
for text, ok in CASES:
    t = time.time(); p = R.edit_plan(text); ts.append(time.time() - t)
    good = p["kind"] in ok
    bad += not good
    print(("PASS " if good else "FAIL ") + f"{text!r} -> {p['kind']} {p['steps'] or ''} (want {sorted(ok)})")
ts.sort()
print(f"{len(CASES) - bad}/{len(CASES)}  median {ts[len(ts)//2]:.2f}s")
sys.exit(1 if bad else 0)
