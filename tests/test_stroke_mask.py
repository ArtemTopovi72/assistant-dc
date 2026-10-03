"""Lettering mask follows the letters, not the OCR box."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image, ImageDraw
import image_lettering_remove as L


def _img(path):
    im = Image.new("RGB", (200, 120), (230, 200, 160))            # beige floor
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 80, 120), fill=(20, 15, 15))               # dark fur on the left
    d.rectangle((60, 50, 150, 56), fill=(255, 255, 255))          # a white "stroke" across both
    d.rectangle((100, 40, 106, 70), fill=(255, 255, 255))
    im.save(path)


def test_mask_is_strokes_not_box():
    d = tempfile.mkdtemp(); p = os.path.join(d, "s.png"); _img(p)
    m = np.asarray(L.stroke_mask(p, [(55, 35, 155, 75)], pad=4, halo=2)) > 0
    box = (155 - 55 + 8) * (75 - 35 + 8)
    assert m[53, 120] and m[55, 70] and m[60, 103]                 # the ink is in
    assert not m[40, 65] and not m[70, 130]                        # fur and floor are not
    assert m.sum() < 0.5 * box


def test_inseparable_falls_back_to_box():
    d = tempfile.mkdtemp(); p = os.path.join(d, "f.png")
    Image.new("RGB", (100, 100), (128, 128, 128)).save(p)
    m = np.asarray(L.stroke_mask(p, [(20, 20, 60, 60)], pad=2, halo=0)) > 0
    assert m[40, 40] and m[20, 20]


if __name__ == "__main__":
    test_mask_is_strokes_not_box(); test_inseparable_falls_back_to_box(); print("2/2 ok")
