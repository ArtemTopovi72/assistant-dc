"""The face check of a mask crop reads its own crop, not another job's.

Bug: image_masks saved the crop to one fixed scratch name per kind
(_pair_facechk.png / _rm_facechk.png) and ran face_box on it. With several bot
workers editing at once, one job could read the other's crop, and a face box
from the wrong picture let a mask take in a face.
"""
import os
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")

from PIL import Image  # noqa: E402

import image_masks as M  # noqa: E402


def test_concurrent_face_checks_each_read_their_own_crop(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "OUTPUT_DIR", tmp_path)
    both_saved = threading.Barrier(2)

    class Idm:
        @staticmethod
        def face_box(path, pad=0.0):
            both_saved.wait(5)              # the other job has written its crop too
            return Image.open(path).getpixel((0, 0))

    seen = {}

    def job(colour):
        seen[colour] = M._face_box_of(Image.new("RGB", (8, 8), colour), "_pair_facechk", Idm)

    a = threading.Thread(target=job, args=((255, 0, 0),))
    b = threading.Thread(target=job, args=((0, 0, 255),))
    a.start(); b.start(); a.join(10); b.join(10)
    assert seen == {(255, 0, 0): (255, 0, 0), (0, 0, 255): (0, 0, 255)}
    left = list((tmp_path / "generated" / "_intermediate").glob("*facechk*"))
    assert left == []                       # scratch files are removed
