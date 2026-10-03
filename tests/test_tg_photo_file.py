"""📎 Photo as file: a settings toggle that also sends every picture via sendDocument.

Run: venv/Scripts/python.exe -m pytest tests/test_tg_photo_file.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
import tg_bot as T
import tg_sessions as S


def _labels(kb):
    return [b["text"] if isinstance(b, dict) else b for row in kb["keyboard"] for b in row]


def test_toggle_is_in_settings_and_resolves_by_label():
    for lang in ("ru", "en"):
        lab = T._b("photo_file", lang)
        assert lab in _labels(T._settings_kb(False, False, lang))
        assert T._LABEL2KEY.get(lab) == "photo_file"


def test_setting_persists():
    s = S._Session(0, {})
    assert s.photo_file is False
    s.photo_file = True
    assert s.to_dict()["photo_file"] is True
