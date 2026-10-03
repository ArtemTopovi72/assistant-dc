"""A requested label is drawn ONCE (live 10-03: 'MSVCP140.dll' came out twice —
the planner had it in an element's desc, and ensure_text_elements added a second
carrier on top)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("core", "imaging", "agent"):
    sys.path.insert(0, os.path.join(ROOT, d))
import ideogram as I
import intent
import pytest


@pytest.fixture(autouse=True)
def _yes(monkeypatch):
    monkeypatch.setattr(intent, "ask_yes", lambda *a, **k: True)

PROMPT = "old DLL files labeled 'MSVCP140.dll' replaced by light, a sign reading 'system fix'"


def _carriers(layout, s):
    return [e for e in layout["elements"] if str(e.get("text") or "").lower() == s.lower()]


def test_label_in_desc_is_not_added_again():
    lay = {"elements": [
        {"desc": "old rusty DLL files labeled 'MSVCP140.dll'", "x": .1, "y": .1, "w": .5, "h": .4},
        {"desc": "glowing light streams", "x": .5, "y": .5, "w": .4, "h": .4}]}
    out = I.ensure_text_elements(lay, PROMPT)
    assert len(_carriers(out, "MSVCP140.dll")) == 1, out["elements"]


def test_two_labels_do_not_share_one_surface():
    lay = {"elements": [
        {"desc": "old rusty digital file fragments", "x": .1, "y": .1, "w": .5, "h": .4},
        {"desc": "a metal sign plate", "x": .6, "y": .6, "w": .3, "h": .3}]}
    out = I.ensure_text_elements(lay, PROMPT)
    descs = [e["desc"] for e in out["elements"] if e.get("text")]
    assert len(descs) == len(set(descs)), descs
