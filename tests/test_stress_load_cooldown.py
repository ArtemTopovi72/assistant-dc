"""voice/stress.py does not retry a failed model load on every word.

Bug: offline (or with the hub down), each English word re-ran
AutoTokenizer.from_pretrained -- a hub download with ~25 s of retries -- so a
voice reply with five English words stalled for minutes, every reply. RUAccent
and silero-stress had the same per-call retry.
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")

import stress  # noqa: E402


def _broken_module(name, attr, counter):
    m = types.ModuleType(name)

    class _Cls:
        @staticmethod
        def from_pretrained(*a, **k):
            counter.append(name)
            raise OSError("hub unreachable")
    for a in attr:
        setattr(m, a, _Cls)
    return m


def test_english_loader_tried_once(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "transformers",
                        _broken_module("transformers", ["AutoTokenizer", "T5ForConditionalGeneration"], calls))
    acc = stress.BilingualAccentor(enable_english=True)
    text = "hello wonderful world, testing english words"
    assert acc._accent_english(text) == text           # unchanged, not crashed
    assert acc._accent_english(text) == text
    assert len(calls) == 1, calls
    # after the cooldown it is tried again
    acc._en_failed_at -= stress.LOAD_RETRY_S + 1
    acc._accent_english("another sentence")
    assert len(calls) == 2


def test_russian_loader_tried_once(monkeypatch):
    calls = []
    pkg = types.ModuleType("ruaccent")
    sub = types.ModuleType("ruaccent.ruaccent")

    class RUAccent:
        def load(self, **k):
            calls.append(1)
            raise OSError("hub unreachable")
    sub.RUAccent = RUAccent
    monkeypatch.setitem(sys.modules, "ruaccent", pkg)
    monkeypatch.setitem(sys.modules, "ruaccent.ruaccent", sub)
    acc = stress.BilingualAccentor(enable_english=False)
    for _ in range(3):
        assert acc("привет мир") == "привет мир"
    assert len(calls) == 1
