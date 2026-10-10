"""«Translate to Russian» with no text of its own translates the reply above.

Live 10-09 (a forwarded voice retold in English): «Translate to Russian» came
back as «Переведи на русский» -- the request itself translated. The note that
points the model at its previous reply rides only a translate request that
brings no text; «переведи: <текст>» keeps translating its own text.
"""
import os
import sys

os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests  # noqa: E402,F401
import intent  # noqa: E402
from graph_compose import translate_previous_note  # noqa: E402

HIST = [{"role": "user", "content": "write about cats"},
        {"role": "assistant", "content": "Cats are independent animals."}]


def _stub(monkeypatch, translate: bool, has_text: bool):
    real = intent.read
    monkeypatch.setattr(intent, "read", lambda ctx, text, *a, **k: dict(real(ctx, text), translate=translate))
    monkeypatch.setattr(intent, "YES_STUB", lambda q, t: has_text)


def test_bare_request_points_at_the_previous_reply(monkeypatch):
    _stub(monkeypatch, translate=True, has_text=False)
    assert "previous reply" in translate_previous_note("Translate to Russian", HIST)


def test_request_with_its_own_text_is_left_alone(monkeypatch):
    _stub(monkeypatch, translate=True, has_text=True)
    assert translate_previous_note("переведи на английский: кошка спит", HIST) == ""


def test_no_reply_above_or_not_a_translation(monkeypatch):
    _stub(monkeypatch, translate=True, has_text=False)
    assert translate_previous_note("Translate to Russian", []) == ""
    _stub(monkeypatch, translate=False, has_text=False)
    assert translate_previous_note("how are you", HIST) == ""
