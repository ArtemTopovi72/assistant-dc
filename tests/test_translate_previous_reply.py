"""«Translate to Russian» with no text of its own translates the reply above.

A forwarded voice retold in English: «Translate to Russian» came
back as «Переведи на русский» -- the request itself translated. The note that
points the model at its previous reply rides only a translate request that
brings no text; «переведи: <текст>» keeps translating its own text.
"""
import os
import sys

os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests  # noqa: E402,F401
import types  # noqa: E402
import intent  # noqa: E402
from graph_compose import translate_previous_note  # noqa: E402

HIST = [{"role": "user", "content": "write about cats"},
        {"role": "assistant", "content": "Cats are independent animals."}]


def _reads(translate: bool, has_text: bool):
    """The model's reads: the message is (not) a translate ask, and (does not) carry its own text."""
    return types.SimpleNamespace(intent_read=lambda ctx, text, *a, **k: dict(intent.FALLBACK, translate=translate),
                                 yes_no=lambda q, t, default=False, wait=True: has_text)


def test_bare_request_points_at_the_previous_reply():
    assert "previous reply" in translate_previous_note("Translate to Russian", HIST,
                                                       ctx=_reads(translate=True, has_text=False))


def test_request_with_its_own_text_is_left_alone():
    assert translate_previous_note("переведи на английский: кошка спит", HIST,
                                   ctx=_reads(translate=True, has_text=True)) == ""


def test_no_reply_above_or_not_a_translation():
    assert translate_previous_note("Translate to Russian", [], ctx=_reads(translate=True, has_text=False)) == ""
    assert translate_previous_note("how are you", HIST, ctx=_reads(translate=False, has_text=False)) == ""
