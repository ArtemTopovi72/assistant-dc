"""The receiving thread never waits for a render's card (2026-10-13): «send it as a
file» was asked of the model on the dispatch path, the model waited for a 25-minute
video chain, and every chat's message sat unqueued -- not even «you're #2 in queue».

Run: venv/Scripts/python.exe -m pytest -q tests/test_dispatch_never_waits_for_card.py
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT] + [os.path.join(ROOT, d) for d in ("agent", "bot", "core", "imaging")]


def _busy_card(answer, asked=None):
    """A live read with a render holding the card; the model would say `answer`."""
    return types.SimpleNamespace(deps=types.SimpleNamespace(
        live=True, card_in_use=lambda: True,
        call_llm_simple=lambda *a, **k: (asked.append(a) if asked is not None else None) or answer))


def test_the_as_file_check_does_not_wait_for_the_card():
    import tg_dispatch
    seen = []
    ctx = types.SimpleNamespace(yes_no=lambda q, t, default=False, wait=True: seen.append(wait) or False)
    assert tg_dispatch._asks_as_file("пришли файлом ту картинку", ctx=ctx) is False
    assert seen == [False]


def test_a_no_wait_check_answers_at_once_while_a_render_holds_the_card():
    import intent
    asked = []
    out = intent.ask_yes("Is {text} a file request (nowait)?", "пришли файлом ту картинку", wait=False,
                         ctx=_busy_card("yes", asked))
    assert out is False and not asked


def test_a_flow_check_still_waits_for_the_model():
    import intent
    assert intent.ask_yes("Is {text} a cat?", "a tabby cat on a sofa", ctx=_busy_card("yes")) is True
