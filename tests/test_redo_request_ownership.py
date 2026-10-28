"""A contradictory redo label cannot replace a self-contained new request."""
from types import SimpleNamespace

from fakes import reads
from graph_personality import _previous_request_for_redo


def test_new_video_after_shopping_is_not_replaced_by_the_old_search():
    request = ('Make a new video based only on text: one red cube rotates on a white table. '
               'Five seconds, no people or speech.')
    previous = 'Find a USB-C to USB-C cable on Ozon under 500 rubles'
    ctx = SimpleNamespace(**reads(
        read=lambda text: {'redo': 'same' if text == request else '', 'wants': ['generate_video']},
        yes=lambda question, text: False))
    assert _previous_request_for_redo(request, [{'role': 'user', 'content': previous}], ctx=ctx) == ''


def test_a_confirmed_bare_repeat_still_uses_the_previous_request():
    previous = 'Draw a blue boat'
    ctx = SimpleNamespace(**reads(
        read=lambda text: {'redo': 'same' if text == 'again' else ''},
        yes=lambda question, text: text == 'again'))
    assert _previous_request_for_redo('again', [{'role': 'user', 'content': previous}], ctx=ctx) == previous
