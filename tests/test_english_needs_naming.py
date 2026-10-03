"""A Russian chat switches to English only when the user names English.

Live 10-03: «Кибернетический Ленин хуяриться на кулаках с чертями в аду…»
was answered in English, and so was everything after it -- the model's read
said reply_language/language_mode "en" for a message that asks nothing of the
kind, and the pin stuck. The read now counts only when English is named.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")

import graph_language as G  # noqa: E402

LENIN = "Кибернетический Ленин дерётся на кулаках с чертями в аду, глаза сияют красным"


def test_a_false_read_does_not_switch_to_english(monkeypatch):
    monkeypatch.setattr(G, "_read", lambda t: {"reply_language": "en", "language_mode": "en"})
    assert not G.asks_for_english(LENIN)
    assert not G.names_english(LENIN)


def test_a_real_request_still_does(monkeypatch):
    monkeypatch.setattr(G, "_read", lambda t: {"reply_language": "en", "language_mode": "en"})
    for t in ("и то же самое по-английски", "ответь на английском", "answer in English please"):
        assert G.asks_for_english(t), t


def test_the_pin_needs_english_named_too():
    src = open(os.path.join(ROOT, "bot", "tg_tasks.py"), encoding="utf-8").read()
    assert '_mode == "en" and _names_en(task.user_text)' in src
