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


# ── the pin does not stick for good (10-03) ──────────────────────────────────
class _S:
    lang_pin = ""
    lang_pin_ts = 0.0
    turn_lang = ""


def test_the_pin_holds_while_in_use_and_ends_after_quiet():
    import tg_tasks as T
    s = _S()
    T._lang_pin_after(s, "en", True, "ru", now=1000.0)
    assert s.lang_pin == "en"
    T._lang_pin_after(s, "", False, "ru", now=1000.0 + 3600)          # an hour on: still English
    assert s.lang_pin == "en"
    T._lang_pin_after(s, "", False, "ru", now=1000.0 + 3600 + T.LANG_PIN_TTL + 1)
    assert s.lang_pin == ""                                           # hours of quiet, Russian: over


def test_an_old_pin_with_no_time_ends_on_a_russian_line():
    import tg_tasks as T
    s = _S(); s.lang_pin = "en"                                      # saved before the timestamp existed
    T._lang_pin_after(s, "", False, "en", now=10.0 ** 9)
    assert s.lang_pin == "en"                                        # English writing keeps it
    s.lang_pin_ts = 0.0
    T._lang_pin_after(s, "", False, "ru", now=10.0 ** 9)
    assert s.lang_pin == ""


def test_choosing_a_language_or_clear_drops_the_pin():
    src = {f: open(os.path.join(ROOT, "bot", f), encoding="utf-8").read()
           for f in ("tg_callbacks.py", "tg_commands.py", "tg_sessions.py")}
    assert 'sess.lang_pin = sess.turn_lang = ""' in src["tg_callbacks.py"]
    assert 'sess.lang_pin = sess.turn_lang = ""' in src["tg_commands.py"]
    assert 'self.lang_pin = self.turn_lang = ""' in src["tg_sessions.py"]


def test_clear_drops_the_pin_for_real():
    import tg_sessions
    s = tg_sessions._Session(7, {"lang_pin": "en", "turn_lang": "en", "lang_pin_ts": 5})
    assert s.lang_pin == "en" and s.to_dict()["lang_pin_ts"] == 5
    s.clear_context()
    assert s.lang_pin == "" and s.turn_lang == ""
