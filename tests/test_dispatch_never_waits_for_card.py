"""The receiving thread never waits for a render's card (2026-10-13): «send it as a
file» was asked of the model on the dispatch path, the model waited for a 25-minute
video chain, and every chat's message sat unqueued -- not even «you're #2 in queue».

Run: venv/Scripts/python.exe -m pytest -q tests/test_dispatch_never_waits_for_card.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT] + [os.path.join(ROOT, d) for d in ("agent", "bot", "core", "imaging")]


def test_the_as_file_check_answers_at_once_while_a_render_holds_the_card(monkeypatch):
    import comfy_client
    import intent
    import llm
    import tg_dispatch
    monkeypatch.setenv("INTENT_LIVE", "1")
    monkeypatch.setattr(comfy_client, "card_in_use", lambda: True)

    asked = []
    monkeypatch.setattr(llm, "call_llm_simple", lambda *a, **k: asked.append(a) or "yes")
    assert tg_dispatch._asks_as_file("пришли файлом ту картинку") is False and not asked


def test_a_flow_check_still_waits_for_the_model(monkeypatch):
    import comfy_client
    import intent
    import llm
    monkeypatch.setenv("INTENT_LIVE", "1")
    monkeypatch.setattr(comfy_client, "card_in_use", lambda: True)
    monkeypatch.setattr(llm, "call_llm_simple", lambda *a, **k: "yes")
    assert intent.ask_yes("Is {text} a cat?", "a tabby cat on a sofa") is True
