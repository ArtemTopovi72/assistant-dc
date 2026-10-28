import threading
from types import SimpleNamespace

import pytest

import comfy_client


def test_exclusive_claim_waits_for_speech(monkeypatch):
    claimed = threading.Event()
    monkeypatch.setattr(comfy_client, '_claim_card_locked', lambda *a, **k: claimed.set())
    with comfy_client.speech_session(timeout=2):
        worker = threading.Thread(target=comfy_client._claim_card, args=('video',))
        worker.start()
        assert not claimed.wait(.1)
    worker.join(timeout=2)
    assert claimed.is_set()
    comfy_client._SPEECH_FREE.set()


def test_speech_waits_for_existing_exclusive():
    entered = threading.Event()
    def speech():
        with comfy_client.speech_session(timeout=2):
            entered.set()
    comfy_client._SPEECH_FREE.clear()
    try:
        worker = threading.Thread(target=speech)
        worker.start()
        assert not entered.wait(.1)
    finally:
        comfy_client._SPEECH_FREE.set()
    worker.join(timeout=2)
    assert entered.is_set()


def test_waiting_speech_can_be_cancelled():
    comfy_client._SPEECH_FREE.clear()
    try:
        with pytest.raises(RuntimeError, match='Cancelled'):
            with comfy_client.speech_session(timeout=2, ctx=SimpleNamespace(is_cancelled=lambda: True)):
                pass
    finally:
        comfy_client._SPEECH_FREE.set()


def test_claim_without_unloading_blocks_speech_until_release():
    ctx = SimpleNamespace(deps=SimpleNamespace(COMFY_EVICT_LLM=False))
    comfy_client._claim_card('video', ctx=ctx)
    try:
        with pytest.raises(TimeoutError):
            with comfy_client.speech_session(timeout=.1):
                pass
    finally:
        comfy_client._release_card(ctx)
    with comfy_client.speech_session(timeout=1):
        pass
