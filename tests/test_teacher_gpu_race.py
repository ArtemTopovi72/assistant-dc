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


def test_whisper_paths_wait_for_exclusive_render(monkeypatch):
    import audio
    calls = []
    monkeypatch.setattr(audio, '_whisper_transcribe_admitted', lambda *a: calls.append('asr') or 'ok')
    monkeypatch.setattr(audio, '_detect_media_language_admitted', lambda *a: calls.append('lang') or 'ru')
    comfy_client._SPEECH_FREE.clear()
    try:
        worker = threading.Thread(target=lambda: (audio._whisper_transcribe(None, 'file'),
                                                   audio.detect_media_language(None, 'file')))
        worker.start()
        worker.join(.2)
        assert calls == []
    finally:
        comfy_client._SPEECH_FREE.set()
    worker.join(2)
    assert calls == ['asr', 'lang']


def test_nested_speech_admission_is_reentrant(monkeypatch):
    import gpu_admission
    import contextlib
    entries = []
    @contextlib.contextmanager
    def admit(**kwargs):
        entries.append(1)
        yield
    monkeypatch.setattr(gpu_admission, 'admit', admit)
    with comfy_client.speech_session():
        with comfy_client.speech_session():
            pass
    assert entries == [1]
