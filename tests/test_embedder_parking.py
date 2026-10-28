import llama_backend as b


def test_embedder_does_not_restart_while_parked(monkeypatch):
    monkeypatch.setattr(b, '_under_tests', lambda: False)
    monkeypatch.setattr(b, '_EMBED_PAUSED', True)
    monkeypatch.setattr(b, '_EMBED_CHECKED', [0.0])
    monkeypatch.setattr(b.subprocess, 'Popen', lambda *a, **k: (_ for _ in ()).throw(AssertionError('started')))
    b.ensure_embedder(min_interval=0)
    assert b._EMBED_CHECKED == [0.0]
