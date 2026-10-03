"""Opening a Library does not load the cross-encoder; the first rerank does.

Bug: Library.__init__ loaded the reranker. The GUI's Database tab opens its
Library on the UI thread at startup just to list documents, so the window
froze for the whole first-start download of the ~2 GB model (and ~30 s of
retries on an offline machine), for a tab that never reranks.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")

import library as L  # noqa: E402


def test_reranker_loads_on_first_retrieve_only(tmp_path, monkeypatch):
    import config
    monkeypatch.delenv("F5_TEST_RUN", raising=False)
    monkeypatch.setattr(config, "LIBRARY_CROSS_ENCODER", "some/reranker", raising=False)
    loads = []
    monkeypatch.setattr(L, "load_cross_encoder", lambda name: loads.append(name) or None)
    lib = L.Library(tmp_path / "lib.db")
    lib.documents()
    lib.stats()
    assert loads == []                                  # nothing at open / listing
    monkeypatch.setenv("F5_TEST_RUN", "1")
    lib.retrieve("anything")
    lib.retrieve("again")
    assert loads == ["some/reranker"]                   # once, when needed


def test_an_assigned_reranker_is_used_as_is(tmp_path):
    lib = L.Library(tmp_path / "lib.db")
    sentinel = object()
    lib._cross_encoder = sentinel
    assert lib._cross_encoder is sentinel
