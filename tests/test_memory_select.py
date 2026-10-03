"""JIT facts: drop only noise, never lose a fact because selection broke."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import memory_select as M

VOCAB = ["кот", "собак", "имя", "артем", "музык", "рок", "город", "москв", "аллерг", "орех"]


def _fake(texts):
    return [M._unit([float(w in t.lower()) + 1e-3 for w in VOCAB]) for t in texts]


FACTS = ["У меня кот Барсик", "Моё имя Иван", "Люблю рок-музыку", "Живу в городе Москва",
         "Аллергия на орехи", "Собака соседа лает", "Кот любит рыбу", "Рок-концерт в субботу",
         "Москва — мой родной город", "Ненавижу орехи в шоколаде"]


def setup_function(_):
    M._CACHE.clear()


def test_small_store_goes_whole(monkeypatch):
    monkeypatch.setattr(M, "_embed", lambda t: (_ for _ in ()).throw(AssertionError("no call")))
    assert M.select(FACTS[:5], "кот") == FACTS[:5]


def test_relevant_plus_newest_in_original_order(monkeypatch):
    monkeypatch.setattr(M, "_embed", _fake)
    got = M.select(FACTS, "что с моим котом?", k=2, newest=1, floor=0.3)
    assert "У меня кот Барсик" in got and "Кот любит рыбу" in got
    assert got[-1] == FACTS[-1]                          # newest kept, order kept
    assert "Живу в городе Москва" not in got
    assert got == [f for f in FACTS if f in got]


def test_no_embedder_keeps_everything(monkeypatch):
    monkeypatch.setattr(M, "_embed", lambda t: None)
    assert M.select(FACTS, "кот") == FACTS


def test_embedder_crash_keeps_everything(monkeypatch):
    def boom(t): raise RuntimeError("lm studio down")
    monkeypatch.setattr(M, "_embed", boom)
    assert M.select(FACTS, "кот") == FACTS


def test_ctx_facts_text_uses_selection(monkeypatch):
    import models
    monkeypatch.setenv("JIT_FACTS", "1")
    monkeypatch.setattr(M, "_embed", _fake)
    ctx = models.Context.__new__(models.Context)
    import threading
    ctx.memory_lock = threading.Lock()
    ctx.pinned_facts = [{"text": f} for f in FACTS]
    full = ctx.facts_text()
    assert full.count("\n") == len(FACTS) - 1             # no query = everything
    sel = ctx.facts_text("аллергия на орехи есть?")
    assert "Аллергия на орехи" in sel and "Барсик" not in sel
