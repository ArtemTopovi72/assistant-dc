"""JIT facts: drop only noise, never lose a fact because selection broke."""
import os, sys, importlib.util
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import memory_select as memory_source


def _fresh_selector():
    spec=importlib.util.spec_from_file_location('isolated_memory_select',memory_source.__file__)
    selector=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    return selector


M=_fresh_selector()

VOCAB = ["кот", "собак", "имя", "артем", "музык", "рок", "город", "москв", "аллерг", "орех"]


def _fake(texts):
    return [M._unit([float(w in t.lower()) + 1e-3 for w in VOCAB]) for t in texts]


FACTS = ["У меня кот Барсик", "Моё имя Иван", "Люблю рок-музыку", "Живу в городе Москва",
         "Аллергия на орехи", "Собака соседа лает", "Кот любит рыбу", "Рок-концерт в субботу",
         "Москва — мой родной город", "Ненавижу орехи в шоколаде"]


def setup_function(_):
    global M
    M=_fresh_selector()


def test_small_store_goes_whole():
    def never(t):
        raise AssertionError("no call")
    assert M.select(FACTS[:5], "кот", embed=never) == FACTS[:5]


def test_relevant_plus_newest_in_original_order():
    got = M.select(FACTS, "что с моим котом?", k=2, newest=1, floor=0.3, embed=_fake)
    assert "У меня кот Барсик" in got and "Кот любит рыбу" in got
    assert got[-1] == FACTS[-1]                          # newest kept, order kept
    assert "Живу в городе Москва" not in got
    assert got == [f for f in FACTS if f in got]


def test_no_embedder_keeps_everything():
    assert M.select(FACTS, "кот", embed=lambda t: None) == FACTS


def test_embedder_crash_keeps_everything():
    def boom(t): raise RuntimeError("lm studio down")
    assert M.select(FACTS, "кот", embed=boom) == FACTS


def test_ctx_facts_text_uses_selection():
    import models, types
    ctx = models.Context.__new__(models.Context)
    ctx.deps = types.SimpleNamespace(jit_facts=lambda: True, embed=_fake)
    import threading
    ctx.memory_lock = threading.Lock()
    ctx.pinned_facts = [{"text": f} for f in FACTS]
    full = ctx.facts_text()
    assert full.count("\n") == len(FACTS) - 1             # no query = everything
    sel = ctx.facts_text("аллергия на орехи есть?")
    assert "Аллергия на орехи" in sel and "Барсик" not in sel


class _Embedder:
    base_url='http://offline'
    model='model-a'

    def __init__(self):
        self.phase=0
        self.dimension=2

    def embed(self,texts):
        hit=[1.,0.] if self.dimension==2 else [1.,0.,100.]
        miss=[0.,1.] if self.dimension==2 else [0.,1.,0.]
        return [hit if text=='question' or text==('fact-a' if self.phase==0 else 'fact-b') else miss for text in texts]


def _select(embed=None):
    return M.select(['fact-a','fact-b'],'question',k=1,newest=0,floor=0.3,keep_all_upto=0,embed=embed)


@pytest.mark.parametrize('changed',['model','base_url'])
@pytest.mark.parametrize('injected',[False,True])
def test_fact_cache_is_scoped_to_embedding_model_and_endpoint(changed,injected):
    source=_Embedder();M._EMBEDDER=source
    embed=source.embed if injected else None
    assert _select(embed)==['fact-a']
    source.phase=1;setattr(source,changed,'changed')
    assert _select(embed)==['fact-b']


def test_dimension_change_keeps_all_facts_then_recovers():
    source=_Embedder();M._EMBEDDER=source
    assert _select()==['fact-a']
    source.dimension=3
    assert _select()==['fact-a','fact-b']
    assert _select()==['fact-a']


@pytest.mark.parametrize('bad',[[float('nan'),0.],[float('inf'),0.],[0.,0.],[1e308,1e308],[None,0.],None])
@pytest.mark.parametrize('where',['query','documents'])
def test_invalid_vectors_keep_all_facts_and_do_not_poison_recovery(bad,where):
    source=_Embedder();M._EMBEDDER=source
    original=source.embed
    def invalid(texts):
        if (texts==['question'])==(where=='query'):
            return [bad] if where=='query' else [bad,[0.,1.]]
        return original(texts)
    source.embed=invalid
    assert _select()==['fact-a','fact-b']
    source.embed=original
    assert _select()==['fact-a']


def test_partial_document_response_is_not_cached():
    source=_Embedder();M._EMBEDDER=source
    original=source.embed
    source.embed=lambda texts:original(texts) if texts==['question'] else [[0.,1.]]
    assert _select()==['fact-a','fact-b']
    source.embed=original
    assert _select()==['fact-a']


def test_injected_embeddings_are_compared_by_cosine_not_raw_magnitude():
    def embed(texts):
        return [[100.,100.] if text=='fact-b' else [1.,0.] for text in texts]
    assert _select(embed)==['fact-a']


@pytest.mark.parametrize('bad',[[float('nan'),0.],[0.,0.],None])
def test_corrupt_cached_vectors_keep_all_facts_then_allow_recovery(bad):
    M._EMBEDDER=_Embedder()
    assert _select()==['fact-a']
    M._CACHE[next(iter(M._CACHE))]=bad
    assert _select()==['fact-a','fact-b']
    assert _select()==['fact-a']


def test_different_injected_embedding_functions_do_not_share_fact_vectors():
    def first(texts):
        return [[1.,0.] if text in ('question','fact-a') else [0.,1.] for text in texts]
    def second(texts):
        return [[1.,0.] if text in ('question','fact-b') else [0.,1.] for text in texts]
    assert _select(first)==['fact-a']
    assert _select(second)==['fact-b']


def test_multiple_query_vectors_keep_all_facts():
    source=_Embedder();M._EMBEDDER=source
    original=source.embed
    source.embed=lambda texts:[[1.,0.],[1.,0.]] if texts==['question'] else original(texts)
    assert _select()==['fact-a','fact-b']
