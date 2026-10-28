"""Dynamic tool retrieval.

On this deployment it is not a token saving, it is a correctness fix: the full
schema set is ~6800 tokens, and a model loaded at 12288 with parallel 2 gives
each request 6144, so a tool-bearing round was rejected outright and the turn
lost. See tests/test_context_overflow.py for the other half of that failure.

The negatives matter more than the positives here. A missing schema is a
capability the model cannot use however well it routes, so retrieval may only
ever narrow from an evidenced match, and no evidence means send everything.
"""
import sys, os, json, importlib.util
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tool_retrieval as retrieval_source
import graph

ALL = graph.TOOL_SCHEMAS


def _fresh_retrieval():
    spec=importlib.util.spec_from_file_location('isolated_tool_retrieval',retrieval_source.__file__)
    retrieval=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retrieval)
    retrieval._embed_on=lambda:False
    return retrieval


T=_fresh_retrieval()


@pytest.fixture(autouse=True)
def isolated_offline_retrieval():
    global T
    T=_fresh_retrieval()


def _names(schemas):
    return sorted(s["function"]["name"] for s in schemas)


# What a message needs is the model's read (agent/intent.py `wants`); these
# stand in for it, so the tests check what retrieval does WITH a read.
WANTS = {
    "нарисуй рыжего кота в скафандре": ["generate_image"],
    "нарисуй кота": ["generate_image"],
    "найди реальное фото Эйфелевой башни": ["find_photo"],
    "добавь ей очки": ["inpaint_image"],
    "у неё пальцы кривые, почини руки": ["fix_hands"],
    "убери мутное пятно слева": ["fix_artifact"],
    "посчитай 15% от 2480": ["calculate"],
    "запомни: мою собаку зовут Байкал": ["remember_fact"],
    "сделай презентацию про историю Рима": ["create_presentation"],
    "сделай короткое видео с волнами": ["generate_video"],
    "какая сейчас погода в Москве": ["weather_forecast"],
    "что нового на этой неделе": ["search"],
    "вставь то, что в буфере обмена": ["read_clipboard"],
    "раздели 987654 на 321 и округли до двух знаков": ["calculate"],
    "найди курс биткоина и посчитай 5% от него": ["search", "calculate"],
}


def _sel(text, **kw):
    kw.setdefault("wants", WANTS.get(text))
    return _names(T.select_tools(text, ALL, **kw))


# --- it picks the right tool ------------------------------------------------

def test_each_intent_reaches_its_tool():
    cases = {
        "нарисуй рыжего кота в скафандре": "generate_image",
        "найди реальное фото Эйфелевой башни": "find_photo",
        "добавь ей очки": "inpaint_image",
        "у неё пальцы кривые, почини руки": "fix_hands",
        "убери мутное пятно слева": "fix_artifact",
        "посчитай 15% от 2480": "calculate",
        "запомни: мою собаку зовут Байкал": "remember_fact",
        "сделай презентацию про историю Рима": "create_presentation",
        "сделай короткое видео с волнами": "generate_video",
        "какая сейчас погода в Москве": "weather_forecast",
        "что нового на этой неделе": "search",
        "вставь то, что в буфере обмена": "read_clipboard",
    }
    for text, want in cases.items():
        assert want in _sel(text), f"{want} dropped for {text!r}: {_sel(text)}"


# --- it actually narrows ----------------------------------------------------

def test_a_matched_turn_is_much_smaller():
    full = len(json.dumps(ALL, ensure_ascii=False))
    got = len(json.dumps(T.select_tools("добавь ей очки", ALL, wants=WANTS["добавь ей очки"]), ensure_ascii=False))
    assert got < full * 0.45, f"{got} vs {full} — barely narrowed"


def test_an_edit_keeps_its_verifier():
    """The escalation guard forces inspect_image; dropping it here would make
    the agent structurally unable to check its own output."""
    assert "inspect_image" in _sel("добавь ей очки")
    assert "inspect_image" in _sel("нарисуй кота")


# --- when it must NOT narrow ------------------------------------------------

def test_no_evidence_means_the_core_set():
    """No cue: the core set, not all 40 schemas. What the words cannot find, the
    STATE pins -- a picture pins the picture tools, a full folder the file
    kit (graph_personality._kit_always); TOOLS_NOMATCH=all restores send-all."""
    import tool_retrieval as R
    assert set(_sel("в каком году высадились на Луну")) == set(R._CORE)
    assert set(_sel("")) == set(R._CORE)


def test_send_all_can_be_restored():
    assert _sel("в каком году высадились на Луну", send_all=True) == _names(ALL)


def test_a_picture_in_the_chat_pins_the_edit_tools():
    import graph_personality as GP
    class C: sandbox = None
    pinned = GP._kit_always(C(), set(), has_image=True)
    assert {"redraw_image", "inpaint_image", "inspect_image"} <= pinned
    assert "redraw_image" not in GP._kit_always(C(), set(), has_image=False)


def test_a_tool_already_used_always_survives():
    """Otherwise the narrowing makes a retry impossible."""
    got = _sel("посчитай 15% от 2480", always={"generate_image"})
    assert "generate_image" in got


def test_it_never_invents_a_tool_that_was_not_offered():
    subset = [s for s in ALL if s["function"]["name"] in ("search", "calculate")]
    got = _names(T.select_tools("нарисуй кота", subset, wants=["generate_image"]))
    assert got == ["calculate", "search"], got


def test_it_never_returns_an_empty_payload():
    """A round with no tools at all is worse than a round with too many."""
    for text in ["добавь ей очки", "привет", "нарисуй кота", ""]:
        for pool in (ALL, ALL[:1], []):
            got = T.select_tools(text, pool)
            assert got or not pool, f"empty for {text!r}"


def test_the_core_fill_does_not_contradict_the_match():
    """Measured on the routing bench: `раздели 987654 на 321 и округли` matched
    calculate, the core fill added search anyway, and on one run of two the
    model computed the answer and then went to the web as well. Offering the
    wrong tool is most of the invitation to use it."""
    got = _sel("раздели 987654 на 321 и округли до двух знаков")
    assert "calculate" in got and "search" not in got, got


def test_a_mixed_request_keeps_both():
    """Only the FILL is suppressed. A turn that genuinely asks for both still
    gets both, because search matched on its own."""
    got = _sel("найди курс биткоина и посчитай 5% от него")
    assert "search" in got and "calculate" in got, got


def test_a_tool_in_use_survives_the_contradiction_rule():
    got = _sel("посчитай 15% от 2480", always={"search"})
    assert "search" in got


@pytest.mark.parametrize('tool,required', [
    ('generate_image',{'inspect_image'}),
    ('run_code',{'install_packages','run_tests','read_file'}),
])
def test_embedding_picks_keep_the_same_dependencies_as_intent_picks(tool,required):
    T._embed_on=lambda:True
    T._embed_pick=lambda *args,**kwargs:{tool}
    got=set(_names(T.select_tools('paraphrase',ALL,wants=[])))
    assert {tool,*required} <= got


def test_weak_embedding_plan_hit_does_not_pull_in_the_file_kit():
    T._embed_on=lambda:True
    T._embed_pick=lambda *args,**kwargs:{'update_plan'}
    got=set(_names(T.select_tools('paraphrase',ALL,wants=[])))
    assert 'update_plan' in got and 'read_file' not in got and 'run_code' not in got


class _FakeEmbedder:
    base_url='http://offline'
    model='model-a'

    def __init__(self):
        self.phase=0
        self.dimension=2
        self.document_batches=0

    def embed(self,texts):
        if texts==['question']:
            return [[1.,0.] if self.dimension==2 else [1.,0.,100.]]
        self.document_batches+=1
        hit=[1.,0.] if self.dimension==2 else [1.,0.,100.]
        miss=[0.,1.] if self.dimension==2 else [0.,1.,0.]
        return [hit if text.startswith(('a:' if self.phase==0 else 'b:')) else miss for text in texts]


def _embedding_catalog():
    embedder=_FakeEmbedder()
    T._EMBEDDER=embedder
    catalog=[{'function':{'name':name,'description':'old'}} for name in ('a','b')]
    return embedder,catalog


@pytest.mark.parametrize('changed',['description','model','base_url'])
def test_embedding_cache_does_not_reuse_another_catalog_or_model(changed):
    embedder,catalog=_embedding_catalog()
    assert T._embed_pick('question',catalog)=={'a'}
    embedder.phase=1
    if changed=='description':
        catalog[0]['function']['description']='new'
    else:
        setattr(embedder,changed,'changed')
    assert T._embed_pick('question',catalog)=={'b'}


def test_changed_query_dimension_reembeds_cached_documents_once():
    embedder,catalog=_embedding_catalog()
    assert T._embed_pick('question',catalog)=={'a'}
    embedder.dimension=3
    assert T._embed_pick('question',catalog)=={'a'}
    assert embedder.document_batches==2


def test_incompatible_document_dimensions_fall_back_without_truncated_cosine():
    embedder,catalog=_embedding_catalog()
    original=embedder.embed
    def mismatch(texts):
        return [[1.,0.,100.]] if texts==['question'] else original(texts)
    embedder.embed=mismatch
    assert T._embed_pick('question',catalog,floor=0.001)==set()
    assert embedder.document_batches==1


@pytest.mark.parametrize('bad',[[float('nan'),0.],[float('inf'),0.],[0.,0.],[1e308,1e308],[None,0.],[True,0.],None])
@pytest.mark.parametrize('source',['query','document','cache'])
def test_invalid_embedding_vectors_do_not_poison_cache_and_healthy_turn_recovers(bad,source):
    embedder,catalog=_embedding_catalog()
    original=embedder.embed
    if source=='cache':
        assert T._embed_pick('question',catalog)=={'a'}
        key=next(iter(T._EMBED_CACHE))
        T._EMBED_CACHE[key][0]=bad
        assert T._embed_pick('question',catalog)=={'a'}
        assert embedder.document_batches==2
    else:
        def invalid(texts):
            return [bad] if texts==['question'] else [bad,[0.,1.]]
        def one_bad_source(texts):
            is_query=texts==['question']
            return invalid(texts) if is_query==(source=='query') else original(texts)
        embedder.embed=one_bad_source
        assert T._embed_pick('question',catalog)==set()
        if source=='query':
            assert embedder.document_batches==0
        embedder.embed=original
    assert T._embed_pick('question',catalog)=={'a'}
