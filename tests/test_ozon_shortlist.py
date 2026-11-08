"""A shortlist across messages yields one verified offer."""
import os
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services"))

import ozon_same
import ozon_shopper
import intent
import tool_ozon_handlers as handler
import tools as agent_tools
import graph_personality as personality


URLS = [f"https://ozon.ru/t/card{i}" for i in range(7)]


def test_failed_review_lookup_cannot_be_reported_as_no_reviews():
    ctx = SimpleNamespace(turn_products=[{'name': 'Рюкзак', 'url': 'https://www.ozon.ru/product/item-123/', 'price': 5000}])
    answer = personality._guard_failed_ozon_reviews('Отзывов на данный товар нет.', ['ozon_product', 'ozon_reviews!'], ctx)
    assert 'не удалось' in answer.lower() and 'Отзывов на данный товар нет' not in answer
    assert 'https://www.ozon.ru/product/item-123/' in answer and '5 000' in answer
    assert personality._guard_failed_ozon_reviews('Отзывы есть.', ['ozon_reviews!', 'ozon_reviews'], ctx) == 'Отзывы есть.'
    skipped = personality._guard_failed_ozon_reviews('Отзывов нет.', ['ozon_product'], ctx, 'Проверь отзывы')
    assert 'не удалось' in skipped.lower() and 'Отзывов нет.' not in skipped
    searched = personality._guard_failed_ozon_reviews('Покупатели довольны.', ['ozon_search'], ctx, 'Найди и проверь отзывы')
    assert 'не удалось' in searched.lower() and 'Покупатели довольны' not in searched
    assert 'https://www.ozon.ru/product/item-123/' in searched
    ctx._ozon_shop_review_checked = True
    assert personality._guard_failed_ozon_reviews('Отзывы изучены магазинным инструментом.',
        ['ozon_search', 'ozon_shop'], ctx, 'Проверь отзывы') == 'Отзывы изучены магазинным инструментом.'


def test_requested_reviews_are_checked_after_the_product_without_retry_loop():
    read = {'ok': True, 'wants': ['ozon_product', 'ozon_reviews']}
    tools = [{'function': {'name': 'ozon_reviews'}}]
    force = personality._forced_ozon_reviews
    assert force(read, {'ozon_product'}, {}, tools) == 'ozon_reviews'
    assert not force(read, {'ozon_product', 'ozon_reviews'}, {}, tools)
    assert not force(read, {'ozon_product'}, {('ozon_reviews', 'x'): 1}, tools)
    assert not force({'ok': True, 'wants': ['ozon_product']}, {'ozon_product'}, {}, tools)


def test_requested_reviews_after_search_even_if_intent_missed_them():
    read = {'ok': True, 'wants': ['ozon_search']}
    tools = [{'function': {'name': 'ozon_reviews'}}]
    products = [{'sku': '123', 'url': 'https://www.ozon.ru/product/item-123/'}]
    force = personality._forced_ozon_reviews
    assert force(read, {'ozon_search'}, {}, tools, 'Найди товар и покажи отзывы', products) == 'ozon_reviews'
    assert not force(read, {'ozon_search'}, {}, tools, 'Найди товар и покажи отзывы', [])
    assert not force(read, {'ozon_search'}, {}, tools, 'Найди товар без отзывов', products)
    assert not force(read, {'ozon_search'}, {}, tools, 'Find it without reviews', products)
    assert not force(read, {'ozon_search', 'ozon_reviews'}, {}, tools, 'Покажи отзывы', products)
    assert not force(read, {'ozon_search'}, {('ozon_reviews', 'x'): 1}, tools, 'Покажи отзывы', products)


def test_intent_routes_many_links_to_shop_before_same_seller():
    out = {"must_call": "ozon_same", "wants": ["ozon_same", "ozon_shop"]}
    intent._same_or_analogue(SimpleNamespace(),
                             f"Выбери лучший {URLS[0]} {URLS[1]} и найди дешевле", out)
    assert out["must_call"] == "ozon_shop"
    assert out["wants"][0] == "ozon_shop"


@pytest.mark.parametrize('marketplace', ['Озоне', 'Ozon'])
def test_explicit_web_then_ozon_request_searches_web_first(monkeypatch, marketplace):
    monkeypatch.setenv("INTENT_LIVE", "1")
    ctx = SimpleNamespace(text_model=lambda *a, **kw: (
        '{"needs_tool":true,"wants":["ozon_shop"],"must_call":""}'))
    out = intent._read(ctx, f"Погугли обзоры хороших рюкзаков, потом найди на {marketplace}", "", "", {})
    assert out["must_call"] == "search"
    assert out["wants"][:2] == ["search", "ozon_shop"]


def test_web_reviews_then_ozon_does_not_skip_research(monkeypatch):
    monkeypatch.setenv("INTENT_LIVE", "1")
    ctx = SimpleNamespace(text_model=lambda *a, **kw: (
        '{"needs_tool":true,"wants":["ozon_search"],"must_call":""}'))
    text = ('Сначала найди конкретные модели по обзорам в интернете, затем проверь '
            'эти же модели на Ozon, цены и отзывы')
    out = intent._read(ctx, text, '', '', {})
    assert out['must_call'] == 'search'
    assert out['wants'][:2] == ['search', 'ozon_shop']


def test_find_review_then_ozon_does_not_skip_research(monkeypatch):
    monkeypatch.setenv('INTENT_LIVE', '1')
    ctx = SimpleNamespace(text_model=lambda *a, **kw: (
        '{"needs_tool":true,"wants":["ozon_shop"],"must_call":""}'))
    text = ('Найди независимый обзор SSD Kingston NV3 SNV3S/1000G, затем поищи '
            'именно эту модель на Ozon. Проверь актуальную цену и реальные отзывы.')
    out = intent._read(ctx, text, '', '', {})
    assert out['must_call'] == 'search'


def test_web_reviews_then_ozon_rejects_unresearched_shop(monkeypatch):
    monkeypatch.setattr(ozon_shopper, 'shop', lambda *a, **kw: pytest.fail('research was skipped'))
    state = {'user_input_original': 'Сначала найди модели по обзорам в интернете, затем проверь на Ozon'}
    result = handler._handle_ozon_shop(SimpleNamespace(), state, {'request': 'рюкзак'})
    assert '[TOOL ERROR]' in result


def test_research_request_cannot_bypass_matching_with_generic_search(monkeypatch):
    monkeypatch.setattr(handler._oz, 'call', lambda *a, **kw: pytest.fail('unvetted search ran'))
    state = {'user_input_original': 'Сначала найди модели по обзорам в интернете, затем проверь на Ozon',
             'research_shop_evidence': 'Review: Timbuk2 Authority Deluxe'}
    result = handler._handle_ozon_search(SimpleNamespace(), state, {'query': 'рюкзак Thule'})
    assert '[TOOL ERROR]' in result and 'ozon_shop' in result


def test_research_failure_gives_exact_reviewed_names(monkeypatch):
    monkeypatch.setattr(handler, '_research_shop_index_fallback', lambda *a: None)
    monkeypatch.setattr(ozon_shopper, 'shop', lambda *a, **kw: pytest.fail('unresearched model searched'))
    state = {'user_input_original': 'Сначала обзоры в интернете, затем проверь на Ozon',
             'research_shop_evidence': 'Review: Timbuk2 Authority Deluxe',
             'research_shop_index_attempted': True,
             'research_shop_candidates': [{'model': 'Timbuk2 Authority Deluxe'}]}
    result = handler._handle_ozon_shop(SimpleNamespace(), state, {'request': 'Thule EnRoute'})
    assert 'Timbuk2 Authority Deluxe' in result


def test_research_brand_prefix_cannot_replace_reviewed_model(monkeypatch):
    monkeypatch.setattr(handler, '_research_shop_index_fallback', lambda *a: None)
    monkeypatch.setattr(ozon_shopper, 'shop', lambda *a, **kw: pytest.fail('generic brand was accepted'))
    state = {'user_input_original': 'Сначала обзоры в интернете, затем проверь на Ozon',
             'research_shop_evidence': 'Review: Peak Design Everyday Backpack V2 30L',
             'research_shop_index_attempted': True,
             'research_shop_candidates': [{'model': 'Peak Design Everyday Backpack V2 30L'}]}
    result = handler._handle_ozon_shop(SimpleNamespace(), state, {'request': 'Peak Design backpack'})
    assert '[TOOL ERROR]' in result
    result = handler._handle_ozon_shop(SimpleNamespace(), state,
                                       {'request': 'Peak Design Everyday Backpack V2 20L'})
    assert '[TOOL ERROR]' in result


def test_empty_backpack_review_search_retries_with_shorter_query():
    queries = []
    def search(_ctx, query):
        queries.append(query)
        return "No results found for your query."
    ctx = SimpleNamespace(deps=SimpleNamespace(run_web_search=search), set_stage=lambda *_: None,
                          web_search_enabled=True)
    state = {"user_input_original": "Погугли рюкзаки, потом Озон"}
    agent_tools._handle_search(ctx, state,
                               {"query": "best durable comfortable 16 inch laptop backpack reviews 2026"})
    assert queries == ["best durable comfortable 16 inch laptop backpack reviews 2026",
                       "16 inch laptop backpack reviews"]
    assert state["research_shop_evidence"] == ""


def test_failed_followup_does_not_discard_research_already_found():
    ctx = SimpleNamespace(deps=SimpleNamespace(run_web_search=lambda *_: 'No results found for your query.'),
                          set_stage=lambda *_: None, web_search_enabled=True)
    state = {'user_input_original': 'Погугли рюкзаки, потом Озон',
             'research_shop_evidence': 'Review: Thule Subterra 2',
             'research_shop_verified': False}
    agent_tools._handle_search(ctx, state, {'query': 'review other model'})
    assert state['research_shop_evidence'] == 'Review: Thule Subterra 2'


@pytest.mark.parametrize('marketplace', ['Озон', 'Ozon'])
def test_next_web_search_keeps_both_models(marketplace):
    result = 'Review of Bellroy Transit Backpack. ' * 12
    ctx = SimpleNamespace(deps=SimpleNamespace(run_web_search=lambda *_: result),
                          set_stage=lambda *_: None, remember=lambda *_: None,
                          web_search_enabled=True)
    state = {'user_input_original': f'Погугли рюкзаки, потом {marketplace}',
             'research_shop_evidence': 'Review of Thule Subterra 2.',
             'research_shop_verified': True}
    agent_tools._handle_search(ctx, state, {'query': 'Bellroy Transit Backpack review'})
    assert 'Thule Subterra 2' in state['research_shop_evidence']
    assert 'Bellroy Transit Backpack' in state['research_shop_evidence']
    assert not state['research_shop_verified']


@pytest.mark.parametrize('marketplace', ['Озоне', 'Ozon'])
def test_research_then_shop_rejects_generic_ozon_request(monkeypatch, marketplace):
    monkeypatch.setattr(ozon_shopper, "shop", lambda *a, **kw: pytest.fail("generic search ran"))
    calls = []
    monkeypatch.setattr(handler, '_research_shop_index_fallback', lambda *a: calls.append(True))
    state = {"user_input_original": f"Погугли обзоры, потом найди на {marketplace}",
             "research_shop_evidence": "Thule Subterra 2 has a padded laptop compartment"}
    out = handler._handle_ozon_shop(SimpleNamespace(), state,
                                    {"request": "рюкзак для ноутбука 16 дюймов"})
    assert "specific model" in out
    assert calls == [True]


def test_failed_research_shop_never_returns_generic_index_links(monkeypatch):
    monkeypatch.setattr(handler, '_research_shop_index_fallback', lambda *a: None)
    monkeypatch.setattr(handler, '_indexed_ozon_links', lambda *a: pytest.fail('generic index search ran'))
    def fail(*args, **kwargs):
        raise RuntimeError('blocked')
    monkeypatch.setattr(ozon_shopper, 'shop', fail)
    state = {'user_input_original': 'Погугли обзоры, потом найди на Ozon',
             'research_shop_evidence': 'Review: Thule Subterra 2'}
    out = handler._handle_ozon_shop(SimpleNamespace(turn_products=[]), state,
                                    {'request': 'Thule Subterra 2'})
    assert '[TOOL ERROR]' in out
    assert handler._NAMED_MODEL.search("Thule Subterra 2 27L")


def test_unavailable_ozon_looks_up_several_researched_models(monkeypatch):
    import search
    monkeypatch.setattr(ozon_shopper, 'shop', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('blocked')))
    monkeypatch.setattr(ozon_shopper, '_llm_json', lambda *a, **kw: {
        'models': ['Thule Subterra 2', 'Bellroy Transit', 'Invented Model']})
    queries = []
    def search_index(ctx, query, max_results):
        queries.append(query)
        if 'Thule' in query:
            return [{'title': 'Thule Subterra 2 backpack', 'href': 'https://ozon.ru/product/thule-subterra-2-123/'}]
        return [{'title': 'Bellroy Transit backpack', 'href': 'https://ozon.ru/product/bellroy-transit-456/'}]
    monkeypatch.setattr(search, 'raw_search_results', search_index)
    state = {'user_input_original': 'Погугли обзоры, потом найди на Озоне',
             'research_shop_evidence': 'Review: Thule Subterra 2 and Bellroy Transit.'}
    handler._handle_ozon_shop(SimpleNamespace(), state, {'request': 'Thule Subterra 2'})
    assert len(queries) == 2 and not any('Invented' in q for q in queries)
    assert [row['model'] for row in state['research_shop_candidates']] == ['Thule Subterra 2', 'Bellroy Transit']
    assert all(row['url'].startswith('https://ozon.ru/product/') for row in state['research_shop_candidates'])


def test_unverified_index_results_are_labelled_not_recommended():
    from graph_personality import _unverified_research_answer
    answer = _unverified_research_answer({'research_shop_candidates': [
        {'model': 'Thule Subterra 2', 'url': 'https://ozon.ru/product/thule-subterra-2-123/'},
        {'model': 'Bellroy Transit', 'url': ''}]})
    assert 'Thule Subterra 2' in answer and 'Bellroy Transit' in answer
    assert 'https://ozon.ru/product/thule-subterra-2-123/' in answer
    assert 'не провер' in answer.lower() and 'цен' in answer.lower()
    assert 'купи' not in answer.lower()


def test_only_verified_request_does_not_return_unverified_shop_links():
    from graph_personality import _unverified_research_answer
    answer = _unverified_research_answer({
        'user_input_original': 'Дай ссылки только на проверенные модели.',
        'research_shop_candidates': [
            {'model': 'Thule Subterra 2', 'url': 'https://ozon.ru/product/thule-subterra-2-123/'}]},
        [{'url': 'https://www.ozon.ru/product/thule-subterra-2-123/'}])
    assert 'Thule Subterra 2' in answer
    assert 'ozon.ru/product/' not in answer


def test_fallback_researches_more_when_only_one_model_is_known(monkeypatch):
    import search
    calls = []
    def more(ctx, state, args):
        calls.append(args['query'])
        state['research_shop_evidence'] += ' Review: Bellroy Transit.'
        return 'ok'
    monkeypatch.setattr(agent_tools, '_handle_search', more)
    monkeypatch.setattr(ozon_shopper, '_llm_json', lambda ctx, system, evidence, *a, **kw: {
        'models': ['Thule Subterra 2', 'Bellroy Transit'] if 'Bellroy' in evidence else ['Thule Subterra 2']})
    monkeypatch.setattr(search, 'raw_search_results', lambda *a: [])
    state = {'user_input_original': 'Погугли обзоры, потом найди на Озоне',
             'research_shop_evidence': 'Review: Thule Subterra 2.'}
    handler._research_shop_index_fallback(SimpleNamespace(), state, ozon_shopper)
    assert len(calls) == 1 and 'Thule Subterra 2' in calls[0]
    assert [row['model'] for row in state['research_shop_candidates']] == ['Thule Subterra 2', 'Bellroy Transit']


def test_verified_only_fallback_does_not_search_unverified_index(monkeypatch):
    import search
    monkeypatch.setattr(ozon_shopper, '_llm_json', lambda *a, **kw: {
        'models': ['Thule Subterra 2', 'Bellroy Transit']})
    calls = []
    monkeypatch.setattr(search, 'raw_search_results', lambda *a: calls.append(a) or [])
    state = {'user_input_original': 'Дай ссылки только на проверенные модели',
             'research_shop_evidence': 'Thule Subterra 2 and Bellroy Transit'}
    handler._research_shop_index_fallback(SimpleNamespace(), state, ozon_shopper)
    assert calls == []
    assert [row['model'] for row in state['research_shop_candidates']] == ['Thule Subterra 2', 'Bellroy Transit']


def test_index_fallback_rejects_wrong_product_and_foreign_domain(monkeypatch):
    import search
    monkeypatch.setattr(ozon_shopper, '_llm_json', lambda *a, **kw: {
        'models': ['Thule Subterra 2', 'Bellroy Transit']})
    monkeypatch.setattr(search, 'raw_search_results', lambda *a: [
        {'title': 'Thule Subterra 2', 'href': 'https://ozon.ru.attacker.example/product/thule/'},
        {'title': 'Different backpack', 'href': 'https://ozon.ru/product/unrelated-123/'}])
    state = {'research_shop_evidence': 'Thule Subterra 2 and Bellroy Transit'}
    handler._research_shop_index_fallback(SimpleNamespace(), state, ozon_shopper)
    assert len(state['research_shop_candidates']) == 2
    assert all(not row['url'] for row in state['research_shop_candidates'])


def test_index_fallback_accepts_inches_in_model_name(monkeypatch):
    import search
    monkeypatch.setattr(ozon_shopper, '_llm_json', lambda *a, **kw: {
        'models': ['MATEIN 15.6″ Travel Backpack', 'Bellroy Transit']})
    monkeypatch.setattr(search, 'raw_search_results', lambda *a: [{
        'title': 'MATEIN 15.6 Travel Backpack',
        'href': 'https://www.ozon.ru/product/matein-123/'}])
    state = {'research_shop_evidence': 'MATEIN 15.6″ Travel Backpack and Bellroy Transit'}
    handler._research_shop_index_fallback(SimpleNamespace(), state, ozon_shopper)
    assert state['research_shop_candidates'][0]['url'] == 'https://www.ozon.ru/product/matein-123/'


def test_research_then_shop_accepts_model_named_in_evidence(monkeypatch):
    monkeypatch.setattr(ozon_shopper, "shop", lambda *a, **kw: (
        {"mode": "one"}, [{"choice": {"sku": "42", "name": "Thule Subterra 2 27L",
                                      "url": "https://ozon.ru/product/42/"},
                          "item": {"need": "рюкзак"}, "decision": {}, "others": []}]))
    monkeypatch.setattr(ozon_shopper, "report", lambda *a, **kw: "FOUND")
    state = {"user_input_original": "Погугли обзоры, потом найди на Озоне",
             "research_shop_evidence": "Thule Subterra 2 has a padded laptop compartment"}
    out = handler._handle_ozon_shop(SimpleNamespace(turn_products=[]), state,
                                    {"request": "Thule Subterra 2 27L рюкзак"})
    assert out == "FOUND"
    assert state["research_shop_verified"] is True


def test_research_shop_does_not_mistake_brand_for_researched_model(monkeypatch):
    monkeypatch.setattr(handler, '_research_shop_index_fallback', lambda *a: None)
    monkeypatch.setattr(ozon_shopper, 'shop', lambda *a, **kw: (
        {'mode': 'one'}, [{'choice': {'sku': '1', 'name': 'The North Face Рюкзак'}}]))
    state = {'user_input_original': 'Погугли обзоры, потом найди на Ozon',
             'research_shop_evidence': 'Review: The North Face Borealis backpack.'}
    result = handler._handle_ozon_shop(SimpleNamespace(turn_products=[]), state,
        {'request': 'The North Face Borealis Commuter 16 inch laptop backpack'})
    assert 'No confirmed Ozon listing' in result
    assert not state.get('research_shop_verified')


def test_researched_model_with_size_is_not_mistaken_for_generic(monkeypatch):
    called = []
    monkeypatch.setattr(ozon_shopper, 'shop', lambda *a, **kw: (
        called.append(True) or {'mode': 'one'}, [{'choice': {'sku': '1',
        'name': 'MATEIN 15.6 Travel Laptop Backpack', 'url': 'https://ozon.ru/product/matein-1/'}}]))
    monkeypatch.setattr(ozon_shopper, 'report', lambda *a, **kw: 'FOUND')
    state = {'user_input_original': 'Погугли обзоры, потом найди на Озоне',
             'research_shop_evidence': 'MATEIN 15.6″ Travel Laptop Backpack is mentioned in a review.'}
    out = handler._handle_ozon_shop(SimpleNamespace(turn_products=[]), state,
                                    {'request': 'MATEIN 15.6″ Travel Laptop Backpack'})
    assert called and out == 'FOUND' and state['research_shop_verified']


def test_missing_exact_shop_listing_runs_index_fallback(monkeypatch):
    monkeypatch.setattr(ozon_shopper, 'shop', lambda *a, **kw: (
        {'mode': 'one'}, [{'choice': {'sku': '1', 'name': 'Unrelated model'}}]))
    calls = []
    monkeypatch.setattr(handler, '_research_shop_index_fallback', lambda ctx, state, shopper: calls.append(True))
    state = {'user_input_original': 'Погугли обзоры, потом найди на Озоне',
             'research_shop_evidence': 'Review: Thule Subterra 2'}
    out = handler._handle_ozon_shop(SimpleNamespace(turn_products=[]), state,
                                    {'request': 'Thule Subterra 2'})
    assert 'No confirmed Ozon listing' in out and calls == [True]


def test_research_then_shop_rejects_different_listing(monkeypatch):
    monkeypatch.setattr(handler, '_research_shop_index_fallback', lambda *a: None)
    monkeypatch.setattr(ozon_shopper, "shop", lambda *a, **kw: (
        {"mode": "one"}, [{"choice": {"sku": "42", "name": "Generic laptop backpack"},
                          "item": {"need": "рюкзак"}, "decision": {}, "others": []}]))
    state = {"user_input_original": "Погугли обзоры, потом найди на Озоне",
             "research_shop_evidence": "Thule Subterra 2 has a padded laptop compartment"}
    out = handler._handle_ozon_shop(SimpleNamespace(turn_products=[]), state,
                                    {"request": "Thule Subterra 2 27L рюкзак"})
    assert "No confirmed Ozon listing" in out
    assert not state.get("research_shop_verified")


def test_research_then_shop_does_not_substitute_model_variant(monkeypatch):
    monkeypatch.setattr(handler, '_research_shop_index_fallback', lambda *a: None)
    monkeypatch.setattr(ozon_shopper, "shop", lambda *a, **kw: (
        {"mode": "one"}, [{"choice": {"sku": "42", "name": "Samsonite Paralux BT Everyday Backpack"},
                          "item": {"need": "рюкзак"}, "decision": {}, "others": []}]))
    state = {"user_input_original": "Погугли обзоры, потом найди на Озоне",
             "research_shop_evidence": "Samsonite Paralux 2-in-1 Travel Backpack review"}
    out = handler._handle_ozon_shop(SimpleNamespace(turn_products=[]), state,
                                    {"request": "Samsonite Paralux 2-1 Travel Backpack"})
    assert "No confirmed Ozon listing" in out
    assert not state.get("research_shop_verified")


def test_collects_seven_recent_user_links_only():
    messages = []
    for url in URLS:
        messages.extend([{"role": "user", "content": f"Рюкзак {url}"},
                         {"role": "assistant", "content": "https://ozon.ru/t/assistant"}])
    links, words = handler._shop_shortlist({"messages": messages,
                                            "user_input_original": "Сравни и выбери лучший"})
    assert links == URLS
    assert "Рюкзак" in words and "assistant" not in words
    messages.insert(0, {"role": "user", "content": "https://ozon.ru/t/stale"})
    messages.insert(1, {"role": "user", "content": "Другая тема"})
    assert handler._shop_shortlist({"messages": messages,
                                    "user_input_original": "Выбери лучший"})[0] == URLS
    assert handler._shop_shortlist({"messages": messages,
                                    "user_input_original": "Подбери один по качеству"})[0] == URLS
    with_note = [{"role": "user", "content": URLS[0] +
                  " [The user's message contains an Ozon product link (). Read that card]"},
                 {"role": "user", "content": URLS[1]}]
    links, captions = handler._shop_shortlist({"messages": with_note,
                                                "user_input_original": "Какой лучше взять?"})
    assert links == URLS[:2] and "The user's message contains" not in captions


def test_forwarded_links_are_shortlist_data():
    quoted = ("Какой и где лучше взять?\n<<<QUOTED MESSAGE -- forwarded from Darya>>>\n"
              f"VIAGGIO {URLS[0]}\nBange {URLS[1]}\n<<<END OF QUOTED MESSAGE>>>")
    links, words = handler._shop_shortlist({"user_input_original": quoted, "messages": []})
    assert links == URLS[:2]
    assert "QUOTED MESSAGE" not in words


def test_current_forwarded_shortlist_does_not_include_old_products():
    quoted = (f"Какой лучше?\n<<<QUOTED MESSAGE>>>\n{URLS[0]}\n{URLS[1]}"
              "\n<<<END OF QUOTED MESSAGE>>>")
    links, _ = handler._shop_shortlist({
        "user_input_original": quoted,
        "messages": [{"role": "user", "content": f"Старый выбор {URLS[2]}"}],
    })
    assert links == URLS[:2]


def test_keeps_chosen_card_when_cheaper_seller_is_found(monkeypatch):
    seen = {}
    winner = {"sku": "1", "url": URLS[0], "name": "Рюкзак", "price": 3000}

    def shop(ctx, req, strategy, pinned):
        seen["pinned"] = pinned
        seen["request"] = req
        return {"mode": "one"}, [{"choice": winner, "item": {"need": "рюкзак"},
                                   "decision": {}, "others": []}]

    monkeypatch.setattr(ozon_shopper, "shop", shop)
    monkeypatch.setattr(ozon_shopper, "report", lambda results, *a: results[0])
    monkeypatch.setattr(ozon_same, "find_same", lambda *a, **kw: {"best": {"cheap": {
        "sku": "2", "url": "https://ozon.ru/product/same-2/", "price": 2000,
        "check": {"verdict": "same", "copy_price": False}}}})
    ctx = SimpleNamespace(turn_products=[])
    state = {"messages": [{"role": "user", "content": u} for u in URLS],
             "user_input_original": "Выбери лучший рюкзак"}
    result = handler._handle_ozon_shop(ctx, state, {"request": "рюкзак"})
    assert seen["pinned"] == URLS
    assert all(url not in seen["request"] for url in URLS)
    assert result["choice"]["sku"] == "1"
    assert result["same_offer_candidate"]["price"] == 2000
    assert [x["sku"] for x in ctx.turn_products] == ["1"]
    answer = ozon_shopper.digest(result)
    assert "Похожее предложение за 2 000 ₽" in answer
    assert "https://ozon.ru/product/same-2/" in answer
    assert "качество не подтверждено" in answer


def test_unverified_offer_cannot_replace_winner(monkeypatch):
    winner = {"sku": "1", "url": URLS[0], "name": "Рюкзак", "price": 3000}
    monkeypatch.setattr(ozon_shopper, "shop", lambda *a, **kw: (
        {"mode": "one"}, [{"choice": winner, "item": {"need": "рюкзак"},
                           "decision": {}, "others": []}]))
    monkeypatch.setattr(ozon_shopper, "report", lambda results, *a: results[0])
    monkeypatch.setattr(ozon_same, "find_same", lambda *a, **kw: {"best": {"cheap": {
        "sku": "copy", "url": "https://ozon.ru/product/copy/", "price": 1000,
        "check": {"verdict": "unclear"}}}})
    state = {"messages": [{"role": "user", "content": u} for u in URLS],
             "user_input_original": "Выбери лучший рюкзак"}
    result = handler._handle_ozon_shop(SimpleNamespace(turn_products=[]), state,
                                       {"request": "рюкзак"})
    assert result["choice"]["sku"] == "1"
