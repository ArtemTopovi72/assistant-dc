"""Raw search titles cannot prove connector variants or other hard requirements."""
import json
from types import SimpleNamespace

import pytest
import tool_ozon_handlers as H


def listing(sku, name):
    return {'sku': str(sku), 'name': name, 'price': 200,
            'url': f'https://www.ozon.ru/product/{sku}/', 'image': f'https://images.invalid/{sku}.jpg'}


@pytest.mark.parametrize('sort', ['popular', 'price'])
@pytest.mark.parametrize('request_field', ['user_input_original', 'user_input'])
def test_search_checks_ambiguous_variants_against_original_request(sort, request_field):
    items = [listing(1, 'USB-C to USB-C 1m'), listing(2, 'USB Type C cable 1m'),
             listing(3, 'Charging cable 1m')]
    details, prompts = [], []
    def card(sku):
        details.append(sku)
        return {'characteristics': {'connectors': 'USB-A to USB-C' if sku == '2' else 'USB-C to USB-C',
                                    'length': '1m'}}
    def model(ctx, system, user, **kw):
        prompts.append(user)
        if 'PRODUCT ASKED FOR:' in user:  # existing cheap-only product class filter
            return '{"other": []}'
        if 'USB-A to USB-C' in user:
            return '{"verified": [2], "inspect": []}'
        return '{"verified": [1], "inspect": [2, 3]}'
    ctx = SimpleNamespace(text_model=model, turn_products=[], deps=SimpleNamespace(
        ozon=SimpleNamespace(search=lambda *a, **k: items, details=card)))
    request = 'Find three USB-C to USB-C cables, exactly 1 meter, under 500 rubles'
    out = H._handle_ozon_search(ctx, {request_field: request},
                               {'query': 'charging cable', 'sort': sort, 'limit': 3})
    assert [x['sku'] for x in ctx.turn_products] == ['1', '3']
    assert 'sku 2 ' not in out
    assert details == ['2', '3']
    assert any(request in prompt for prompt in prompts)
    assert '2 verified' in out and 'do not substitute' in out.lower()


def test_unverified_requirements_are_not_recommended_when_card_is_incomplete():
    prompts = []
    def model(ctx, system, user, **kw):
        prompts.append(user)
        return '{"verified": [], "inspect": [1]}'
    ctx = SimpleNamespace(text_model=model, turn_products=[], deps=SimpleNamespace(
        ozon=SimpleNamespace(search=lambda *a, **k: [listing(2, 'USB Type C cable')],
                             details=lambda sku: {'characteristics': {}})))
    out = H._handle_ozon_search(ctx, {'user_input_original': 'USB-C to USB-C, 1m'},
                               {'query': 'USB cable', 'limit': 3})
    assert not ctx.turn_products
    assert 'sku 2 ' not in out and 'verified' in out.lower()
    assert len(prompts) == 2


def test_unavailable_verifier_does_not_claim_raw_results_match():
    ctx = SimpleNamespace(text_model=lambda *a, **k: '', turn_products=[], deps=SimpleNamespace(
        ozon=SimpleNamespace(search=lambda *a, **k: [listing(1, 'USB Type C cable')],
                             details=lambda sku: pytest.fail('no candidate was selected'))))
    out = H._handle_ozon_search(ctx, {}, {'query': 'USB-C to USB-C 1m'})
    assert not ctx.turn_products and 'verified' in out.lower()


def test_ambiguous_shortlist_has_a_fixed_card_read_bound():
    import ozon_shopper as S
    opened = []
    def model(ctx, system, user, **kw):
        return json.dumps({'verified': [], 'inspect': list(range(1, 31))})
    def card(sku):
        opened.append(sku)
        return {'characteristics': {'connector': 'unknown'}}
    ctx = SimpleNamespace(text_model=model, deps=SimpleNamespace(ozon=SimpleNamespace(details=card)))
    assert S.verified_matches(ctx, 'USB-C to USB-C 1m',
                              [listing(n, 'Type C cable') for n in range(30)], 10) == []
    assert len(opened) == S.FINALISTS


@pytest.mark.parametrize('key', ['verified', 'inspect'])
@pytest.mark.parametrize('invalid', ['12', {'1': False}, 1, True, [[1]], [True], [1.0], [None]])
def test_malformed_selection_never_verifies_or_opens_a_product(key, invalid):
    import ozon_shopper as S
    data = {'verified': [], 'inspect': [], key: invalid}
    ctx = SimpleNamespace(text_model=lambda *a, **k: json.dumps(data), deps=SimpleNamespace(
        ozon=SimpleNamespace(details=lambda sku: pytest.fail('malformed selection opened a card'))))
    assert S.verified_matches(ctx, 'USB-C to USB-C 1m',
                              [listing(1, 'Type C cable'), listing(2, 'Type C cable')], 2) == []


def test_valid_numeric_selection_accepts_only_integer_and_digit_string_ids():
    import ozon_shopper as S
    ctx = SimpleNamespace(text_model=lambda *a, **k: '{"verified": [1, "2"]}')
    assert [x['sku'] for x in S.verified_matches(ctx, 'cable',
                [listing(1, 'Cable'), listing(2, 'Cable')], 2)] == ['1', '2']
