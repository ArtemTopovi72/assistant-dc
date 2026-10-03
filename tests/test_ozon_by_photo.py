"""Ozon's photo search: the picture finds the product, the words add wishes and
the priority (one 📷 button; 💸/⭐/🚚 or words choose what matters)."""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ozon_client as O
import ozon_shopper as S


def test_photo_item_searches_by_the_photo_and_keeps_the_budget(monkeypatch):
    calls = []
    monkeypatch.setattr(O, "image_search_path", lambda p: calls.append(p) or "/search-by-image?image_id=1")
    monkeypatch.setattr(O, "search_by_image", lambda path, limit, page=1: [
        {"sku": f"{page}a", "name": "Чайник стеклянный", "price": 1500},
        {"sku": f"{page}b", "name": "Чайник стеклянный", "price": 9000}])
    monkeypatch.setattr(O, "search", lambda *a, **k: (_ for _ in ()).throw(AssertionError("text search")))
    out = S.gather({"need": "чайник", "queries": [], "image": "x.jpg", "max_price": 3000})
    assert calls == ["x.jpg"]                         # uploaded once, pages reuse it
    assert out and all(x["price"] <= 3000 for x in out)


def test_tool_needs_a_picture():
    import tool_ozon_handlers as H
    ctx = types.SimpleNamespace(last_image_path=None, user_id="t")
    r = H._handle_ozon_shop.__wrapped__(ctx, {}, {"request": "подешевле", "by_photo": True}) \
        if hasattr(H._handle_ozon_shop, "__wrapped__") else H._handle_ozon_shop(ctx, {}, {"request": "подешевле", "by_photo": True})
    assert "no picture" in r


def test_one_button_in_the_ozon_menu():
    import tg_keyboards as K
    rows = [[b["text"] if isinstance(b, dict) else b for b in r] for r in K._ozon_kb("ru")["keyboard"]]
    flat = [t for r in rows for t in r]
    assert flat.count("📷 Найти по фото") == 1 and all(len(r) <= 2 for r in rows)
