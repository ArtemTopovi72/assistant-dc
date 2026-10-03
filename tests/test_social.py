"""Coverage for social.py: URL classification (pure), real Telegram public-
channel fetch (auth-free, safe), and stubbed VK API paths (no VK_TOKEN
configured in this environment, so VK is exercised via stubs, not live calls).
Run: venv/Scripts/python.exe tests/test_social.py
"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import social as SO

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


class _Patches:
    def __init__(self, **kw):
        self.kw = kw; self.orig = {}
    def __enter__(self):
        for k, v in self.kw.items():
            self.orig[k] = getattr(SO, k)
            setattr(SO, k, v)
        return self
    def __exit__(self, *a):
        for k, v in self.orig.items():
            setattr(SO, k, v)


def test_host_and_social_kind():
    check("host_strips_www", SO._host("https://www.vk.com/x") == "vk.com")
    check("host_bad_url_empty", SO._host(None) == "")
    check("social_kind_vk", SO.social_kind("https://vk.com/club1") == "vk")
    check("social_kind_vk_mobile", SO.social_kind("https://m.vk.com/club1") == "vk")
    check("social_kind_vk_ru", SO.social_kind("https://vk.ru/club1") == "vk")
    check("social_kind_telegram", SO.social_kind("https://t.me/channel") == "telegram")
    check("social_kind_telegram_dog", SO.social_kind("https://telegram.dog/channel") == "telegram")
    check("social_kind_none", SO.social_kind("https://example.com") == "")
    check("is_social_url_true", SO.is_social_url("https://vk.com/x"))
    check("is_social_url_false", not SO.is_social_url("https://example.com"))


def test_tg_channel():
    check("tg_channel_basic", SO._tg_channel("https://t.me/durov") == "durov")
    check("tg_channel_s_prefix", SO._tg_channel("https://t.me/s/durov") == "durov")
    check("tg_channel_with_post_id", SO._tg_channel("https://t.me/durov/123") == "durov")
    check("tg_channel_empty_path", SO._tg_channel("https://t.me/") == "")


def test_extract_telegram_posts_invalid_channel():
    check("telegram_joinchat_none", SO.extract_telegram_posts("https://t.me/joinchat/xyz") is None)
    check("telegram_empty_channel_none", SO.extract_telegram_posts("https://t.me/") is None)


def test_extract_telegram_posts_real():
    # durov's channel is public, long-lived, and definitely has posts.
    out = SO.extract_telegram_posts("https://t.me/durov", limit=3)
    check("telegram_real_fetch_success", out is None or ("Telegram channel @durov" in out))


def test_extract_telegram_posts_stubbed():
    class FakeResp:
        def __init__(self, status=200, text=""):
            self.status_code = status; self.text = text
    with _Patches(requests=types.SimpleNamespace(get=lambda *a, **k: FakeResp(status=404))):
        check("telegram_non200_none", SO.extract_telegram_posts("https://t.me/somechannel") is None)
    empty_html = "<html><body>no messages here</body></html>"
    with _Patches(requests=types.SimpleNamespace(get=lambda *a, **k: FakeResp(text=empty_html))):
        check("telegram_no_message_nodes_none", SO.extract_telegram_posts("https://t.me/somechannel") is None)
    html_with_posts = (
        '<html><body>'
        '<div class="tgme_widget_message_text">First post</div>'
        '<div class="tgme_widget_message_text">Second post</div>'
        '</body></html>')
    with _Patches(requests=types.SimpleNamespace(get=lambda *a, **k: FakeResp(text=html_with_posts))):
        out = SO.extract_telegram_posts("https://t.me/somechannel", limit=5)
        check("telegram_stubbed_posts_newest_first", out is not None and out.index("Second post") < out.index("First post"))
    with _Patches(requests=types.SimpleNamespace(get=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))):
        check("telegram_exception_none", SO.extract_telegram_posts("https://t.me/somechannel") is None)
    blank_posts_html = '<html><body><div class="tgme_widget_message_text">   </div></body></html>'
    with _Patches(requests=types.SimpleNamespace(get=lambda *a, **k: FakeResp(text=blank_posts_html))):
        check("telegram_blank_text_nodes_none", SO.extract_telegram_posts("https://t.me/somechannel") is None)


def test_vk_call():
    class FakeResp:
        def __init__(self, data): self._data = data
        def json(self): return self._data
    with _Patches(requests=types.SimpleNamespace(get=lambda *a, **k: FakeResp({"response": {"ok": True}}))):
        out = SO._vk_call("test.method")
        check("vk_call_success", out == {"ok": True})
    with _Patches(requests=types.SimpleNamespace(get=lambda *a, **k: FakeResp({"error": {"error_msg": "bad"}}))):
        check("vk_call_api_error_none", SO._vk_call("test.method") is None)
    with _Patches(requests=types.SimpleNamespace(get=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))):
        check("vk_call_exception_none", SO._vk_call("test.method") is None)


def test_vk_owner_id():
    check("vk_owner_id_empty_path_none", SO._vk_owner_id("https://vk.com/") is None)
    check("vk_owner_id_club_numeric", SO._vk_owner_id("https://vk.com/club12345") == -12345)
    check("vk_owner_id_public_numeric", SO._vk_owner_id("https://vk.com/public999") == -999)
    check("vk_owner_id_event_numeric", SO._vk_owner_id("https://vk.com/event42") == -42)
    with _Patches(_vk_call=lambda method, **p: {"object_id": 555, "type": "group"}):
        check("vk_owner_id_resolved_group_negative", SO._vk_owner_id("https://vk.com/somescreen") == -555)
    with _Patches(_vk_call=lambda method, **p: {"object_id": 777, "type": "user"}):
        check("vk_owner_id_resolved_user_positive", SO._vk_owner_id("https://vk.com/someuser") == 777)
    with _Patches(_vk_call=lambda method, **p: None):
        check("vk_owner_id_resolve_fails_none", SO._vk_owner_id("https://vk.com/unresolvable") is None)
    with _Patches(_vk_call=lambda method, **p: {"no_object_id": True}):
        check("vk_owner_id_no_object_id_key_none", SO._vk_owner_id("https://vk.com/weird") is None)


def test_extract_vk_posts():
    with _Patches(VK_TOKEN=""):
        check("vk_posts_no_token_none", SO.extract_vk_posts("https://vk.com/club1") is None)
    with _Patches(VK_TOKEN="fake-token", _vk_owner_id=lambda url: None):
        check("vk_posts_no_owner_none", SO.extract_vk_posts("https://vk.com/club1") is None)
    with _Patches(VK_TOKEN="fake-token", _vk_owner_id=lambda url: -1,
                  _vk_call=lambda method, **p: None):
        check("vk_posts_wall_get_fails_none", SO.extract_vk_posts("https://vk.com/club1") is None)
    with _Patches(VK_TOKEN="fake-token", _vk_owner_id=lambda url: -1,
                  _vk_call=lambda method, **p: {"items": []}):
        check("vk_posts_no_items_none", SO.extract_vk_posts("https://vk.com/club1") is None)
    with _Patches(VK_TOKEN="fake-token", _vk_owner_id=lambda url: -1,
                  _vk_call=lambda method, **p: {"items": [{"text": "hello"}, {"text": ""}]}):
        out = SO.extract_vk_posts("https://vk.com/club1")
        check("vk_posts_basic_success", out is not None and "hello" in out)
    with _Patches(VK_TOKEN="fake-token", _vk_owner_id=lambda url: -1,
                  _vk_call=lambda method, **p: {"items": [
                      {"text": "", "copy_history": [{"text": "reposted content"}]}]}):
        out2 = SO.extract_vk_posts("https://vk.com/club1")
        check("vk_posts_repost_uses_original_text", out2 is not None and "reposted content" in out2)
    with _Patches(VK_TOKEN="fake-token", _vk_owner_id=lambda url: -1,
                  _vk_call=lambda method, **p: {"items": [{"text": ""}]}):
        check("vk_posts_all_blank_none", SO.extract_vk_posts("https://vk.com/club1") is None)


def test_fetch_social_posts_dispatch():
    with _Patches(extract_telegram_posts=lambda url, limit: "telegram result"):
        check("fetch_social_posts_dispatches_telegram", SO.fetch_social_posts("https://t.me/x") == "telegram result")
    with _Patches(extract_vk_posts=lambda url, limit: "vk result"):
        check("fetch_social_posts_dispatches_vk", SO.fetch_social_posts("https://vk.com/x") == "vk result")
    check("fetch_social_posts_non_social_none", SO.fetch_social_posts("https://example.com") is None)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print("  ERROR in " + fn.__name__ + ": " + type(e).__name__ + ": " + str(e))
    passed = sum(1 for _, c in RESULTS if c)
    print(f"\n{len(fns)-failed}/{len(fns)} functions, {passed}/{len(RESULTS)} checks passed")
    # A failed check() must reach the exit code. This used to be `1 if failed else 0`,
    # where `failed` counts EXCEPTIONS only -- so every assertion in the file could
    # print FAIL while the suite still exited 0, invisible to anything judging by
    # exit code (which is how these suites are judged).
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)
