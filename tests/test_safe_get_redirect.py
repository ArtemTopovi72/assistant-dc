"""safe_get must re-check every redirect hop (SSRF via 302 -> localhost)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dr_urls

class R:
    def __init__(s, loc=None): s.headers = {"location": loc} if loc else {}; s.is_redirect = bool(loc)
    def close(s): pass

calls = []
def get(url, **k):
    calls.append(url)
    return R("http://127.0.0.1:8000/view") if "evil" in url else R()

dr_urls._is_safe_public_url = lambda u: "127.0.0.1" not in u
assert dr_urls.safe_get("https://evil.example/a", getter=get) is None
assert calls == ["https://evil.example/a"], calls
assert dr_urls.safe_get("https://ok.example/", getter=get) is not None
print("PASS safe_get redirect")
