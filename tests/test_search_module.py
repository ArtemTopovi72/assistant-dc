"""Coverage for search.py: ddgs fetch/format/distill pipeline, image search,
image download validation, face/grayscale gating, reference-photo fetch loop.
The ddgs client, page reader, getter, face detector and model reads are handed in
on the context (or as parameters), so every branch runs deterministically without
live network calls and nothing global is rebound.
Run: venv/Scripts/python.exe tests/test_search_module.py
"""
import os, sys, tempfile, types, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import offline_guard; offline_guard.public_dns()   # the SSRF guard resolves hosts: no network needed
from pathlib import Path
import search as SE

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + str(detail)))
    if not cond and os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))

_TMP = Path(tempfile.mkdtemp(prefix="searchmod_"))


class FakeCtx:
    """`FakeCtx(name=stand_in, ...)`: the search module takes `call_llm_simple` as the
    context's text model, `yes_no` as its yes/no read, every other name from `ctx.deps`."""
    def __init__(self, yes_no=None, **deps):
        self.api_lock = threading.Lock()
        self.api_min_interval = 0.0
        self.last_api_call_time = 0.0
        self.yes_no = yes_no or (lambda question, text, default=False, wait=True: default)
        if "call_llm_simple" in deps:
            self.text_model = deps.pop("call_llm_simple")
        self.deps = types.SimpleNamespace(**deps)


def _fake_ddgs(text_results=None, news_results=None, images_results=None, raises=None):
    class FakeDDGS:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def text(self, query, **kw):
            if raises:
                raise raises
            return text_results or []
        def news(self, query, **kw):
            if raises:
                raise raises
            return news_results or []
        def images(self, query, **kw):
            if raises:
                raise raises
            return images_results or []
    return FakeDDGS


def test_domain():
    check("domain_strips_www", SE._domain("https://www.example.com/x") == "example.com")
    check("domain_bad_url", isinstance(SE._domain(None), (str, type(None))))


def test_raw_search_results_success():
    ddgs = _fake_ddgs(text_results=[
        {"title": "A", "href": "https://a.com/1", "body": "text a"},
        {"title": "B", "url": "https://b.com/1", "body": "text b"},  # 'url' key variant
        {"title": "", "href": "", "body": "no href, skipped"},
    ])
    out = SE.raw_search_results(FakeCtx(DDGS=ddgs), "query")
    check("raw_search_results_basic", len(out) == 2 and out[0]["domain"] == "a.com")
    check("raw_search_results_url_key_accepted", out[1]["href"] == "https://b.com/1")


def test_raw_search_results_news():
    ddgs = _fake_ddgs(news_results=[
        {"title": "News A", "url": "https://news.com/1", "body": "news text", "date": "2024-01-01"}])
    out = SE.raw_search_results(FakeCtx(DDGS=ddgs), "query", news=True)
    check("raw_search_results_news_has_date", out[0].get("date") == "2024-01-01")


def test_raw_search_results_empty_then_retry_succeeds():
    calls = {"n": 0}
    class FakeDDGS:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def text(self, query, **kw):
            calls["n"] += 1
            return [] if calls["n"] == 1 else [{"title": "T", "href": "https://x.com/1", "body": ""}]
    ddgs = FakeDDGS
    out = SE.raw_search_results(FakeCtx(DDGS=ddgs), "query")
    check("raw_search_results_retries_on_empty", len(out) == 1 and calls["n"] == 2)


def test_raw_search_results_exception_then_retry_succeeds():
    calls = {"n": 0}
    class FakeDDGS:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def text(self, query, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return [{"title": "T", "href": "https://x.com/1", "body": ""}]
    ddgs = FakeDDGS
    out = SE.raw_search_results(FakeCtx(DDGS=ddgs), "query")
    check("raw_search_results_exception_then_retry", len(out) == 1)


def test_raw_search_results_exception_twice_empty():
    ddgs = _fake_ddgs(raises=RuntimeError("boom"))
    out = SE.raw_search_results(FakeCtx(DDGS=ddgs), "query")
    check("raw_search_results_both_attempts_fail_empty", out == [])


def test_search_duckduckgo_and_fetch():
    # Derived from config, not hardcoded, so a settings change is not a test failure.
    from config import SEARCH_MAX_PER_DOMAIN as CAP
    over = CAP + 1
    ddgs = _fake_ddgs(text_results=(
        [{"title": "A", "href": "https://a.com/0", "body": "x" * 600}]
        + [{"title": f"A{i}", "href": f"https://a.com/{i}",
            "body": f"over the per-domain cap {i}"} for i in range(1, over)]
        + [{"title": "B", "href": "https://b.com/1", "body": "text b"}]))
    out = SE.search_duckduckgo_and_fetch(FakeCtx(DDGS=ddgs), "query")
    a_kept = sum(1 for ln in out.splitlines() if "https://a.com/" in ln)
    check("search_and_fetch_caps_results_per_domain", a_kept == CAP,
          f"kept {a_kept} from a.com, cap is {CAP}")
    check("search_and_fetch_keeps_other_domains", "https://b.com/1" in out)
    check("search_and_fetch_numbering_is_contiguous",
          all(f"[{i}]" in out for i in range(1, CAP + 2)), out[:200])
    check("search_and_fetch_snippet_capped", "…" in out)

    ddgs = _fake_ddgs(text_results=[])
    check("search_and_fetch_no_results_sentinel", SE.search_duckduckgo_and_fetch(FakeCtx(DDGS=ddgs), "q") == SE.NO_RESULTS)

    ddgs = _fake_ddgs(text_results=[{"title": "T", "href": "", "body": ""}])  # all missing href -> no parts
    check("search_and_fetch_all_missing_href_sentinel", SE.search_duckduckgo_and_fetch(FakeCtx(DDGS=ddgs), "q") == SE.NO_RESULTS)


def test_distill_search_results():
    ctx = FakeCtx(call_llm_simple=lambda ctx, sys_, usr, temperature=0.2, max_tokens=0, prefill=None: "Distilled facts.")
    out = SE.distill_search_results(ctx, "q", "raw text")
    check("distill_search_results_basic", out == "Distilled facts.")
    ctx = FakeCtx(call_llm_simple=lambda *a, **k: None)
    check("distill_search_results_none_becomes_empty", SE.distill_search_results(ctx, "q", "raw") == "")


def test_run_web_search():
    # The follow-up round has its own test; these pin round one.
    ctx = FakeCtx(SEARCH_FOLLOWUP_ROUND=False, search_duckduckgo_and_fetch=lambda ctx, q, **_kw: SE.NO_RESULTS)
    check("run_web_search_no_results_passthrough", SE.run_web_search(ctx, "q") == SE.NO_RESULTS)
    ctx = FakeCtx(SEARCH_FOLLOWUP_ROUND=False, search_duckduckgo_and_fetch=lambda ctx, q, **_kw: "raw results here",
                  SEARCH_DISTILL=False)
    check("run_web_search_distill_disabled_returns_raw", SE.run_web_search(ctx, "q") == "raw results here")
    ctx = FakeCtx(SEARCH_FOLLOWUP_ROUND=False, search_duckduckgo_and_fetch=lambda ctx, q, **_kw: "raw results here",
                  SEARCH_DISTILL=True,
                  distill_search_results=lambda ctx, q, raw: "good distilled facts")
    check("run_web_search_distill_success", SE.run_web_search(ctx, "q") == "good distilled facts")
    ctx = FakeCtx(SEARCH_FOLLOWUP_ROUND=False, search_duckduckgo_and_fetch=lambda ctx, q, **_kw: "raw results here",
                  SEARCH_DISTILL=True,
                  distill_search_results=lambda ctx, q, raw: "insufficient information found")
    check("run_web_search_distill_insufficient_falls_back_raw", SE.run_web_search(ctx, "q") == "raw results here")
    ctx = FakeCtx(SEARCH_FOLLOWUP_ROUND=False, search_duckduckgo_and_fetch=lambda ctx, q, **_kw: "raw results here",
                  SEARCH_DISTILL=True,
                  distill_search_results=lambda ctx, q, raw: "")
    check("run_web_search_distill_empty_falls_back_raw", SE.run_web_search(ctx, "q") == "raw results here")


def test_image_search_urls():
    # img1 is LANDSCAPE and img2 portrait, so the portrait bonus has something to
    # overtake (equal-area or pixel-area ordering would not exercise it).
    ddgs = _fake_ddgs(images_results=[
        {"image": "https://a.com/img1.jpg", "width": 200, "height": 100},
        {"image": "https://a.com/img2.jpg", "width": 100, "height": 200},  # portrait
        {"url": "https://a.com/img3.jpg", "width": 50, "height": 50},
        {"image": "ftp://bad.com/img.jpg", "width": 999, "height": 999},  # non-http skipped
    ])
    out = SE.image_search_urls(FakeCtx(DDGS=ddgs), "query")
    check("image_search_urls_portrait_ranked_first", out[0] == "https://a.com/img2.jpg")
    check("image_search_urls_non_http_skipped", "ftp://bad.com/img.jpg" not in out)
    check("image_search_urls_url_key_accepted", "https://a.com/img3.jpg" in out)

    ddgs = _fake_ddgs(raises=RuntimeError("boom"))
    check("image_search_urls_exception_twice_empty", SE.image_search_urls(FakeCtx(DDGS=ddgs), "q") == [])


def test_download_image():
    import numpy as np
    from PIL import Image
    good_bytes = _TMP / "good.jpg"
    Image.fromarray((np.random.rand(300, 300, 3) * 255).astype("uint8")).save(good_bytes, "JPEG", quality=95)

    class FakeResp:
        def __init__(self, status=200, content=b"", headers=None):
            self.status_code = status; self.content = content
            self.headers = headers or {"Content-Type": "image/jpeg"}

    get = lambda *a, **k: FakeResp(content=good_bytes.read_bytes())
    out = SE.download_image(FakeCtx(safe_get=get), "https://x.com/img.jpg", str(_TMP / "dl1.jpg"))
    check("download_image_success", out is True and Path(_TMP / "dl1.jpg").exists())

    get = lambda *a, **k: FakeResp(status=404)
    check("download_image_non200_false", not SE.download_image(FakeCtx(safe_get=get), "https://x.com/img.jpg", str(_TMP / "dl2.jpg")))

    get = lambda *a, **k: FakeResp(content=b"html error page", headers={"Content-Type": "text/html"})
    check("download_image_wrong_content_type_false", not SE.download_image(FakeCtx(safe_get=get), "https://x.com/nofile", str(_TMP / "dl3.jpg")))

    get = lambda *a, **k: FakeResp(content=b"x", headers={"Content-Type": "image/jpeg"})
    check("download_image_too_small_false", not SE.download_image(FakeCtx(safe_get=get), "https://x.com/img.jpg", str(_TMP / "dl4.jpg")))

    tiny_img = _TMP / "tiny.jpg"
    Image.new("RGB", (50, 50), (1, 2, 3)).save(tiny_img, "JPEG")
    get = lambda *a, **k: FakeResp(content=tiny_img.read_bytes())
    check("download_image_too_small_dims_false", not SE.download_image(FakeCtx(safe_get=get), "https://x.com/img.jpg", str(_TMP / "dl5.jpg")))

    get = lambda *a, **k: FakeResp(content=b"not a real image at all but long enough to pass min_bytes" * 200)
    check("download_image_corrupt_image_false", not SE.download_image(FakeCtx(safe_get=get), "https://x.com/img.jpg", str(_TMP / "dl6.jpg")))

    def raiser(*a, **k):
        raise RuntimeError("network boom")
    get = raiser
    check("download_image_exception_false", not SE.download_image(FakeCtx(safe_get=get), "https://x.com/img.jpg", str(_TMP / "dl7.jpg")))

    get = lambda *a, **k: FakeResp(content=good_bytes.read_bytes(), headers={})  # no content-type but url ext ok
    check("download_image_no_content_type_url_ext_ok", SE.download_image(FakeCtx(safe_get=get), "https://x.com/img.jpg", str(_TMP / "dl8.jpg")))


def test_has_clear_face():
    import numpy as np
    from PIL import Image
    noface_path = _TMP / "noface.jpg"
    Image.fromarray((np.random.rand(200, 200, 3) * 255).astype("uint8")).save(noface_path)

    def has(face_box):
        return SE._has_clear_face(str(noface_path), face_box=face_box)
    check("has_clear_face_no_face_false", has(lambda path, pad=0.0: None) is False)
    check("has_clear_face_big_face_true", has(lambda path, pad=0.0: (10, 10, 190, 190)) is True)
    check("has_clear_face_tiny_face_false", has(lambda path, pad=0.0: (0, 0, 5, 5)) is False)

    def raiser(path, pad=0.0):
        raise RuntimeError("boom")
    check("has_clear_face_exception_false", has(raiser) is False)


def test_is_grayscale():
    from PIL import Image
    gray_path = _TMP / "gray.jpg"
    Image.new("RGB", (100, 100), (128, 128, 128)).save(gray_path)
    check("is_grayscale_true_for_gray_image", SE._is_grayscale(str(gray_path)))

    color_path = _TMP / "color.jpg"
    Image.new("RGB", (100, 100), (255, 0, 0)).save(color_path)
    check("is_grayscale_false_for_color_image", not SE._is_grayscale(str(color_path)))

    check("is_grayscale_missing_file_false", not SE._is_grayscale(str(_TMP / "nope.jpg")))


def test_fetch_reference_photo():
    ctx = FakeCtx(image_search_urls=lambda ctx, q, **k: [])
    check("fetch_ref_photo_no_urls_none", SE.fetch_reference_photo(ctx, "q", str(_TMP / "ref1.jpg")) is None)

    ctx = FakeCtx(image_search_urls=lambda ctx, q, **k: ["https://x.com/1"],
                  download_image=lambda ctx, url, dest, **k: False)
    check("fetch_ref_photo_download_fails_none", SE.fetch_reference_photo(ctx, "q", str(_TMP / "ref2.jpg")) is None)

    ctx = FakeCtx(image_search_urls=lambda ctx, q, **k: ["https://x.com/1"],
                  download_image=lambda ctx, url, dest, **k: True,
                  _has_clear_face=lambda p: False)
    check("fetch_ref_photo_no_face_none", SE.fetch_reference_photo(ctx, "q", str(_TMP / "ref3.jpg")) is None)

    ctx = FakeCtx(image_search_urls=lambda ctx, q, **k: ["https://x.com/1"],
                  download_image=lambda ctx, url, dest, **k: True,
                  _has_clear_face=lambda p: True, _is_grayscale=lambda p: False)
    out = SE.fetch_reference_photo(ctx, "q", str(_TMP / "ref4.jpg"))
    check("fetch_ref_photo_color_success", out == str(_TMP / "ref4.jpg"))

    dest = _TMP / "ref5.jpg"
    dest.write_bytes(b"placeholder")
    ctx = FakeCtx(image_search_urls=lambda ctx, q, **k: ["https://x.com/1", "https://x.com/2"],
                  download_image=lambda ctx, url, d, **k: True,
                  _has_clear_face=lambda p: True, _is_grayscale=lambda p: True)
    out2 = SE.fetch_reference_photo(ctx, "q", str(dest))
    check("fetch_ref_photo_grayscale_fallback_used", out2 == str(dest))

    ctx = FakeCtx(image_search_urls=lambda ctx, q, **k: ["https://x.com/1"],
                  download_image=lambda ctx, url, d, **k: True)
    out3 = SE.fetch_reference_photo(ctx, "q", str(_TMP / "ref6.jpg"), require_face=False, prefer_color=False)
    check("fetch_ref_photo_no_gates_accepts_first", out3 == str(_TMP / "ref6.jpg"))


def test_top_results_are_opened_and_read():
    # Snippets are teasers: the top results must be opened, in parallel, and
    # their page text must reach the model instead of the 500-char snippet.
    ddgs = _fake_ddgs(text_results=[
        {"title": f"T{i}", "href": f"https://s{i}.com/a", "body": "teaser"} for i in range(7)])
    seen, lock, peak = [], threading.Lock(), {"now": 0, "max": 0}
    gate = threading.Barrier(5, timeout=60)   # a failsafe, not a timing assumption: -j 6 load broke 5 s
    def page(url):
        with lock:
            seen.append(url); peak["now"] += 1; peak["max"] = max(peak["max"], peak["now"])
        try: gate.wait()
        except threading.BrokenBarrierError: pass
        with lock: peak["now"] -= 1
        return "FULLTEXT " + url + " " + "word " * 50
    ctx = FakeCtx(DDGS=ddgs, _read_page=page)
    out = SE.search_duckduckgo_and_fetch(ctx, "query")
    check("top results opened (with spares)", len(seen) == 7, seen)
    check("opened in parallel", peak["max"] == 5, peak)
    check("page text replaces the snippet", out.count("FULLTEXT") == 5, out[:300])
    check("the rest keep their snippet", "teaser" in out)


def test_refused_pages_are_backfilled_from_further_down():
    ddgs = _fake_ddgs(text_results=[
        {"title": f"T{i}", "href": f"https://s{i}.com/a", "body": "teaser"} for i in range(10)])
    ctx = FakeCtx(DDGS=ddgs, _read_page=lambda u: "" if u[9] in "01" else "PAGE " + u)
    out = SE.search_duckduckgo_and_fetch(ctx, "query")
    check("five pages read despite two refusals", out.count("PAGE ") == 5, out[:400])
    check("taken in rank order", "s6.com" in out.split("PAGE ")[-1] and "PAGE https://s7" not in out)


def test_best_passages_finds_the_answer_deep_in_the_page():
    page = "Главная страница сайта. " + "Меню и реклама разделов. " * 200 + "Курс биткоина сегодня составляет 83 805 долларов. " + "Подвал сайта. " * 50
    got = SE.best_passages(page, "курс биткоина сегодня", 400)
    check("answer sentence kept", "83 805" in got, got)
    check("page opening kept", got.startswith("Главная"), got[:60])
    check("within budget", len(got) <= 420, len(got))
    menu = "Главная Новости Крипто Спорт Экономика Политика Бизнес " * 20
    got = SE.best_passages(menu + ". Курс биткоина сегодня 83 805 долларов. " + "Подвал. " * 300,
                           "курс биткоина", 300)
    check("a run-on menu is not kept as the opening", "Спорт" not in got, got[:80])


def test_news_queries_ask_for_last_week():
    # the model's read is stubbed; the phrases run live in bench/intent_sweep_live.py
    def news(q, t, default=False, wait=True):
        return "recent news" in q and ("новост" in t or "news" in t)
    seen = {}
    ctx = FakeCtx(yes_no=news, raw_search_results=lambda c, q, m=None, **k: seen.update(k) or [])
    SE.search_duckduckgo_and_fetch(ctx, "новости Москвы")
    check("news query limited to a week", seen.get("timelimit") == "w", seen)
    seen.clear()
    ctx = FakeCtx(yes_no=news, raw_search_results=lambda c, q, m=None, **k: seen.update(k) or [])
    SE.search_duckduckgo_and_fetch(ctx, "latest AI news")
    check("english news query limited too", seen.get("timelimit") == "w", seen)


def test_followup_round_fills_a_gap():
    searched = []
    def fetch(ctx, q, max_results=None):
        searched.append(q)
        return "first round" if len(searched) == 1 else "second round about the price"
    distilled = []
    def distill(ctx, q, raw):
        distilled.append(raw)
        return "facts: " + raw
    ctx = FakeCtx(search_duckduckgo_and_fetch=fetch, distill_search_results=distill, SEARCH_DISTILL=True,
                  SEARCH_FOLLOWUP_ROUND=True,
                  call_llm_simple=lambda *a, **k: "iphone 17 price russia")
    out = SE.run_web_search(ctx, "iphone 17 release and price")
    check("the model's follow-up query is searched", searched == ["iphone 17 release and price",
                                                                  "iphone 17 price russia"], searched)
    check("both rounds are distilled together", "first round" in out and "second round" in out, out)
    searched.clear()
    ctx = FakeCtx(search_duckduckgo_and_fetch=fetch, distill_search_results=distill, SEARCH_DISTILL=True,
                  SEARCH_FOLLOWUP_ROUND=True, call_llm_simple=lambda *a, **k: "NONE")
    out = SE.run_web_search(ctx, "q")
    check("NONE means one round only", searched == ["q"] and out == "facts: first round", (searched, out))
    ctx = FakeCtx(call_llm_simple=lambda *a, **k: "Q")
    check("same query is ignored", SE.followup_query(ctx, "q", "x") == "")
    ctx = FakeCtx(call_llm_simple=lambda *a, **k: "кто выиграл чемпионат мира 2022 счёт")
    check("a year the user did not give is dropped",
          SE.followup_query(ctx, "последний чемпионат мира", "x") == "кто выиграл чемпионат мира счёт")


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
    # A failed check() must reach the exit code; `failed` counts exceptions only.
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)
