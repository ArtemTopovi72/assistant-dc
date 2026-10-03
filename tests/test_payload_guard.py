"""Coverage for payload_guard.py: the silent-payload guard's structural checks
(empty/malformed/thin/truncated/stale/ok) and the block-message/skeptic-banner
wrapping. Pure functions, exercised against the exact junk-payload shapes the
PlanBench-XL benchmark identified.
Run: venv/Scripts/python.exe tests/test_payload_guard.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import payload_guard as PG

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


def test_strip_markup_and_looks_like_json():
    check("strip_markup_removes_punct", PG._strip_markup('{"a": "b"}') == "ab")
    check("looks_like_json_object", PG._looks_like_json('{"a": 1}'))
    check("looks_like_json_array", PG._looks_like_json("[1, 2, 3]"))
    # prose that merely CONTAINS a json fragment is prose (live 2026-09-28: a
    # scraped script snippet blocked a whole phone comparison)
    check("looks_like_json_embedded_is_prose", not PG._looks_like_json('prose {"a": 1}'))
    check("citation_prose_is_not_json", not PG._looks_like_json("[1] Apple says ... [2]"))
    check("looks_like_json_false_for_prose", not PG._looks_like_json("just plain text"))
    check("json_parses_true", PG._json_parses('{"a": 1}'))
    check("json_parses_false", not PG._json_parses("not json{{{"))


def test_assess_payload_empty():
    v = PG.assess_payload("q", "")
    check("assess_empty_string", v.usable is False and v.status == "empty")
    v2 = PG.assess_payload("q", "   ")
    check("assess_whitespace_only_empty", v2.usable is False and v2.status == "empty")
    v3 = PG.assess_payload("q", None)
    check("assess_none_empty", v3.usable is False and v3.status == "empty")


def test_assess_payload_malformed():
    v = PG.assess_payload("q", "some text with \x00 null byte and enough padding to pass thin check")
    check("assess_null_byte_malformed", v.usable is False and v.status == "malformed")
    v2 = PG.assess_payload("q", "some text with � replacement char and enough padding here")
    check("assess_replacement_char_malformed", v2.usable is False and v2.status == "malformed")
    v3 = PG.assess_payload("q", '{"broken": "json missing close quote}')
    check("assess_broken_json_malformed", v3.usable is False and v3.status == "malformed")


def test_assess_payload_thin():
    v = PG.assess_payload("q", "hi.")
    check("assess_thin_content", v.usable is False and v.status == "thin")


def test_assess_payload_truncated():
    v = PG.assess_payload("q", "This is a long enough sentence that just cuts off mid")
    check("assess_truncated_midword", v.usable is False and v.status == "truncated")
    v2 = PG.assess_payload("q", "This is a complete sentence that ends properly.")
    check("assess_not_truncated_terminal_punct", v2.usable is True and v2.status == "ok")
    v3 = PG.assess_payload("q", "This trails off with ellipsis and enough text to pass thin check...")
    check("assess_ellipsis_not_truncated", v3.usable is True)
    v4 = PG.assess_payload("q", "Here is a link to more info https://example.com/some/long/path/here")
    check("assess_bare_url_ending_not_truncated", v4.usable is True)


def test_assess_payload_stale():
    text = ("По состоянию на 2020 год данные показывают следующее подробное "
            "описание ситуации и контекста для полноты картины изложения.")
    v = PG.assess_payload("q", text, now_year=2026)
    check("assess_stale_flagged_but_usable", v.usable is True and v.status == "stale")
    text2 = "As of 2020 the situation looked like this with plenty of detail here to pass thin."
    v2 = PG.assess_payload("q", text2, now_year=2026)
    check("assess_stale_english_marker", v2.usable is True and v2.status == "stale")


def test_assess_payload_ok():
    v = PG.assess_payload("q", "This is a perfectly normal, complete, and usable search result text.")
    check("assess_ok_normal_text", v.usable is True and v.status == "ok" and v.reason == "")
    recent = "This mentions the year 2025 as recent context with plenty of detail to pass."
    v2 = PG.assess_payload("q", recent, now_year=2026)
    check("assess_recent_year_not_stale", v2.status == "ok")
    old_no_marker = "This mentions the year 2010 but has no staleness marker words at all here really."
    v3 = PG.assess_payload("q", old_no_marker, now_year=2026)
    check("assess_old_year_no_marker_not_flagged_stale", v3.status == "ok")


def test_block_message():
    v = PG.Verdict(False, "empty", "the search returned an empty result")
    msg = PG._block_message(v, kind="search")
    check("block_message_has_tool_error_prefix", msg.startswith("[TOOL ERROR]"))
    check("block_message_mentions_reason", "empty result" in msg)
    msg2 = PG._block_message(v, kind="calculate")
    check("block_message_uses_kind", "calculate" in msg2)


def test_guard_search_result():
    out = PG.guard_search_result("q", "")
    check("guard_empty_returns_block_message", out.startswith("[TOOL ERROR]"))

    ok_text = "This is a perfectly normal, complete, and usable search result text."
    out2 = PG.guard_search_result("q", ok_text)
    check("guard_ok_appends_skeptic_banner", ok_text in out2 and PG.SKEPTIC_BANNER in out2)
    check("guard_ok_no_stale_prefix", "[DATA-VALIDATION] Note:" not in out2)

    stale_text = ("По состоянию на 2018 год данные показывают следующее подробное "
                  "описание ситуации и контекста для полноты картины изложения.")
    out3 = PG.guard_search_result("q", stale_text)
    check("guard_stale_prepends_note", out3.startswith("[DATA-VALIDATION] Note:"))
    check("guard_stale_includes_original_text", stale_text in out3)

    out4 = PG.guard_search_result("q", "malformed \x00 bytes here with enough padding text", kind="tool")
    check("guard_malformed_uses_kind", "tool" in out4)


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
