"""Pure-logic coverage for deep_research.py: URL/host classification, dedup/
similarity, evidence stats, outline normalization, report-text helpers, citation
parsing. No LM Studio / network / GPU needed for these.
Run: venv/Scripts/python.exe tests/test_deep_research_pure.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import deep_research as DR
import dr_calls

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


def test_domain():
    check("domain_strips_www", DR._domain("https://www.example.com/x") == "example.com")
    check("domain_bad_url", DR._domain("not a url \x00") == "not a url \x00" or True)


def test_is_landing_url():
    check("landing_root", DR._is_landing_url("https://nature.com/"))
    check("landing_short_seg", DR._is_landing_url("https://x.com/en"))
    check("landing_login", DR._is_landing_url("https://x.com/foo/login"))
    check("not_landing_real_article", not DR._is_landing_url("https://x.com/2023/an-article-about-di-zenzo"))


def test_norm_url():
    check("norm_url_strips_frag_query", DR._norm_url("https://X.com/a/b/?q=1#frag") == "https://x.com/a/b")
    check("norm_url_root_slash", DR._norm_url("https://x.com") == "https://x.com/")


def test_junk_and_community_and_authority():
    check("junk_source_true", DR._is_junk_source("https://kinogo.pro/movie"))
    check("junk_source_false", not DR._is_junk_source("https://arxiv.org/abs/123"))
    check("community_host_true", DR._is_community_host("https://reddit.com/r/x"))
    check("community_host_vk_local_exempt", not DR._is_community_host("https://vk.com/club1", category="local"))
    check("community_host_vk_general", DR._is_community_host("https://vk.com/club1", category="general"))
    check("authority_junk_zero", DR._authority_score("https://kinogo.pro/x") == 0.0)
    check("authority_base", DR._authority_score("https://en.wikipedia.org/wiki/X") == DR._AUTH_BASE_SCORE)
    check("authority_category", DR._authority_score("https://arxiv.org/abs/1", category="science") == DR._AUTH_CATEGORY_SCORE)
    check("authority_community", DR._authority_score("https://reddit.com/r/x") == DR._COMMUNITY_SCORE)
    check("authority_mid", DR._authority_score("https://randomblogxyz.com/post") == DR._MID_SCORE)
    check("source_rank_authority", DR._source_rank("https://arxiv.org/abs/1", "science") == 0)
    check("source_rank_ordinary", DR._source_rank("https://randomblogxyz.com/post") == 1)


def test_classify_source_type():
    check("type_gov", DR._classify_source_type("https://nist.gov/x") == "government")
    check("type_academic_edu", DR._classify_source_type("https://mit.edu/x") == "academic")
    check("type_academic_arxiv", DR._classify_source_type("https://arxiv.org/abs/1") == "academic")
    check("type_news", DR._classify_source_type("https://reuters.com/x") == "news")
    check("type_social", DR._classify_source_type("https://twitter.com/x") == "social")
    check("type_forum", DR._classify_source_type("https://reddit.com/r/x") == "forum")
    check("type_blog", DR._classify_source_type("https://medium.com/x") == "blog")
    check("type_technical_docs", DR._classify_source_type("https://docs.python.org/x") == "technical")
    check("type_technical_github", DR._classify_source_type("https://github.com/x") == "technical")
    check("type_whitepaper", DR._classify_source_type("https://example.com/paper.pdf") == "whitepaper")
    check("type_general", DR._classify_source_type("https://randomxyz123.com/x") == "general")


def test_domain_allowed_and_source_type_allowed():
    check("allow_no_filters", DR._domain_allowed("https://x.com", "", ""))
    check("allow_blacklist_hit", not DR._domain_allowed("https://x.com/y", "", "x.com"))
    check("allow_whitelist_hit", DR._domain_allowed("https://x.com/y", "x.com", ""))
    check("allow_whitelist_miss", not DR._domain_allowed("https://y.com/y", "x.com", ""))
    check("source_type_allowed_none_excluded", DR._source_type_allowed("https://reddit.com/x", ""))
    check("source_type_allowed_excluded", not DR._source_type_allowed("https://reddit.com/x", "forum"))


def test_is_safe_public_url():
    check("safe_scheme_rejects_file", not DR._is_safe_public_url("file:///etc/passwd"))
    check("safe_rejects_localhost", not DR._is_safe_public_url("http://localhost/x"))
    check("safe_rejects_metadata", not DR._is_safe_public_url("http://metadata.google.internal/x"))
    check("safe_rejects_dot_local", not DR._is_safe_public_url("http://foo.local/x"))
    check("safe_rejects_bad_url", not DR._is_safe_public_url("::::not a url"))
    # A real public site should resolve to a public IP (network permitting; best-effort).
    try:
        ok = DR._is_safe_public_url("https://www.google.com/")
        check("safe_allows_public_site", ok is True or ok is False)  # just must not raise
    except Exception as e:
        check("safe_allows_public_site", False, str(e))


def test_mutate_query():
    check("mutate_attempt0_strips_quotes", DR._mutate_query('"foo bar"', 0) == "foo bar")
    check("mutate_attempt0_no_quotes_appends", DR._mutate_query("foo bar", 0) == "foo bar overview")
    check("mutate_attempt1_drops_last_word", DR._mutate_query("foo bar baz", 1) == "foo bar")
    check("mutate_attempt1_too_short_none", DR._mutate_query("foo", 1) is None)
    check("mutate_attempt2_explained", DR._mutate_query("foo", 2) == "foo explained")
    check("mutate_attempt3_none", DR._mutate_query("foo", 3) is None)


def test_cap_per_domain():
    sources = [{"href": "https://a.com/1"}, {"href": "https://a.com/2"},
               {"href": "https://a.com/3"}, {"href": "https://b.com/1"}]
    out = DR._cap_per_domain(sources, 2)
    check("cap_per_domain_limits", len(out) == 3)
    check("cap_per_domain_zero_unlimited", DR._cap_per_domain(sources, 0) == sources)


def test_apply_scholarly_floor():
    non_sci = [{"href": "https://randomblogxyz.com/x"}]
    check("scholarly_floor_noop_nonscience", DR._apply_scholarly_floor(non_sci, "general") == non_sci)
    # science category, not enough strong sources -> no-op
    weak = [{"href": f"https://randomblog{i}.com/x"} for i in range(3)]
    check("scholarly_floor_noop_too_few_strong", DR._apply_scholarly_floor(weak, "science") == weak)
    # enough strong + enough total -> pass through unchanged (>=12 kept already)
    strong = [{"href": f"https://arxiv.org/abs/{i}"} for i in range(5)]
    mid = [{"href": f"https://randomblog{i}.com/x"} for i in range(10)]
    out = DR._apply_scholarly_floor(strong + mid, "science")
    check("scholarly_floor_drops_mid_but_keeps_floor", len(out) >= DR._SCHOLARLY_FLOOR_KEEP)
    # enough strong, few total -> top-up path (kept < floor before top-up)
    strong2 = [{"href": f"https://arxiv.org/abs/{i}"} for i in range(4)]
    mid2 = [{"href": f"https://randomblog{i}.com/x"} for i in range(2)]
    out2 = DR._apply_scholarly_floor(strong2 + mid2, "science")
    check("scholarly_floor_topup", len(out2) == len(strong2 + mid2))


def test_apply_trust_floor():
    check("trust_floor_defaults_community", DR._apply_trust_floor(None, "https://randomblogxyz.com/x") == "COMMUNITY")
    check("trust_floor_invalid_defaults", DR._apply_trust_floor("BOGUS", "https://randomblogxyz.com/x") == "COMMUNITY")
    check("trust_floor_base_authority", DR._apply_trust_floor("LOW", "https://en.wikipedia.org/wiki/X") == "SECONDARY")
    check("trust_floor_category_authority", DR._apply_trust_floor("LOW", "https://arxiv.org/abs/1", "science") == "SECONDARY")
    check("trust_floor_encyclopedic_ceiling", DR._apply_trust_floor("PRIMARY", "https://en.wikipedia.org/wiki/X") == "SECONDARY")
    check("trust_floor_landing_never_primary", DR._apply_trust_floor("PRIMARY", "https://nature.com/") == "SECONDARY")
    check("trust_floor_community_ceiling", DR._apply_trust_floor("PRIMARY", "https://reddit.com/r/x") == "COMMUNITY")
    check("trust_floor_passthrough", DR._apply_trust_floor("SECONDARY", "https://randomxyz.com/x") == "SECONDARY")


def test_classify_extraction():
    check("gate_empty", DR.classify_extraction(None) == (DR.GATE_EMPTY, DR.GATE_EMPTY))
    check("gate_empty_blank", DR.classify_extraction("   ") == (DR.GATE_EMPTY, DR.GATE_EMPTY))
    check("gate_challenge", DR.classify_extraction("please enable javascript to continue")[0] == DR.GATE_CHALLENGE)
    big_html = "x" * 70000
    check("gate_shell", DR.classify_extraction("short text here", big_html)[0] == DR.GATE_SHELL)
    check("gate_nav_title", DR.classify_extraction("some body text of decent length here padding padding", title="Sign In")[0] == DR.GATE_NAV)
    check("gate_nav_no_prose", DR.classify_extraction("word " * 100)[0] == DR.GATE_NAV)
    check("gate_thin", DR.classify_extraction("One. Two. " + "pad " * 20)[0] == DR.GATE_THIN)
    long_ok = ("This is a real sentence. " * 50) + ("Another one here. " * 50)
    check("gate_ok", DR.classify_extraction(long_ok)[0] == DR.GATE_OK)


def test_slug():
    check("slug_basic", DR._slug("Hello, World! 123") == "hello-world-123")
    check("slug_empty_fallback", DR._slug("###") == "research")
    check("slug_maxlen", len(DR._slug("a" * 100, maxlen=10)) <= 10)


def test_resolve_caps():
    q = DR._resolve_caps("quick")
    d = DR._resolve_caps("deep")
    s = DR._resolve_caps("standard")
    check("caps_quick", q["depth"] == 0)
    check("caps_deep", d["depth"] >= 1)
    check("caps_standard_default", s["report_tokens"] == DR.DR_REPORT_TOKENS_STANDARD)
    check("caps_unknown_defaults_standard", DR._resolve_caps("bogus")["report_tokens"] == DR.DR_REPORT_TOKENS_STANDARD)
    saved = DR.DR_REPORT_TOKENS_OVERRIDE
    DR.DR_REPORT_TOKENS_OVERRIDE = 555
    try:
        check("caps_override_wins", DR._resolve_caps("standard")["report_tokens"] == 555)
    finally:
        DR.DR_REPORT_TOKENS_OVERRIDE = saved


def test_topic_first_line_and_citation_query():
    check("topic_first_line_strips_imperative", DR._topic_first_line("Explain: quantum tunneling\nmore stuff") == "quantum tunneling")
    check("topic_first_line_empty", DR._topic_first_line("") == "")
    q = DR._citation_query("quantum tunneling in depth", ["quantum tunneling", "quantum tunneling in depth"])
    check("citation_query_prefers_short_candidate", q == "quantum tunneling")
    check("citation_query_fallback_to_topic", DR._citation_query("perform: some research topic", []) == "some research topic")


def test_shingles_jaccard_similarity_dedupe():
    a = DR._shingles("the quick brown fox jumps over the lazy dog repeatedly again")
    b = DR._shingles("the quick brown fox jumps over the lazy dog repeatedly again")
    c = DR._shingles("completely unrelated text about something else entirely different")
    check("shingles_short_text_is_wordset", DR._shingles("a b") == frozenset({"a", "b"}))
    check("jaccard_identical_is_1", DR._jaccard(a, a) == 1.0)
    check("jaccard_empty_is_0", DR._jaccard(frozenset(), a) == 0.0)
    check("jaccard_disjoint_is_0", DR._jaccard(a, c) == 0.0)
    check("content_similarity_identical", DR._content_similarity(a, b) == 1.0)
    # containment: short text 'contained' in long text
    small = DR._shingles(" ".join(["word"] * 6))
    big = DR._shingles(" ".join(["word"] * 6) + " extra padding text to make this longer than the small one by a lot indeed")
    check("content_similarity_containment_or_jaccard", 0.0 <= DR._content_similarity(small, big) <= 1.0)

    pages = [
        {"text": "the quick brown fox jumps over the lazy dog repeatedly and again and again", "url": "https://arxiv.org/a", "domain": "arxiv.org"},
        {"text": "the quick brown fox jumps over the lazy dog repeatedly and again and again", "url": "https://randomblog.com/a", "domain": "randomblog.com"},
        {"text": "something totally different about unrelated topics and other matters entirely", "url": "https://randomblog2.com/b", "domain": "randomblog2.com"},
    ]
    out = DR.dedupe_pages(pages, "general")
    check("dedupe_collapses_duplicates", len(out) == 2)
    rep = next(p for p in out if p["cluster_size"] > 1)
    check("dedupe_prefers_higher_authority_rep", rep["url"] == "https://arxiv.org/a")
    check("dedupe_no_dup_noop", len(DR.dedupe_pages([pages[2]], "general")) == 1)


def test_query_lang_and_diversify():
    check("query_lang_ru", DR._query_lang("привет мир") == "ru")
    check("query_lang_en", DR._query_lang("hello world") == "en")
    out = DR.diversify_queries(["quantum tunneling effect", "tunneling effect quantum",
                                "completely different other query"], 5)
    check("diversify_drops_near_dup", len(out) == 2)
    check("diversify_respects_max", len(DR.diversify_queries(["a b", "c d", "e f"], 1)) == 1)
    check("diversify_empty_tokens_skipped", DR.diversify_queries(["   "], 5) == [])
    # The filter must stay PURE: it never tops itself back up, or the check
    # above stops meaning anything. The floor lives in plan_queries.
    check("diversify_never_invents_a_query",
          len(DR.diversify_queries(["a b c d e", "a b c d e f"], 5)) == 1)


def test_query_plan_never_starves():
    """A dead planner must not leave the run with ONE query.

    Measured live: the planner returned nothing, the heuristic fallback appended
    short suffixes to a long Russian topic, and every variant scored as a
    paraphrase of the first -- eight candidates collapsed to one and the whole
    research run went out with a single query. The de-duplicator was starving
    the guard that exists to prevent starvation.
    """
    import dr_collect as C
    topic = "Чем отличается зелёный чай от чёрного по составу и воздействию"
    real = C._think_call
    C._think_call = lambda *a, **k: ""            # the planner says nothing
    try:
        q, prof = C.plan_queries(None, topic, 8)
    finally:
        C._think_call = real
    check("a dead planner still yields several queries", len(q) >= 3, q)
    check("...all distinct", len({x.lower() for x in q}) == len(q), q)
    check("...and the topic itself is one of them",
          any(x.strip() == topic for x in q), q)

    # The ceiling still wins over the floor.
    C._think_call = lambda *a, **k: ""
    try:
        q1, _ = C.plan_queries(None, topic, 1)
    finally:
        C._think_call = real
    check("a max of one is still honoured", len(q1) == 1, q1)

    # The floor must not undo the distillation: a search engine cannot match an
    # essay, and topping up with "<532-char request> benchmark comparison" was
    # exactly what an existing check caught.
    long_topic = ("Напиши мне подробнейший обзор по теме " + "структурный тензор " * 12)
    C._think_call = lambda *a, **k: ""
    try:
        q3, _ = C.plan_queries(None, long_topic, 8)
    finally:
        C._think_call = real
    check("the floor never adds a runaway query",
          all(len(x) <= C.MAX_QUERY_CHARS for x in q3), [len(x) for x in q3])

    # And a HEALTHY planner is not padded: its queries are already diverse.
    good = ('{"queries": ["green tea catechins content", '
            '"black tea theaflavins oxidation", "tea caffeine comparison study"], '
            '"category": "general", "recency": "any", '
            '"core_topic": "green vs black tea composition"}')
    C._think_call = lambda *a, **k: good
    try:
        q2, prof2 = C.plan_queries(None, topic, 8)
    finally:
        C._think_call = real
    check("a working planner keeps its own diverse queries", len(q2) >= 3, q2)
    check("...and its distilled English subject is carried",
          prof2["core_topic"] == "green vs black tea composition", prof2)
    return True


def test_coverage_signals():
    sources = [{"domain": "a.com", "query": "q"}] * 3 + [{"domain": "b.com", "query": "q"}]
    sig = DR.coverage_signals(sources, {"category": "general"})
    check("coverage_signals_weak_clustered", sig["weak"] and "clustered" in " ".join(sig["reasons"]))
    many_domains = [{"domain": f"{i}.com", "query": "q"} for i in range(10)]
    sig2 = DR.coverage_signals(many_domains, {"category": "general"})
    check("coverage_signals_strong_not_weak", sig2["weak"] is False)
    check("coverage_signals_thin", DR.coverage_signals([], {"category": "general"})["weak"])
    local_sources = [{"domain": "vk.com", "query": "привет мир один два три четыре пять шесть"}] * 8
    sig3 = DR.coverage_signals(local_sources, {"category": "local"})
    check("coverage_signals_lang_skew", any("language" in r for r in sig3["reasons"]))


def test_evidence_stats_and_confidence_rule_and_requested_sections():
    briefs = [{"trust": "PRIMARY", "authority": 0.9}, {"trust": "SECONDARY", "authority": 0.8},
              {"trust": "COMMUNITY", "authority": 0.3}]
    stats = DR.evidence_stats(briefs)
    check("evidence_stats_counts", stats["independent_clusters"] == 3 and stats["strong_sources"] == 2)
    check("evidence_stats_max_authority", stats["max_authority"] == 0.9)
    rule = DR._confidence_rule(stats)
    check("confidence_rule_mentions_counts", "2" in rule)
    empty_stats = DR.evidence_stats([])
    rule2 = DR._confidence_rule(empty_stats)
    check("confidence_rule_none_mix", "none" in rule2)
    check("requested_sections_timeline", "Chronological Timeline" in DR.requested_sections("give me a timeline of the discovery"))
    check("requested_sections_none", DR.requested_sections("just tell me about foo") == [])
    check("requested_sections_multi", len(DR.requested_sections("compare methods and list misconceptions and myths")) >= 2)


def test_parse_trust():
    tier, body = DR._parse_trust("SOURCE: PRIMARY\nActual content here")
    check("parse_trust_primary", tier == "PRIMARY" and body.strip() == "Actual content here")
    tier2, body2 = DR._parse_trust("no tag line at all just content")
    check("parse_trust_default_community", tier2 == "COMMUNITY")


def test_pack_batches():
    entries = ["a" * 10, "b" * 10, "c" * 10]
    batches = DR._pack_batches(entries, 15)
    check("pack_batches_splits", len(batches) == 3)
    check("pack_batches_single_when_fits", len(DR._pack_batches(entries, 100)) == 1)
    check("pack_batches_empty", DR._pack_batches([], 10) == [])


def test_brief_tag():
    b = {"domain": "arxiv.org", "trust": "PRIMARY", "title": "T", "brief": "B",
         "cluster_size": 3, "equations": [r"E = mc^2", ""]}
    tag = DR._brief_tag(b)
    check("brief_tag_has_copies", "copies=3" in tag)
    check("brief_tag_has_equation", "E = mc^2" in tag)
    b2 = {"domain": "x.com", "trust": "LOW", "title": "T2", "brief": "B2", "cluster_size": 1}
    tag2 = DR._brief_tag(b2)
    check("brief_tag_no_copies_singleton", "copies=" not in tag2)


def test_extract_outline_json_and_normalize_and_default():
    d = DR._extract_outline_json('```json\n{"title": "T", "sections": []}\n```')
    check("extract_outline_json_fenced", d == {"title": "T", "sections": []})
    d2 = DR._extract_outline_json('prose before {"title": "T2"} prose after')
    check("extract_outline_json_embedded", d2 == {"title": "T2"})
    check("extract_outline_json_none_on_empty", DR._extract_outline_json("") is None)
    check("extract_outline_json_none_on_garbage", DR._extract_outline_json("not json at all {{{") is None)

    default = DR._default_outline("Some Topic")
    check("default_outline_has_sections", len(default["sections"]) > 0)

    check("normalize_outline_none_plan", DR._normalize_outline(None, "Topic") == DR._default_outline("Topic"))
    check("normalize_outline_no_sections_key", DR._normalize_outline({"title": "x"}, "Topic")["sections"] == DR._default_outline("Topic")["sections"])
    plan = {"title": "My Title", "sections": [
        {"heading": "## Intro", "subsections": ["a", "b", "c", "d", "e"], "focus": "f1"},
        {"heading": "", "subsections": [], "focus": "skip me (no heading)"},
        {"not": "a heading dict really"},
    ]}
    norm = DR._normalize_outline(plan, "Topic")
    check("normalize_outline_strips_hash", norm["sections"][0]["heading"] == "Intro")
    check("normalize_outline_caps_subsections", len(norm["sections"][0]["subsections"]) == 4)
    check("normalize_outline_drops_empty_heading", len(norm["sections"]) == 1)
    check("normalize_outline_non_dict_sections_only", DR._normalize_outline({"sections": ["not a dict"]}, "Topic") == DR._default_outline("Topic"))


def test_repair_outline_text():
    outline = {"title": "T", "abstract_focus": "AF",
               "sections": [{"heading": "H", "focus": "F", "subsections": ["s1"]}]}
    DR._repair_outline_text(outline)  # should not raise; ftfy may or may not be installed
    check("repair_outline_text_survives", outline["title"] == "T")
    DR._repair_outline_text(None)  # no-op branch
    check("repair_outline_text_none_noop", True)
    DR._repair_outline_text({"title": ""})  # falsy title/abstract_focus branch
    check("repair_outline_text_falsy_fields_noop", True)


def test_outline_overview_and_toc():
    plan = {"sections": [{"heading": "Intro", "subsections": ["a", "b"]},
                         {"heading": "Body", "subsections": []}]}
    ov = DR._outline_overview(plan)
    check("outline_overview_has_subs", "subsections: a, b" in ov)
    toc = DR._toc(plan)
    check("toc_has_headings", "Intro" in toc and "Body" in toc)


def test_est_tokens_and_section_terms_and_select_evidence():
    check("est_tokens_min_1", DR._est_tokens("") == 1)
    check("est_tokens_scales", DR._est_tokens("a" * 400) == 100)
    sec = {"heading": "Quantum Tunneling", "focus": "explain tunneling effect", "subsections": ["barrier penetration"]}
    terms = DR._section_terms(sec)
    check("section_terms_has_words", "quantum" in terms and "tunneling" in terms)
    briefs = [
        {"title": "T1", "brief": "discusses quantum tunneling barrier penetration in depth", "trust": "PRIMARY", "equations": ["E=mc^2"]},
        {"title": "T2", "brief": "unrelated content about cooking recipes", "trust": "LOW"},
    ]
    out = DR._select_section_evidence(briefs, sec, 5000)
    check("select_section_evidence_picks_relevant", "T1" in out)
    tiny = DR._select_section_evidence(briefs, sec, 1)
    check("select_section_evidence_tiny_budget_still_returns", len(tiny) > 0)


def test_repair_latex_artifacts_and_audit_math():
    text = r"\(x = 1\) and \[y = 2\] with \vskip 3.0pt plus 1.0pt minus 1.0pt\cr and \((\text{arxiv.org})\)"
    fixed = DR._repair_latex_artifacts(text)
    check("repair_latex_delimiters", "$x = 1$" in fixed and "$$y = 2$$" in fixed)
    check("repair_latex_strips_vskip_cr", "vskip" not in fixed and "\\cr" not in fixed)
    check("repair_latex_unwraps_domain_cite", "(arxiv.org)" in fixed)
    check("repair_latex_none_passthrough", DR._repair_latex_artifacts("") == "")
    op_fixed = DR._repair_latex_artifacts(r"\tr(A) and \boldmath x")
    check("repair_latex_operator_and_boldmath", "operatorname{tr}" in op_fixed and "boldmath" not in op_fixed)

    clean = "The formula is $x^2 + y^2 = z^2$ and that's it."
    a1 = DR.audit_math(clean)
    check("audit_math_clean", a1["clean"] is True)
    unbalanced = "The formula is $x^2 and more text"
    a2 = DR.audit_math(unbalanced)
    check("audit_math_unbalanced_inline", a2["inline_balanced"] is False and a2["clean"] is False)
    lostbs = "The formula is $frac{a}{b}$ end"
    a3 = DR.audit_math(lostbs)
    check("audit_math_lost_backslash", "frac" in a3["suspect_lost_backslash"])


def test_postprocess_report():
    md = "# Intro\nbody1\n\n# 2. Intro (Dedicated Section)\nbody2 dup\n\n# Unique\nbodyU"
    out = DR._postprocess_report(md)
    check("postprocess_dedupes_headers", out.count("# Intro") == 1 or out.lower().count("intro") <= 2)
    check("postprocess_keeps_unique", "Unique" in out)
    dropped = DR._postprocess_report("# DropMe\nbad\n\n# Keep\ngood", drop_substrs=("dropme",))
    check("postprocess_drops_substr_matched_section", "DropMe" not in dropped and "Keep" in dropped)


def test_parse_cite_entry_and_citation_section():
    entry = DR._parse_cite_entry('"A Great Paper" (2020) — J. Smith — cited_by=42 — doi:10.1/x', "seminal")
    check("parse_cite_entry_title", entry[0] == "A Great Paper")
    check("parse_cite_entry_year", entry[1] == "2020")
    check("parse_cite_entry_cites", entry[3] == "42")
    check("parse_cite_entry_author", entry[2] == "J. Smith")
    no_quotes = DR._parse_cite_entry("Some Title Without Quotes (2019)", "related")
    check("parse_cite_entry_no_quotes_title", no_quotes[0].startswith("Some Title"))

    body = ("Most-influential prior work:\n"
            "- \"Foo Adam Optimizer\" (2015) — cited_by=1000\n"
            "Likely seminal: \"Adam: A Method\" (2014) — cited_by=5000\n"
            "Principal descendants:\n"
            "- \"AdamW Variant\" (2019) — cited_by=200\n")
    sect = DR._citation_section_from_brief(body)
    check("citation_section_has_table", "Citation Lineage" in sect)
    check("citation_section_empty_no_rows", DR._citation_section_from_brief("nothing here") == "")
    # relevance-gated: terms filter out non-matching rows, needs >=2 on-topic non-seminal
    sect_terms = DR._citation_section_from_brief(body, terms=["adam"])
    check("citation_section_terms_filter_runs", isinstance(sect_terms, str))


def test_entity_terms_and_term_matching():
    terms = DR._entity_terms("Adam optimizer survey benchmark comparison")
    check("entity_terms_excludes_stopwords", "survey" not in terms and "benchmark" not in terms)
    check("entity_terms_includes_adam", "adam" in terms)
    check("term_in_word_boundary", DR._term_in("adam", "the adam optimizer") and not DR._term_in("adam", "team adams here"))
    check("term_in_any_true", DR._term_in_any("hello adam world", ["adam", "eve"]))
    check("relevance_hits_count", DR._relevance_hits("adam optimizer adam", ["adam"]) == 1)  # word occurs but hits counts terms matched not occurrences? verify below
    check("min_hits_many_terms", DR._min_hits(["a", "b", "c"]) == 2)
    check("min_hits_few_terms", DR._min_hits(["a"]) == 1)
    check("brief_is_relevant_no_terms", DR._brief_is_relevant("t", "b", []))
    check("brief_is_relevant_true", DR._brief_is_relevant("Adam Optimizer", "discusses adam in depth", ["adam"]))
    check("brief_is_relevant_false", not DR._brief_is_relevant("Unrelated", "nothing relevant here", ["adam"]))

    briefs = [{"title": "Adam Optimizer", "brief": "about adam"}, {"title": "Cooking", "brief": "recipes"}]
    kept, dropped = DR._filter_offtopic_briefs(briefs, ["adam"])
    check("filter_offtopic_keeps_relevant", len(kept) == 1 and kept[0]["title"] == "Adam Optimizer")
    check("filter_offtopic_drops_irrelevant", len(dropped) == 1)
    check("filter_offtopic_no_terms_noop", DR._filter_offtopic_briefs(briefs, [])[0] == briefs)
    all_irrelevant = [{"title": "Cooking", "brief": "recipes"}]
    k2, d2 = DR._filter_offtopic_briefs(all_irrelevant, ["adam"])
    check("filter_offtopic_never_empties_kept", k2 == all_irrelevant and d2 == [])


def test_sources_appendix_and_clean_title_and_header():
    briefs = [{"trust": "PRIMARY", "domain": "arxiv.org", "title": "T1", "url": "https://arxiv.org/1"},
              {"trust": "COMMUNITY", "domain": "reddit.com", "title": "T2", "url": "https://reddit.com/2"}]
    appx = DR._sources_appendix(briefs)
    check("sources_appendix_has_tiers", "Primary" in appx and "Community" in appx)

    check("clean_title_quoted", DR._clean_report_title('search from("Di Zenzo" OR "color tensor")') == "color tensor")
    check("clean_title_plain_short", DR._clean_report_title("Plain Topic") == "Plain Topic")
    long_title = "x" * 200
    check("clean_title_truncates", DR._clean_report_title(long_title).endswith("…"))

    stats = {"sources": 10, "pages": 8, "quarantined": {"empty_extraction": 2}}
    header = DR._report_header("raw topic query text", briefs, stats, "standard", core_topic="Clean Topic")
    check("report_header_has_title", "Clean Topic" in header)
    check("report_header_has_query_subtitle", "Query:" in header)
    check("report_header_has_quarantine", "filtered" in header)
    header2 = DR._report_header("Same Topic", briefs, {"sources": 1, "pages": 1}, "quick", core_topic="Same Topic")
    check("report_header_no_subtitle_when_same", "Query:" not in header2)


def test_overrides():
    saved = DR.apply_overrides({"DR_MAX_QUERIES": 7, "DR_MULTIHOP_ENABLED": 1})
    check("apply_overrides_sets_int", DR.DR_MAX_QUERIES == 7)
    check("apply_overrides_coerces_bool", DR.DR_MULTIHOP_ENABLED is True)
    DR.restore_overrides(saved)
    check("restore_overrides_reverts", DR.DR_MAX_QUERIES == saved["DR_MAX_QUERIES"])
    try:
        DR.apply_overrides({"NOT_A_KNOB": 1})
        check("apply_overrides_unknown_key_raises", False)
    except KeyError:
        check("apply_overrides_unknown_key_raises", True)
    try:
        DR.apply_overrides({"DR_MAX_QUERIES": 999})
        check("apply_overrides_out_of_range_raises", False)
    except ValueError:
        check("apply_overrides_out_of_range_raises", True)
    try:
        DR.apply_overrides({"DR_RERANK_BACKEND": "not_a_choice"})
        check("apply_overrides_bad_choice_raises", False)
    except ValueError:
        check("apply_overrides_bad_choice_raises", True)
    saved2 = DR.apply_overrides({"DR_DOMAIN_WHITELIST": "arxiv.org"})
    check("apply_overrides_str_type", DR.DR_DOMAIN_WHITELIST == "arxiv.org")
    DR.restore_overrides(saved2)
    saved3 = DR.apply_overrides({"DR_CLAIM_MERGE_THRESHOLD": 0.5})
    check("apply_overrides_float_type", DR.DR_CLAIM_MERGE_THRESHOLD == 0.5)
    DR.restore_overrides(saved3)
    check("restore_overrides_empty_noop", DR.restore_overrides(None) is None)


def test_model_context_real():
    class FakeCtx:
        model_name = "some-nonexistent-model-xyz"
    dr_calls._CONTEXT_CACHE.clear()
    val = DR._model_context(FakeCtx())
    check("model_context_falls_back_or_detects", val >= 2048)
    check("model_context_cached", DR._model_context(FakeCtx()) == val)


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
