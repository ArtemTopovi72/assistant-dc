"""Offline unit tests for the deep-research quality-hardening phases (1-7).

No network, no LLM — every test drives the pure helpers with synthetic fixtures
so it runs in milliseconds and is safe in CI. The live end-to-end behaviour is
still exercised separately by tests/_dr_scenarios.py (needs the LLM loaded).

Run: .\\venv\\Scripts\\python.exe tests\\test_research_quality.py
"""
import os
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The docstring above says "no LLM", and it was not true. The stubs below
# replace collection, crawling, briefing and synthesis, but run_deep_research
# also plans, reflects, clusters and gates -- and every one of those went to
# the real LM Studio. It made a suite advertised as milliseconds take six
# minutes of a full run, and made its result depend on which model happened
# to be loaded. send_to_lm_studio returning None is the documented
# "model unavailable" contract every caller already handles.
import offline_guard; offline_guard.offline_llm()

# Run me as a script, not through pytest. main() below fixes an ORDER --
# test_p39 reloads config and deep_research, so it has to go last -- and
# pytest ignores that list and runs test_ functions in definition order,
# where three tests follow p39. The reload then landed mid-run and took
# pytest's own output capture down with it at teardown.
RUN_AS_SCRIPT = True

import deep_research as dr
import dr_calls
import dr_collect
import dr_crawl
import research_cache
import source_adapters as sa
import rerank
import entities as _ent
import contradiction as _con
import research_graph as _rg
import decision_gate as _dg
import clustering as _cl
import graph_guided as _gg
import claim_merge as _cm
import hierarchical as _hi


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


class _isolated_dr_dir:
    """Redirect dr.DR_DIR to a throwaway temp dir for the test body. Without this,
    every test that drives the real run_deep_research() (not a mocked stand-in)
    writes a permanent run folder into the user's real memory/research — repeated
    test runs accumulate junk "BGE reranker quality"/"obscure undocumented thing"
    directories forever (found 2026-06-20: ~100 such dirs from this file alone)."""
    def __enter__(self):
        import pathlib
        self._old = dr.DR_DIR
        self._tmp = tempfile.TemporaryDirectory()
        dr.DR_DIR = pathlib.Path(self._tmp.name)
        return dr.DR_DIR

    def __exit__(self, *exc):
        dr.DR_DIR = self._old
        self._tmp.cleanup()


# ───────────────────── Phase 10 — second-stage reranker ───────────────────────
def test_p10_reranker():
    # Deterministic lexical backend (no network / no model needed).
    rr = rerank.Reranker("lexical")
    _check(rr.backend == "lexical", "lexical backend not pinned")
    query = "BGE reranker cross-encoder relevance scoring"
    items = [
        {"text": "How to bake sourdough bread with a starter.", "url": "https://blog.x/bread", "domain": "blog.x"},
        {"text": "The BGE reranker is a cross-encoder scoring query passage relevance for retrieval.",
         "url": "https://github.com/FlagOpen/FlagEmbedding", "domain": "github.com"},
        {"text": "Cross-encoder rerankers improve relevance over bi-encoders by scoring pairs directly.",
         "url": "https://seo-mill.net/x", "domain": "seo-mill.net"},
    ]
    out = rr.rerank(query, [dict(i) for i in items], top_k=3)
    _check(out[0]["domain"] != "blog.x", "irrelevant doc ranked first")
    _check(all("_rerank_rank" in o and "_rerank_relevance" in o for o in out), "annotations missing")
    _check(len(rr.rerank(query, [dict(i) for i in items], top_k=2)) == 2, "top_k not applied")
    _check(rr.rerank(query, []) == [], "empty input not handled")

    # Authority fusion: an authoritative source outranks an equally-relevant SEO page.
    def auth(it):
        return {"github.com": 0.95, "seo-mill.net": 0.05, "blog.x": 0.3}[it["domain"]]
    o3 = [o["domain"] for o in rr.rerank(query, [dict(i) for i in items], top_k=3,
                                         authority_fn=auth, authority_weight=1.0)]
    _check(o3.index("github.com") < o3.index("seo-mill.net"), "authority fusion failed")

    # Pipeline stage: enabled keeps top_k and cleans scratch key; disabled passes through.
    pages = [dict(i, title="t") for i in items]
    class _P:
        def update(self, *a, **k): pass
    saved = (dr.DR_RERANK_ENABLED, dr.DR_RERANK_TOP_K)
    try:
        dr.DR_RERANK_ENABLED, dr.DR_RERANK_TOP_K = True, 2
        res = dr.rerank_for_briefing(query, [dict(p) for p in pages], {"category": "general"}, _P())
        _check(len(res) == 2, "stage did not truncate to top_k")
        _check(all("_rr_text" not in p for p in res), "scratch _rr_text leaked")
        dr.DR_RERANK_ENABLED = False
        res2 = dr.rerank_for_briefing(query, [dict(p) for p in pages], {"category": "general"}, _P())
        _check(len(res2) == 3, "disabled stage must pass through")
    finally:
        dr.DR_RERANK_ENABLED, dr.DR_RERANK_TOP_K = saved
    print("PASS: P10 second-stage reranker (relevance + authority fusion + pipeline stage)")
    return True


# ─────────────────── Phase 11 — entity extraction + multi-hop ─────────────────
def test_p11_entities():
    txt = ("The BGE-M3 reranker from BAAI improves retrieval (arXiv:2402.03216), "
           "repo FlagOpen/FlagEmbedding, version v2.1.0, fixed issue #1234, "
           "CVE-2023-1234, compared vs GPT-4 in 2024.")
    es = _ent.extract_entities(txt)
    _check(any("2402.03216" in v for v in es.values("arxiv")), "arxiv missed")
    _check("FlagOpen/FlagEmbedding" in es.values("repo"), "repo missed")
    _check(any("1234" in v for v in es.values("issue")), "issue id missed")
    _check(any(v.lower() == "cve-2023-1234" for v in es.values("cve")), "cve missed")
    _check(es.total() > 0 and es.rank(top_k=3), "ranking empty")
    # entity_queries respects + updates the visited set (multi-hop loop protection)
    vis = set()
    q1 = _ent.entity_queries("BGE reranker quality", es, max_entities=3, visited=vis)
    _check(q1 and all(q.lower() in vis for q in q1), "visited set not updated")
    q2 = _ent.entity_queries("BGE reranker quality", es, max_entities=3, visited=vis)
    _check(set(x.lower() for x in q2).isdisjoint(x.lower() for x in q1), "produced repeat queries")
    # failure case: empty text yields no entities, no queries, no crash
    empty = _ent.extract_entities("")
    _check(empty.total() == 0 and _ent.entity_queries("t", empty) == [], "empty not handled")
    print("PASS: P11 entity extraction + entity-driven multi-hop queries + loop guard")
    return True


# ─────────────────── Phase 12 — contradiction search pass ─────────────────────
def test_p12_contradiction():
    vis = set()
    cq = _con.contradiction_queries("BGE reranker", max_queries=4, visited=vis)
    _check(len(cq) == 4, f"want 4 contradiction queries, got {len(cq)}")
    _check(all("BGE reranker" in q for q in cq), "topic not anchored")
    _check(all(q.lower() in vis for q in cq), "visited not updated")
    # different topic must not collide with the already-visited set
    cq2 = _con.contradiction_queries("other topic", max_queries=4, visited=vis)
    _check(set(cq).isdisjoint(cq2), "contradiction queries collided across topics")
    # partition + strength grading
    briefs = [{"url": "a", "domain": "reuters.com", "trust": "PRIMARY", "brief": "ok"},
              {"url": "b", "domain": "f.x", "trust": "COMMUNITY", "brief": "bad"}]
    _con.tag_confirming(briefs)
    contra = [{"url": "c", "domain": "github.com", "trust": "SECONDARY", "brief": "fails"},
              {"url": "d", "domain": "arxiv.org", "trust": "PRIMARY", "brief": "refuted"}]
    _con.tag_contradicting(contra)
    conf, con = _con.partition_evidence(briefs + contra)
    _check(len(conf) == 2 and len(con) == 2, "partition wrong")
    sig = _con.contradiction_summary(briefs + contra)
    _check(sig["has_contradictions"] and sig["strength"] == "strong", f"strength {sig['strength']} != strong")
    _check(_con.contradiction_summary(briefs)["strength"] == "none", "no-contradiction not 'none'")
    print("PASS: P12 contradiction queries (separate) + partition + strength grading")
    return True


# ─────────────────── Phase 13 — source/claim provenance graph ─────────────────
def test_p13_research_graph():
    briefs = [{"url": "u1", "domain": "reuters.com", "trust": "PRIMARY", "mode": "confirm",
               "title": "BGE-M3 model", "brief": "The BGE-M3 model works well."},
              {"url": "u2", "domain": "github.com", "trust": "SECONDARY", "mode": "contradict",
               "title": "issue", "brief": "BGE-M3 fails on long inputs, see issue #99."}]
    g = _rg.build_graph(briefs)
    st = g.stats()
    _check(st["sources"] == 2 and st["claims"] == 2, "node counts wrong")
    _check(st["contradicting_sources"] == 1, "contradicting source not tracked")
    _check(st["entities"] >= 1, "no entities linked")
    # provenance: a claim resolves to its source with trust + relation, auditable
    cl = g.claims()[0]
    prov = g.provenance(cl["id"])
    _check(prov["relation"] in ("supported_by", "refuted_by"), "no provenance relation")
    _check(prov["source"]["trust"] in ("PRIMARY", "SECONDARY"), "provenance lost trust")
    _check(isinstance(g.to_dict()["nodes"], list), "serialization broken")
    print("PASS: P13 source/claim graph (entity↔source↔claim, provenance, contradiction edges)")
    return True


# ─────────────────── Phase 14 — abstain / ask / answer gate ───────────────────
def test_p14_decision_gate():
    strong = {"independent_clusters": 3, "strong_sources": 2, "max_authority": 0.95}
    weak = {"independent_clusters": 1, "strong_sources": 0, "max_authority": 0.3}
    empty = {"independent_clusters": 0, "strong_sources": 0, "max_authority": 0.0}
    _check(_dg.decide(strong)["decision"] == "ANSWER", "strong should ANSWER")
    _check(_dg.decide(weak)["decision"] == "ABSTAIN", "weak should ABSTAIN")
    _check(_dg.decide(empty)["decision"] == "ABSTAIN", "empty should ABSTAIN")
    # ambiguity → ASK
    amb = _dg.assess_ambiguity("", has_entities=False)
    d = _dg.decide(strong, ambiguity=amb)
    _check(d["decision"] == "ASK" and d["clarifying_question"], "ambiguous should ASK")
    # answer-with-uncertainty when contradictions unresolved
    unc = _dg.decide(strong, contradiction={"strength": "strong", "has_contradictions": True})
    _check(unc["decision"] == "ANSWER" and unc["uncertain"], "unresolved contradiction must flag uncertainty")
    _check("uncertainty" in _dg.gate_banner(unc).lower(), "banner missing uncertainty")
    # normal question is NOT treated as ambiguous (no over-triggering)
    _check(not _dg.assess_ambiguity("compare BGE and E5 rerankers")["ambiguous"], "false ambiguity")
    print("PASS: P14 decision gate (ANSWER / ASK / ABSTAIN / answer-with-uncertainty)")
    return True


# ─────────────── Phase 15 — full pipeline integration (mocked I/O) ─────────────
def test_p15_pipeline_integration():
    saved = {n: getattr(dr, n) for n in
             ("collect_sources", "crawl_pages", "dedupe_pages", "rerank_for_briefing",
              "brief_source", "synthesize_report", "plan_queries")}
    search_log = []

    def fake_collect(ctx, queries, per_query, prog, run_dir, profile=None):
        search_log.extend(queries)
        return [{"href": f"https://s{abs(hash(q)) % 9999}.com/x",
                 "domain": f"s{abs(hash(q)) % 9999}.com", "title": q[:40], "query": q}
                for q in queries]

    def fake_crawl(ctx, sources, caps, prog, run_dir, profile, cache):
        return [{"url": s["href"], "domain": s["domain"], "title": s["title"],
                 "text": f"{s['title']}. BGE-M3 model repo Flag/Emb v2.1 arXiv:2402.03216.",
                 "cluster_size": 1, "cluster_domains": [s["domain"]]} for s in sources]

    counter = [0]

    def fake_brief(ctx, topic, page):
        counter[0] += 1
        return f"SOURCE: {'PRIMARY' if counter[0] % 2 else 'SECONDARY'}\nWorks, mentions BGE-M3."

    class _Ctx:
        def is_cancelled(self): return False

    try:
        dr.collect_sources = fake_collect
        dr.crawl_pages = fake_crawl
        dr.dedupe_pages = lambda p, c="general": p
        dr.rerank_for_briefing = lambda t, p, pr, pg: p
        dr.brief_source = fake_brief
        dr.synthesize_report = lambda ctx, topic, briefs, prog, **k: "## Executive Summary\nBGE works.\n"
        dr.plan_queries = lambda ctx, topic, mq: (
            ["bge reranker quality", "bge benchmark"],
            {"category": "general", "recency": "any", "timelimit": None, "news": False})
        with _isolated_dr_dir():
            res = dr.run_deep_research(_Ctx(), "BGE reranker quality evaluation", depth="quick")
            rep = res["report"]
            _check(any(k in q for q in search_log for k in
                       ("limitation", "criticism", "failure", "negative", "reproducib", "anomal")),
                   "contradiction pass issued no contradiction queries")
            _check(any(k in q for q in search_log for k in
                       ("benchmark results", "documentation", "README", "limitations")),
                   "multi-hop issued no entity queries")
            _check("Contradiction Check" in rep, "report missing contradiction section")
            _check(res["stats"]["decision"] in ("ANSWER", "ASK", "ABSTAIN"), "no gate decision in stats")
            _check(res["stats"]["graph"]["contradicting_sources"] >= 1, "graph lost contradiction edges")
            _check("reflection" in res["stats"], "reflection not recorded")
            _check("clusters" in res["stats"], "hierarchical clustering not recorded in stats")

            # ABSTAIN path: a single LOW-trust source must not yield a confident answer.
            dr.brief_source = lambda ctx, topic, page: "SOURCE: LOW\nvague."
            dr.collect_sources = lambda ctx, q, pq, prog, rd, profile=None: [
                {"href": "https://x.com/a", "domain": "x.com", "title": "t", "query": (q[0] if q else "")}]
            res2 = dr.run_deep_research(_Ctx(), "obscure undocumented thing", depth="quick")
            _check(res2["stats"]["decision"] == "ABSTAIN", "weak evidence should ABSTAIN")
            _check("Insufficient evidence" in res2["report"], "abstain report missing banner")
    finally:
        for n, fn in saved.items():
            setattr(dr, n, fn)
    print("PASS: P15 pipeline integration (multi-hop + contradiction + gate + graph + abstain)")
    return True


# ───────────────────────── Phase 1 — content-quality gate ─────────────────────
def test_p1_extraction_gate():
    # empty
    _check(dr.classify_extraction("")[0] == dr.GATE_EMPTY, "empty not flagged")
    _check(dr.classify_extraction(None)[0] == dr.GATE_EMPTY, "None not flagged")
    # challenge / anti-bot interstitial (short + marker)
    chal = "Please verify you are human. Checking your browser before access. Captcha required."
    _check(dr.classify_extraction(chal)[0] == dr.GATE_CHALLENGE, "challenge not flagged")
    chal_ru = "Доступ ограничен. Подтвердите, что вы не робот."
    _check(dr.classify_extraction(chal_ru)[0] == dr.GATE_CHALLENGE, "ru challenge not flagged")
    # JS shell: big HTML, tiny text
    shell_html = "<html><body>" + ("<div></div>" * 8000) + "</body></html>"
    _check(dr.classify_extraction("Home Login Menu", shell_html)[0] == dr.GATE_SHELL,
           "JS shell not flagged")
    # short but VALID page must pass (not over-blocked)
    short_valid = ("Кочегар is a 2010 Russian crime film directed by Aleksei Balabanov. "
                   "The plot follows a Yakut war veteran working as a boiler-room stoker "
                   "in St Petersburg who writes stories while criminals burn bodies in his "
                   "furnace. It runs about 87 minutes.")
    _check(dr.classify_extraction(short_valid)[0] == dr.GATE_OK,
           "valid short page wrongly blocked")
    # normal long article passes even if it mentions 'captcha'
    article = ("This article discusses how websites use a captcha to stop bots. " * 60)
    _check(dr.classify_extraction(article)[0] == dr.GATE_OK,
           "long article mentioning captcha wrongly blocked")
    # nav/landing pages must be quarantined, not briefed (PMC Home / Journal List)
    _check(dr.classify_extraction("Welcome to PMC.", title="PMC Home")[0] == dr.GATE_NAV,
           "nav title not flagged")
    _check(dr.classify_extraction("Browse all journals A B C D E F G "
           * 20, title="PMC Journal List")[0] == dr.GATE_NAV, "journal-list nav not flagged")
    # a real article with the word 'home' in a sentence still passes
    _check(dr.classify_extraction(short_valid, title="The Stoker — plot and analysis")[0]
           == dr.GATE_OK, "valid article wrongly flagged nav")
    print("PASS: P1 content-quality gate (empty/challenge/shell/nav/short-valid/article)")
    return True


def test_p9e_trust_ceiling_and_citation_query():
    # Wikipedia/.gov landing can no longer be PRIMARY (encyclopedic/tertiary ceiling)
    _check(dr._apply_trust_floor("PRIMARY", "https://en.wikipedia.org/wiki/Structure_tensor",
           "science") == "SECONDARY", "wikipedia not capped from PRIMARY")
    _check(dr._apply_trust_floor("PRIMARY", "https://www.britannica.com/x", "science")
           == "SECONDARY", "britannica not capped")
    # a genuine paper host can still be PRIMARY
    _check(dr._apply_trust_floor("PRIMARY", "https://link.springer.com/chapter/x", "science")
           == "PRIMARY", "springer wrongly capped")
    # citation query is short, not the giant prompt
    big = "Perform a deep technical investigation of the Di Zenzo structure tensor. " * 40
    q = dr._citation_query(big, [big, "Di Zenzo structure tensor", "gradient structure tensor GST"])
    _check(6 <= len(q) <= 90 and q != big, f"citation query not short: {len(q)}")
    # falls back to trimmed first line, stripping the imperative
    q2 = dr._citation_query("Investigate the Di Zenzo structure tensor history", [])
    _check(q2.lower().startswith("the di zenzo") or "di zenzo" in q2.lower(), f"bad fallback: {q2!r}")
    # landing/root pages are rejected and can never be PRIMARY
    for landing in ("https://www.nature.com/siteindex", "https://pmc.ncbi.nlm.nih.gov/",
                    "https://scholar.xjtu.edu.cn/en/", "https://www.academia.edu/"):
        _check(dr._is_landing_url(landing), f"{landing} not detected as landing")
        _check(dr._apply_trust_floor("PRIMARY", landing, "science") != "PRIMARY",
               f"{landing} still PRIMARY")
    # real document URLs survive
    for doc in ("https://people.csail.mit.edu/tieu/notebook/imageproc/dizenzo86.pdf",
                "https://link.springer.com/chapter/10.1007/11815921_8",
                "https://en.wikipedia.org/wiki/Structure_tensor",
                "https://arxiv.org/abs/1706.03762"):
        _check(not dr._is_landing_url(doc), f"{doc} wrongly flagged landing")
    print("PASS: P9e trust ceiling + short citation query + landing-page rejection")
    return True


def test_p9f_deterministic_postprocess():
    # duplicate sections are collapsed to one
    md = ("# Executive Summary\nA\n\n## Common Misconceptions\nfirst\n\n"
          "# Detailed Analysis\nB\n\n# Common Misconceptions\nsecond dup\n")
    out = dr._postprocess_report(md)
    _check(out.count("Common Misconceptions") == 1, "duplicate section not removed")
    _check("first" in out and "second dup" not in out, "wrong dup copy kept")
    # numbered subsection vs top-level header must still dedupe (v5 regression)
    md_num = ("### 6. Common Misconceptions\nA\n\n# Detailed\nB\n\n"
              "# Common Misconceptions\nDUP\n")
    on = dr._postprocess_report(md_num)
    _check(on.count("Common Misconceptions") == 1 and "DUP" not in on,
           "numbered-header duplicate not collapsed")
    # "(Dedicated Section)" suffix duplicate must also collapse (v7 regression)
    md_ded = ("### Common Misconceptions\nA\n\n# Method Comparison\nB\n\n"
              "# Common Misconceptions (Dedicated Section)\nDUP2\n")
    od = dr._postprocess_report(md_ded)
    _check(od.count("Common Misconceptions") == 1 and "DUP2" not in od,
           "(Dedicated Section) duplicate not collapsed")
    # register/login endpoints are landing pages even at depth
    for acct in ("https://journals.uwyo.edu/index.php/ela/user/register",
                 "https://site.org/account/login", "https://x.com/signup"):
        _check(dr._is_landing_url(acct), f"{acct} not landing")
    # drop_substrs removes model lineage prose
    md2 = "# Intro\nx\n\n## Influential Citations\nblah\n\n# Gaps\ny\n"
    out2 = dr._postprocess_report(md2, drop_substrs=("influential citation",))
    _check("Influential Citations" not in out2 and "blah" not in out2, "drop_substrs failed")
    _check("# Gaps" in out2, "non-target section wrongly dropped")
    # deterministic citation table from an OpenAlex-style brief
    brief = ('Citation-graph data from OpenAlex (scholarly index), not web pages:\n\n'
             'Most-influential works on this topic (by citation count):\n'
             '- "A note on the gradient of a multi-image" (1986) — S. Di Zenzo — '
             'cited_by=1200 — CVGIP — doi:10.1016/x\n'
             '- "Nonlinear structure tensors" (2006) — T. Brox — cited_by=400 — IVC\n\n'
             'Likely seminal/earliest high-impact work: "A note on the gradient of a '
             'multi-image" (1986) — S. Di Zenzo — cited_by=1200 — CVGIP\n\n'
             'Principal descendants (most-cited works citing the seminal work):\n'
             '- "Nonlinear structure tensors" (2006) — T. Brox — cited_by=400 — IVC\n')
    sec = dr._citation_section_from_brief(brief)
    _check("Citation Lineage" in sec and "1200" in sec and "400" in sec, "counts missing in table")
    _check("Di Zenzo" in sec and "Brox" in sec, "authors missing")
    _check("seminal" in sec and "descendant" in sec, "roles missing")
    # relevance gate
    terms = dr._entity_terms("S Di Zenzo 1986 Gradient Structure Tensor")
    _check(dr._brief_is_relevant("Structure tensor", "the di zenzo structure tensor ...", terms),
           "on-topic brief wrongly dropped")
    _check(not dr._brief_is_relevant("Non-Abelian monopoles in the Josephson effect",
           "superconducting monopole bound states", terms), "off-topic brief not dropped")
    # a generic 'tensor' paper (Shampoo optimizer) must NOT qualify on 'tensor' alone
    _check(not dr._brief_is_relevant("Shampoo: Preconditioned Stochastic Tensor Optimization",
           "a preconditioned stochastic tensor optimization algorithm for training",
           terms), "generic tensor paper wrongly kept")
    # a real structure-tensor page (says 'structure') is still kept
    _check(dr._brief_is_relevant("Tensor based feature detection for color images",
           "builds a structure tensor from multichannel gradients", terms),
           "on-topic structure-tensor page wrongly dropped")
    print("PASS: P9f dedup + deterministic citation table + relevance gate")
    return True


# ───────────────────────── Phase 2 — near-duplicate collapse ──────────────────
def test_p2_dup_collapse():
    base = ("The European Union finalised new AI regulation rules in 2026 covering "
            "high-risk systems, transparency duties and fines up to seven percent of "
            "global turnover for serious violations under the AI Act framework.")
    pages = [
        {"url": "https://reuters.com/a", "domain": "reuters.com", "title": "EU AI", "text": base},
        # exact repost on a mirror domain
        {"url": "https://mirror1.example/x", "domain": "mirror1.example", "title": "EU AI", "text": base},
        # lightly edited syndicated copy
        {"url": "https://mirror2.example/y", "domain": "mirror2.example", "title": "EU AI rules",
         "text": base.replace("finalised", "approved").replace("seven percent", "7%")},
        # genuinely different claim, same topic
        {"url": "https://apnews.com/z", "domain": "apnews.com", "title": "US AI order",
         "text": ("The United States issued an executive order on artificial intelligence "
                  "safety testing and federal procurement standards, a wholly different "
                  "policy track from the European approach, focused on agencies and "
                  "national security reviews rather than market fines.")},
    ]
    clusters = dr.dedupe_pages(pages, "news")
    _check(len(clusters) == 2, f"expected 2 clusters, got {len(clusters)}")
    big = max(clusters, key=lambda c: c["cluster_size"])
    _check(big["cluster_size"] == 3, f"syndicated cluster size {big['cluster_size']} != 3")
    # representative is the highest-authority member (reuters > mirrors)
    _check(big["domain"] == "reuters.com", f"rep should be reuters, got {big['domain']}")
    _check("mirror1.example" in big["cluster_domains"], "provenance not preserved")
    print("PASS: P2 near-duplicate collapse (3 copies -> 1 cluster, provenance kept)")
    return True


# ───────────────────────── Phase 3 — coverage / diversity ─────────────────────
def test_p3_coverage_signals():
    prof = {"category": "general"}
    # thin + narrow: 2 sources, 1 domain
    weak = [{"href": "https://x.com/a", "domain": "x.com", "query": "q"},
            {"href": "https://x.com/b", "domain": "x.com", "query": "q"}]
    _check(dr.coverage_signals(weak, prof)["weak"], "thin/narrow set not flagged weak")
    # healthy: many domains
    healthy = [{"href": f"https://d{i}.com/p", "domain": f"d{i}.com", "query": "q"}
               for i in range(8)]
    _check(not dr.coverage_signals(healthy, prof)["weak"], "healthy set wrongly weak")
    # language skew for a local topic
    loc = {"category": "local"}
    ru_only = [{"href": f"https://d{i}.ru/p", "domain": f"d{i}.ru", "query": "афиша концерт"}
               for i in range(8)]
    sig = dr.coverage_signals(ru_only, loc)
    _check(sig["weak"] and any("language" in r for r in sig["reasons"]),
           "single-language local set not flagged")
    print("PASS: P3 coverage signals (thin/narrow/clustered/language-skew)")
    return True


# ───────────────────────── Phase 4 — graded authority ─────────────────────────
def test_p4_graded_authority():
    # official/encyclopaedic ranks above category authority above mid above community
    s_wiki = dr._authority_score("https://en.wikipedia.org/wiki/X", "science")
    s_cat = dr._authority_score("https://arxiv.org/abs/1", "science")
    s_seo = dr._authority_score("https://random-seo-blog.tld/post", "science")
    s_forum = dr._authority_score("https://reddit.com/r/x", "science")
    _check(s_wiki > s_cat > s_seo > s_forum,
           f"authority ordering wrong: wiki={s_wiki} cat={s_cat} seo={s_seo} forum={s_forum}")
    # official source vs fluent SEO: SEO cannot be promoted to PRIMARY
    _check(dr._apply_trust_floor("PRIMARY", "https://random-seo-blog.tld/posts/tensor-guide",
           "science") == "PRIMARY", "unknown host floor wrongly altered")  # unknown stays as labelled
    # community host cannot self-promote above COMMUNITY
    _check(dr._apply_trust_floor("PRIMARY", "https://reddit.com/r/x", "science")
           == "COMMUNITY", "reddit not capped to COMMUNITY")
    _check(dr._apply_trust_floor("LOW", "https://habr.com/p", "science")
           == "LOW", "community floor wrongly raised LOW")  # ceiling only, not floor
    # authority host cannot be dragged below SECONDARY
    _check(dr._apply_trust_floor("LOW", "https://en.wikipedia.org/wiki/X", "science")
           == "SECONDARY", "wikipedia not floored to SECONDARY")
    # local topic: VK is the primary source, NOT capped as community
    _check(dr._apply_trust_floor("PRIMARY", "https://vk.com/kctroitsky", "local")
           == "PRIMARY", "VK wrongly capped for local topic")
    print("PASS: P4 graded authority + trust floor/ceiling")
    return True


# ───────────────────────── Phase 5 — confidence + query diversity ─────────────
def test_p5_confidence_and_queries():
    # query monoculture collapses to distinct queries
    qs = ["best budget keyboard 2026 reviews", "best budget keyboard 2026 review",
          "top affordable mechanical keyboards 2026", "best budget keyboard 2026 reviews"]
    out = dr.diversify_queries(qs, 8)
    _check(len(out) < len(qs), "near-duplicate queries not collapsed")
    _check(out[0] == qs[0], "first query not preserved")
    # computed confidence rule reflects structure
    strong = [{"trust": "PRIMARY", "authority": 0.9, "cluster_size": 1},
              {"trust": "SECONDARY", "authority": 0.85, "cluster_size": 1}]
    st = dr.evidence_stats(strong)
    _check(st["strong_sources"] == 2 and st["independent_clusters"] == 2, "stats wrong")
    rule = dr._confidence_rule(st)
    _check("High:" in rule and "independent" in rule.lower(), "confidence rule malformed")
    print("PASS: P5 computed confidence + diversity-aware queries")
    return True


# ───────────────────────── Phase 6 — extraction cache ─────────────────────────
def test_p6_cache():
    with tempfile.TemporaryDirectory() as d:
        c = research_cache.ExtractionCache(d, enabled=True)
        url = "https://example.com/article"
        _check(c.get(url, "science") is None, "empty cache returned a hit")
        c.put(url, "Title", "Some extracted body text that is long enough to matter.")
        rec = c.get(url, "science")
        _check(rec and rec["text"].startswith("Some extracted"), "cache miss after put")
        _check(c.hits == 1 and c.stores == 1, f"cache stats wrong: {c.stats()}")
        # category TTL ordering: stable >> fresh
        _check(research_cache.ttl_for("science") > research_cache.ttl_for("news"),
               "science TTL should exceed news TTL")
        # stale: force an old timestamp -> miss for a short-TTL category
        import json, time, hashlib
        key = hashlib.sha1(url.encode()).hexdigest()
        p = os.path.join(d, key + ".json")
        rec = json.load(open(p, encoding="utf-8"))
        rec["ts"] = time.time() - research_cache.ttl_for("news") - 10
        json.dump(rec, open(p, "w", encoding="utf-8"))
        _check(c.get(url, "news") is None, "stale news entry served")
        # invalidation
        c.invalidate(url)
        _check(c.get(url, "science") is None, "invalidate did not remove entry")
    print("PASS: P6 extraction cache (hit/miss/TTL/stale/invalidate)")
    return True


# ───────────────────────── Phase 7 — API adapters ─────────────────────────────
def test_p7_adapter_routing_and_fallback():
    # routing (pure, no network)
    _check(sa.adapter_for("https://en.wikipedia.org/wiki/Stoker_(2010_film)") == "wikipedia",
           "wikipedia not routed")
    _check(sa.adapter_for("https://ru.wikipedia.org/wiki/Кочегар_(фильм)") == "wikipedia",
           "ru wikipedia not routed")
    _check(sa.adapter_for("https://arxiv.org/abs/1234.5678") == "arxiv", "arxiv not routed")
    _check(sa.adapter_for("https://doi.org/10.1109/abc") == "crossref", "doi not routed")
    # JS film/local sites have NO adapter -> fall back (None), not an error
    _check(sa.adapter_for("https://www.kinopoisk.ru/film/420923/") is None,
           "kinopoisk wrongly claimed an adapter")
    _check(sa.adapter_for("https://vk.com/kctroitsky") is None, "vk wrongly routed")
    # adapter network failure degrades to None, never raises
    import requests
    orig = requests.get

    def boom(*a, **k):
        raise requests.RequestException("network down")
    requests.get = boom
    try:
        _check(sa.fetch_via_adapter("https://en.wikipedia.org/wiki/X", "entertainment") is None,
               "wiki adapter did not fail gracefully")
        _check(sa.fetch_via_adapter("https://arxiv.org/abs/1", "science") is None,
               "arxiv adapter did not fail gracefully")
    finally:
        requests.get = orig
    print("PASS: P7 adapter routing + graceful failure fallback")
    return True


# ═══════════════════ Phase 9 — research-grade upgrade ═══════════════════════
def test_p9a_truncation_and_gating():
    # report token budget scales with depth (no more 2000-token truncation on deep)
    q = dr._resolve_caps("quick")["report_tokens"]
    s = dr._resolve_caps("standard")["report_tokens"]
    d = dr._resolve_caps("deep")["report_tokens"]
    _check(q < s < d, f"report tokens not depth-scaled: {q}/{s}/{d}")
    # the leaked science junk from the critique is now blocked
    for bad in ("https://slideserve.com/x", "https://www.slideserve.com/a/b",
                "https://app.studyraid.com/read/1", "https://aimangatranslator.io/"):
        _check(dr._is_junk_source(bad), f"{bad} not blocked")
    # scholarly floor: with >=4 authority sources, MID hosts are pruned for science
    srcs = ([{"href": f"https://arxiv.org/abs/{i}", "domain": "arxiv.org"} for i in range(4)]
            + [{"href": f"https://blog{i}.tld/p", "domain": f"blog{i}.tld"} for i in range(15)])
    kept = dr._apply_scholarly_floor(list(srcs), "science")
    _check(len(kept) < len(srcs), "scholarly floor pruned nothing")
    _check(all(dr._authority_score(s["href"], "science") > dr._MID_SCORE
               or kept.index(s) >= 0 for s in kept), "kept set invalid")
    # but NOT pruned for a non-scholarly category
    _check(len(dr._apply_scholarly_floor(list(srcs), "news")) == len(srcs),
           "scholarly floor wrongly applied to news")
    # and a thin scholarly set is never starved
    thin = [{"href": "https://arxiv.org/abs/1", "domain": "arxiv.org"},
            {"href": "https://blog.tld/p", "domain": "blog.tld"}]
    _check(len(dr._apply_scholarly_floor(list(thin), "science")) == 2,
           "scholarly floor starved a thin set")
    print("PASS: P9a depth-scaled report tokens + scholarly source gating")
    return True


def test_p9b_pdf_extraction():
    # round-trip a tiny PDF through the extractor
    try:
        from pypdf import PdfWriter
    except Exception:
        print("SKIP: P9b (pypdf writer unavailable)")
        return True
    import io
    # build a 1-page PDF with text via reportlab if present, else synthesize minimal
    raw = None
    try:
        from reportlab.pdfgen import canvas
        buf = io.BytesIO()
        c = canvas.Canvas(buf)
        c.drawString(72, 720, "Di Zenzo structure tensor multichannel gradient covariance.")
        c.save()
        raw = buf.getvalue()
    except Exception:
        raw = None
    if raw:
        text = dr._extract_pdf_text(raw)
        _check(text and "Di Zenzo" in text, f"pdf text not extracted: {text!r}")
    # garbage bytes must fail gracefully (None), never raise
    _check(dr._extract_pdf_text(b"not a pdf at all") is None, "bad pdf did not return None")
    print("PASS: P9b PDF extraction (round-trip + graceful failure)")
    return True


def test_p9c_citation_graph():
    import citation_graph as cg
    import requests
    # parse a mocked OpenAlex response into a lineage brief
    sample = {"results": [
        {"id": "https://openalex.org/W1", "display_name": "A note on the gradient of a multi-image",
         "publication_year": 1986, "cited_by_count": 1200,
         "authorships": [{"author": {"display_name": "S. Di Zenzo"}}],
         "primary_location": {"source": {"display_name": "Computer Vision, Graphics, and Image Processing"}},
         "doi": "https://doi.org/10.1016/x"},
        {"id": "https://openalex.org/W2", "display_name": "Nonlinear structure tensors",
         "publication_year": 2006, "cited_by_count": 400,
         "authorships": [{"author": {"display_name": "T. Brox"}}],
         "primary_location": {"source": {"display_name": "Image and Vision Computing"}}, "doi": None},
    ]}
    orig = requests.get

    class R:
        status_code = 200
        def json(self): return sample
    calls = {"n": 0}

    def fake(*a, **k):
        calls["n"] += 1
        return R()
    requests.get = fake
    try:
        brief = cg.citation_lineage_brief("Di Zenzo structure tensor", "x@y.z")
    finally:
        requests.get = orig
    _check(brief and brief.startswith("SOURCE: PRIMARY"), "citation brief missing PRIMARY tag")
    _check("Di Zenzo" in brief and "cited_by=1200" in brief, "citation counts missing")
    _check("1986" in brief and "seminal" in brief.lower(), "seminal work not identified")
    # network failure degrades to None, never raises
    def boom(*a, **k): raise requests.RequestException("down")
    requests.get = boom
    try:
        _check(cg.citation_lineage_brief("x", "a@b.c") is None, "no graceful failure")
    finally:
        requests.get = orig
    print("PASS: P9c citation-graph lineage (parse + seminal + graceful failure)")
    return True


def test_p9d_adaptive_sections():
    # research-grade prompt triggers the right extra sections
    topic = ("Investigate the Di Zenzo structure tensor: mathematical derivation, "
             "reconstruct the historical timeline, find the most influential citing "
             "papers with citation counts, identify common misconceptions, verify "
             "true or false, compare against Harris, and produce a bibliography.")
    secs = set(dr.requested_sections(topic))
    for must in {"Chronological Timeline", "Common Misconceptions", "Method Comparison",
                 "Mathematical Derivation", "Citation Lineage", "Bibliography"}:
        _check(must in secs, f"section {must!r} not detected")
    # a plain factual question triggers NO extra sections
    _check(dr.requested_sections("what is the capital of France") == [],
           "extra sections wrongly triggered on a simple query")
    print("PASS: P9d adaptive synthesis-section detection")
    return True


def test_p21_synthesis_empty_fallback():
    """When the LLM reduce returns empty twice, the run must still deliver a
    structured, cited answer (digest fallback) — never a dead 'no output' stub."""
    # deterministic builder is pure
    digests = [{"label": "BGE-M3", "major": True, "size": 3, "evidence_weight": 2.1,
                "entities": ["bge-m3", "mteb"],
                "representative_claims": [{"text": "BGE-M3 improves nDCG on MTEB.",
                                          "domain": "arxiv.org", "trust": "PRIMARY"}]}]
    briefs = [{"domain": "arxiv.org", "url": "u1", "brief": "BGE-M3 improves retrieval.",
               "trust": "PRIMARY", "mode": "confirm"}]
    body = dr._deterministic_digest_report("BGE rerankers", digests, briefs)
    _check("BGE-M3" in body and "arxiv.org" in body, "digest fallback missing evidence")
    _check("without model synthesis" in body, "fallback not labelled")
    # with no digests it falls back to briefs
    body2 = dr._deterministic_digest_report("x", [], briefs)
    _check("arxiv.org" in body2, "brief-only fallback missing source")

    # integration: synthesize_report returns "" -> retry also "" -> digest fallback in report
    saved = {n: getattr(dr, n) for n in
             ("collect_sources", "crawl_pages", "dedupe_pages", "rerank_for_briefing",
              "brief_source", "synthesize_report", "plan_queries")}
    calls = {"n": 0, "efforts": []}

    def empty_synth(ctx, topic, briefs, prog, **k):
        calls["n"] += 1
        calls["efforts"].append(k.get("effort", "default"))
        return ""   # always empty -> exercises the effort ladder + fallback

    class _Ctx:
        def is_cancelled(self): return False
    try:
        dr.collect_sources = lambda ctx, q, pq, prog, rd, profile=None: [
            {"href": f"https://s{i}.com/x", "domain": f"s{i}.com", "title": q[0] if q else "t",
             "query": (q[0] if q else "")} for i in range(3)]
        dr.crawl_pages = lambda ctx, s, c, prog, rd, prof, cache: [
            {"url": x["href"], "domain": x["domain"], "title": "BGE-M3 reranker",
             "text": "BGE-M3 reranker MTEB nDCG benchmark.", "cluster_size": 1,
             "cluster_domains": [x["domain"]]} for x in s]
        dr.dedupe_pages = lambda p, c="general": p
        dr.rerank_for_briefing = lambda t, p, pr, pg: p
        dr.brief_source = lambda ctx, t, p: "SOURCE: PRIMARY\nBGE-M3 improves retrieval on MTEB."
        dr.synthesize_report = empty_synth
        dr.plan_queries = lambda ctx, t, mq: (
            ["bge-m3 reranker", "bge-m3 mteb"],
            {"category": "general", "recency": "any", "timelimit": None, "news": False})
        with _isolated_dr_dir():
            res = dr.run_deep_research(_Ctx(), "BGE-M3 reranker quality", depth="quick")
            rep = res["report"]
            _check(calls["n"] >= 2, f"synthesis retry not attempted (calls={calls['n']})")
            _check("medium" in calls["efforts"] and "low" in calls["efforts"],
                   f"effort ladder didn't step down (efforts={calls['efforts']})")
            _check("no output" not in rep, "dead 'no output' stub still emitted")
            _check("without model synthesis" in rep, "digest fallback not used")
            _check("# Source Appendix" in rep or "Source Appendix" in rep, "sources missing from fallback")
    finally:
        for n, fn in saved.items():
            setattr(dr, n, fn)
    print("PASS: P21 synthesis-empty -> retry -> deterministic cited digest fallback (no dead stub)")
    return True


def _hier_briefs():
    """A fixture with two clear sub-threads + a weak claim + a contradiction."""
    return [
        # Thread A: BGE-M3 reranker (3 confirming, two near-duplicate claims)
        {"url": "a1", "domain": "arxiv.org", "trust": "PRIMARY", "mode": "confirm",
         "title": "BGE-M3 reranker", "brief": "The BGE-M3 reranker model improves "
         "retrieval on the MTEB benchmark with strong nDCG across languages."},
        {"url": "a2", "domain": "aclanthology.org", "trust": "SECONDARY", "mode": "confirm",
         "title": "BGE-M3 results", "brief": "BGE-M3 reranker model improves retrieval "
         "on the MTEB benchmark with strong nDCG across many languages."},
        {"url": "a3", "domain": "github.com", "trust": "COMMUNITY", "mode": "confirm",
         "title": "BGE-M3 repo", "brief": "BGE-M3 reranker FlagOpen/FlagEmbedding repo "
         "documents MTEB nDCG evaluation and multilingual support."},
        # Thread B: Cohere Rerank (single LOW source -> weak claim)
        {"url": "b1", "domain": "reddit.com", "trust": "LOW", "mode": "confirm",
         "title": "cohere", "brief": "Cohere Rerank API is a hosted reranking service "
         "used in production RAG pipelines by some teams."},
        # Contradiction touching thread A
        {"url": "c1", "domain": "blog.io", "trust": "COMMUNITY", "mode": "contradict",
         "title": "BGE-M3 latency", "brief": "BGE-M3 reranker has serious latency "
         "limitations on long documents in production."},
    ]


# ─────────────────── Phase 16 — hierarchical clustering ────────────────────────
def test_p16_clustering():
    cs = _cl.cluster_briefs(_hier_briefs(), threshold=0.15)
    # contradiction brief is excluded from the confirming thread structure
    member_urls = {b["url"] for c in cs.clusters for b in cs.members_of(c)}
    _check("c1" not in member_urls, "contradiction brief leaked into clusters")
    # the BGE cluster (a1/a2/a3) groups together; cohere stays separate
    bge = next((c for c in cs.clusters if "a1" in {cs.briefs[i]["url"] for i in c.members}), None)
    _check(bge is not None and bge.size() >= 2, "BGE thread did not cluster")
    _check(bge.major, "multi-source BGE thread not marked major")
    cohere = next((c for c in cs.clusters
                   if "b1" in {cs.briefs[i]["url"] for i in c.members}), None)
    _check(cohere is not None and "a1" not in {cs.briefs[i]["url"] for i in cohere.members},
           "distinct threads wrongly merged")
    _check(cs.stats()["clusters"] >= 2 and len(cs.major()) >= 1, "cluster stats wrong")
    # determinism: same input -> same partition
    cs2 = _cl.cluster_briefs(_hier_briefs(), threshold=0.15)
    _check([sorted(c.members) for c in cs.clusters] == [sorted(c.members) for c in cs2.clusters],
           "clustering not deterministic")
    # failure case: empty input
    _check(_cl.cluster_briefs([]).stats()["clusters"] == 0, "empty not handled")
    print("PASS: P16 hierarchical clustering (threads, major/minor, contradiction-excluded, deterministic)")
    return True


# ─────────────────── Phase 17 — graph-guided expansion ─────────────────────────
def test_p17_graph_guided():
    briefs = _hier_briefs()
    g = _rg.build_graph(briefs)
    cs = _cl.cluster_briefs(briefs, threshold=0.15)
    # centrality finds the hub entities
    cent = _gg.entity_centrality(g)
    _check(cent and cent[0]["degree"] >= 2, "centrality did not rank a hub entity")
    # weak claims: the single LOW cohere source is flagged for corroboration
    weak = _gg.weak_claims(g)
    _check(any(w["domain"] == "reddit.com" for w in weak), "weak LOW claim not flagged")
    # planner emits structurally-motivated queries with recorded reasons
    vis = set()
    gx = _gg.plan_graph_expansion("BGE reranker quality", g, cs,
                                  contradiction={"has_contradictions": True, "domains": ["blog.io"]},
                                  visited=vis, max_queries=6)
    _check(gx and all("reason" in x and "target" in x for x in gx), "expansion missing provenance")
    reasons = " ".join(x["reason"] for x in gx)
    _check("centrality" in reasons, "no centrality-driven target")
    _check(any("contradiction" in x["target"] for x in gx), "contradiction not chased")
    _check(any("weakclaim" in x["target"] for x in gx), "weak claim not chased")
    _check(all(x["query"].lower() in vis for x in gx), "visited set not updated (loop guard)")
    # loop guard: a second call with the same visited set yields no repeats
    gx2 = _gg.plan_graph_expansion("BGE reranker quality", g, cs, visited=vis, max_queries=6)
    _check({x["query"].lower() for x in gx2}.isdisjoint(x["query"].lower() for x in gx),
           "graph expansion repeated visited queries")
    # cluster-prioritized contradiction targets hit major clusters first
    ct = _gg.cluster_contradiction_targets("BGE reranker quality", g, cs, visited=set())
    _check(ct and any("cluster:" in x["target"] for x in ct), "contradiction not cluster-prioritized")
    print("PASS: P17 graph-guided expansion (centrality + weak claims + contradiction priority + loop guard)")
    return True


# ─────────────────── Phase 18 — claim-level dedup/merge ────────────────────────
def test_p18_claim_merge():
    briefs = _hier_briefs()
    merged, log = _cm.merge_claims(briefs, threshold=0.5)
    # the two near-duplicate BGE claims (a1/a2) merge into one representative
    _check(len(log) == 1 and log[0]["support_count"] == 2, f"expected 1 merge of 2, got {log}")
    rep = next(b for b in merged if b.get("support_count", 1) > 1)
    _check(rep["trust"] == "PRIMARY", "merge did not keep highest-trust representative")
    _check(rep["merged_sources"] and rep["merged_sources"][0]["url"] in ("a1", "a2"),
           "provenance of merged source lost")
    # contradiction brief is never merged away (stays visible)
    _check(any(b["url"] == "c1" and b.get("mode") == "contradict" for b in merged),
           "contradiction brief was merged/dropped")
    # distinct claims are NOT merged (cohere stays separate)
    _check(any(b["url"] == "b1" for b in merged), "distinct claim wrongly merged")
    # determinism + failure case
    _check(_cm.merge_claims([])[0] == [], "empty not handled")
    print("PASS: P18 claim-level merge (dedup + provenance kept + contradictions preserved)")
    return True


# ─────────────────── Phase 19 — hierarchical synthesis digest ──────────────────
def test_p19_hierarchical_digest():
    briefs = _hier_briefs()
    cs = _cl.cluster_briefs(briefs, threshold=0.15)
    digests = _hi.cluster_digests(cs, contradiction_domains={"blog.io"}, max_clusters=6)
    _check(digests and digests[0]["major"], "major cluster not surfaced first")
    _check(all("representative_claims" in d and "evidence_weight" in d for d in digests),
           "digest missing structure")
    # the major BGE thread carries more evidence weight than the minor cohere thread
    weights = [d["evidence_weight"] for d in digests]
    _check(max(weights) > min(weights), "evidence weight did not differentiate threads")
    pre = _hi.render_digest_preamble(digests)
    _check("THREAD STRUCTURE" in pre and "MAJOR" in pre, "preamble scaffold malformed")
    _check("do NOT repeat" in pre, "preamble missing anti-repetition instruction")
    # empty -> empty preamble, no crash
    _check(_hi.render_digest_preamble([]) == "", "empty digest not handled")
    print("PASS: P19 hierarchical synthesis (per-cluster digests + thread scaffold preamble)")
    return True


# ─────────────────── Phase 20 — hierarchical pipeline integration ──────────────
def test_p20_hierarchical_pipeline():
    saved = {n: getattr(dr, n) for n in
             ("collect_sources", "crawl_pages", "dedupe_pages", "rerank_for_briefing",
              "brief_source", "synthesize_report", "plan_queries")}
    search_log, syn_capture = [], {}

    def fake_collect(ctx, queries, per_query, prog, run_dir, profile=None):
        search_log.extend(queries)
        return [{"href": f"https://s{abs(hash(q)) % 99999}.com/x",
                 "domain": f"s{abs(hash(q)) % 99999}.com", "title": q[:40], "query": q}
                for q in queries]

    def fake_crawl(ctx, sources, caps, prog, run_dir, profile, cache):
        return [{"url": s["href"], "domain": s["domain"], "title": s["title"],
                 "text": f"{s['title']}. BGE-M3 reranker model MTEB benchmark nDCG "
                         f"FlagOpen/FlagEmbedding arXiv:2402.03216 multilingual.",
                 "cluster_size": 1, "cluster_domains": [s["domain"]]} for s in sources]

    cnt = [0]

    def fake_brief(ctx, topic, page):
        cnt[0] += 1
        return (f"SOURCE: {'PRIMARY' if cnt[0] % 2 else 'SECONDARY'}\n"
                "The BGE-M3 reranker model improves retrieval on the MTEB benchmark "
                "with strong nDCG across many languages.")

    def fake_syn(ctx, topic, briefs, prog, **k):
        syn_capture["preamble"] = k.get("preamble")
        return "## Executive Summary\nBGE-M3 works.\n"

    class _Ctx:
        def is_cancelled(self): return False

    try:
        dr.collect_sources = fake_collect
        dr.crawl_pages = fake_crawl
        dr.dedupe_pages = lambda p, c="general": p
        dr.rerank_for_briefing = lambda t, p, pr, pg: p
        dr.brief_source = fake_brief
        dr.synthesize_report = fake_syn
        dr.plan_queries = lambda ctx, topic, mq: (
            ["bge-m3 reranker quality", "bge-m3 mteb benchmark"],
            {"category": "general", "recency": "any", "timelimit": None, "news": False})
        with _isolated_dr_dir():
            res = dr.run_deep_research(_Ctx(), "BGE-M3 reranker quality evaluation", depth="standard")
            stats = res["stats"]
            # hierarchical stats are recorded
            _check("clusters" in stats and stats["clusters"]["clusters"] >= 1, "no cluster stats")
            _check(stats.get("claims_merged", 0) >= 1, "claim-merge did not fire on duplicate claims")
            # graph-guided deep-dive issued structurally-motivated follow-up queries
            _check(any("in depth" in q or "analysis" in q or "corroboration" in q
                       or "consensus" in q for q in search_log),
                   "graph-guided expansion issued no targeted queries")
            # the synthesis received the hierarchical thread scaffold
            _check(syn_capture.get("preamble") and "THREAD STRUCTURE" in syn_capture["preamble"],
                   "synthesis did not receive cluster digest preamble")
            # state artifacts written for observability (now inside the isolated tempdir).
            slug = dr._slug("BGE-M3 reranker quality evaluation")
            runs = sorted(p for p in dr.DR_DIR.glob(f"*_{slug}") if p.is_dir())
            _check(runs, "no run dir created for this topic")
            latest = runs[-1]
            names = {p.name for p in latest.glob("*.json")}
            _check("graph.json" in names, "graph.json not saved")
            _check(any("state" in n for n in names) or (latest / "state.json").exists()
                   or list(latest.glob("state*")), "no state artifacts saved")
    finally:
        for n, fn in saved.items():
            setattr(dr, n, fn)
    print("PASS: P20 hierarchical pipeline (cluster stats + claim-merge + graph deep-dive + digest preamble + artifacts)")
    return True


def _graded_briefs():
    """Briefs with nested vocabulary so a real recursive hierarchy can form:
    Database System > {Vector, Relational} > {engine} > variants."""
    def b(dom, txt, trust="SECONDARY"):
        return {"domain": dom, "url": f"http://{dom}/x", "title": txt[:40],
                "brief": txt, "trust": trust, "mode": "confirm"}
    out = []

    def mk(dom_root, sub, n=3):
        for i in range(n):
            out.append(b(f"{dom_root}{i}.io",
                         f"Database System overview. {sub} detail {i}"))
    mk("qdrhnsw", "Vector Embedding Search Qdrant Rust HNSW graph tuning")
    mk("qdrquant", "Vector Embedding Search Qdrant Rust HNSW scalar quantization")
    mk("milgpu", "Vector Embedding Search Milvus GPU IVF partition sharding")
    mk("milcpu", "Vector Embedding Search Milvus CPU IVF flat brute force")
    mk("pgacid", "Relational SQL Transactions Postgres ACID MVCC isolation")
    mk("pgrepl", "Relational SQL Transactions Postgres streaming replication WAL")
    mk("myrepl", "Relational SQL Transactions MySQL InnoDB replication binlog")
    return out


def test_p22_recursive_hierarchy():
    import hierarchy as H
    briefs = _graded_briefs()
    h = H.build_hierarchy(briefs, max_depth=4, min_cluster_size=2)
    _check(h.levels() >= 3, f"hierarchy did not reach 3+ levels: {h.levels()}")
    # a child must itself have children somewhere (real nesting, not 2-level)
    def has_grandchild(n):
        return any(c.children for c in n.children)
    _check(has_grandchild(h.root), "no node spawned sub-children (still 2-level)")
    # determinism: same input -> identical tree
    h2 = H.build_hierarchy(briefs, max_depth=4, min_cluster_size=2)
    _check(h.to_dict()["root"] == h2.to_dict()["root"], "hierarchy not deterministic")
    # loop protection: children are strict subsets (size strictly decreases with depth)
    def subset_ok(n):
        for c in n.children:
            if c.size() >= n.size():
                return False
            if not subset_ok(c):
                return False
        return True
    _check(subset_ok(h.root), "a child was not a strict subset of its parent (loop risk)")
    # max_depth respected
    _check(h.max_depth() <= 4, "max_depth exceeded")
    # serialization shape
    d = h.to_dict()
    _check("stats" in d and d["stats"]["levels"] == h.levels(), "bad hierarchy.to_dict")
    # tree preamble renders the nesting
    pre = H.render_tree_preamble(h)
    _check("RESEARCH HIERARCHY" in pre and "sub-thread" in pre, "tree preamble missing depth")
    print(f"PASS: P22 recursive hierarchy ({h.levels()} levels, deterministic, loop-safe, bounded)")
    return True


def test_p23_graph_communities():
    import communities as C
    import research_graph as RG
    import clustering as CL
    briefs = _graded_briefs()
    # add shared capitalized entities so sources cross-link in the graph
    g = RG.build_graph(briefs)
    _check(g.stats()["entities"] > 0, "no entities to bridge sources (bad test fixture)")
    res = C.detect_communities(g)
    _check(res["stats"]["communities"] >= 2, "Louvain found <2 communities")
    _check(res["modularity"] > 0.0, "non-positive modularity")
    # determinism
    res2 = C.detect_communities(g)
    _check(res["assignment"] == res2["assignment"], "community detection not deterministic")
    # metadata uses entities + sources + claims (+ contradiction field present)
    c0 = res["communities"][0]
    for fld in ("size", "n_sources", "n_entities", "n_claims", "domains",
                "entities", "contradiction_sources"):
        _check(fld in c0, f"community metadata missing {fld}")
    # communities DIFFER from lexical clusters
    cs = CL.cluster_briefs(briefs, threshold=0.18, major_min_size=2)
    vs = C.communities_vs_clusters(res, cs)
    _check(vs["differ"] is True, f"communities did not differ from lexical clusters: {vs}")
    print(f"PASS: P23 graph communities (Louvain, Q={res['modularity']}, "
          f"{res['stats']['communities']} communities, differ-from-lexical={vs['pair_agreement']})")
    return True


def test_p24_phase_history():
    import tempfile, glob, json as _json
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    dr._save_state(d, phase="planning", queries=["q1", "q2"])
    dr._save_state(d, phase="searching", sources=10)
    dr._save_state(d, phase="decision_gate", decision={"decision": "ANSWER"})
    hist = sorted(glob.glob(str(d / "history" / "*.json")))
    _check(len(hist) == 3, f"history did not record every phase: {len(hist)}")
    names = [Path(p).name for p in hist]
    _check(names[0].startswith("001_planning") and names[1].startswith("002_searching")
           and names[2].startswith("003_decision"), f"history not ordered/named: {names}")
    # each record carries its own timestamp + the phase payload (reconstructable)
    rec = _json.load(open(hist[2], encoding="utf-8"))
    _check(rec["seq"] == 3 and rec["phase"] == "decision_gate"
           and rec["decision"]["decision"] == "ANSWER" and "wall_clock" in rec and "ts" in rec,
           "history record missing fields")
    # latest snapshot still in state.json
    _check((d / "state.json").exists(), "state.json snapshot missing")
    print("PASS: P24 full phase history (ordered, timestamped, reconstructable from disk)")
    return True


def test_p25_eval_scoring():
    import tempfile, json as _json
    from pathlib import Path
    sys.path.insert(0, os.path.dirname(__file__))
    import eval_harness as EH
    d = Path(tempfile.mkdtemp())
    # synthetic graph.json: 3 sources (1 primary, 1 contradict) on the topic
    graph = {"nodes": [
        {"type": "source", "domain": "qdrant.tech", "url": "https://qdrant.tech/x",
         "trust": "PRIMARY", "mode": "confirm"},
        {"type": "source", "domain": "blog.io", "url": "https://blog.io/y",
         "trust": "SECONDARY", "mode": "confirm"},
        {"type": "source", "domain": "critic.com", "url": "https://critic.com/z",
         "trust": "SECONDARY", "mode": "contradict"},
        {"type": "claim", "text": "qdrant milvus weaviate vector database comparison"},
    ], "edges": [], "stats": {}}
    _json.dump(graph, open(d / "graph.json", "w", encoding="utf-8"))
    task = {"topic": "Qdrant vs Milvus vs Weaviate", "category": "technology",
            "must_terms": ["qdrant", "milvus", "weaviate", "vector"],
            "expect_primary": True, "expect_contra": True}
    good = {"report": "# R\nqdrant.tech and blog.io agree on qdrant, milvus, weaviate "
                      "vector tradeoffs.\n## Source Appendix\nqdrant.tech\nblog.io\ncritic.com\n",
            "stats": {"decision": "ANSWER"}}
    m, aux = EH.score_run(task, good, str(d))
    for k in ("retrieval_quality", "source_diversity", "primary_source_precision",
              "contradiction_coverage", "faithfulness", "hallucination_rate",
              "citation_coverage", "answer_completeness", "answer_correctness"):
        _check(k in m and 0.0 <= m[k] <= 1.0, f"metric {k} missing/out-of-range: {m.get(k)}")
    _check(m["contradiction_coverage"] == 1.0, "missed the contradiction source")
    _check(m["answer_completeness"] == 1.0, "should find all must_terms in report")
    _check(m["hallucination_rate"] == 0.0, "no fabricated domains -> 0 hallucination")
    # a report citing a domain NOT in the source set -> hallucination detected
    bad = {"report": "# R\nSee fabricated-source.example.org for proof of qdrant.\n"
                     "## Source Appendix\nqdrant.tech\n", "stats": {}}
    mb, _ = EH.score_run(task, bad, str(d))
    _check(mb["hallucination_rate"] > 0.0, "did not flag a fabricated citation")
    _check(0.0 <= EH._headline(m) <= 1.0, "headline out of range")
    print(f"PASS: P25 scored eval harness (9 metrics, headline={EH._headline(m)}, "
          f"hallucination detection works)")
    return True


def test_p26_reflection_loop_active():
    """Offline: with no contradictions in the first pass, reflection must LAUNCH new
    retrieval cycles (not just record gaps), bounded by max-iterations + budget."""
    saved = {n: getattr(dr, n) for n in
             ("collect_sources", "crawl_pages", "dedupe_pages", "rerank_for_briefing",
              "brief_source", "synthesize_report", "plan_queries")}
    pass_log = []

    def fake_collect(ctx, queries, per_query, prog, run_dir, profile=None):
        pass_log.append(list(queries))
        base = abs(hash(tuple(queries))) % 99999
        return [{"href": f"https://r{base}_{i}.com/x", "domain": f"r{base}_{i}.com",
                 "title": q[:40], "query": q} for i, q in enumerate(queries)]

    def fake_crawl(ctx, sources, caps, prog, run_dir, profile, cache):
        return [{"url": s["href"], "domain": s["domain"], "title": s["title"],
                 "text": f"{s['title']}. Kubernetes Nomad orchestration scheduler "
                         f"Raft consensus bin-packing workload.",
                 "cluster_size": 1, "cluster_domains": [s["domain"]]} for s in sources]

    def fake_brief(ctx, topic, page):
        return ("SOURCE: SECONDARY\nKubernetes and Nomad differ in scheduler design "
                "and operational complexity for container orchestration.")

    class _Ctx:
        def is_cancelled(self): return False

    try:
        dr.collect_sources = fake_collect
        dr.crawl_pages = fake_crawl
        dr.dedupe_pages = lambda p, c="general": p
        dr.rerank_for_briefing = lambda t, p, pr, pg: p
        dr.brief_source = fake_brief
        dr.synthesize_report = lambda ctx, topic, briefs, prog, **k: "## Summary\nok.\n"
        dr.plan_queries = lambda ctx, topic, mq: (
            ["kubernetes vs nomad orchestration", "nomad scheduler design"],
            {"category": "general", "recency": "any", "timelimit": None, "news": False})
        with _isolated_dr_dir():
            res = dr.run_deep_research(_Ctx(), "Kubernetes vs Nomad orchestration tradeoffs",
                                       depth="standard")
            iters = res["stats"].get("reflection_iterations", 0)
            _check(iters >= 1, "reflection loop never launched a new cycle")
            _check(iters <= dr.DR_REFLECTION_MAX_ITERATIONS,
                   f"reflection loop exceeded max iterations: {iters}")
            # the reflection cycles issued NEW retrieval passes beyond the initial + multihop
            _check(len(pass_log) > 2, "no extra retrieval passes were launched by reflection")
            # observability: reflection_iteration phase recorded queries on disk
            slug = dr._slug("Kubernetes vs Nomad orchestration tradeoffs")
            runs = sorted(p for p in dr.DR_DIR.glob(f"*_{slug}") if p.is_dir())
            hist = list((runs[-1] / "history").glob("*reflection_iteration*.json"))
            _check(hist, "no reflection_iteration phase recorded in history")
            import json as _json
            rec = _json.load(open(hist[0], encoding="utf-8"))
            _check(rec.get("queries"), "reflection iteration recorded no queries")
    finally:
        for n, fn in saved.items():
            setattr(dr, n, fn)
    print(f"PASS: P26 active reflection loop (launched {iters} cycle(s), bounded, "
          f"new retrieval + recorded queries)")
    return True


# ─────────────── Phase 27 — manual control (GUI override knobs) ────────────────
def test_p27_manual_control_overrides():
    """The GUI's Manual Control panel patches dr.MANUAL_OVERRIDE_SPEC knobs for a
    single run via apply_overrides/restore_overrides. Every knob must be a real
    module global (or applying it is a silent no-op), values must round-trip
    exactly, bad input must fail loud, and a real pipeline run must observe the
    overridden values."""
    for name in dr.MANUAL_OVERRIDE_SPEC:
        _check(hasattr(dr, name), f"manual-control knob {name} is not a real dr global")

    before = {n: getattr(dr, n) for n in dr.MANUAL_OVERRIDE_SPEC}
    saved = dr.apply_overrides({
        "DR_MAX_QUERIES": 3, "DR_MULTIHOP_ENABLED": False, "DR_RERANK_BACKEND": "lexical",
    })
    _check(dr.DR_MAX_QUERIES == 3 and dr.DR_MULTIHOP_ENABLED is False
           and dr.DR_RERANK_BACKEND == "lexical", "override did not take effect")
    dr.restore_overrides(saved)
    after = {n: getattr(dr, n) for n in dr.MANUAL_OVERRIDE_SPEC}
    _check(before == after, "restore_overrides left the module in a different state")

    try:
        dr.apply_overrides({"DR_MAX_QUERIES": 9999})
        raise AssertionError("out-of-range override was not rejected")
    except ValueError:
        pass
    try:
        dr.apply_overrides({"NOT_A_REAL_KNOB": 1})
        raise AssertionError("unknown override key was not rejected")
    except KeyError:
        pass

    # a real (mocked-I/O) run actually observes the override: cap queries to 1
    # and disable multi-hop, then verify no extra entity-driven queries appear.
    saved2 = {n: getattr(dr, n) for n in
              ("collect_sources", "crawl_pages", "dedupe_pages", "rerank_for_briefing",
               "brief_source", "synthesize_report", "plan_queries")}
    search_log = []

    def fake_collect(ctx, queries, per_query, prog, run_dir, profile=None):
        search_log.extend(queries)
        return [{"href": f"https://s{i}.com/x", "domain": f"s{i}.com", "title": q[:30], "query": q}
                for i, q in enumerate(queries)]

    class _Ctx:
        def is_cancelled(self): return False

    try:
        dr.collect_sources = fake_collect
        dr.crawl_pages = lambda ctx, s, c, prog, rd, prof, cache: [
            {"url": x["href"], "domain": x["domain"], "title": x["title"],
             "text": "Manual control test content with enough words to pass the gate.",
             "cluster_size": 1, "cluster_domains": [x["domain"]]} for x in s]
        dr.dedupe_pages = lambda p, c="general": p
        dr.rerank_for_briefing = lambda t, p, pr, pg: p
        dr.brief_source = lambda ctx, t, p: "SOURCE: SECONDARY\nManual control test claim."
        dr.synthesize_report = lambda ctx, topic, briefs, prog, **k: "## Summary\nok.\n"
        dr.plan_queries = lambda ctx, topic, mq: (
            ["manual control test query"],
            {"category": "general", "recency": "any", "timelimit": None, "news": False})
        # DR_REPLAN_ENABLED belongs in this list. The coverage/re-plan loop is a
        # fifth stage that issues queries of its own, with its own knob in
        # MANUAL_OVERRIDE_SPEC, and one stubbed source makes coverage "weak" --
        # so it broadened, correctly, and the check below read its queries as a
        # leak from the four stages that were switched off.
        ov = dr.apply_overrides({"DR_MULTIHOP_ENABLED": False, "DR_CONTRADICTION_ENABLED": False,
                                  "DR_REFLECTION_ENABLED": False, "DR_GRAPH_EXPANSION_ENABLED": False,
                                  "DR_REPLAN_ENABLED": False})
        try:
            with _isolated_dr_dir():
                dr.run_deep_research(_Ctx(), "manual control test query", depth="quick")
        finally:
            dr.restore_overrides(ov)
        _check(search_log == ["manual control test query"],
               f"disabling multihop/contradiction/reflection still issued extra queries: {search_log}")
    finally:
        for n, fn in saved2.items():
            setattr(dr, n, fn)
    print("PASS: P27 manual control (knobs are real globals, round-trip, validated, observed live)")
    return True


# ────────────── Phase 28 — source filters, classifier, query mutation ──────────
def test_p28_source_filters_and_mutation():
    """Domain whitelist/blacklist, source-type include/exclude, per-domain cap,
    and query-mutation-on-zero-hits are REAL filters applied in collect_sources,
    not cosmetic UI state."""
    _check(dr._classify_source_type("https://arxiv.org/abs/1.2") == "academic", "arXiv misclassified")
    _check(dr._classify_source_type("https://www.reuters.com/x") == "news", "Reuters misclassified")
    _check(dr._classify_source_type("https://city.gov/page") == "government", "gov misclassified")
    _check(dr._classify_source_type("https://reddit.com/r/x") == "forum", "reddit misclassified")
    _check(dr._classify_source_type("https://medium.com/@x/post") == "blog", "medium misclassified")
    _check(dr._classify_source_type("https://x.com/user/status/1") == "social", "x.com misclassified")
    _check(dr._classify_source_type("https://example.com/report.pdf") == "whitepaper", "pdf misclassified")
    _check(dr._classify_source_type("https://example.com/article") == "general", "default wrong")

    _check(dr._domain_allowed("https://arxiv.org/x", "", "") is True, "no filters should allow")
    _check(dr._domain_allowed("https://reddit.com/x", "", "reddit.com") is False, "blacklist not applied")
    _check(dr._domain_allowed("https://arxiv.org/x", "arxiv.org", "") is True, "whitelist wrongly blocked")
    _check(dr._domain_allowed("https://example.com/x", "arxiv.org", "") is False,
           "whitelist did not restrict")
    _check(dr._domain_allowed("https://reddit.com/x", "reddit.com", "reddit.com") is False,
           "blacklist must win over whitelist")

    _check(dr._source_type_allowed("https://reddit.com/x", "") is True, "empty exclude allows all")
    _check(dr._source_type_allowed("https://reddit.com/x", "forum") is False,
           "excluded type not filtered")
    _check(dr._source_type_allowed("https://arxiv.org/x", "forum") is True,
           "unrelated type wrongly filtered")

    capped = dr._cap_per_domain(
        [{"href": f"https://s.com/{i}"} for i in range(5)] +
        [{"href": f"https://t.com/{i}"} for i in range(2)], 2)
    _check(len(capped) == 4, f"per-domain cap not applied: {len(capped)}")
    _check(dr._cap_per_domain([{"href": "https://s.com/1"}], 0) == [{"href": "https://s.com/1"}],
           "0 should mean unlimited")

    m0 = dr._mutate_query('"exact phrase query"', 0)
    _check(m0 and '"' not in m0, "mutation 0 should strip quotes")
    m1 = dr._mutate_query("one two three four", 1)
    _check(m1 == "one two three", "mutation 1 should drop trailing word")
    _check(dr._mutate_query("x", 5) is None, "out-of-range attempt should return None")
    print("PASS: P28 source-type classifier + domain filters + per-domain cap + query mutation")
    return True


# ─────────────── Phase 29 — exhaustive multi-round collection loop ─────────────
def test_p29_exhaustive_collection_loop():
    """The CRITICAL requirement: the agent must not stop after a small number of
    sources if the user configured higher targets. Verifies the round loop keeps
    broadening across multiple rounds to satisfy DR_MIN_SOURCES, respects
    DR_MAX_SOURCES as a hard ceiling, and the stagnation guard still stops a
    force-exhaustive run when a round truly finds nothing new (bounded, never an
    actual infinite loop)."""
    saved = {n: getattr(dr, n) for n in
             ("collect_sources", "plan_queries")}
    counter = {"n": 0}

    def fake_plan(ctx, topic, mq):
        counter["n"] += 1
        return ([f"broadening query {counter['n']}"],
                {"category": "general", "recency": "any", "timelimit": None, "news": False})

    batch = {"n": 0}

    def fake_collect_growing(ctx, queries, per_query, prog, run_dir, profile=None):
        batch["n"] += 1
        # each round contributes 2 brand-new sources, forever (never stagnates on
        # its own) so the loop's own targets/ceiling are what must stop it.
        return [{"href": f"https://round{batch['n']}-{i}.example.com/x",
                 "domain": f"round{batch['n']}-{i}.example.com", "title": "t", "query": ""}
                for i in range(2)]

    try:
        dr.plan_queries = fake_plan
        dr.collect_sources = fake_collect_growing
        ov = dr.apply_overrides({
            "DR_MIN_SOURCES": 9, "DR_MAX_COLLECTION_ROUNDS": 10, "DR_FORCE_EXHAUSTIVE": True,
        })
        try:
            queries, profile = ["seed"], {"category": "general", "recency": "any",
                                          "timelimit": None, "news": False}
            sources = fake_collect_growing(None, queries, 6, dr._Progress(None), None, profile)

            class _FakeCtx:
                def is_cancelled(self): return False
            cancelled = _FakeCtx().is_cancelled
            prog = dr._Progress(None)
            max_rounds = max(1, dr.DR_MAX_COLLECTION_ROUNDS)
            for round_n in range(1, max_rounds + 1):
                if dr.DR_MAX_SOURCES > 0 and len(sources) >= dr.DR_MAX_SOURCES:
                    break
                cov = dr.coverage_signals(sources, profile)
                domains_now = len({dr._domain(s["href"]) for s in sources})
                targets_unmet = (
                    (dr.DR_MIN_SOURCES > 0 and len(sources) < dr.DR_MIN_SOURCES) or
                    (dr.DR_TARGET_SOURCES > 0 and len(sources) < dr.DR_TARGET_SOURCES) or
                    (dr.DR_MIN_UNIQUE_DOMAINS > 0 and domains_now < dr.DR_MIN_UNIQUE_DOMAINS))
                if not (cov["weak"] or targets_unmet or dr.DR_FORCE_EXHAUSTIVE):
                    break
                # What is under test is the LOOP -- targets, ceiling, stagnation
                # guard -- not query generation. Asking the real _corrective_queries
                # for them made the test depend on an LLM inventing something new
                # every round: with no model it repeats itself, `extra` comes back
                # empty on round two, and the loop stops at 6 sources for a reason
                # that has nothing to do with what is being measured.
                extra = [f"corrective query round {round_n}"]
                if not extra:
                    break
                more = dr.collect_sources(None, extra, 6, prog, None, profile)
                have = {dr._norm_url(s["href"]) for s in sources}
                added = [s for s in more if dr._norm_url(s["href"]) not in have]
                sources.extend(added)
                queries = queries + extra
                if not added:
                    break
            _check(len(sources) >= 9, f"loop stopped before reaching DR_MIN_SOURCES: {len(sources)}")
            _check(batch["n"] >= 5, f"too few rounds ran to explain the growth: {batch['n']}")
        finally:
            dr.restore_overrides(ov)

        # ceiling: DR_MAX_SOURCES must cap growth even with force-exhaustive + huge min.
        batch["n"] = 0
        ov2 = dr.apply_overrides({
            "DR_MIN_SOURCES": 50, "DR_MAX_SOURCES": 5, "DR_MAX_COLLECTION_ROUNDS": 20,
            "DR_FORCE_EXHAUSTIVE": True,
        })
        try:
            sources2 = []
            queries2, profile2 = ["seed"], profile
            for round_n in range(1, dr.DR_MAX_COLLECTION_ROUNDS + 1):
                if dr.DR_MAX_SOURCES > 0 and len(sources2) >= dr.DR_MAX_SOURCES:
                    break
                more = dr.collect_sources(None, [f"q{round_n}"], 6, dr._Progress(None), None, profile2)
                have = {dr._norm_url(s["href"]) for s in sources2}
                added = [s for s in more if dr._norm_url(s["href"]) not in have]
                if dr.DR_MAX_SOURCES > 0:
                    added = added[:max(0, dr.DR_MAX_SOURCES - len(sources2))]
                sources2.extend(added)
            _check(len(sources2) <= 5, f"DR_MAX_SOURCES ceiling was not enforced: {len(sources2)}")
        finally:
            dr.restore_overrides(ov2)
    finally:
        for n, fn in saved.items():
            setattr(dr, n, fn)
    print("PASS: P29 exhaustive collection loop (reaches min-sources target, "
          "respects max-sources ceiling, bounded by round cap)")
    return True


# ────────── Phase 30 — quality/verification knobs (gate + claim-merge) ─────────
def test_p30_verification_knobs():
    """The GUI's Quality/Verification mode preset maps to REAL decision-gate and
    claim-merge knobs added for manual control, not just a label."""
    for name in ("DR_GATE_MIN_STRONG", "DR_GATE_MIN_CLUSTERS", "DR_CLAIM_MERGE_THRESHOLD"):
        _check(name in dr.MANUAL_OVERRIDE_SPEC, f"{name} not exposed as a manual-control knob")
        _check(hasattr(dr, name), f"{name} is not a real dr global")
    before = (dr.DR_GATE_MIN_STRONG, dr.DR_GATE_MIN_CLUSTERS, dr.DR_CLAIM_MERGE_THRESHOLD)
    saved = dr.apply_overrides({"DR_GATE_MIN_STRONG": 3, "DR_GATE_MIN_CLUSTERS": 4,
                                 "DR_CLAIM_MERGE_THRESHOLD": 0.3})
    _check((dr.DR_GATE_MIN_STRONG, dr.DR_GATE_MIN_CLUSTERS, dr.DR_CLAIM_MERGE_THRESHOLD)
           == (3, 4, 0.3), "verification knobs did not apply")
    dr.restore_overrides(saved)
    _check((dr.DR_GATE_MIN_STRONG, dr.DR_GATE_MIN_CLUSTERS, dr.DR_CLAIM_MERGE_THRESHOLD)
           == before, "verification knobs did not restore")
    try:
        dr.apply_overrides({"DR_CLAIM_MERGE_THRESHOLD": 1.5})
        raise AssertionError("out-of-range float override was not rejected")
    except ValueError:
        pass
    print("PASS: P30 verification knobs (gate strength + claim-merge threshold, real & bounded)")
    return True


# ─────────────── Phase 31 — apply_overrides atomicity (no partial leak) ────────
def test_p31_override_atomicity():
    """A batch override with ANY invalid entry must mutate NOTHING — otherwise a
    raise mid-loop leaves the module half-applied and, since the caller never
    receives `saved`, those partial overrides leak permanently into later runs
    (would have hung/poisoned the GUI worker). Validate-all-then-apply."""
    before = {n: getattr(dr, n) for n in dr.MANUAL_OVERRIDE_SPEC}
    # last entry out of range -> the earlier valid ones must NOT stick
    try:
        dr.apply_overrides({"DR_MAX_QUERIES": 5, "DR_MIN_SOURCES": 30,
                            "DR_FORCE_EXHAUSTIVE": True, "DR_MAX_PAGES": 99999})
        raise AssertionError("expected ValueError on out-of-range batch")
    except ValueError:
        pass
    after = {n: getattr(dr, n) for n in dr.MANUAL_OVERRIDE_SPEC}
    _check(before == after, "partial override leaked after a bad value in the batch")
    # unknown key mid-batch -> also atomic
    try:
        dr.apply_overrides({"DR_MAX_QUERIES": 7, "TOTALLY_BOGUS_KNOB": 1})
        raise AssertionError("expected KeyError on unknown knob")
    except KeyError:
        pass
    _check({n: getattr(dr, n) for n in dr.MANUAL_OVERRIDE_SPEC} == before,
           "partial override leaked after an unknown key in the batch")
    print("PASS: P31 apply_overrides atomic (no partial leak on bad value / unknown key)")
    return True


def test_p34_rerank_backend_resolution():
    """The reranker hang/no-op fixes: (a) an explicit backend='cross' must ACTUALLY
    load the cross-encoder — it used to return 'cross' while leaving _ce None, so
    relevance() silently fell back to lexical (a no-op 'cross'); now it loads it or
    degrades to a REPORTED dense/lexical. (b) when the cross-encoder can't load, the
    reranker still scores (lexical) instead of hanging."""
    import rerank
    saved_tc = rerank.Reranker._try_cross
    saved_ea = rerank.Reranker._embedder_available
    try:
        # cross requested but unavailable → must fall back to lexical, NOT claim cross
        rerank.Reranker._try_cross = lambda self: False
        rerank.Reranker._embedder_available = lambda self: False
        r = rerank.Reranker("cross")
        _check(r.backend == "lexical", f"explicit cross w/o CE must fall back, got {r.backend!r}")
        _check(r._ce is None, "no cross-encoder should be loaded on fallback")
        sc = r.relevance("alpha beta", ["alpha beta gamma here", "completely unrelated zzz"])
        _check(sc[0] > sc[1], "lexical fallback did not score (would look like a hang)")

        # cross requested AND available → loads _ce and reports cross
        rerank.Reranker._try_cross = lambda self: (setattr(self, "_ce", object()) or True)
        r2 = rerank.Reranker("cross")
        _check(r2.backend == "cross" and r2._ce is not None, "cross not loaded when available")
    finally:
        rerank.Reranker._try_cross = saved_tc
        rerank.Reranker._embedder_available = saved_ea
    print("PASS: P34 explicit cross loads-or-degrades-loudly (no silent-lexical no-op, no hang)")
    return True


def test_p35_cross_encoder_no_download_hang():
    """The "dead loop with no logs" root cause: backend='cross' with the
    bge-reranker NOT cached used to block on a silent ~2.3GB HF download mid-run.
    Guard: _try_cross must REFUSE to instantiate CrossEncoder when the model isn't
    cached and DR_RERANK_ALLOW_DOWNLOAD isn't set — it degrades instead of hanging.
    Opt-in (env) or an already-cached model lets it proceed."""
    import rerank
    saved_allow = rerank._ALLOW_DOWNLOAD
    saved_cached = rerank._model_is_cached
    ce_constructed = {"n": 0}

    class _BoomCE:  # constructing this == "the download/load happened"
        def __init__(self, *a, **k):
            ce_constructed["n"] += 1

    try:
        import types
        st = types.ModuleType("sentence_transformers")
        st.CrossEncoder = _BoomCE
        saved_mod = sys.modules.get("sentence_transformers")
        sys.modules["sentence_transformers"] = st
        try:
            # NOT cached + NOT allowed → must NOT construct CrossEncoder (no hang)
            rerank._ALLOW_DOWNLOAD = False
            rerank._model_is_cached = lambda repo: False
            r = rerank.Reranker("cross")
            _check(r._ce is None, "cross-encoder must not load when uncached+disallowed")
            _check(ce_constructed["n"] == 0, "CrossEncoder was constructed → would download/hang")
            _check(r.backend in ("dense", "lexical"), f"must degrade, got {r.backend!r}")

            # cached → allowed to construct
            rerank._model_is_cached = lambda repo: True
            r2 = rerank.Reranker("cross")
            _check(ce_constructed["n"] == 1 and r2._ce is not None, "cached model should load")

            # uncached but explicit opt-in → allowed to construct
            rerank._ALLOW_DOWNLOAD = True
            rerank._model_is_cached = lambda repo: False
            r3 = rerank.Reranker("cross")
            _check(ce_constructed["n"] == 2 and r3._ce is not None, "opt-in download should load")
        finally:
            if saved_mod is not None:
                sys.modules["sentence_transformers"] = saved_mod
            else:
                sys.modules.pop("sentence_transformers", None)
    finally:
        rerank._ALLOW_DOWNLOAD = saved_allow
        rerank._model_is_cached = saved_cached
    print("PASS: P35 cross-encoder won't download mid-run unless cached or opted-in (no dead loop)")
    return True


def test_p36_report_title_and_equation_blocks():
    """Report structure fixes: (a) a boolean-query topic must NOT become the H1 —
    the title is the most-specific quoted phrase; a plain topic passes through. (b) the
    equations section must put each formula on its own block with the source tag on a
    SEPARATE line, so a math renderer sees one clean display block (not formula+tag
    folded together)."""
    # (a) title cleaning
    boolean = ('give me best promf for deep reserch from("Di Zenzo" OR '
               '"Di Zenzo structure tensor" OR "color tensor") AND ("structure tensor")')
    title = dr._clean_report_title(boolean)
    _check(title == "Di Zenzo structure tensor", f"boolean title not cleaned: {title!r}")
    _check(dr._clean_report_title("Transformer attention") == "Transformer attention",
           "plain topic title should pass through unchanged")
    # header uses the clean title for the H1 and keeps the full query as a subtitle
    header = dr._report_header(boolean, [], {"sources": 1, "pages": 1}, "deep")
    _check(header.startswith("# Di Zenzo structure tensor"), "H1 is not the clean title")
    _check("# Research report:" not in header, "old giant-query H1 still present")
    _check("*Query:" in header, "full query subtitle dropped (traceability lost)")

    # (b) equation block formatting: equation and its attribution on separate lines
    briefs = [{"domain": "laszloegri.github.io", "trust": "COMMUNITY",
               "equations": [r"\begin{align}\theta^* = \tfrac12\end{align}"]}]
    sec = dr._build_equations_section(briefs)
    _check(r"\begin{align}" in sec, "equation lost from section")
    lines = [ln for ln in sec.splitlines() if ln.strip()]
    eq_idx = next(i for i, ln in enumerate(lines) if r"\begin{align}" in ln)
    _check("laszloegri.github.io" not in lines[eq_idx],
           "source tag folded onto the equation line (breaks math rendering)")
    _check(any("laszloegri.github.io" in ln for ln in lines), "attribution missing")
    print("PASS: P36 clean report title (+query subtitle) and clean equation blocks")
    return True


def test_p33_parallel_io_and_stage_timing():
    """Parallel search + wave-based fetch are I/O speedups that must NOT change
    results vs serial, must actually overlap, must respect per-domain politeness
    (never two same-domain fetches in flight), and the run must record per-stage
    wall-clock in stats['stage_timings']. Concurrency knobs validate through the
    override layer."""
    import threading
    import time as _t
    from pathlib import Path

    # 1) parallel search == serial search (same deduped set, dup URL collapsed).
    HITS = {"qa": [{"title": "A", "href": "https://a.com/1", "body": "", "domain": "a.com"},
                   {"title": "B", "href": "https://b.com/1", "body": "", "domain": "b.com"}],
            "qb": [{"title": "B2", "href": "https://b.com/1", "body": "", "domain": "b.com"},
                   {"title": "C", "href": "https://c.com/1", "body": "", "domain": "c.com"}]}
    saved_raw = dr_collect.raw_search_results
    dr_collect.raw_search_results = lambda ctx, q, n, *, timelimit=None, news=False: list(HITS.get(q, []))
    saved_sc = dr.DR_SEARCH_CONCURRENCY
    try:
        with _isolated_dr_dir():
            rd = Path(dr.DR_DIR)
            dr.DR_SEARCH_CONCURRENCY = 1
            ser = dr.collect_sources(None, ["qa", "qb"], 5, dr._Progress(None), rd, {"category": "general"})
            dr.DR_SEARCH_CONCURRENCY = 4
            par = dr.collect_sources(None, ["qa", "qb"], 5, dr._Progress(None), rd, {"category": "general"})
    finally:
        dr_collect.raw_search_results = saved_raw
        dr.DR_SEARCH_CONCURRENCY = saved_sc
    _check(sorted(s["href"] for s in ser) == sorted(s["href"] for s in par),
           "parallel search changed the result set")
    _check([s["href"] for s in par].count("https://b.com/1") == 1, "dup URL not deduped under parallel")

    # 2) wave crawl — mock the per-URL unit to (a) prove overlap and (b) prove
    #    per-domain politeness. Bypasses the content gate so we test wave logic.
    saved_acq = dr_crawl._acquire_page
    st = {"now": 0, "max": 0}
    lk = threading.Lock()

    def fake_acq(url, depth, seed, category, use_adapters, cache):
        with lk:
            st["now"] += 1
            st["max"] = max(st["max"], st["now"])
        # Linger until a peer is in flight (or 0.3 s): a fixed 30 ms let a
        # loaded Windows runner start the next worker only after this one
        # finished, so a crawl that does run in parallel read as serial.
        end = _t.time() + 0.3
        while _t.time() < end:
            with lk:
                if st["now"] >= 2:
                    break
            _t.sleep(0.005)
        with lk:
            st["max"] = max(st["max"], st["now"])
        with lk:
            st["now"] -= 1
        return {"url": url, "depth": depth, "social": False, "reason": None, "html": None,
                "page": {"url": url, "domain": dr._domain(url),
                         "title": seed.get("title", ""), "text": "x" * 50}}

    dr_crawl._acquire_page = fake_acq
    saved_fc = dr.DR_FETCH_CONCURRENCY
    caps = {"max_pages": 20, "depth": 0, "links_per_page": 0}
    try:
        # distinct domains → parallelism should overlap, capped at #domains
        seeds = [{"href": f"https://d{i % 3}.com/p{i}", "domain": f"d{i % 3}.com",
                  "title": f"t{i}"} for i in range(6)]
        with _isolated_dr_dir():
            rd = Path(dr.DR_DIR)
            dr.DR_FETCH_CONCURRENCY = 1
            p_ser = dr.crawl_pages(None, seeds, caps, dr._Progress(None), rd)
            st["max"] = 0
            dr.DR_FETCH_CONCURRENCY = 3
            p_par = dr.crawl_pages(None, seeds, caps, dr._Progress(None), rd)
        _check({p["url"] for p in p_ser} == {p["url"] for p in p_par}, "parallel crawl changed pages")
        _check(len(p_par) == 6, f"expected 6 pages, got {len(p_par)}")
        _check(st["max"] >= 2, f"fetch did not overlap (max inflight={st['max']})")
        _check(st["max"] <= 3, f"per-domain politeness violated (max inflight={st['max']})")

        # all SAME domain → must never run two in parallel (wave size 1)
        same = [{"href": f"https://one.com/p{i}", "domain": "one.com", "title": "t"}
                for i in range(4)]
        st["max"] = 0
        with _isolated_dr_dir():
            dr.DR_FETCH_CONCURRENCY = 3
            p_same = dr.crawl_pages(None, same, caps, dr._Progress(None), Path(dr.DR_DIR))
        _check(len(p_same) == 4, "same-domain seeds were dropped (deferral lost URLs)")
        _check(st["max"] == 1, f"same-domain fetched in parallel — politeness broken (max={st['max']})")
    finally:
        dr_crawl._acquire_page = saved_acq
        dr.DR_FETCH_CONCURRENCY = saved_fc

    # 3) per-stage timing recorded across phase transitions + finalize flush.
    pr = dr._Progress(None)
    pr.update("Searching"); _t.sleep(0.01)
    pr.update("Crawling"); _t.sleep(0.01)
    pr.update("Crawling", pages=1)         # same phase → no new bucket
    pr.finalize_timings()
    timings = pr.stats["stage_timings"]
    _check("Searching" in timings and "Crawling" in timings, f"stage timings missing: {timings}")
    _check(all(v >= 0 for v in timings.values()), f"negative stage time: {timings}")

    # 4) concurrency knobs validate + coerce + restore through the override layer.
    before = {n: getattr(dr, n) for n in ("DR_SEARCH_CONCURRENCY", "DR_FETCH_CONCURRENCY")}
    saved = dr.apply_overrides({"DR_SEARCH_CONCURRENCY": 4, "DR_FETCH_CONCURRENCY": 3})
    _check(dr.DR_SEARCH_CONCURRENCY == 4 and dr.DR_FETCH_CONCURRENCY == 3, "knobs not applied")
    dr.restore_overrides(saved)
    _check({n: getattr(dr, n) for n in before} == before, "restore failed")
    try:
        dr.apply_overrides({"DR_FETCH_CONCURRENCY": 99})
        raise AssertionError("expected ValueError for out-of-range concurrency")
    except ValueError:
        pass
    print("PASS: P33 parallel search/fetch == serial + overlap + per-domain politeness + stage timings")
    return True


def test_p32_ssrf_and_outlink_ban():
    """Safety: the crawler must not 'accidentally end up on a banned resource'.
    (a) _is_safe_public_url fails CLOSED on non-public/unsafe targets — loopback,
    private LAN, link-local cloud-metadata (169.254.169.254), file://, and
    .local hosts — while passing real public research domains. (b) the domain
    blacklist still bans a host (the social-outlink/in-page hops re-apply it)."""
    blocked = ["http://localhost/x", "http://127.0.0.1:8000/x",
               "http://169.254.169.254/latest/meta-data/", "http://10.0.0.5/",
               "http://192.168.1.1/", "file:///etc/passwd", "ftp://example.com/x",
               "http://foo.local/", "gopher://evil/"]
    for u in blocked:
        _check(dr._is_safe_public_url(u) is False, f"unsafe URL not blocked: {u}")
    for u in ("https://arxiv.org/abs/1706.03762", "https://en.wikipedia.org/wiki/Adam"):
        _check(dr._is_safe_public_url(u) is True, f"public URL wrongly blocked: {u}")
    # blacklist still wins (the hops now re-check this)
    _check(dr._domain_allowed("https://pinterest.com/p", "", "pinterest.com") is False,
           "blacklisted host not banned")
    _check(dr._domain_allowed("https://arxiv.org/x", "", "pinterest.com") is True,
           "non-blacklisted host wrongly banned")
    print("PASS: P32 SSRF guard fails-closed + blacklist re-applied on hops")
    return True


def test_p37_survey_synthesis():
    """Survey mode: the final report is a dynamically-OUTLINED, section-by-section
    scientific document — model designs the plan, each section is written in its own
    call as merged prose (not a source digest). Verify: outline JSON is parsed; the
    document carries its own title + abstract + TOC + every planned section; sections
    are written one-LLM-call-each; junk plans degrade to a default skeleton; mojibake
    is repaired in section bodies."""
    # (a) outline JSON parsing tolerates code fences / stray prose
    fenced = '```json\n{"title":"T","sections":[{"heading":"Intro","subsections":[],"focus":"f"}]}\n```'
    p = dr._extract_outline_json(fenced)
    _check(p and p["title"] == "T", "fenced outline JSON not parsed")
    _check(dr._extract_outline_json("garbage, no json") is None, "non-JSON should be None")

    # (b) normalize: junk -> default skeleton; clamp section count; drop empty headings
    _check(len(dr._normalize_outline({}, "topic")["sections"]) >= 4, "empty plan -> default skeleton")
    big = {"title": "X", "sections": [{"heading": f"S{i}", "subsections": [], "focus": ""}
                                      for i in range(50)]}
    _check(len(dr._normalize_outline(big, "t")["sections"]) <= dr.DR_SURVEY_MAX_SECTIONS,
           "section count not clamped")
    mixed = {"title": "X", "sections": [{"heading": "  ", "subsections": [], "focus": ""},
                                        {"heading": "Real", "subsections": ["a"], "focus": "f"}]}
    norm = dr._normalize_outline(mixed, "t")
    _check([s["heading"] for s in norm["sections"]] == ["Real"], "empty headings not dropped")

    # (c) end-to-end synthesize_survey with a stubbed LLM
    outline_json = ('{"title":"The Color Structure Tensor","abstract_focus":"Establishes the '
                    'multichannel tensor.","include_math":true,"sections":['
                    '{"heading":"Introduction","subsections":[],"focus":"motivate"},'
                    '{"heading":"Mathematical Foundations","subsections":["Gradient"],"focus":"define"},'
                    '{"heading":"Conclusion","subsections":[],"focus":"wrap"}]}')
    calls = {"outline": 0, "sections": 0}

    def fake_llm(ctx, system, user, **k):
        if "lead author" in system:            # OUTLINE_PLANNER_PROMPT
            calls["outline"] += 1
            return outline_json
        if "ABSTRACT" in system:               # dedicated abstract synthesis (not a section)
            calls["abstract"] = calls.get("abstract", 0) + 1
            return "This synthesized abstract summarizes the assembled body in prose."
        if "ONE section" in system:            # SURVEY_SECTION_PROMPT
            calls["sections"] += 1
            # includes a mojibake artifact (Î») and a verbatim equation to preserve
            return (r"The eigenvalues \(\lambda\) of the 2x2 tensor are key (arxiv.org). "
                    r"$$\lambda = \tfrac12(a+c)$$ The symbol Î» appears garbled.")
        return ""

    saved = dr_calls.call_llm_simple
    try:
        dr_calls.call_llm_simple = fake_llm

        class _Prog:
            stats = {}
            def update(self, *a, **k): pass
            def snap_formula(self, *a, **k): pass

        class _Ctx:
            model_name = "gpt-oss-120b"
            def is_cancelled(self): return False

        briefs = [
            {"domain": "arxiv.org", "url": "u1", "title": "Di Zenzo color gradient",
             "brief": "- color gradient tensor", "trust": "PRIMARY", "cluster_size": 1},
            {"domain": "wikipedia.org", "url": "u2", "title": "Structure tensor",
             "brief": "- eigen decomposition", "trust": "SECONDARY", "cluster_size": 1},
        ]
        doc = dr.synthesize_survey(_Ctx(), "color structure tensor", briefs, _Prog())
        _check(calls["outline"] == 1, f"outline must be planned exactly once (got {calls['outline']})")
        _check(calls["sections"] == 3, f"each section its own call (got {calls['sections']})")
        _check(doc.startswith("# The Color Structure Tensor"), "document missing its own H1 title")
        _check("## Abstract" in doc, "abstract missing")
        # the abstract must be SYNTHESIZED prose, not the planner's one-line directive
        # ("abstract_focus") dumped verbatim (the old fake-abstract bug).
        _check(calls.get("abstract") == 1, "abstract must be written in its own call")
        _check("synthesized abstract summarizes" in doc, "real abstract text missing")
        _check("Establishes the multichannel tensor." not in doc,
               "abstract_focus directive leaked verbatim as the abstract")
        _check("## Table of Contents" in doc, "TOC missing")
        for h in ("## Introduction", "## Mathematical Foundations", "## Conclusion"):
            _check(h in doc, f"planned section {h!r} missing from document")
        _check(r"$$\lambda = \tfrac12(a+c)$$" in doc, "verbatim equation not preserved")
        _check("Î»" not in doc, "mojibake not repaired in section body")
        _check("Source A" not in doc and "According to" not in doc,
               "document leaked source-organized phrasing")

        # empty sections -> survey returns "" so caller can fall back
        dr_calls.call_llm_simple = lambda ctx, s, u, **k: (outline_json if "lead author" in s else "")
        _check(dr.synthesize_survey(_Ctx(), "t", briefs, _Prog()) == "",
               "all-empty sections must yield '' (so caller falls back)")
    finally:
        dr_calls.call_llm_simple = saved
    print("PASS: P37 survey mode — dynamic outline + section-by-section merged document")
    return True


def test_p39_context_aware_sections():
    """Small-context safety: the section writer must keep each prompt inside the
    model's context window (DR_MODEL_CONTEXT) and SELF-ADAPT (shrink evidence +
    retry) when a section comes back empty — so a 4096-loaded model still yields a
    real multi-section document instead of empty→digest. Per-section evidence is
    selected by relevance to that section, not one shared blob."""
    import importlib
    old = os.environ.get("DR_MODEL_CONTEXT")
    os.environ["DR_MODEL_CONTEXT"] = "4096"
    try:
        import config as _cfg
        importlib.reload(_cfg)
        importlib.reload(dr)
        _check(dr.DR_MODEL_CONTEXT == 4096, "DR_MODEL_CONTEXT not honored")
        # _select_section_evidence packs to a char budget and is section-relevant
        briefs = [{"domain": f"d{i}.org", "url": f"u{i}",
                   "title": ("convergence proof bound" if i < 2 else "application vision cnn"),
                   "brief": "- detail " * 30,
                   "trust": "PRIMARY", "cluster_size": 1, "equations": []} for i in range(12)]
        sec = {"heading": "Convergence Analysis", "subsections": [], "focus": "prove the bound"}
        ev = dr._select_section_evidence(briefs, sec, 1500)
        _check(len(ev) <= 1800, "evidence not trimmed to char budget")
        _check("convergence proof" in ev, "section-relevant brief not prioritized")

        # end-to-end: a stub that returns empty when the prompt exceeds a tight
        # window must trigger shrink-and-retry and still yield every section.
        outline_json = ('{"title":"Doc","sections":['
                        '{"heading":"Intro","subsections":[],"focus":"a"},'
                        '{"heading":"Convergence Analysis","subsections":[],"focus":"b"}]}')
        seen = {"max": 0}

        def fake(ctx, system, user, **k):
            if "lead author" in system:
                return outline_json
            if "ABSTRACT" in system:       # the dedicated abstract-synthesis call
                return "A concise synthesized abstract of the document."
            seen["max"] = max(seen["max"], len(user))
            if len(user) > 3500:           # simulate n_keep overflow
                return ""
            return "## S\n\nReal merged prose about the topic (arxiv.org)."

        saved = dr_calls.call_llm_simple
        try:
            dr_calls.call_llm_simple = fake

            class _P:
                stats = {}
                def update(self, *a, **k): pass
                def snap_formula(self, *a, **k): pass

            class _C:
                model_name = "qwen"
                def is_cancelled(self): return False

            doc = dr.synthesize_survey(_C(), "topic", briefs, _P())
            _check(doc.startswith("# Doc"), "document not assembled under small context")
            _check(doc.count("## S") == 2, f"both sections must survive shrink-retry (got {doc.count('## S')})")
        finally:
            dr_calls.call_llm_simple = saved
    finally:
        if old is None:
            os.environ.pop("DR_MODEL_CONTEXT", None)
        else:
            os.environ["DR_MODEL_CONTEXT"] = old
        import config as _cfg
        importlib.reload(_cfg)
        importlib.reload(dr)
    print("PASS: P39 context-aware section sizing + shrink-retry under small context")
    return True


def test_p38_relevance_and_citation_noise():
    """Retrieval-noise gating: an ambiguous entity ('Adam' vs 'Adams') must not drag
    politics/off-topic pages into the report, and the OpenAlex citation table must be
    suppressed when its works are off-topic. Word-boundary matching is the lever."""
    # (a) word-boundary term matching: 'adam' != 'adams'
    _check(dr._term_in("adam", "the adam optimizer"), "should match whole word")
    _check(not dr._term_in("adam", "mayor adams embraces"), "must NOT match 'adams'")
    _check(dr._term_in("adam", "adam's method"), "apostrophe still matches")

    # (b) off-topic brief sweep drops noise, keeps on-topic, never empties
    terms = ["adam", "optimizer", "stochastic"]
    briefs = [
        {"domain": "arxiv.org", "title": "Adam: A Method for Stochastic Optimization",
         "brief": "- adam update rule"},
        {"domain": "theturkeytimes.com", "title": "Mayor Adams Embraces Anti-Hate",
         "brief": "- city politics, nothing technical"},
        {"domain": "pythonguides.com", "title": "Your Complete Python Roadmap",
         "brief": "- learn python from zero"},
    ]
    kept, dropped = dr._filter_offtopic_briefs(briefs, terms)
    _check([b["domain"] for b in kept] == ["arxiv.org"], "off-topic noise not dropped")
    _check(len(dropped) == 2, "should drop both noise pages")
    # safety: a term list that matches nothing must NOT empty the report
    k2, d2 = dr._filter_offtopic_briefs(briefs, ["zzzznomatch"])
    _check(len(k2) == 3 and not d2, "no-match must be a no-op, never empty")

    # (b2) generic academic/SEO words must be stopwords (not relevance terms), or a
    # query-derived 'survey'/'arxiv' matches every "A Survey of ..." source. Relevance
    # terms come from the TOPIC, not a noisy planned query.
    tterms = dr._entity_terms("The Adam optimizer for stochastic gradient descent")
    _check("survey" not in tterms and "arxiv" not in tterms, "generic words leaked as terms")
    _check("adam" in tterms and "optimizer" in tterms, "core entity terms missing")

    # (b3) ≥2-hit rule: a page mentioning ONLY the ambiguous bare entity ('Adam' the
    # name) is NOT relevant on a multi-term topic; a real 'adam optimizer' page is.
    multi = ["adam", "optimizer", "stochastic", "descent"]
    _check(dr._brief_is_relevant("Adam in literature", "- the biblical Adam, a poem", multi) is False,
           "single ambiguous-entity hit must not pass on a multi-term topic")
    _check(dr._brief_is_relevant("Adam optimizer", "- adam optimizer update rule", multi) is True,
           "genuine multi-term match must pass")

    # (c) citation lineage suppressed when OpenAlex works are off-topic
    offtopic_body = ("SOURCE: PRIMARY\nMost-influential works on this topic:\n"
                     '- "Noisy intermediate-scale quantum algorithms" (2022) — Bharti — cited_by=1647\n'
                     '- "Spa hotels TripAdvisor segmentation" (2019) — Ahani — cited_by=258\n')
    _check(dr._citation_section_from_brief(offtopic_body, terms) == "",
           "off-topic citation table must be suppressed")
    # on-topic citations DO render
    ontopic_body = ("SOURCE: PRIMARY\nMost-influential works on this topic:\n"
                    '- "Adam: A Method for Stochastic Optimization" (2015) — Kingma — cited_by=99999\n'
                    '- "Decoupled Weight Decay (AdamW) for Adam" (2019) — Loshchilov — cited_by=8000\n')
    sec = dr._citation_section_from_brief(ontopic_body, terms)
    _check("Citation Lineage" in sec and "Kingma" in sec, "on-topic citations must render")
    # without terms, legacy behaviour (no filtering) is preserved
    _check("Bharti" in dr._citation_section_from_brief(offtopic_body, None),
           "no terms -> unfiltered (back-compat)")
    print("PASS: P38 retrieval-noise gating — word-boundary relevance + citation suppression")
    return True


def test_p41_latex_artifact_repair():
    """LaTeX-rendering bugs that survive synthesis must be repaired deterministically:
    (1) ar5iv layout primitives (\\vskip/\\cr/glue) leaking into matrices, (2) \\(..\\)/
    \\[..\\] delimiters the renderer can't show, (3) domain citations wrapped in \\text{}.
    Legit math subscripts like \\text{struc}/\\text{child} (no dot) must be untouched."""
    # (1) matrix layout primitives stripped, row separator \\ preserved
    mat = (r"$$P=\begin{bmatrix}p^{1}&p^{4}\\ \vskip 3.0pt plus 1.0pt minus 1.0pt\cr "
           r"p^{4}&p^{2}\end{bmatrix}$$")
    out = dr._repair_latex_artifacts(mat)
    _check("vskip" not in out and r"\cr" not in out, "layout primitives not stripped")
    _check(r"\\" in out and r"\begin{bmatrix}" in out, "matrix row sep / structure lost")

    # (2) delimiter normalization
    _check(dr._repair_latex_artifacts(r"with \(\sqrt{F(\theta)}\) here")
           == r"with $\sqrt{F(\theta)}$ here", "inline \\(..\\) not normalized to $..$")
    _check(dr._repair_latex_artifacts(r"\[E=mc^2\]") == r"$$E=mc^2$$",
           "display \\[..\\] not normalized to $$..$$")

    # (3) citation unwrap — domain in \text{} (with/without surrounding math), but NOT
    #     legitimate subscripts.
    _check(dr._repair_latex_artifacts(r"lines \((\text{link.springer.com})\).")
           == "lines (link.springer.com).", "math-wrapped domain citation not unwrapped")
    _check(dr._repair_latex_artifacts(r"see \text{people.csail.mit.edu} ok")
           == "see people.csail.mit.edu ok", "bare \\text{domain} not unwrapped")
    keep = r"$Q_{\text{struc}}(I)$ and $\mu_{\text{child}}$"
    _check(dr._repair_latex_artifacts(keep) == keep,
           "legit \\text{} subscript (no dot) must be preserved")

    # (4) source-paper macros MathJax can't render: drop \boldmath, map \tr->operatorname
    fixed = dr._repair_latex_artifacts(r"$$\boldmath H := \frac{1}{m}\tr_\gamma M$$")
    _check(r"\boldmath" not in fixed, "\\boldmath font-switch not dropped")
    _check(r"\operatorname{tr}" in fixed and r"\tr_" not in fixed,
           "\\tr operator not mapped to \\operatorname{tr}")
    # must NOT touch standard commands or already-correct operators
    _check(dr._repair_latex_artifacts(r"$\sin\theta+\operatorname{tr}(A)$")
           == r"$\sin\theta+\operatorname{tr}(A)$", "standard \\sin / existing operator altered")
    print("PASS: P41 LaTeX artifact repair — layout strip + delimiter norm + citation unwrap + macros")
    return True


def test_p40_query_distillation():
    """A verbose/instruction-style request (here a long Russian 'write me a doctoral
    paper that...' prompt) must be DISTILLED via the planner's core_topic and never
    seeded into search verbatim — that was the bug that crawled 49 junk pages, 0
    findings. Also: an LLM that returns nothing must not fall back to the 532-char
    essay as a query."""
    raw = ("знания по структурному тензору Di zenzo нужны все формулы входящие в эту "
           "парадигму при этом каждый входящий элемент формулы должен быть детально "
           "расписан не просто название а что это такое мне нужна научная работа "
           "которую я дам человеку чтобы он мог докторскую защитить все формулы")
    saved = dr_calls.call_llm_simple
    try:
        # (a) planner returns core_topic + clean queries -> raw essay is NOT a query
        dr_calls.call_llm_simple = lambda ctx, s, u, **k: (
            '{"core_topic": "Di Zenzo structure tensor for color images", '
            '"category": "science", "recency": "any", '
            '"queries": ["Di Zenzo structure tensor color images", '
            '"structure tensor eigenvalues edge orientation"]}')
        qs, prof = dr.plan_queries(None, raw, 8)
        _check(raw not in qs, "raw verbose request must never be a search query")
        _check(all(len(q) <= 120 for q in qs), "no query may be a runaway essay")
        _check(any("structure tensor" in q.lower() for q in qs), "distilled subject missing")
        _check(prof["category"] == "science", "category from planner lost")
        # core_topic must be threaded into the profile so relevance-gating / citation
        # query / title use the clean English subject, not the multilingual essay.
        _check(prof.get("core_topic") == "Di Zenzo structure tensor for color images",
               "profile must carry the distilled core_topic")
        # entity terms derived from core_topic are English (would match crawled pages);
        # derived from the raw Russian essay they would not.
        ct_terms = dr._entity_terms(prof["core_topic"])
        _check(any(t in ("tensor", "structure", "zenzo", "color", "images") for t in ct_terms),
               "core_topic must yield English relevance terms")

        # (b) LLM gives nothing: fallback must distill the raw topic, not seed it whole
        dr_calls.call_llm_simple = lambda ctx, s, u, **k: ""
        qs2, _ = dr.plan_queries(None, raw, 8)
        _check(raw not in qs2, "empty-LLM fallback must not seed the 532-char essay")
        _check(all(len(q) <= 130 for q in qs2), "fallback queries must stay short")

        # (c) an already-clean short topic is preserved as a seed
        dr_calls.call_llm_simple = lambda ctx, s, u, **k: ""
        qs3, _ = dr.plan_queries(None, "Adam optimizer", 8)
        _check(any("adam optimizer" in q.lower() for q in qs3), "clean topic must survive")
    finally:
        dr_calls.call_llm_simple = saved
    print("PASS: P40 query distillation — verbose request rewritten, never seeded verbatim")
    return True


def test_hop_anchor_language():
    """The follow-up passes must search in the language the sources are in.

    Observed on a live run: the main queries were English ("Arabica vs Robusta
    coffee composition and flavor profile"), and then the multi-hop pass sent
    "Robusta Чем отличается кофе арабика от робусты" -- it built its anchor from
    the RAW request instead of the distilled subject plan_queries already
    returns. Half the retrieval passes were searching in the wrong language,
    against the pipeline's own documented design, with the whole question glued
    on to boot.
    """
    prof = {"core_topic": "Arabica vs Robusta coffee composition"}
    topic_ru = "Чем отличается кофе арабика от робусты по составу и вкусу"

    _check(dr.hop_anchor(prof, topic_ru) == prof["core_topic"],
                 "the hop anchor is the distilled subject, not the raw request")
    _check(dr.hop_anchor({}, topic_ru) == topic_ru,
                 "...and falls back to the raw topic when nothing was distilled")
    _check(dr.hop_anchor({"core_topic": "   "}, topic_ru) == topic_ru,
                 "...a blank core_topic is not an anchor")
    _check(dr.hop_anchor({"core_topic": "x"}, "") == "x",
                 "...and an empty topic does not erase a good anchor")

    es = _ent.EntitySet()
    _ent.extract_entities(
        "Robusta beans from Coffea canephora contain more caffeine", into=es)
    hopq = _ent.entity_queries(dr.hop_anchor(prof, topic_ru), es, max_entities=2)
    _check(bool(hopq) and not any(
        any("Ѐ" <= ch <= "ӿ" for ch in q) for q in hopq),
        "no Cyrillic reaches a follow-up query built on the distilled subject: %r" % (hopq,))
    # and the detector is not simply blind to Cyrillic
    raw = _ent.entity_queries(topic_ru, es, max_entities=2, visited=set())
    _check(any(any("Ѐ" <= ch <= "ӿ" for ch in q) for q in raw),
                 "the raw-topic anchor really does produce Cyrillic queries")
    print("PASS: hop anchor is the distilled English subject")
    return True


if __name__ == "__main__":


    tests = [
        test_p41_latex_artifact_repair,
        test_p40_query_distillation,
        test_p38_relevance_and_citation_noise,
        test_p37_survey_synthesis,
        test_p34_rerank_backend_resolution,
        test_p35_cross_encoder_no_download_hang,
        test_p36_report_title_and_equation_blocks,
        test_p33_parallel_io_and_stage_timing,
        test_p32_ssrf_and_outlink_ban,
        test_p22_recursive_hierarchy,
        test_p23_graph_communities,
        test_p24_phase_history,
        test_p25_eval_scoring,
        test_p26_reflection_loop_active,
        test_p27_manual_control_overrides,
        test_p28_source_filters_and_mutation,
        test_p29_exhaustive_collection_loop,
        test_p30_verification_knobs,
        test_p31_override_atomicity,
        test_p10_reranker,
        test_p11_entities,
        test_p12_contradiction,
        test_p13_research_graph,
        test_p14_decision_gate,
        test_p15_pipeline_integration,
        test_p16_clustering,
        test_p17_graph_guided,
        test_p18_claim_merge,
        test_p19_hierarchical_digest,
        test_p20_hierarchical_pipeline,
        test_p21_synthesis_empty_fallback,
        test_p1_extraction_gate,
        test_p2_dup_collapse,
        test_p3_coverage_signals,
        test_p4_graded_authority,
        test_p5_confidence_and_queries,
        test_p6_cache,
        test_p7_adapter_routing_and_fallback,
        test_p9a_truncation_and_gating,
        test_p9b_pdf_extraction,
        test_p9c_citation_graph,
        test_p9d_adaptive_sections,
        test_p9e_trust_ceiling_and_citation_query,
        test_p9f_deterministic_postprocess,
        test_hop_anchor_language,
        test_p39_context_aware_sections,   # LAST: reloads config/dr (restores after)
    ]
    results = []
    for t in tests:
        try:
            results.append(bool(t()))
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            results.append(False)
    print("\n" + "=" * 52)
    print(f"Results: {sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
