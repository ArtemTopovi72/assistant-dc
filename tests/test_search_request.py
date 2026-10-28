"""Keyword rewrites must not erase the user's source and output requirements."""
from types import SimpleNamespace

import search
import tools


REQUEST = "Find the official Widget 4 release page and list two features with sources."


def test_search_handler_preserves_original_request_without_changing_engine_query():
    seen = {}

    def run(ctx, query):
        seen.update(request=getattr(ctx, "search_request", ""), query=query)
        return search.NO_RESULTS

    ctx = SimpleNamespace(
        deps=SimpleNamespace(run_web_search=run),
        set_stage=lambda stage: None,
        remember=lambda *args: None,
    )
    state = {"user_input_original": REQUEST, "user_input": "translated request"}
    tools._handle_search(ctx, state, {"query": "Widget 4 features"})
    assert seen == {"request": REQUEST, "query": "Widget 4 features"}
    assert not hasattr(ctx, "search_request")


def test_distillation_receives_full_request_alongside_retrieval_query():
    seen = []
    ctx = SimpleNamespace(
        search_request=REQUEST,
        choice=lambda *args: "0",
        text_model=lambda ctx, system, user, **kw: seen.append(user) or "facts",
    )
    assert search.distill_search_results(ctx, "Widget 4 features", "page text") == "facts"
    assert REQUEST in seen[0]
    assert "Widget 4 features" in seen[0] and "page text" in seen[0]


def test_followup_checks_original_requirements_after_keyword_rewrite():
    seen = []
    ctx = SimpleNamespace(
        search_request=REQUEST,
        text_model=lambda ctx, system, user, **kw: seen.append(user) or "NONE",
    )
    assert search.followup_query(ctx, "Widget 4 features", "third party facts") == ""
    assert REQUEST in seen[0]
