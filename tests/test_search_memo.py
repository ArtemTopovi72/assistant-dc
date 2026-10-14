"""One search asked the model the same two questions up to five times (outbound filter x2,
item count x3): a 92 s turn. Within a turn each is asked once."""
import os, sys, types
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import search as S


def ctx_with(calls):
    return types.SimpleNamespace(
        turn_queries=[],
        yes_no=lambda q, t, default=None: calls.append("yn") or False,
        choice=lambda q, t, opts, default: calls.append("ch") or "0")


def test_each_question_is_asked_once_per_turn():
    calls = []
    c = ctx_with(calls)
    for _ in range(3):
        assert S.wanted_count("q", ctx=c) == 0
        assert S.outbound_blocked("q", ctx=c) is False
    assert calls.count("ch") == 1 and calls.count("yn") == 1, calls


def test_a_new_turn_asks_again():
    calls = []
    c = ctx_with(calls)
    S.wanted_count("q", ctx=c)
    c.turn_queries = []
    S.wanted_count("q", ctx=c)
    assert calls.count("ch") == 2


def test_a_bare_context_remembers_nothing():
    calls = []
    c = ctx_with(calls)
    del c.turn_queries
    S.wanted_count("q", ctx=c)
    S.wanted_count("q", ctx=c)
    assert calls.count("ch") == 2
