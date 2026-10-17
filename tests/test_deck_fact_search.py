"""The deck's web facts are searched by the topic, not by the whole outline.

The agent hands create_presentation the outline («How to care for a cat (for children):
1. Introduction - meet…»); searched as is, it came back as Pinterest, Chess.com,
Duolingo and Google Keep, and those were the sources slide of a children's deck.
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, d) for d in ("services", "agent", "core", "research")]
import slides  # noqa: E402

asked = []
JUNK = [{"title": "Discover recipes, home ideas, style inspiration", "href": "https://ca.pinterest.com/",
         "domain": "pinterest.com", "body": "Discover recipes, home ideas, style inspiration."},
        {"title": "Chess.com - Play Chess Online", "href": "https://chess.com/", "domain": "chess.com",
         "body": "Play chess online for free."}]
GOOD = [{"title": "Teaching Kids to Care for Cats", "href": "https://example.org/kids-cats",
         "domain": "example.org", "body": "Cats need fresh water every day."}]


def search(ctx, q, max_results=8):
    asked.append(q)
    return JUNK + GOOD


ctx = types.SimpleNamespace(deps=types.SimpleNamespace(raw_search_results=search))
digest, sources = slides.gather_facts(
    ctx, "How to care for a cat (for children): 1. Introduction - meeting your new friend; 2. Food")
assert asked == ["How to care for a cat (for children)"], asked
assert [s["url"] for s in sources] == ["https://example.org/kids-cats"], sources
assert "pinterest" not in digest and "fresh water" in digest, digest
print("3/3 checks passed")
