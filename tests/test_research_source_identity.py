"""A source's hostname, not URL credentials or a lookalike, drives trust."""
import pytest

import dr_policy
import dr_urls


@pytest.mark.parametrize("url", [
    "https://wikipedia.org.attacker.example/article",
    "https://fakewikipedia.org/article",
    "https://agency.gov.attacker.example/article",
    "https://university.edu.attacker.example/article",
    "https://wikipedia.org@attacker.example/article",
])
def test_lookalike_authority_does_not_raise_low_trust(url):
    assert dr_urls._authority_score(url) == dr_urls._MID_SCORE
    assert dr_policy._apply_trust_floor("LOW", url) == "LOW"


@pytest.mark.parametrize("url", [
    "https://en.wikipedia.org/article", "https://www.nist.gov/article",
    "https://mit.edu/article", "https://www.cam.ac.uk/article",
])
def test_authority_hosts_keep_their_trust_floor(url):
    assert dr_policy._apply_trust_floor("LOW", url) == "SECONDARY"


def test_source_identity_ignores_credentials_and_port():
    assert dr_urls._domain("https://wikipedia.org@www.example.com:8443/article") == "example.com"
    sources = [{"href": "https://example.com:443/a"},
               {"href": "https://example.com:8443/b"}]
    assert dr_urls._cap_per_domain(sources, 1) == sources[:1]


@pytest.mark.parametrize("category,host", [
    ("news", "reuters.com"), ("science", "nature.com"),
    ("product", "github.com"), ("health", "who.int"),
    ("jobs", "linkedin.com"), ("local", "afisha.ru"),
])
def test_category_authority_uses_real_domain_boundaries(category, host):
    url = f"https://{host}.attacker.example/article"
    assert dr_urls._authority_score(url, category) == dr_urls._MID_SCORE
    assert dr_policy._apply_trust_floor("LOW", url, category) == "LOW"
    assert dr_urls._authority_score(f"https://sub.{host}/article", category) == dr_urls._AUTH_CATEGORY_SCORE


@pytest.mark.parametrize("category,host", [
    ("science", "link.springer.com"), ("science", "www.sciencedirect.com"),
    ("news", "bbc.co.uk"), ("jobs", "careers.example.com"),
    ("jobs", "jobs.example.com"), ("finance", "investor.example.com"),
])
def test_explicit_category_domain_and_semantic_rules_keep_working(category, host):
    assert dr_urls._authority_score(f"https://{host}/article", category) == dr_urls._AUTH_CATEGORY_SCORE
