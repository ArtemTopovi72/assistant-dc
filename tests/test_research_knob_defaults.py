"""Research tab's knob defaults are the real values: the knobs live behind the
dr_settings proxy, so vars(deep_research) saw none of them and every default
came back None (the tab showed blank fields)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import research_client


def test_knob_defaults_are_filled():
    d = research_client.InProcessResearchClient().knob_defaults()
    assert d and all(v is not None for v in d.values()), d
