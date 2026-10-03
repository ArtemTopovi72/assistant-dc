"""SkillOpt plumbing, offline: the RRSI critic, edit application, source rewrite."""
import os, sys, copy, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tools
from bench import skillopt as S
from bench import skillopt_apply as A

DESC = {s["function"]["name"]: s["function"].get("description", "") for s in tools.TOOL_SCHEMAS}


def test_critic_rejects_quoting_a_case():
    text = S.CASES[5]["text"]
    words = text.split()
    assert len(words) >= 3
    e = {"tool": "search", "op": "append", "new": "e.g. " + " ".join(words[:3])}
    assert S.critic(e, DESC) and "quotes case text" in S.critic(e, DESC)


def test_critic_rejects_bloat_and_unknown_tool():
    assert "grows" in S.critic({"tool": "search", "op": "append", "new": "x" * 400}, DESC)
    assert S.critic({"tool": "no_such", "op": "append", "new": "fine"}, DESC) == "unknown tool"


def test_critic_passes_a_general_boundary():
    e = {"tool": "search", "op": "append",
         "new": "Not for writing or running code; use run_code for computation."}
    assert S.critic(e, DESC) is None


def test_apply_edits_does_not_mutate_base():
    base = copy.deepcopy(tools.TOOL_SCHEMAS)
    new = S.apply_edits(base, [{"tool": "search", "op": "append", "new": "ZZZ."}])
    assert base == tools.TOOL_SCHEMAS
    assert [s for s in new if s["function"]["name"] == "search"][0]["function"]["description"].endswith("ZZZ.")
    with pytest.raises(ValueError):
        S.apply_edits(base, [{"tool": "search", "op": "replace", "old": "not there", "new": "x"}])


def test_source_rewrite_round_trips(tmp_path):
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(A.__file__))), "agent/tool_descriptions.py")
    src = open(src_path, encoding="utf-8").read()
    edits = [{"tool": "search", "op": "append", "new": "Проверка \"кавычек\" и юникода."},
             {"tool": "calculate", "op": "replace",
              "old": DESC["calculate"][:30], "new": "REPLACED HEAD "}]
    out, changes = A.apply_to_source(src, tools.TOOL_SCHEMAS, edits)
    p = tmp_path / "td.py"
    p.write_text(out, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("td", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert m._SEARCH_DESC == DESC["search"].rstrip() + " Проверка \"кавычек\" и юникода."
    assert m._CALCULATE_DESC.startswith("REPLACED HEAD ")
    # everything else byte-identical in value
    orig = A._constants(src)
    for name, (val, _, _) in A._constants(out).items():
        if name not in ("_SEARCH_DESC", "_CALCULATE_DESC"):
            assert val == orig[name][0], name
