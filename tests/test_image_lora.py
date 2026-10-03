"""LoRA injection, checked against the REAL workflow file.

A fixture graph with one loader would pass every check here and still miss the
defect that matters: workflow_ideogram4.json loads the model twice (the
positive and the negative branch), and an adapter wired into only one of them
is silently undone by the other. (It used to be the old model's two-stage
workflow_turbo.json, removed with old-model.)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import image_lora as L

_WF = Path(__file__).resolve().parent.parent / "workflows" / "image" / "workflow_ideogram4.json"


def _graph():
    return json.loads(_WF.read_text(encoding="utf-8"))


def _model_loaders(g):
    return [n for n, v in g.items() if v.get("class_type") in L._MODEL_LOADERS]


def test_every_sampling_stage_gets_the_adapter():
    g = _graph()
    loaders = _model_loaders(g)
    assert len(loaders) >= 2, "the real workflow is expected to have two model loaders"
    out = L.inject_lora(g, "neurostepan.safetensors", 0.9)
    lora_nodes = [n for n, v in out.items()
                  if v["class_type"] == "LoraLoaderModelOnly"]
    assert len(lora_nodes) == len(loaders)


def test_no_consumer_still_reads_the_bare_model():
    """The whole point: nothing may bypass the adapter."""
    g = _graph()
    loaders = set(_model_loaders(g))
    out = L.inject_lora(g, "x.safetensors")
    for nid, node in out.items():
        if node["class_type"] == "LoraLoaderModelOnly":
            continue                     # this one legitimately reads the loader
        for val in node.get("inputs", {}).values():
            if isinstance(val, list) and len(val) == 2:
                assert str(val[0]) not in loaders, (
                    "%s still samples with the un-adapted model" % nid)


def test_strength_is_carried_through():
    out = L.inject_lora(_graph(), "x.safetensors", 0.55)
    for v in out.values():
        if v["class_type"] == "LoraLoaderModelOnly":
            assert v["inputs"]["strength_model"] == 0.55


def test_round_trip_restores_the_original_graph():
    g = _graph()
    assert L.strip_lora(L.inject_lora(g, "x.safetensors")) == g


def test_a_graph_with_nothing_to_attach_to_is_refused():
    for bad in ({}, {"9": {"class_type": "SaveImage", "inputs": {}}}):
        try:
            L.inject_lora(bad, "x.safetensors")
        except ValueError:
            continue
        raise AssertionError("returned a graph that renders a stranger")


def test_empty_lora_name_is_refused():
    try:
        L.inject_lora(_graph(), "")
    except ValueError:
        return
    raise AssertionError("accepted an empty adapter name")


def test_trigger_is_added_once():
    assert L.prompt_with_trigger("in a red coat", "neurostepan") == \
        "neurostepan, in a red coat"
    # Already present, in any case: do not double the token.
    assert L.prompt_with_trigger("photo of NeuroStepan outside", "neurostepan") == \
        "photo of NeuroStepan outside"
    assert L.prompt_with_trigger("", "neurostepan") == "neurostepan"


def test_every_wire_still_points_at_a_real_node():
    """Caught by mutation: a broken injection left references to nodes that
    were never created, and the 'nothing bypasses the adapter' check above
    passed precisely because the dangling id was no longer a loader id.
    ComfyUI would reject such a graph at submit time, far from the cause."""
    out = L.inject_lora(_graph(), "x.safetensors")
    for nid, node in out.items():
        for key, val in node.get("inputs", {}).items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                assert val[0] in out, "%s.%s -> missing node %s" % (nid, key, val[0])
