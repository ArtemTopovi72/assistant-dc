"""H3 Fun ControlNet graph: the patch sits between the model chain and the sampler."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import video as V
import video_control as C


def test_canny_graph_wiring():
    wf = C.build_control_workflow("anime style", "clip.mp4", frames=124, width=1344, height=768, seed=7)
    s = wf[V.N_SAMPLER]["inputs"]
    assert s["model"] == [C.N_APPLY, 0]
    ap = wf[C.N_APPLY]["inputs"]
    assert ap["model"] == ["2", 0] and ap["model_patch"] == ["35", 0]
    assert ap["control_video"] == ["34", 0] and wf["34"]["class_type"] == "Canny"
    assert wf["32"]["inputs"]["length"] == 124
    assert (wf["33"]["inputs"]["width"], wf["33"]["inputs"]["height"]) == (1344, 768)
    assert wf[V.N_COND]["inputs"]["prompt"] == "anime style"
    assert "first_frame" not in wf[V.N_COND]["inputs"]


def test_raw_control_and_first_frame():
    wf = C.build_control_workflow("x", "pose.mp4", kind="raw", first_frame="me.png", seed=1)
    assert "34" not in wf and wf[C.N_APPLY]["inputs"]["control_video"] == ["33", 0]
    assert wf[V.N_COND]["inputs"]["first_frame"] == ["20", 0]
    assert wf["20"]["inputs"]["image"] == "me.png"


def test_unknown_kind_refused():
    with pytest.raises(ValueError):
        C.build_control_workflow("x", "a.mp4", kind="depth")
