import os, sys
os.environ.setdefault("F5_TEST_RUN", "1")
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(R, d) for d in ("media", "agent", "services", "core", "voice", "")]
import video


def test_overrides_patch_known_nodes_only():
    wf = {"8": {"inputs": {"sampler_name": "er_sde"}}}
    os.environ["VIDEO_WF_OVERRIDES"] = '{"8.sampler_name": "euler", "99.x": 1}'
    try:
        video._apply_overrides(wf)
    finally:
        del os.environ["VIDEO_WF_OVERRIDES"]
    assert wf == {"8": {"inputs": {"sampler_name": "euler"}}}
    os.environ["VIDEO_WF_OVERRIDES"] = '{"90": {"class_type": "X", "inputs": {}}, "8.model": ["90", 0]}'
    try:
        video._apply_overrides(wf)
    finally:
        del os.environ["VIDEO_WF_OVERRIDES"]
    assert wf["90"]["class_type"] == "X" and wf["8"]["inputs"]["model"] == ["90", 0]


if __name__ == "__main__":
    test_overrides_patch_known_nodes_only(); print("ok")
