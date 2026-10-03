import json, os, sys
os.environ.setdefault("F5_TEST_RUN", "1")
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(R, d) for d in ("media", "agent", "services", "core", "voice", "")]
import video


def test_readiness_checks_the_unets_the_graphs_load():
    """missing_weights() must ask for the files the workflows really load, not a stale build."""
    req = video.REQUIRED_FILES
    for name in ("workflow_video_h3.json", "workflow_video_h3_ref.json"):
        wf = json.load(open(os.path.join(R, "workflows", "video", name), encoding="utf-8"))
        for node in wf.values():
            unet = node["inputs"].get("unet_name")
            if unet:
                assert unet in req.get("diffusion_models", []) + req.get("unet", []), (name, unet)


if __name__ == "__main__":
    test_readiness_checks_the_unets_the_graphs_load(); print("ok")
