"""Continuing a clip pins its last 22 frames + 1 s of sound into the new clip's own
timeline (H3 Motion Context) instead of feeding the tail in as a <Video 1> reference,
and the pinned head is trimmed off picture and sound before the clip is saved."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"

import video as V

ok = []


def check(c, m):
    ok.append(bool(c)); print(("ok   " if c else "FAIL ") + m)


V._upload = lambda p, kind="image": os.path.basename(p)

for two in (False, True):
    wf = V.build_workflow("next", mode="ref2va", width=1344, height=768, frames=124, seed=7,
                          images=["seed.jpg"], two_stage=two, context_video="tail.mp4")
    by = {v["class_type"]: k for k, v in wf.items()}
    mc = by.get("MiniMaxH3MotionContext")
    check(mc is not None, f"two_stage={two}: a Motion Context node is added")
    if not mc:
        continue
    i = wf[mc]["inputs"]
    check(i["context_length"] == "22" and i["conditioning"] == [V.N_COND, 0]
          and i["latent"] == [V.N_COND, 1], "pins 22 frames on the clip's own conditioning")
    comp = i["context_frames"][0]
    check(wf[comp]["class_type"] == "GetVideoComponents" and i["context_audio"] == [comp, 1],
          "frames and sound both come from the previous clip")
    check(wf[V.N_SAMPLER]["inputs"]["positive"] == [mc, 0], "the sampler reads the pinned conditioning")
    check(not any(k.startswith("ref_videos.") for k in wf[V.N_COND]["inputs"]),
          "the previous clip is NOT a reference video")
    trim = by["MiniMaxH3MotionContextTrim"]
    cv = wf[wf[V.N_SAVE]["inputs"]["video"][0]]["inputs"]
    check(cv["images"] == [trim, 0] and cv["audio"] == [trim, 1]
          and wf[trim]["inputs"]["trim_frames"] == [mc, 1], "the pinned head is trimmed before saving")
    if two:
        g = next(v for v in wf.values() if v["class_type"] == "CFGGuider")
        check(g["inputs"]["positive"] == [V.N_COND, 0], "the refine keeps the plain conditioning")

wf = V.build_workflow("x", mode="ref2va", width=1344, height=768, frames=124, seed=7,
                      images=["seed.jpg"], two_stage=False)
check(not any(v["class_type"] == "MiniMaxH3MotionContext" for v in wf.values()),
      "no context video, no Motion Context node")

line = "He smiles and says «Вот теперь отлично!»"
check(V.estimate_seconds(V.CONTINUE_CTX_PREFIX + line) == V.estimate_seconds(line),
      "the continuation template does not lengthen the clip")

# a continuation with no picture keeps the previous clip's portrait shape, not 16:9
seen = {}
V.engine_available = lambda ctx=None: (True, "")
V.probe = lambda p: {"width": 768, "height": 1024, "seconds": 2.5, "has_audio": True}
V._context_ir_on = lambda: False
V.build_workflow = lambda *a, **k: seen.update(k) or {}
V._normalize_save_node = lambda wf: wf
import comfy_client
comfy_client._submit_and_poll = lambda *a, **k: None
comfy_client.last_failure = lambda: "x"
V.generate_video(None, V.CONTINUE_CTX_PREFIX + line, context_video="tail.mp4", seed=5)
check((seen.get("width"), seen.get("height")) == V.fit_canvas(768, 1024),
      f"continuation canvas follows the clip: {seen.get('width')}x{seen.get('height')}")
check(seen.get("mode") == "t2va" and seen.get("context_video") == "tail.mp4",
      "pinned continuation runs the plain FL2V graph")

os.environ["VIDEO_MOTION_CONTEXT"] = "0"
check(V.motion_context_on() is False, "VIDEO_MOTION_CONTEXT=0 turns it off")
os.environ.pop("VIDEO_MOTION_CONTEXT")

print(f"\n{sum(ok)}/{len(ok)}")
sys.exit(0 if all(ok) else 1)
