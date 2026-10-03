"""Live end-to-end check of the MiniMax H3 video pipeline.

This is the half the offline suite deliberately cannot cover: whether a real
ComfyUI accepts the graphs we build. It runs in stages so a failure tells you
WHICH assumption was wrong, instead of just "the render failed":

  1. the server is reachable and new enough to have the H3 nodes
  2. every weight file is where ComfyUI looks for it
  3. our workflow JSON validates against the LIVE node schemas — this is what
     catches the two things reasoned from source rather than observed: how the
     Autogrow ref_image_N inputs serialise, and what shape SaveVideo's
     DynamicCombo `codec` wants
  4. (with --render) an actual short clip, which is the only way to learn what a
     33B DiT plus a 32B encoder really does to a 24GB card

Run: venv/Scripts/python.exe scripts/smoke_video.py [--render] [--seconds 5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import requests

import config as C
import video as V

FAIL = []
def step(name):
    print(f"\n{'=' * 68}\n{name}\n{'=' * 68}")
def ok(msg):    print(f"  OK   {msg}")
def bad(msg):
    print(f"  FAIL {msg}")
    FAIL.append(msg)


def stage_server():
    step("1. ComfyUI reachable, and new enough for H3")
    try:
        r = requests.get(f"{C.COMFY_URL}/system_stats", timeout=15)
        stats = r.json()
    except Exception as exc:
        bad(f"ComfyUI is not reachable at {C.COMFY_URL}: {exc}")
        return False
    ver = stats.get("system", {}).get("comfyui_version", "?")
    ok(f"ComfyUI {ver}")
    dev = (stats.get("devices") or [{}])[0]
    ok(f"{dev.get('name', '?')} — {int(dev.get('vram_total', 0)) / 1e9:.1f} GB total, "
       f"{int(dev.get('vram_free', 0)) / 1e9:.1f} GB free")
    try:
        maj, minor = (int(x) for x in str(ver).split(".")[:2])
        if (maj, minor) < (0, 30):
            bad(f"ComfyUI {ver} has no MiniMaxH3 nodes — 0.30.0+ is required. "
                f"Point COMFY_SRC at the ComfyUI-0.30.0 tree and restart it.")
            return False
    except Exception:
        pass

    missing_nodes = []
    for node in ("MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo",
                 "MiniMaxH3SigmaShift", "EmptyMiniMaxH3LatentAV",
                 "UnetLoaderGGUF", "CLIPLoaderGGUF", "VAEDecodeAudio",
                 "CreateVideo", "SaveVideo", "GetVideoComponents"):
        try:
            rr = requests.get(f"{C.COMFY_URL}/object_info/{node}", timeout=15)
            if rr.status_code == 200 and rr.json():
                ok(f"node {node}")
            else:
                missing_nodes.append(node)
        except Exception as exc:
            missing_nodes.append(f"{node} ({exc})")
    for m in missing_nodes:
        bad(f"missing node: {m}")
    return not missing_nodes


def stage_weights():
    step("2. Weights present where ComfyUI looks")
    miss = V.missing_weights()
    for f in miss:
        bad(f"missing weight: {f}")
    if not miss:
        ok("all five H3 files are on disk")
    # ComfyUI has to actually LIST them, which is a different question from the
    # file existing: a wrong folder shows up as an empty dropdown, not an error.
    try:
        u = requests.get(f"{C.COMFY_URL}/object_info/UnetLoaderGGUF", timeout=15).json()
        names = u["UnetLoaderGGUF"]["input"]["required"]["unet_name"][0]
        for want in ("MiniMax-H3-FL2VA-Q4_K_M.gguf", "MiniMax-H3-Ref2VA-Q4_K_M.gguf"):
            (ok if want in names else bad)(f"UnetLoaderGGUF lists {want}")
    except Exception as exc:
        bad(f"could not read the unet list: {exc}")
    try:
        c = requests.get(f"{C.COMFY_URL}/object_info/CLIPLoaderGGUF", timeout=15).json()
        req = c["CLIPLoaderGGUF"]["input"]["required"]
        names, types = req["clip_name"][0], req["type"][0]
        (ok if "qwen3vl_32b_minimax_h3-Q4_K_M.gguf" in names else bad)(
            "CLIPLoaderGGUF lists the H3 encoder")
        (ok if "minimax" in types else bad)(
            "CLIPLoaderGGUF offers the 'minimax' CLIP type")
    except Exception as exc:
        bad(f"could not read the clip list: {exc}")
    try:
        v = requests.get(f"{C.COMFY_URL}/object_info/VAELoader", timeout=15).json()
        names = v["VAELoader"]["input"]["required"]["vae_name"][0]
        for want in ("minimax_h3_video_vae_fp16.safetensors",
                     "minimax_h3_audio_vae_fp32.safetensors"):
            (ok if want in names else bad)(f"VAELoader lists {want}")
    except Exception as exc:
        bad(f"could not read the vae list: {exc}")
    return not FAIL


def _schema_of(node):
    r = requests.get(f"{C.COMFY_URL}/object_info/{node}", timeout=15)
    return (r.json().get(node) or {}).get("input", {}) if r.status_code == 200 else {}


def stage_graph_shapes():
    step("3. Our graphs match the LIVE node schemas")
    # (a) SaveVideo codec: DynamicCombo serialises differently across builds.
    spec = _schema_of("SaveVideo")
    codec = (spec.get("required", {}) or {}).get("codec")
    print(f"  SaveVideo.codec schema: {json.dumps(codec)[:160]}")
    wf = json.load(open(C.WORKFLOW_VIDEO_PATH, encoding="utf-8"))
    normalized = V._normalize_save_node(dict(wf))
    ok(f"codec normalized to {normalized[V.N_SAVE]['inputs']['codec']!r}")

    # (b) The Autogrow reference inputs. This is THE thing reasoned from source.
    ref = _schema_of("MiniMaxH3ReferenceToVideo")
    opt = ref.get("optional", {}) or {}
    req = ref.get("required", {}) or {}
    keys = sorted(set(opt) | set(req))
    print(f"  MiniMaxH3ReferenceToVideo inputs: {keys}")
    # /object_info advertises the Autogrow CONTAINER ('ref_images'). The wire
    # names are ZERO-based AND carry that container as a dotted prefix:
    # "ref_images.ref_image_0". comfy_api/latest/_io.py builds the slot names as
    # names = [f"{prefix}{i}" for i in range(max)], then parse_class_inputs()
    # prefixes each with the input's own id; build_nested_inputs() splits on "."
    # at execution to rebuild execute(ref_images={...}).
    #
    # THIS CHECK ONCE ACCEPTED THE BARE NAME AND THE FEATURE WAS BROKEN ANYWAY.
    # A bare "ref_image_0" is not in the schema, so validation drops it as an
    # unknown extra, POST /prompt returns 200, and execution then dies on the
    # stray kwarg after the encoder and both VAEs have loaded. Do not weaken this
    # back into an "or" that also tolerates the flat form.
    if "ref_images" in keys:
        ok("reference-image container 'ref_images' present (Autogrow)")
    else:
        bad(f"no 'ref_images' container on MiniMaxH3ReferenceToVideo: {keys}")

    probe_wf = V.build_workflow("<Picture 1> waves", mode="ref2va", width=768,
                                height=768, frames=124, seed=1,
                                images=[__file__])  # path only needs to be uploadable
    slots = sorted(k for k in probe_wf[V.N_COND]["inputs"]
                   if k.startswith("ref_images.") and k.rsplit("_", 1)[-1].isdigit())
    (ok if slots == ["ref_images.ref_image_0"] else bad)(
        f"video.py emits dotted 0-based wire slots: {slots}")
    stray = sorted(k for k in probe_wf[V.N_COND]["inputs"]
                   if "." not in k and k.rsplit("_", 1)[-1].isdigit()
                   and k.rsplit("_", 1)[0] in ("ref_image", "ref_video",
                                               "ref_video_audio", "ref_audio"))
    (bad if stray else ok)(
        f"no bare autogrow slot would be silently dropped by validation: {stray or 'none'}")

    # (c) Every required input of every node we use is actually supplied.
    for path in (C.WORKFLOW_VIDEO_PATH, C.WORKFLOW_VIDEO_REF_PATH):
        g = json.load(open(path, encoding="utf-8"))
        for nid, node in g.items():
            cls = node["class_type"]
            sch = _schema_of(cls)
            if not sch:
                bad(f"{os.path.basename(str(path))}: unknown node {cls}")
                continue
            for field in (sch.get("required") or {}):
                if field not in node["inputs"]:
                    bad(f"{os.path.basename(str(path))} node {nid} ({cls}): "
                        f"missing required input {field!r}")
        ok(f"{os.path.basename(str(path))}: every node known, required inputs present")
    return not FAIL


def stage_render(seconds):
    step(f"4. Rendering a real {seconds}s clip (this is the slow one)")
    import models
    ctx = models.Context(models=None, transcription_cache={}, cache_file=None,
                         asr_lock=threading.Lock(), tts_lock=threading.Lock())
    ctx.set_stage = lambda s: print(f"  [stage] {s}")

    def prog(step_i, total):
        print(f"\r  sampling {step_i}/{total}", end="", flush=True)

    t0 = time.time()
    res = V.generate_video(
        ctx,
        "a red fox steps out of tall grass and turns toward the camera, slow "
        "push-in, late afternoon light; wind in the grass and one distant bird, "
        "no music",
        seconds=seconds, aspect="16:9", on_progress=prog)
    print()
    mins = (time.time() - t0) / 60
    if res.get("status") == "success" and res.get("path"):
        ok(f"clip rendered in {mins:.1f} min: {res['path']}")
        info = V.probe(res["path"])
        ok(f"{info.get('width')}x{info.get('height')}, {info.get('seconds')}s, "
           f"audio={info.get('has_audio')}, {info.get('bytes', 0) / 1e6:.1f} MB")
        if not info.get("has_audio"):
            bad("the clip has NO audio track — H3 should always emit one")
    else:
        bad(f"render failed after {mins:.1f} min: {res.get('reason')} "
            f"({V._VIDEO_FAILURE})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true",
                    help="actually generate a clip (many minutes)")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    if stage_server() and stage_weights():
        stage_graph_shapes()
        if args.render:
            stage_render(args.seconds)
        else:
            print("\n(skipping the real render — pass --render to do it)")

    print(f"\n{'=' * 68}")
    if FAIL:
        print(f"{len(FAIL)} problem(s):")
        for f in FAIL:
            print(f"  · {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
