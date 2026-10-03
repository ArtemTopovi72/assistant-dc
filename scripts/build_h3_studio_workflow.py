"""Build the one pre-wired H3 "studio" workflow: text, images, video, audio.

Everything is connected up front and the reference loaders ship BYPASSED
(mode 4), so the graph runs as plain text->video the moment it opens. To use a
reference you unbypass that one node (select it, Ctrl+B) and pick a file —
no rewiring, no hunting for which slot takes what.

Why bypass rather than delete: MiniMaxH3ReferenceToVideo's ref_* inputs are
COMFY_AUTOGROW_V3 templates. Re-creating those slots by hand in the editor is
fiddly and easy to get wrong (they are 0-based, and the PROMPT refers to them
1-based as <Picture 1>). Bypassing keeps the wiring intact and inert.

Widget order and types come from the live server's /object_info, so this cannot
drift from the installed nodes.

Run (ComfyUI must be up):
  venv/Scripts/python.exe scripts/build_h3_studio_workflow.py --out <path.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import requests
from api_to_ui_workflow import convert  # noqa: E402  (same scripts/ dir)

COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8000")
BYPASS = 4  # LiteGraph "never" mode == ComfyUI bypass (Ctrl+B)

# The reference loaders. Unbypass one and its slot goes live.
REF_NODES = ("20", "21", "22", "30", "31", "40")

DEFAULT_PROMPT = (
    "a red fox steps out of tall grass and turns toward the camera, "
    "slow push-in, late afternoon light; wind in the grass and one distant "
    "bird, no music"
)


def first_file(cls: str, widget: str) -> str:
    try:
        d = requests.get(f"{COMFY_URL}/object_info/{cls}", timeout=20).json()[cls]
        spec = d["input"]["required"][widget]
        opts = spec[0] if isinstance(spec[0], list) else []
        return opts[0] if opts else ""
    except Exception:
        return ""


def build_keyframe_api() -> dict:
    """FL2VA: "animate THIS picture and keep its framing".

    Distinct from the Ref2VA graph in a way that matters and is easy to get
    wrong: here an image becomes an actual FRAME of the output, so the result
    starts as your picture. In Ref2VA an image is only a REFERENCE — the model
    composes a fresh shot from it and reframes the subject, which reads as
    "it cropped my image" even though nothing is cropped.

    first_frame is stretched to the canvas (crop "disabled"); last_frame gets an
    aspect-preserving COVER-CROP (crop "center") in the node. So a mismatched
    aspect distorts the first frame but genuinely crops the last one — set the
    canvas from the image with legal_canvas() to avoid both.
    """
    img = first_file("LoadImage", "image")
    return {
        "1": {"class_type": "UnetLoaderGGUF",
              "inputs": {"unet_name": "MiniMax-H3-FL2VA-Q4_K_M.gguf"}},
        "2": {"class_type": "MiniMaxH3SigmaShift",
              "inputs": {"model": ["1", 0], "shift_video": 12.0,
                         "shift_audio": 3.0}},
        "3": {"class_type": "CLIPLoaderGGUF",
              "inputs": {"clip_name": "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
                         "type": "minimax"}},
        "4": {"class_type": "VAELoader",
              "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "5": {"class_type": "VAELoader",
              "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "20": {"class_type": "LoadImage", "inputs": {"image": img}},
        "21": {"class_type": "LoadImage", "inputs": {"image": img}},
        "6": {"class_type": "MiniMaxH3ImageToVideo",
              "inputs": {"clip": ["3", 0], "vae": ["4", 0],
                         "prompt": DEFAULT_PROMPT,
                         "width": 1344, "height": 768, "length": 124,
                         "first_frame": ["20", 0],
                         "last_frame": ["21", 0]}},
        "7": {"class_type": "ConditioningZeroOut",
              "inputs": {"conditioning": ["6", 0]}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["2", 0], "positive": ["6", 0],
                         "negative": ["7", 0], "latent_image": ["6", 1],
                         "seed": 0, "steps": 8, "cfg": 1.0,
                         "sampler_name": "euler", "scheduler": "simple",
                         "denoise": 1.0}},
        "9": {"class_type": "VAEDecode",
              "inputs": {"samples": ["8", 0], "vae": ["4", 0]}},
        "10": {"class_type": "VAEDecodeAudio",
               "inputs": {"samples": ["8", 0], "vae": ["5", 0]}},
        "11": {"class_type": "CreateVideo",
               "inputs": {"images": ["9", 0], "fps": 24.0, "audio": ["10", 0]}},
        "12": {"class_type": "SaveVideo",
               "inputs": {"video": ["11", 0],
                          "filename_prefix": "video/h3_keyframe",
                          "format": "auto", "codec": "auto"}},
    }


def build_api() -> dict:
    img = first_file("LoadImage", "image")
    return {
        # ── engine ────────────────────────────────────────────────────────
        "1": {"class_type": "UnetLoaderGGUF",
              "inputs": {"unet_name": "MiniMax-H3-Ref2VA-Q4_K_M.gguf"}},
        "2": {"class_type": "MiniMaxH3SigmaShift",
              "inputs": {"model": ["1", 0], "shift_video": 12.0,
                         "shift_audio": 3.0}},
        "3": {"class_type": "CLIPLoaderGGUF",
              "inputs": {"clip_name": "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
                         "type": "minimax"}},
        "4": {"class_type": "VAELoader",
              "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "5": {"class_type": "VAELoader",
              "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},

        # ── reference inputs (shipped bypassed) ───────────────────────────
        "20": {"class_type": "LoadImage", "inputs": {"image": img}},
        "21": {"class_type": "LoadImage", "inputs": {"image": img}},
        "22": {"class_type": "LoadImage", "inputs": {"image": img}},
        "30": {"class_type": "LoadVideo", "inputs": {"file": ""}},
        "31": {"class_type": "GetVideoComponents",
               "inputs": {"video": ["30", 0]}},
        "40": {"class_type": "LoadAudio", "inputs": {"audio": ""}},

        # ── conditioning + AV latent ──────────────────────────────────────
        # Autogrow slots are 0-BASED and carry the container name as a DOTTED
        # prefix ("ref_images.ref_image_0"); execute() receives them regrouped
        # as ref_images={...}. A bare "ref_image_0" is not in the schema, so
        # POST /prompt happily returns 200 and execution then dies with
        # "unexpected keyword argument" after the encoder has already loaded.
        # The prompt still cites them 1-based as <Picture 1>.
        "6": {"class_type": "MiniMaxH3ReferenceToVideo",
              "inputs": {"clip": ["3", 0], "vae": ["4", 0],
                         "audio_vae": ["5", 0],
                         "prompt": DEFAULT_PROMPT,
                         "width": 1344, "height": 768, "length": 124,
                         "ref_image_size": "match",
                         "ref_images.ref_image_0": ["20", 0],
                         "ref_images.ref_image_1": ["21", 0],
                         "ref_images.ref_image_2": ["22", 0],
                         "ref_videos.ref_video_0": ["31", 0],
                         "ref_video_audios.ref_video_audio_0": ["31", 1],
                         "ref_audios.ref_audio_0": ["40", 0]}},
        "7": {"class_type": "ConditioningZeroOut",
              "inputs": {"conditioning": ["6", 0]}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["2", 0], "positive": ["6", 0],
                         "negative": ["7", 0], "latent_image": ["6", 1],
                         "seed": 0, "steps": 8, "cfg": 1.0,
                         "sampler_name": "euler", "scheduler": "simple",
                         "denoise": 1.0}},
        "9": {"class_type": "VAEDecode",
              "inputs": {"samples": ["8", 0], "vae": ["4", 0]}},
        "10": {"class_type": "VAEDecodeAudio",
               "inputs": {"samples": ["8", 0], "vae": ["5", 0]}},
        "11": {"class_type": "CreateVideo",
               "inputs": {"images": ["9", 0], "fps": 24.0, "audio": ["10", 0]}},
        "12": {"class_type": "SaveVideo",
               "inputs": {"video": ["11", 0],
                          "filename_prefix": "video/h3_studio",
                          "format": "auto", "codec": "auto"}},
    }


TITLES = {
    "1": "① H3 Ref2VA transformer",
    "2": "flow shift (video 12 / audio 3)",
    "3": "② H3 text+vision encoder",
    "4": "③ Visual VAE", "5": "④ Audio VAE",
    "20": "🖼 Reference image 1  → <Picture 1>",
    "21": "🖼 Reference image 2  → <Picture 2>",
    "22": "🖼 Reference image 3  → <Picture 3>",
    "30": "🎬 Reference video    → <Video 1>",
    "31": "   split video into frames + sound",
    "40": "🔊 Reference audio    → <Audio 1>",
    "6": "★ PROMPT + size + length",
    "7": "negative (inert: model is CFG-distilled)",
    "8": "sample the packed A/V latent",
    "9": "decode video", "10": "decode audio",
    "11": "mux — fps MUST stay 24", "12": "💾 save → <output dir>/video",
}

POS = {
    "1": (0, 0), "2": (0, 170), "3": (0, 340), "4": (0, 500), "5": (0, 640),
    "20": (470, 0), "21": (470, 430), "22": (470, 860),
    "30": (470, 1290), "31": (470, 1470), "40": (470, 1650),
    "6": (1000, 300), "7": (1000, 900),
    "8": (1450, 300),
    "9": (1900, 200), "10": (1900, 460),
    "11": (2300, 260), "12": (2300, 520),
}

SIZES = {"20": (330, 380), "21": (330, 380), "22": (330, 380),
         "6": (420, 330), "8": (400, 290)}

NOTE = """MiniMax H3 — studio workflow

RUN AS-IS  →  text-to-video. Just edit the prompt in the pink node and hit Run.

TO USE A REFERENCE
  1. click the loader you want (image / video / audio)
  2. Ctrl+B to unbypass it (colour returns)
  3. pick or upload your file
  4. mention it in the prompt as <Picture 1>, <Video 1>, <Audio 1>
     (wires are 0-based, the prompt is 1-based - that is normal)

HARD RULES - changing these breaks the render
  cfg      MUST stay 1.0  (weights are CFG-distilled; negatives do nothing)
  length   MUST satisfy  n % 17 == 5   ->  5, 22, 39, 56, ... 124 = 5.2s
  size     768 short edge, 768*1344 area cap, both axes multiple of 32
  codec/format  leave on "auto" (a non-string here fails AFTER the whole render)

SPEED  ~55 s/step at 8 steps => ~8 min for 5s.
  Unload LM Studio first (lms unload --all). If the chat model is resident
  the render thrashes and takes 20x longer.

Audio is generated by the model itself - you get a stereo track for free.

WHERE THE FILE LANDS
  Comfy Desktop  ->  Documents\\ComfyUI\\output\\video\\
  (the assistant's own renders go to the project's runtime\\video instead)

LENGTH vs FPS - the one that silently desyncs the sound
  H3 is a 24 fps model. CreateVideo's fps does NOT change what was generated,
  it only decides how fast those frames are played.
  Lower it to 16 and the SAME frames stretch over 1.5x longer, so the (complete)
  audio ends early and looks like "the sound cut out at 12 seconds".
  Want a longer clip WITH sound? raise `length` (frames), never lower fps.
      length 124 -> 5.2s     length 294 -> 12.25s     length 362 -> 15.1s"""


KEY_TITLES = {
    "1": "① H3 FL2VA transformer (keyframes)",
    "2": "flow shift (video 12 / audio 3)",
    "3": "② H3 text+vision encoder",
    "4": "③ Visual VAE", "5": "④ Audio VAE",
    "20": "🖼 FIRST frame — your picture (stretched to canvas)",
    "21": "🖼 LAST frame — optional; COVER-CROPPED, keep aspect equal",
    "6": "★ PROMPT + size + length",
    "7": "negative (inert: model is CFG-distilled)",
    "8": "sample the packed A/V latent",
    "9": "decode video", "10": "decode audio",
    "11": "mux — fps MUST stay 24", "12": "💾 save → <output dir>/video",
}

KEY_POS = {
    "1": (0, 0), "2": (0, 170), "3": (0, 340), "4": (0, 500), "5": (0, 640),
    "20": (470, 0), "21": (470, 430),
    "6": (1000, 100), "7": (1000, 620),
    "8": (1450, 100), "9": (1900, 60), "10": (1900, 320),
    "11": (2300, 120), "12": (2300, 380),
}

KEY_NOTE = """MiniMax H3 — KEYFRAME workflow (FL2VA)

USE THIS when you have a picture and want IT animated, framing intact.
Your image becomes an actual frame of the output.

Do NOT use the studio (Ref2VA) workflow for that: there an image is only a
REFERENCE. The model composes a NEW shot from it and reframes the subject,
which looks like it cropped your picture. Nothing is cropped - it is simply a
different task.

  1 image  -> leave "LAST frame" bypassed. Your picture is frame 0.
  2 images -> unbypass "LAST frame" (Ctrl+B). Video morphs first -> last.

AVOIDING DISTORTION / CROP
  first_frame is STRETCHED to width x height  (crop disabled)
  last_frame  is COVER-CROPPED to width x height  (crop center)
  So a canvas whose aspect differs from your image squashes the first frame
  and really does crop the last one.

  Get the right canvas for your picture:
      venv/Scripts/python.exe scripts/build_h3_studio_workflow.py --canvas-for <image>

  You CANNOT have exact 16:9 here. H3's cap is 768*1344 px with axes on a
  multiple of 32, so 16:9 at the trained 768 short edge would need 1365 wide.
  1344x768 (=7:4) is the closest legal canvas - a 1.6% difference.

cfg stays 1.0 | length % 17 == 5 | ~55 s/step at 8 steps

LENGTH vs FPS - the one that silently desyncs the sound
  H3 is a 24 fps model. CreateVideo's fps does NOT change what was generated,
  it only decides how fast those frames are played.
  Lower it to 16 and the SAME frames stretch over 1.5x longer, so the (complete)
  audio ends early and looks like "the sound cut out at 12 seconds".
  Want a longer clip WITH sound? raise `length` (frames), never lower fps.
      length 124 -> 5.2s     length 294 -> 12.25s     length 362 -> 15.1s"""


def legal_canvas(width: int, height: int) -> tuple[int, int]:
    """Best legal H3 canvas for a source image: smallest aspect error.

    fit_canvas-style rounding of each axis independently can drift the aspect
    (1920x1080 -> 1344x768 turns 1.778 into 1.750). Here we search every /32
    pair under the pixel cap and take the closest aspect, preferring more
    pixels on a tie, so the stretch/crop the node applies is as small as the
    model's grid allows.
    """
    return canvas_candidates(width, height)[0][0]


def canvas_candidates(width: int, height: int, limit: int = 3):
    """Legal canvases for this aspect, best first, as ((w,h), err%, short_edge).

    Ranking deliberately favours the TRAINED short edge over a perfect aspect.
    A pure "smallest aspect error" search recommends 1024x576 for a 16:9 photo:
    exact aspect, but the short edge drops 768 -> 576 on a model trained at 768.
    That trades visible quality for a 1.6% aspect difference nobody can see.
    So: keep the short edge as close to 768 as the grid allows, and only then
    minimise aspect error.
    """
    target = width / height
    rows = []
    for h in range(512, 1345, 32):
        for w in range(512, 1345, 32):
            if w * h > MAX_PIXELS:
                continue
            short = min(w, h)
            err = abs((w / h) - target) / target * 100.0
            # short-edge shortfall dominates; aspect error breaks the tie
            rows.append((((BASE_SHORT_EDGE - short) if short < BASE_SHORT_EDGE else 0,
                          round(err, 2), -(w * h)), (w, h), err, short))
    rows.sort(key=lambda r: r[0])
    return [(r[1], r[2], r[3]) for r in rows[:limit]]


BASE_SHORT_EDGE = 768


MAX_PIXELS = 768 * 1344


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--mode", choices=("studio", "keyframe"), default="studio",
                    help="studio = Ref2VA omni-reference; keyframe = FL2VA first/last frame")
    ap.add_argument("--canvas-for", default="",
                    help="print the best legal canvas for this image and exit")
    args = ap.parse_args()

    if args.canvas_for:
        from PIL import Image
        with Image.open(args.canvas_for) as im:
            w, h = im.size
        print(f"image        : {w}x{h}  (aspect {w/h:.4f})")
        cands = canvas_candidates(w, h, limit=4)
        (cw, ch), err, short = cands[0]
        print(f"USE          : width={cw}  height={ch}   "
              f"(aspect {cw/ch:.4f}, off by {err:.2f}%, short edge {short})")
        print()
        print("  other legal canvases:")
        for (aw, ah), aerr, ashort in cands[1:]:
            print(f"    {aw}x{ah:<5}  off by {aerr:5.2f}%   short edge {ashort}")
        print()
        print("  A sub-2% aspect difference is invisible; a short edge below 768")
        print("  is a real quality loss (H3 was trained at 768). Prefer the top row.")
        return 0

    if not args.out:
        ap.error("--out is required unless --canvas-for is used")

    if args.mode == "keyframe":
        ui = convert(build_keyframe_api())
        titles, positions, note, bypass_ids = KEY_TITLES, KEY_POS, KEY_NOTE, ("21",)
    else:
        ui = convert(build_api())
        titles, positions, note, bypass_ids = TITLES, POS, NOTE, REF_NODES
    return _finish(ui, args.out, titles, positions, note, bypass_ids, args.mode)


def _finish(ui, out_path, titles, positions, note, bypass_ids, mode) -> int:
    for n in ui["nodes"]:
        nid = str(n["id"])
        if nid in titles:
            n["title"] = titles[nid]
        if nid in positions:
            n["pos"] = list(positions[nid])
        if nid in SIZES:
            n["size"] = list(SIZES[nid])
        if nid in bypass_ids:
            n["mode"] = BYPASS
        # The autogrow ref_* slots are templates, so the generic converter
        # cannot type them from /object_info. Type them here or the editor
        # draws them as wildcards and refuses the reconnect.
        for inp in n.get("inputs", []):
            name = inp["name"]
            if name.startswith(("ref_audios.", "ref_video_audios.")):
                inp["type"] = "AUDIO"          # audio FIRST: ref_video_audios.*
            elif name.startswith(("ref_images.", "ref_videos.")):
                inp["type"] = "IMAGE"          # ref_videos carry frames, not VIDEO
    for l in ui["links"]:
        if l[5] in (None, "*"):
            l[5] = "IMAGE"

    note_pos = [1000, 1290] if mode == "studio" else [1000, 900]
    ui["nodes"].append({
        "id": 99, "type": "Note", "pos": note_pos, "size": [620, 470],
        "flags": {}, "order": 99, "mode": 0, "inputs": [], "outputs": [],
        "title": "📖 READ ME", "properties": {}, "widgets_values": [note],
        "color": "#432", "bgcolor": "#653",
    })
    ui["last_node_id"] = 99
    if mode == "studio":
        ui["groups"] = [
            {"title": "engine (load once)", "bounding": [-30, -60, 420, 830],
             "color": "#3f789e", "font_size": 24, "flags": {}},
            {"title": "REFERENCES — bypassed; Ctrl+B to enable",
             "bounding": [440, -60, 390, 1900], "color": "#a1309b",
             "font_size": 24, "flags": {}},
            {"title": "generate", "bounding": [970, -60, 880, 1180],
             "color": "#8A8", "font_size": 24, "flags": {}},
            {"title": "decode + save", "bounding": [1870, -60, 800, 800],
             "color": "#b58b2a", "font_size": 24, "flags": {}},
        ]
    else:
        ui["groups"] = [
            {"title": "engine (load once)", "bounding": [-30, -60, 420, 830],
             "color": "#3f789e", "font_size": 24, "flags": {}},
            {"title": "YOUR PICTURE = an actual frame",
             "bounding": [440, -60, 390, 880], "color": "#a1309b",
             "font_size": 24, "flags": {}},
            {"title": "generate", "bounding": [970, -60, 880, 780],
             "color": "#8A8", "font_size": 24, "flags": {}},
            {"title": "decode + save", "bounding": [1870, -60, 800, 660],
             "color": "#b58b2a", "font_size": 24, "flags": {}},
        ]

    out = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(ui, fh, indent=1, ensure_ascii=False)
    byp = sum(1 for n in ui["nodes"] if n.get("mode") == BYPASS)
    print(f"wrote {out}")
    print(f"  mode={mode}  {len(ui['nodes'])} nodes ({byp} bypassed), "
          f"{len(ui['links'])} links")
    return 0


if __name__ == "__main__":
    sys.exit(main())
