"""Video generation (MiniMax H3) — offline, with ComfyUI/LM-Studio/GPU intercepted.

Nothing here touches a GPU, a 60GB checkpoint or a running ComfyUI: every external
boundary (upload, submit/poll, ffmpeg, the Telegram API) is replaced. What IS
exercised is the part that decides what gets rendered and what gets delivered:

  * the model's hard grids — H3 only accepts frame counts where n % 17 == 5 and a
    768-short-edge / 1MP canvas in multiples of 32. Sending anything else either
    errors in the node or renders a shape the model never saw.
  * mode routing — any image (or a reference video/audio) goes through the
    omni-reference checkpoint (Ref2VA). The keyframe checkpoint (FL2VA, modes
    i2va/flf2va) is disabled: live, 2026-09-19, every job through it on this
    deployment hung in KSampler and needed a manual /interrupt after 45-50+
    minutes, 3 times running, while Ref2VA reliably finished in 8-12 minutes.
    See video.pick_mode's docstring.
  * the reference vocabulary — <Picture i>/<Video k>/<Audio j>, 1-based per type,
    which the prompt has to match or the model cannot tell the references apart.
  * refusing to promise a video when the engine is not actually available, and
    never reporting success without a real file.

Run: venv/Scripts/python.exe tests/test_video_generation.py
"""
import json
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The model's reads are stubbed here; the phrases run live in bench/intent_rest_live.py.
import intent
intent.YES_STUB = lambda q, t: "camera to move" in q and any(w in t.lower() for w in ("zoom", "pan"))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
os.environ.setdefault("F5_TEST_RUN", "1")
import logging; logging.basicConfig(level=logging.CRITICAL)

import config as C
import video as V
import models
import comfy_client
import image as _img

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


_TMP = tempfile.mkdtemp(prefix="video_gen_test_")
# generate_video copies its result out of ComfyUI's tree into config.OUTPUT_DIR
# (_adopt_output). Left alone, this suite drops fixture clips into the user's real
# runtime/ directory alongside genuine renders — a test must not litter the
# output the app delivers from.
V.OUTPUT_DIR = _TMP
def _fake_img(tag):
    p = os.path.join(_TMP, f"{tag}.png")
    with open(p, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    return p
def _fake_vid(tag):
    p = os.path.join(_TMP, f"{tag}.mp4")
    with open(p, "wb") as fh:
        fh.write(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * (17 * 1024))
    return p
def _fake_aud(tag):
    p = os.path.join(_TMP, f"{tag}.wav")
    with open(p, "wb") as fh:
        fh.write(b"RIFF" + b"\x00" * 4 + b"WAVEfmt " + b"\x00" * 64)
    return p

def make_ctx():
    return models.Context(models=None, transcription_cache={}, cache_file=None,
                          asr_lock=threading.Lock(), tts_lock=threading.Lock())


print("=" * 70)
print("1. The frame grid: H3 only accepts n %% 17 == 5")
print("=" * 70)

for raw in (1, 5, 6, 100, 124, 200, 300, 361, 500, 99999):
    got = V.snap_frames(raw)
    check(f"snap_frames({raw}) = {got} is on the grid and in range",
          got % 17 == 5 and 5 <= got <= C.VIDEO_MAX_FRAMES, got)
check("snap_frames never rounds DOWN below the request (until the cap)",
      V.snap_frames(100) >= 100 and V.snap_frames(124) == 124)
check("5 seconds is the documented 124 frames", V.seconds_to_frames(5) == 124,
      V.seconds_to_frames(5))
check("a 15s request stays inside the trained range",
      V.seconds_to_frames(15) <= C.VIDEO_MAX_FRAMES, V.seconds_to_frames(15))

print()
print("=" * 70)
print("2. The canvas: 768 short edge, 1MP cap, every axis a multiple of 32")
print("=" * 70)

for aspect in V.ASPECTS:
    w, h = V.resolve_size(aspect)
    check(f"{aspect} -> {w}x{h}: axes /32, under the pixel cap, 768 short edge",
          w % 32 == 0 and h % 32 == 0 and w * h <= C.VIDEO_MAX_PIXELS
          and min(w, h) <= C.VIDEO_SHORT_EDGE + 32, (w, h, w * h))
check("16:9 is the documented 1344x768", V.resolve_size("16:9") == (1344, 768),
      V.resolve_size("16:9"))
check("portrait 9:16 is its transpose", V.resolve_size("9:16") == (768, 1344),
      V.resolve_size("9:16"))
w, h = V.fit_canvas(4000, 2000)
check("an oversized source is scaled down, not passed through",
      w * h <= C.VIDEO_MAX_PIXELS and w % 32 == 0, (w, h))
check("a degenerate 0x0 source does not crash or divide by zero",
      V.fit_canvas(0, 0)[0] > 0)

print()
print("=" * 70)
print("3. Mode routing — keyframes vs omni-reference")
print("=" * 70)

check("no input at all -> t2va", V.pick_mode() == "t2va")
# FL2VA-checkpoint modes (i2va/flf2va) are disabled live, 2026-09-19: every
# real job through that checkpoint on this deployment hung in KSampler and
# never finished on its own (3/3 required a manual /interrupt after 45-50+
# minutes), while ref2va reliably finishes in 8-12 minutes. See pick_mode's
# docstring. Any image now routes through the checkpoint that actually works.
check("one image -> ref2va (FL2VA checkpoint hangs on this deployment)",
      V.pick_mode(["a"]) == "ref2va")
check("two images -> ref2va (FL2VA checkpoint hangs on this deployment)",
      V.pick_mode(["a", "b"]) == "ref2va")
check("three images -> ref2va (too many to be keyframes anyway)",
      V.pick_mode(["a", "b", "c"]) == "ref2va")
check("a reference VIDEO forces ref2va even with one image",
      V.pick_mode(["a"], ["v"]) == "ref2va")
check("reference AUDIO alone also forces ref2va",
      V.pick_mode([], [], ["s"]) == "ref2va")

print()
print("=" * 70)
print("4. Reference tags are 1-based per type, in attach order")
print("=" * 70)

check("two images, one video, one audio",
      V.reference_tags(2, 1, 1) == ["<Picture 1>", "<Picture 2>", "<Video 1>", "<Audio 1>"],
      V.reference_tags(2, 1, 1))
check("nothing attached -> no tags", V.reference_tags() == [])
check("nine images is the documented ceiling",
      len(V.reference_tags(9)) == 9 and V.MAX_REF_IMAGES == 9)
# The single most confusable thing in this feature: the PROMPT vocabulary is
# 1-based (<Picture 1>) while the WIRE slots are 0-based (ref_image_0). Both are
# correct, they are just different namespaces, and conflating them either drops
# the first reference or points the model at the wrong picture.
check("the tag for the FIRST reference is <Picture 1>, not <Picture 0>",
      V.reference_tags(1) == ["<Picture 1>"], V.reference_tags(1))

print()
print("=" * 70)
print("5. The built graph wires the right nodes for each mode")
print("=" * 70)

_uploaded = []
V._upload = lambda p, kind="image": (_uploaded.append((p, kind)) or os.path.basename(p))

wf = V.build_workflow("a fox in the grass", mode="t2va", width=1344, height=768,
                      frames=124, seed=42)
check("t2va uses the keyframe conditioning node",
      wf[V.N_COND]["class_type"] == "MiniMaxH3ImageToVideo", wf[V.N_COND]["class_type"])
check("...and attaches no keyframes",
      "first_frame" not in wf[V.N_COND]["inputs"], wf[V.N_COND]["inputs"].keys())
check("prompt/width/height/length reach the conditioning node",
      wf[V.N_COND]["inputs"]["prompt"] == "a fox in the grass"
      and wf[V.N_COND]["inputs"]["length"] == 124
      and wf[V.N_COND]["inputs"]["width"] == 1344)
check("the seed reaches the sampler", wf[V.N_SAMPLER]["inputs"]["seed"] == 42)
check("cfg stays 1.0 — the weights are CFG-distilled",
      wf[V.N_SAMPLER]["inputs"]["cfg"] == 1.0, wf[V.N_SAMPLER]["inputs"]["cfg"])
check("the video half is decoded with the VIDEO vae",
      wf["9"]["inputs"]["vae"] == [V.N_VAE_VIDEO, 0])
check("the audio half is decoded with the AUDIO vae",
      wf["10"]["inputs"]["vae"] == [V.N_VAE_AUDIO, 0])
check("both decode the SAME sampled latent (one packed AV latent)",
      wf["9"]["inputs"]["samples"] == wf["10"]["inputs"]["samples"] == [V.N_SAMPLER, 0])
check("the muxer runs at 24fps with the generated audio",
      wf["11"]["inputs"]["fps"] == 24.0 and wf["11"]["inputs"]["audio"] == ["10", 0])

_uploaded.clear()
i1, i2 = _fake_img("k1"), _fake_img("k2")
wf2 = V.build_workflow("walk", mode="flf2va", width=768, height=768, frames=124,
                       seed=1, images=[i1, i2])
check("flf2va attaches BOTH keyframes",
      "first_frame" in wf2[V.N_COND]["inputs"] and "last_frame" in wf2[V.N_COND]["inputs"],
      list(wf2[V.N_COND]["inputs"]))
check("...as LoadImage nodes", all(
      wf2[wf2[V.N_COND]["inputs"][s][0]]["class_type"] == "LoadImage"
      for s in ("first_frame", "last_frame")))
check("both keyframes were uploaded", len(_uploaded) == 2, _uploaded)

_uploaded.clear()
imgs = [_fake_img(f"r{i}") for i in range(3)]
vid = _fake_vid("refclip")
wf3 = V.build_workflow("<Picture 1> enters the room from <Video 1>", mode="ref2va",
                       width=1344, height=768, frames=124, seed=7,
                       images=imgs, videos=[vid])
check("ref2va uses the omni-reference conditioning node",
      wf3[V.N_COND]["class_type"] == "MiniMaxH3ReferenceToVideo")
ci = wf3[V.N_COND]["inputs"]
# ZERO-based on the wire AND dotted with the container name. COMFY_AUTOGROW_V3
# builds slot names as [f"{prefix}{i}" for i in range(max)], then
# parse_class_inputs() prefixes each with the input's own id:
# finalize_prefix(["ref_images"], "ref_image_0") -> "ref_images.ref_image_0".
#
# A BARE "ref_image_0" is the trap this suite exists to catch. It is not in the
# schema, so validation ignores it as an unknown extra and POST /prompt returns
# 200 — execution then dies on the stray kwarg after the VAEs have loaded. This
# suite once asserted the bare names, so it stayed green while the shipped
# workflow could not render a single reference frame. Assert the EXACT key, not
# a suffix or a substring, or the bare form passes again.
#
# The PROMPT tags are a separate, 1-based vocabulary (<Picture 1>) — see part 4.
check("reference images are dotted, 0-based ref_images.ref_image_N",
      all(f"ref_images.ref_image_{i}" in ci for i in (0, 1, 2)), list(ci))
check("...and slot 3 was not created for only three images",
      "ref_images.ref_image_3" not in ci, list(ci))
check("the reference video is ref_videos.ref_video_0",
      "ref_videos.ref_video_0" in ci, list(ci))
check("...and its own soundtrack is index-paired as ref_video_audios.ref_video_audio_0",
      "ref_video_audios.ref_video_audio_0" in ci, list(ci))
# .get, not [] — a missing key here is exactly the bug under test, and an
# IndexError would abort the run before the structural guards below report.
_rv = ci.get("ref_videos.ref_video_0")
_ra = ci.get("ref_video_audios.ref_video_audio_0")
check("the soundtrack comes from GetVideoComponents' audio output, not the images one",
      _ra is not None and _rv is not None and _ra[1] == 1 and _rv[1] == 0, (_rv, _ra))
# Structural guard: NO autogrow slot may be posted bare, in any family. This is
# the check that fails if someone adds a fourth reference family and forgets the
# prefix — the three checks above only cover the families that exist today.
_FAMILIES = {"ref_image": "ref_images", "ref_video": "ref_videos",
             "ref_video_audio": "ref_video_audios", "ref_audio": "ref_audios"}
_bare = []
for k in ci:
    if "." in k:
        continue
    stem = k.rsplit("_", 1)[0]
    if k.rsplit("_", 1)[-1].isdigit() and stem in _FAMILIES:
        _bare.append(k)
check("no autogrow slot is posted without its container prefix", not _bare, _bare)
# ...and every dotted slot's prefix must be the container that owns it, so
# "ref_videos.ref_video_audio_0" (right shape, wrong container) is still caught.
_misfiled = [k for k in ci if "." in k
             and _FAMILIES.get(k.split(".", 1)[1].rsplit("_", 1)[0]) != k.split(".", 1)[0]]
check("every dotted slot sits under the container that declares it",
      not _misfiled, _misfiled)
check("the video went through the video upload path",
      any(k == "video" for _p, k in _uploaded), _uploaded)
check("ref_image_size is set", ci.get("ref_image_size") == "match")
check("every node reference points at a node that exists",
      all(str(v[0]) in wf3 for v in ci.values() if isinstance(v, list)),
      [v for v in ci.values() if isinstance(v, list) and str(v[0]) not in wf3])

# Standalone reference AUDIO had NO coverage at all — the graph above only
# carries images and a video, so the ref_audios family was never built by any
# test. It is reachable from the bot (attach a voice message with no video), so
# a wrong slot name there fails exactly like ref_image_0 did, at execution.
_uploaded.clear()
wf3a = V.build_workflow("speak like <Audio 1>", mode="ref2va", width=768, height=768,
                        frames=124, seed=3, images=[_fake_img("a1")],
                        audios=[_fake_aud("voice")])
cia = wf3a[V.N_COND]["inputs"]
check("standalone reference audio is ref_audios.ref_audio_0",
      "ref_audios.ref_audio_0" in cia, list(cia))
check("...and not the bare form", "ref_audio_0" not in cia, list(cia))
check("...loaded through LoadAudio",
      wf3a[cia["ref_audios.ref_audio_0"][0]]["class_type"] == "LoadAudio",
      wf3a[cia["ref_audios.ref_audio_0"][0]]["class_type"])
check("the audio went through the audio upload path",
      any(k == "audio" for _p, k in _uploaded), _uploaded)

print()
print("=" * 70)
print("6. More than nine reference images cannot be attached")
print("=" * 70)

_uploaded.clear()
many = [_fake_img(f"m{i}") for i in range(15)]
wf4 = V.build_workflow("crowd", mode="ref2va", width=768, height=768, frames=124,
                       seed=1, images=many)
# NB "ref_image_size" is a plain SETTING that also starts with "ref_image" —
# count only dotted, numbered slots, or the setting inflates the total. The
# dotted prefix happens to separate them cleanly, but anything that
# prefix-matches these names still has to make the distinction explicitly.
n_refs = sum(1 for k in wf4[V.N_COND]["inputs"]
             if k.startswith("ref_images.") and k.rsplit("_", 1)[-1].isdigit())
check("attached images are capped at the model's 9", n_refs == 9, n_refs)
check("the ref_image_size SETTING was not consumed as a reference slot",
      wf4[V.N_COND]["inputs"].get("ref_image_size") == "match",
      wf4[V.N_COND]["inputs"].get("ref_image_size"))
check("...and it stayed a bare setting, NOT dotted into the container",
      not any(k.endswith(".ref_image_size") for k in wf4[V.N_COND]["inputs"]),
      [k for k in wf4[V.N_COND]["inputs"] if "ref_image_size" in k])
check("the numbered slots are contiguous from 0 (the wire is 0-based)",
      all(f"ref_images.ref_image_{i}" in wf4[V.N_COND]["inputs"] for i in range(0, 9)))
check("...and never reach index 9, which the node does not define (max=9 -> 0..8)",
      "ref_images.ref_image_9" not in wf4[V.N_COND]["inputs"])

print()
print("=" * 70)
print("7. Nothing is promised when the engine is not available")
print("=" * 70)

_real_missing = V.missing_weights
V.missing_weights = lambda: ["models/unet/MiniMax-H3-FL2VA-Q4_K_M.gguf"]
try:
    ok, why = V.engine_available(None)
    check("engine_available is False when weights are missing", not ok)
    check("...and says so in terms a user can act on", "download" in why.lower(), why)
    out = V.generate_video(make_ctx(), "a fox")
    check("generate_video refuses rather than pretending", out["status"] == "fail", out)
    check("...and returns no path", out.get("path") is None, out)
finally:
    V.missing_weights = _real_missing

V.missing_weights = lambda: []
V._server_has_h3_nodes = lambda: False
try:
    ok, why = V.engine_available(None)
    check("engine_available is False when ComfyUI lacks the H3 nodes", not ok)
    check("...and names the version needed", "0.30" in why, why)
    V._server_has_h3_nodes = lambda: None      # down, not too old
    ok, why = V.engine_available(None)
    check("a ComfyUI that does not answer is reported as not running", not ok and "not running" in why, why)
finally:
    V.missing_weights = _real_missing

print()
print("=" * 70)
print("8. A render that produces no file is reported as a failure, not a success")
print("=" * 70)

V.missing_weights = lambda: []
V._server_has_h3_nodes = lambda: True
V._normalize_save_node = lambda wf: wf
_real_submit = comfy_client._submit_and_poll
try:
    comfy_client._submit_and_poll = lambda *a, **k: None
    out = V.generate_video(make_ctx(), "a fox in the grass")
    check("no file -> status fail", out["status"] == "fail", out)
    check("...and path is None", out.get("path") is None, out)

    made = _fake_vid("result")
    _seen_kw = {}
    def _submit(*a, **k):
        _seen_kw.update(k); return made
    comfy_client._submit_and_poll = _submit
    out2 = V.generate_video(make_ctx(), "a fox in the grass", seconds=5, aspect="16:9")
    # The DiT is ~19.4 GB; beside a resident chat model it ran 6 GB offloaded
    # at 190 s/step (live, 2026-09-12). The render gets the whole card.
    check("the video render claims the card EXCLUSIVELY (chat model evicted)",
          _seen_kw.get("exclusive") is True, _seen_kw)
    check("a real file -> status success", out2["status"] == "success", out2)
    check("the reported duration matches the snapped frame count",
          out2["seconds"] == V.frames_to_seconds(124), out2)
    check("the reported size is the fitted canvas",
          (out2["width"], out2["height"]) == (1344, 768), out2)
    check("mode was recorded", out2["mode"] == "t2va", out2)

    ctx = make_ctx()
    V.generate_video(ctx, "a fox")
    check("ctx.last_video_path is set so 'that video' resolves later",
          bool(getattr(ctx, "last_video_path", None)))
finally:
    comfy_client._submit_and_poll = _real_submit
    V.missing_weights = _real_missing

print()
print("=" * 70)
print("9. The poller accepts a VIDEO file (it used to demand a decodable image)")
print("=" * 70)

good = _fake_vid("ok_clip")
check("a real mp4 passes the video validator", V._valid_video_file(good))
tiny = os.path.join(_TMP, "tiny.mp4")
open(tiny, "wb").write(b"x" * 100)
check("a truncated write is rejected", not V._valid_video_file(tiny))
check("a PNG is not accepted as a video", not V._valid_video_file(_fake_img("nope")))
check("a missing path is rejected", not V._valid_video_file(
      os.path.join(_TMP, "does_not_exist.mp4")))
import inspect
src = inspect.getsource(_img._poll_history)
check("image._poll_history takes a validator instead of hardcoding the image check",
      "validate" in inspect.signature(_img._poll_history).parameters and
      "validate or _valid_image_file" in src)

print()
print("=" * 70)
print("10. The agent tool refuses honestly and never claims a phantom clip")
print("=" * 70)

import tools as T
check("generate_video is registered", "generate_video" in T._BY_NAME)
spec = T._BY_NAME["generate_video"]
desc = spec.schema["function"]["description"]
check("its description teaches the four modes",
      all(s in desc for s in ("text only", "ONE image", "TWO images", "reference")))
check("...and warns that it is slow", "SLOW" in desc or "minutes" in desc)

_real_avail = V.engine_available
try:
    V.engine_available = lambda ctx=None: (False, "the H3 weights are not downloaded yet")
    msg = T._BY_NAME["generate_video"].handler(make_ctx(), {}, {"description": "a fox"})
    check("an unavailable engine yields a [TOOL ERROR]", msg.startswith("[TOOL ERROR]"), msg[:80])
    check("...that forbids claiming a video was made",
          "not claim" in msg.lower() or "no clip" in msg.lower(), msg[:160])
finally:
    V.engine_available = _real_avail

st = {}
msg2 = T._BY_NAME["generate_video"].handler(make_ctx(), st, {"description": ""})
check("an empty description is refused before any render",
      msg2.startswith("[TOOL ERROR]"), msg2[:80])

print()
print("=" * 70)
print("11. A clip costs the whole turn's render budget, not one render")
print("=" * 70)

cap = max(1, int(getattr(C, "IMAGE_MAX_RENDERS_PER_TURN", 3) or 3))
st2 = {}
check("a video spends the entire budget in one call",
      T._render_budget_exhausted(st2, "generate_video", cost=cap) is None
      and st2["_renders_used"] >= cap, st2)
check("...so a SECOND clip in the same turn is refused",
      (T._render_budget_exhausted(st2, "generate_video", cost=cap) or "").startswith("[TOOL ERROR]"))
st3 = {}
check("a still image still costs exactly one",
      T._render_budget_exhausted(st3, "generate_image") is None and st3["_renders_used"] == 1)
check("the refusal talks about a clip when the caller is video",
      "clip" in (T._render_budget_exhausted({"_renders_used": 99}, "generate_video") or ""))
check("...and about a picture when the caller is an image",
      "picture" in (T._render_budget_exhausted({"_renders_used": 99}, "generate_image") or ""))

print()
print("=" * 70)
print("12. The video keys survive the graph (AgentState must DECLARE them)")
print("=" * 70)

# StateGraph is built on the AgentState TypedDict: a key a node writes but the
# schema does not name is DROPPED from the state the graph returns. That is
# exactly how the presentation path was lost once — the deck was really built and
# the user was told it was ready, while no file was ever sent. A clip costs
# minutes of GPU, so the same slip is worse here.
import models as _models
_ann = getattr(_models.AgentState, "__annotations__", {})
for key in ("video_path", "video_status", "video_seconds"):
    check(f"AgentState declares {key!r}", key in _ann, sorted(_ann))

# And prove it end to end through a real graph round-trip, not just the annotation.
import graph as _graph
try:
    _g = _graph.build_graph(make_ctx())
    _sig = getattr(_g, "schema", None) or getattr(_g, "state_schema", None)
    _declared = getattr(_sig, "__annotations__", _ann)
    check("the compiled graph's schema carries video_path",
          "video_path" in _declared, sorted(_declared))
except Exception as _exc:
    check("the graph builds so its schema can be inspected", False, _exc)

# The delivery sites must key off those exact names, or declaring them is moot.
import inspect as _inspect
import tg_bot as _tgb
# Follow the METHOD on the class, not the module file. Reading the source of
# tg_bot.py went stale the moment the monolith was split: delivery now lives in
# tg_tasks.py and is mixed into TelegramBot, so the grep found nothing and
# reported a missing feature that was working the whole time.
_tg_src = ""
for _n, _m in _inspect.getmembers(_tgb.TelegramBot, _inspect.isfunction):
    _s = _inspect.getsource(_m)
    if "[video delivery]" in _s:
        _tg_src = _s
        break
check("the turn has a video-delivery block", bool(_tg_src))
check("it reads final['video_path']", 'final.get("video_path")' in _tg_src)
check("and checks video_status before delivering", '"video_status"' in _tg_src)

print()
print("=" * 70)
print("13. Config matches what the node actually accepts")
print("=" * 70)

check("fps is 24 (the model's native rate)", C.VIDEO_FPS == 24)
check("the default frame count is on the grid", C.VIDEO_DEFAULT_FRAMES % 17 == 5)
check("the max frame count is on the grid", C.VIDEO_MAX_FRAMES % 17 == 5)
check("the pixel cap matches the node's 768*1344", C.VIDEO_MAX_PIXELS == 768 * 1344)
check("the canvas multiple is 32", C.VIDEO_CANVAS_MULTIPLE == 32)
check("cfg default is 1.0 (CFG-distilled weights)", C.VIDEO_CFG == 1.0)
check("the video job timeout is far longer than an image job",
      C.VIDEO_JOB_TIMEOUT > getattr(C, "COMFY_JOB_TIMEOUT", 1900))
for p in (C.WORKFLOW_VIDEO_PATH, C.WORKFLOW_VIDEO_REF_PATH):
    check(f"{os.path.basename(str(p))} exists and is valid JSON",
          os.path.exists(p) and isinstance(json.load(open(p, encoding="utf-8")), dict))

print()
print("=" * 70)
print("14. A static-camera directive is added unless motion was requested")
print("=" * 70)
print("Live, 2026-09-19: 'the video model doesn't preserve the original at all")
print("on complex requests, sometimes crops without being asked' -- researched")
print("against MiniMax H3's own prompt guide: an unstated camera improvises a")
print("shot on complex prompts; the documented fix is an explicit locked camera.")

check("no motion words -> the lock is appended",
      "no camera movement" in V._lock_framing_unless_requested("a fox in the grass"))
check("'zoom in' -> left alone, the user's own motion request is not fought",
      V._lock_framing_unless_requested("slow zoom in on her face")
      == "slow zoom in on her face")
check("'pan across' -> left alone",
      V._lock_framing_unless_requested("the camera pans across the room")
      == "the camera pans across the room")
check("empty description -> stays empty (generate_video's own empty-check catches it)",
      V._lock_framing_unless_requested("") == "")

V.missing_weights = lambda: []
V._server_has_h3_nodes = lambda: True
V._normalize_save_node = lambda wf: wf
_real_submit2 = comfy_client._submit_and_poll
try:
    _seen_wf = {}
    def _capture(ctx, wf, **k):
        _seen_wf["wf"] = wf
        return _fake_vid("locked")
    comfy_client._submit_and_poll = _capture
    V.generate_video(make_ctx(), "the girl waves and smiles")
    prompt_sent = _seen_wf["wf"][V.N_COND]["inputs"]["prompt"]
    check("the actual workflow prompt carries the lock",
          "no camera movement" in prompt_sent, prompt_sent)
    check("...on top of, not instead of, the user's own description",
          prompt_sent.startswith("the girl waves and smiles"), prompt_sent)

    _seen_wf.clear()
    V.generate_video(make_ctx(), "the camera slowly zooms in on her face")
    prompt_sent2 = _seen_wf["wf"][V.N_COND]["inputs"]["prompt"]
    check("an explicit zoom request reaches the node untouched",
          prompt_sent2 == "the camera slowly zooms in on her face", prompt_sent2)
finally:
    comfy_client._submit_and_poll = _real_submit2
    V.missing_weights = _real_missing

_wf3 = V.build_workflow("a cat", mode="t2va", width=512, height=288, frames=25, seed=1, steps=3)
_wf4 = V.build_workflow("a cat", mode="t2va", width=512, height=288, frames=25, seed=1, steps=4)
check("t2va at 3 steps runs the TaoMate 3-step LoRA at full strength",
      "taomate" in _wf3[V.N_LORA]["inputs"]["lora_name"] and _wf3[V.N_LORA]["inputs"]["strength_model"] == 1.0,
      _wf3[V.N_LORA]["inputs"])
check("t2va at 4 steps keeps the 4-step FL2V LoRA",
      "fl2v_turbo_4step" in _wf4[V.N_LORA]["inputs"]["lora_name"], _wf4[V.N_LORA]["inputs"])

print()
print(f"{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
