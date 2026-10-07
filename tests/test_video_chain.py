"""Continuation as a Herrgott chain (H3 Infinite Continuation Suite v1.4): each part keeps
its full AV latent, the next part copies the previous tail into its own latent under the
native denoise mask, and the suite's stitch node joins the saved parts. Live 10-07 the chef
chain had no reset at the seams; the pinned-frames path reset after the pins."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "core", "agent", "media", "bot"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ.setdefault("F5_TEST_RUN", "1")

import video as V  # noqa: E402


def _by(wf, cls):
    return [k for k, v in wf.items() if v["class_type"] == cls]


def test_part_one_starts_the_chain_and_keeps_its_latent(monkeypatch):
    monkeypatch.setattr(V, "_upload", lambda p, kind="image": "start.png")
    wf = V.build_chain_part("a chef chops", chain="h3_continuous/chX", idx=1, width=768, height=1024,
                            seconds=6.0, seed=5, first_frame="photo.jpg")
    st = wf[_by(wf, "H3ContinuousStartV14")[0]]["inputs"]
    assert st["prompt"] == "a chef chops" and (st["width"], st["height"]) == (768, 1024)
    assert wf[st["first_frame"][0]]["inputs"]["image"] == "start.png"
    keep = wf[_by(wf, "H3ContinuousSaveLatent")[0]]["inputs"]
    assert keep["filename_prefix"] == "h3_continuous/chX/clip" and keep["clip_index"] == 1
    assert keep["latent"] == [V.N_SAMPLER, 0] and "head_context_frames" not in keep
    assert wf[V.N_SAMPLER]["inputs"]["steps"] == V.HERRGOTT_STEPS
    assert not _by(wf, "H3ContinuousContinueV14") and not _by(wf, "MiniMaxH3ImageToVideo")


def test_part_three_continues_part_two():
    wf = V.build_chain_part("he stirs", chain="h3_continuous/chX", idx=3, width=768, height=1024,
                            seconds=6.0, seed=5)
    ld = wf[_by(wf, "H3ContinuousLoadLatent")[0]]["inputs"]
    assert ld == {"latent_path": "h3_continuous/chX", "clip_index": 2}   # the chain's own folder
    c = _by(wf, "H3ContinuousContinueV14")[0]
    ci = wf[c]["inputs"]
    assert ci["previous_latent"][1] == 0 and ci["handover"][1] == 3
    assert ci["duration_mode"] == "Net New Content" and ci["masked_context_frames"] == "39"
    keep = wf[_by(wf, "H3ContinuousSaveLatent")[0]]["inputs"]
    assert keep["clip_index"] == 3 and keep["head_context_frames"] == [c, 2]
    assert wf[V.N_SAMPLER]["inputs"]["positive"] == [c, 0]
    assert wf[V.N_SAMPLER]["inputs"]["latent_image"] == [c, 1]


def test_stitch_joins_the_whole_chain():
    s = V.build_chain_stitch("h3_continuous/chX", 3)["stitch"]["inputs"]
    assert s["latent_prefix"] == "h3_continuous/chX/clip" and (s["first_clip"], s["last_clip"]) == (1, 3)


def test_a_clip_is_found_again_by_its_bytes(monkeypatch, tmp_path):
    monkeypatch.setattr(V, "_CHAINS", tmp_path / "chains.json")
    monkeypatch.setattr(V, "OUTPUT_DIR_COMFY", tmp_path)
    (tmp_path / "h3_continuous" / "chX").mkdir(parents=True)
    monkeypatch.setattr(V, "probe", lambda p: {"seconds": 6.5, "width": 768, "height": 1024})
    a = tmp_path / "a.mp4"
    a.write_bytes(b"clip a")
    V.remember_chain(str(a), "h3_continuous/chX", 2)
    back, other = tmp_path / "from_tg.mp4", tmp_path / "b.mp4"
    back.write_bytes(b"clip a")
    other.write_bytes(b"clip b")
    monkeypatch.setattr(V, "probe", lambda p: {"seconds": 9.0, "width": 768, "height": 1024})
    assert V.chain_of(str(back)) == ("h3_continuous/chX", 2)
    assert V.chain_of(str(other)) == ()


class _Ctx:
    reference_images = []
    voice_choice = None
    continue_tail = continue_src = ""

    def set_stage(self, *_): pass
    def is_cancelled(self): return False
    def remember(self, *a): pass
    def memory_text(self): return ""


def _old_join(*a):
    raise AssertionError("a chain is joined by the stitch node, not the old join")


def _wire(monkeypatch, tmp_path):
    import tool_image_handlers as H
    import tg_continue
    calls, stitched, kept = [], [], []

    import comfy_client

    def fake_gen(ctx, description, **kw):
        out = tmp_path / f"part{len(calls) + 1}.mp4"
        out.write_bytes(b"v")
        calls.append((description, dict(kw, card=comfy_client.card_in_use())))
        return {"path": str(out), "status": "success", "seconds": 6.0}
    monkeypatch.setattr(V, "generate_video", fake_gen)
    monkeypatch.setattr(V, "engine_available", lambda ctx: (True, ""))
    monkeypatch.setattr(V, "herrgott_on", lambda: True)
    monkeypatch.setattr(V, "motion_context_on", lambda: False)
    monkeypatch.setattr(V, "bridge_part", lambda ctx, f, part: "BRIDGE. " + part)
    monkeypatch.setattr(V, "prepare_prompt",
                        lambda ctx, d, **kw: calls.append(("PREP", {"card": comfy_client.card_in_use()}))
                        or (d, 6.0))
    monkeypatch.setattr(tg_continue, "seed_frame", lambda src, out: open(out, "wb").write(b"j") > 0)
    monkeypatch.setattr(tg_continue, "cut_tail", lambda src, out: open(out, "wb").write(b"t") > 0)
    monkeypatch.setattr(V, "stitch_chain",
                        lambda ctx, ch, n: stitched.append((ch, n)) or str(tmp_path / "joined.mp4"))
    monkeypatch.setattr(V, "remember_chain", lambda p, ch, n: kept.append((os.path.basename(p), ch, n)))
    monkeypatch.setattr(V, "probe", lambda p: {"seconds": 18.0})
    monkeypatch.setattr(V, "join_continuation", _old_join)
    monkeypatch.setattr(V, "join_pinned", _old_join)
    return H, calls, stitched, kept


LONG = ("The man puts on a formal suit, takes a broom and begins sweeping the floor. He sweeps for "
        "several seconds, then stops, looks at the camera and says «Ладно, извините, я сейчас всё "
        "уберу, честное слово». He then resumes sweeping, knocks over a bucket, slips on the water "
        "and falls. He gets up, laughs and says «Ну и денёк».")


def test_a_long_script_is_one_chain_stitched_once(monkeypatch, tmp_path):
    H, calls, stitched, kept = _wire(monkeypatch, tmp_path)
    asked = []
    monkeypatch.setattr(V, "bridge_script",
                        lambda ctx, ps: asked.append(len(calls)) or [ps[0]] + ["BRIDGE. " + p for p in ps[1:]])

    def no_vision(*a):
        raise AssertionError("no model call between the parts of a chain (it reloads the LLM)")
    monkeypatch.setattr(V, "bridge_part", no_vision)
    st = {}
    H._handle_generate_video(_Ctx(), st, {"description": LONG, "use_current_images": False})
    n = len(V.split_script(LONG))
    preps = [kw for d, kw in calls if d == "PREP"]
    assert len(preps) == n and calls[:n] == [("PREP", {"card": False})] * n   # all prepared first, model loaded
    del calls[:n]
    assert all(kw["prepared"] for _, kw in calls)
    chains = {kw["chain"][0] for _, kw in calls}
    assert len(calls) == n and len(chains) == 1, calls
    assert [kw["chain"][1] for _, kw in calls] == list(range(1, n + 1))
    assert all(d.startswith(V.CONTINUE_CTX_PREFIX + "BRIDGE. ") for d, _ in calls[1:]), calls
    assert not calls[0][0].startswith(V.CONTINUE_CTX_PREFIX)
    assert stitched == [(chains.pop(), n)] and st["video_path"].endswith("joined.mp4")
    assert kept and kept[-1][2] == n
    assert asked == [0]                                  # bridges written once, before any render
    assert all(kw["card"] for _, kw in calls)            # the card held through the whole chain


def test_continue_on_a_chain_clip_goes_on_from_its_latent(monkeypatch, tmp_path):
    H, calls, stitched, kept = _wire(monkeypatch, tmp_path)
    src, tail = tmp_path / "sent_back.mp4", tmp_path / "tail.mp4"
    src.write_bytes(b"s")
    tail.write_bytes(b"t")
    monkeypatch.setattr(V, "chain_of", lambda p: ("h3_continuous/chY", 2) if p == str(src) else ())
    ctx = _Ctx()
    ctx.continue_src, ctx.continue_tail = str(src), str(tail)
    H._handle_generate_video(ctx, {}, {"description": "He smiles and says «Готово»."})
    (d, kw), = calls
    assert kw["chain"] == ("h3_continuous/chY", 3) and kw["size_from"] == str(src)
    assert not kw.get("videos") and not kw.get("images") and not kw.get("context_video")
    assert d.startswith(V.CONTINUE_CTX_PREFIX + "BRIDGE. ")
    assert stitched == [("h3_continuous/chY", 3)] and kept[-1][1:] == ("h3_continuous/chY", 3)


def test_voices_keep_the_old_path(monkeypatch, tmp_path):
    H, calls, stitched, kept = _wire(monkeypatch, tmp_path)
    ctx = _Ctx()
    voice = tmp_path / "v.wav"
    voice.write_bytes(b"w")
    ctx.anim_voices = [str(voice)]
    H._handle_generate_video(ctx, {}, {"description": "He says «Привет».", "use_current_images": False})
    assert calls and not calls[0][1].get("chain") and not stitched


def test_a_prepared_part_calls_no_model(monkeypatch, tmp_path):
    """Inside a chain the card is held and the chat model is out: generate_video must not
    reach for it (live 10-07: the Context-IR rewrite met an unloaded model every part)."""
    def boom(*a, **k):
        raise AssertionError("model call during a prepared render")
    for name in ("estimate_seconds", "to_context_ir", "mark_speech_stress"):
        monkeypatch.setattr(V, name, boom)
    monkeypatch.setattr(V, "engine_available", lambda ctx: (True, ""))
    monkeypatch.setattr(V, "_context_ir_on", lambda: True)
    monkeypatch.setattr(V, "build_chain_part", lambda *a, **k: (_ for _ in ()).throw(V.VideoUnavailable("stop")))
    r = V.generate_video(None, "ready prompt", seconds=6.0, prepared=True, chain=("h3_continuous/chZ", 2))
    assert r["reason"] == "stop"


def test_a_chain_part_samples_like_a_plain_clip():
    """Live 10-07: the chain ran the FL2V turbo LoRA at 6 steps and came back soapy;
    the plain t2va clip (TaoMate 3-step LoRA) was crisp. Same LoRA, same steps."""
    plain = V.build_workflow("a chef", mode="t2va", width=1344, height=768, frames=226, seed=1,
                             images=[], videos=[], audios=[], steps=V.VIDEO_STEPS, ctx=None)
    for idx in (1, 2):
        part = V.build_chain_part("a chef", chain="h3_continuous/chX", idx=idx, width=1344,
                                  height=768, seconds=9.4, seed=1)
        assert part[V.N_LORA]["inputs"] == plain[V.N_LORA]["inputs"]
        assert part[V.N_SAMPLER]["inputs"]["steps"] == plain[V.N_SAMPLER]["inputs"]["steps"]
