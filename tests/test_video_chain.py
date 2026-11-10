"""Continuation as a Herrgott chain (H3 Infinite Continuation Suite v1.4): each part keeps
its full AV latent, the next part copies the previous tail into its own latent under the
native denoise mask, and the suite's stitch node joins the saved parts.

The renderer, the clip tools and the models are handed in on the context
(`video_engine`, `clip_tools`, `text_model`, ...): nothing is written over a module."""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "core", "agent", "media", "bot"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ.setdefault("F5_TEST_RUN", "1")
os.environ.setdefault("VIDEO_AUTO_MAX_PARTS", "99")   # long chains are tested here; the 2-part cap has its own test

import comfy_client  # noqa: E402
import tg_continue  # noqa: E402
import tool_image_handlers as H  # noqa: E402
import video as V  # noqa: E402
import fakes as _fakes  # noqa: E402
import pytest


@pytest.fixture(autouse=True)
def _allow_long_chains(monkeypatch):
    monkeypatch.setattr(V, 'AUTO_MAX_PARTS', 99)


_split = lambda t, **k: V.split_script(t, reads=_fakes.script_model, **k)  # noqa: E731  the sentence read, scripted
_est = lambda t, **k: V.estimate_seconds(t, reads=_fakes.script_model, **k)  # noqa: E731
from tests.fakes import Over  # noqa: E402


def _by(wf, cls):
    return [k for k, v in wf.items() if v["class_type"] == cls]


def _uploads(name):
    return types.SimpleNamespace(video_engine=Over(V, _upload=lambda p, kind="image": name))


def test_part_one_starts_the_chain_and_keeps_its_latent():
    wf = V.build_chain_part("a chef chops", chain="h3_continuous/chX", idx=1, width=768, height=1024,
                            seconds=6.0, seed=5, first_frame="photo.jpg", ctx=_uploads("start.png"))
    st = wf[_by(wf, "H3ContinuousStartV14")[0]]["inputs"]
    assert st["prompt"] == "a chef chops" and (st["width"], st["height"]) == (768, 1024)
    assert wf[st["first_frame"][0]]["inputs"]["image"] == "start.png"
    keep = wf[_by(wf, "H3ContinuousSaveLatent")[0]]["inputs"]
    assert keep["filename_prefix"] == "h3_continuous/chX/clip" and keep["clip_index"] == 1
    assert keep["latent"] == [V.N_SAMPLER, 0] and "head_context_frames" not in keep
    assert wf[V.N_SAMPLER]["inputs"]["steps"] == V.HERRGOTT_STEPS
    assert not _by(wf, "H3ContinuousContinueV14") and not _by(wf, "MiniMaxH3ImageToVideo")


def test_text_only_chain_starts_with_text_adapter(monkeypatch):
    monkeypatch.setattr(V, 'HERRGOTT_STEPS', 3)
    monkeypatch.setattr(V, 'VIDEO_T2VA_LORA_3STEP', 'text.safetensors')
    monkeypatch.setattr(V, 'VIDEO_FL2VA_LORA_3STEP', 'frame.safetensors')
    wf = V.build_chain_part('A plant sways', chain='h3_continuous/chX', idx=1,
                            width=1024, height=576, seconds=5, seed=5)
    assert wf[V.N_LORA]['inputs']['lora_name'] == 'text.safetensors'
    with_frame = V.build_chain_part('A plant sways', chain='h3_continuous/chX', idx=1,
                                    width=1024, height=576, seconds=5, seed=5,
                                    first_frame='plant.png', ctx=_uploads('plant.png'))
    assert with_frame[V.N_LORA]['inputs']['lora_name'] == 'frame.safetensors'


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


def test_longer_context_only_for_safe_static_scene(tmp_path, monkeypatch):
    import json
    import torch
    from safetensors.torch import save_file

    monkeypatch.setattr(V, 'OUTPUT_DIR_COMFY', tmp_path)
    folder = tmp_path / 'h3_continuous' / 'chX'
    folder.mkdir(parents=True)
    latent = folder / 'clip_00001.safetensors'
    def save(end):
        handover = {'available': True, 'frame_count': 175, 'handover_end_frame': end}
        save_file({'video': torch.zeros(1)}, str(latent), metadata={'handover_json': json.dumps(handover)})
    ctx = types.SimpleNamespace(video_chain_static=True)
    save(140)
    assert V._chain_context('h3_continuous/chX', 2, 1152, 576, ctx) == '90'
    wf = V.build_chain_part('Two people speak.', chain='h3_continuous/chX', idx=2,
                            width=1152, height=576, seconds=6, seed=5, ctx=ctx)
    assert wf['cond']['inputs']['masked_context_frames'] == '90'
    assert V._chain_context('h3_continuous/chX', 2, 1344, 768, ctx) == '39'
    assert V._chain_context('h3_continuous/chX', 2, 1152, 576, None) == '39'
    save(72)
    assert V._chain_context('h3_continuous/chX', 2, 1152, 576, ctx) == '39'
    latent.unlink()
    assert V._chain_context('h3_continuous/chX', 2, 1152, 576, ctx) == '39'
    latent.write_bytes(b'not a latent')
    assert V._chain_context('h3_continuous/chX', 2, 1152, 576, ctx) == '39'


def test_stitch_joins_the_whole_chain():
    s = V.build_chain_stitch("h3_continuous/chX", 3)["stitch"]["inputs"]
    assert s["latent_prefix"] == "h3_continuous/chX/clip" and (s["first_clip"], s["last_clip"]) == (1, 3)


def test_a_clip_is_found_again_by_its_bytes(tmp_path):
    (tmp_path / "h3_continuous" / "chX").mkdir(parents=True)
    seconds = {"s": 6.5}
    ctx = types.SimpleNamespace(video_engine=Over(
        V, OUTPUT_DIR_COMFY=tmp_path,
        probe=lambda p: {"seconds": seconds["s"], "width": 768, "height": 1024}))
    a = tmp_path / "a.mp4"
    a.write_bytes(b"clip a")
    V.remember_chain(str(a), "h3_continuous/chX", 2, ctx)
    back, other = tmp_path / "from_tg.mp4", tmp_path / "b.mp4"
    back.write_bytes(b"clip a")
    other.write_bytes(b"clip b")
    seconds["s"] = 9.0                       # a different length: only the bytes can match
    assert V.chain_of(str(back), ctx) == ("h3_continuous/chX", 2)
    assert V.chain_of(str(other), ctx) == ()


class _Ctx:
    deps = types.SimpleNamespace(script_read=_fakes.script_model)
    reference_images = []
    voice_choice = None
    continue_tail = continue_src = ""

    def set_stage(self, *_): pass
    def is_cancelled(self): return False
    def remember(self, *a): pass
    def memory_text(self): return ""


def test_video_references_keep_distinct_files_with_the_same_prefix(tmp_path):
    first, second, duplicate = (tmp_path / name for name in ('a.png', 'b.png', 'copy.png'))
    prefix = b'header' * 11000
    first.write_bytes(prefix + b'A')
    second.write_bytes(prefix + b'B')
    duplicate.write_bytes(first.read_bytes())
    state = {'image_paths': [str(first), str(second), str(duplicate)]}
    assert H._current_image_paths(_Ctx(), state) == [str(first), str(second)]


def _old_join(*a):
    raise AssertionError("a chain is joined by the stitch node, not the old join")


class _Rig:
    """A fake renderer and clip tools for the handler; `ctx()` hands them in."""

    def __init__(self, tmp_path, chains=None, **engine):
        self.calls, self.stitched, self.kept = [], [], []
        chains = chains or {}

        def gen(ctx, description, **kw):
            out = tmp_path / f"part{len(self.calls) + 1}.mp4"
            out.write_bytes(b"v")
            self.calls.append((description, dict(kw, card=comfy_client.card_in_use(grace=0))))
            return {"path": str(out), "status": "success", "seconds": 6.0}

        def prep(ctx, d, **kw):
            self.calls.append(("PREP", {"card": comfy_client.card_in_use(grace=0)}))
            return d, 6.0

        self.engine = Over(V, **{
            "generate_video": gen,
            "engine_available": lambda ctx: (True, ""),
            "herrgott_on": lambda: True,
            "motion_context_on": lambda: False,
            "bridge_part": lambda ctx, f, part: "BRIDGE. " + part,
            "explicit_silence": lambda ctx, request: False,
            "bridge_script": lambda ctx, ps: ps,
            "mark_inserts": lambda ctx, ps: ps,
            "prepare_prompt": prep,
            "stitch_chain": lambda ctx, ch, n: self.stitched.append((ch, n)) or str(tmp_path / "joined.mp4"),
            "remember_chain": lambda p, ch, n, ctx=None: self.kept.append((os.path.basename(p), ch, n)),
            "chain_of": lambda p, ctx=None: chains.get(p, ()),
            "probe": lambda p: {"seconds": 18.0},
            "join_continuation": _old_join,
            "join_pinned": _old_join,
            **engine})
        self.clips = Over(tg_continue,
                          seed_frame=lambda src, out: open(out, "wb").write(b"j") > 0,
                          cut_tail=lambda src, out: open(out, "wb").write(b"t") > 0,
                          first_frame=lambda src, out, skip=0: False)

    def ctx(self, **kw):
        c = _Ctx()
        c.video_engine, c.clip_tools = self.engine, self.clips
        c.yes_no = lambda q, t, default=False, wait=True: default
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    def renders(self):
        return [(d, kw) for d, kw in self.calls if d != "PREP"]


LONG = ("The man puts on a formal suit, takes a broom and begins sweeping the floor. He sweeps for "
        "several seconds, then stops, looks at the camera and says «Ладно, извините, я сейчас всё "
        "уберу, честное слово». He then resumes sweeping, knocks over a bucket, slips on the water "
        "and falls. He gets up, laughs and says «Ну и денёк».")


def _no_vision(*a, **k):
    raise AssertionError("no model call between the parts of a chain (it reloads the LLM)")


def test_new_video_does_not_use_old_image_when_tool_defaults_true(tmp_path):
    old = tmp_path / "old.png"
    old.write_bytes(b"old image")
    rig = _Rig(tmp_path, drawn_over=lambda *a: False)
    ctx = rig.ctx(last_image_path=str(old), text_model=lambda *a, **k: "")
    ctx.yes_no = lambda q, t, default=False, **kw: False if "picture already in the chat" in q else default
    state = {"user_input_original": "Сгенерируй видео: цветок колышется на столе"}
    H._handle_generate_video(ctx, state, {"description": "A plant sways on a table", "seconds": 5,
                                          "use_current_images": True})
    assert rig.renders() and not rig.renders()[0][1].get("images")


def test_chain_continuations_use_the_previous_state_not_the_original_picture(tmp_path):
    still = tmp_path / 'source.png'
    still.write_bytes(b'source')
    first = tmp_path / 'first.mp4'
    first.write_bytes(b'video')
    rig = _Rig(tmp_path)
    parts = [('<Picture 1> moves.', 5.2), ('<Picture 1> continues.', 5.2)]
    last = H._render_chain_parts(rig.ctx(), str(first), parts, {}, 'chain', 1, reference=str(still))
    assert last == 2
    prompt, params = rig.renders()[0]
    assert not params.get('reference')
    assert '<Picture 1>' not in prompt
    assert 'same subjects from the previous shot' in prompt


def test_a_long_script_is_one_chain_stitched_once(tmp_path):
    asked = []
    rig = _Rig(tmp_path, bridge_part=_no_vision,
               bridge_script=lambda ctx, ps: asked.append(len(rig.calls)) or [ps[0]] + ["BRIDGE. " + p for p in ps[1:]])
    st = {}
    H._handle_generate_video(rig.ctx(), st, {"description": LONG, "use_current_images": False})
    calls = rig.calls
    n = len(_split(LONG))
    preps = [kw for d, kw in calls if d == "PREP"]
    assert len(preps) == n and calls[:n] == [("PREP", {"card": False})] * n   # all prepared first, model loaded
    del calls[:n]
    assert all(kw["prepared"] for _, kw in calls)
    chains = {kw["chain"][0] for _, kw in calls}
    assert len(calls) == n and len(chains) == 1, calls
    assert [kw["chain"][1] for _, kw in calls] == list(range(1, n + 1))
    assert all(d.startswith(V.CONTINUE_CTX_PREFIX + "BRIDGE. ") for d, _ in calls[1:]), calls
    assert not calls[0][0].startswith(V.CONTINUE_CTX_PREFIX)
    assert rig.stitched == [(chains.pop(), n)] and st["video_path"].endswith("joined.mp4")
    assert rig.kept and rig.kept[-1][2] == n
    assert asked == [0]                                  # bridges written once, before any render
    assert all(kw["card"] for _, kw in calls)            # the card held through the whole chain


def _continued(tmp_path, rig_kw=None, **ctx_kw):
    src, tail = tmp_path / "sent_back.mp4", tmp_path / "tail.mp4"
    src.write_bytes(b"s")
    tail.write_bytes(b"t")
    rig = _Rig(tmp_path, chains={str(src): ("h3_continuous/chY", 2)}, **(rig_kw or {}))
    return rig, rig.ctx(continue_src=str(src), continue_tail=str(tail), **ctx_kw), src


def test_continue_on_a_chain_clip_goes_on_from_its_latent(tmp_path):
    rig, ctx, src = _continued(tmp_path)
    H._handle_generate_video(ctx, {}, {"description": "He smiles and says «Готово»."})
    (d, kw), = rig.calls
    assert kw["chain"] == ("h3_continuous/chY", 3) and kw["size_from"] == str(src)
    assert not kw.get("videos") and not kw.get("images") and not kw.get("context_video")
    assert d.startswith(V.CONTINUE_CTX_PREFIX + "BRIDGE. ")
    assert rig.stitched == [("h3_continuous/chY", 3)] and rig.kept[-1][1:] == ("h3_continuous/chY", 3)


def test_continuation_does_not_replace_speech_with_a_sign(tmp_path):
    rig, ctx, _ = _continued(tmp_path)
    H._handle_generate_video(ctx, {'user_input_original': 'Пусть он говорит привет рядом с вывеской «СТОП».'},
                             {'description': 'He says "привет".'})
    assert rig.renders() and 'привет' in rig.renders()[0][0]
    assert 'СТОП' not in rig.renders()[0][0]


def test_continuation_does_not_overwrite_a_sign_with_speech(tmp_path):
    rig, ctx, _ = _continued(tmp_path)
    result = H._handle_generate_video(ctx, {'user_input_original': 'Пусть он скажет «привет».'},
                                      {'description': 'The sign reads "STOP".'})
    assert '[TOOL ERROR]' in result and not rig.renders()


def test_continuation_reads_requested_duration_when_tool_omits_it(tmp_path, monkeypatch):
    monkeypatch.setattr(H, 'intent_read', lambda ctx: lambda *a: {'song_seconds': 25})
    rig, ctx, _ = _continued(tmp_path, video_plan_ask=True)
    state = {'user_input_original': 'Продолжи на 25 секунд'}
    result = H._handle_generate_video(ctx, state, {'description': 'He waves.'})
    assert '[NOT MADE YET]' in result
    assert state['video_plan_requested_seconds'] == 25
    assert not rig.renders()


def test_each_prepared_chain_part_has_its_own_action_scope(tmp_path):
    seen, drafts = [], []
    def prepare(ctx, description, **kw):
        seen.append(ctx.video_part_request)
        drafts.append(description)
        return description, 6.0
    parts = ['He waves.', 'He raises a cup.']
    rig, ctx, _ = _continued(tmp_path, rig_kw={'prepare_prompt': prepare}, video_plan_parts=parts)
    H._handle_generate_video(ctx, {}, {'description': LONG})
    assert seen == parts
    assert 'raises a cup' not in drafts[0]


def test_the_joins_of_a_chain_are_looked_at_once_after_the_card_is_free(tmp_path):
    # The joins check lived only on the old join path; the live chain path never ran it.
    rig = _Rig(tmp_path)
    rig.clips = Over(rig.clips, first_frame=lambda src, out, skip=0: open(out, "wb").write(b"f") > 0)
    seen = []
    ctx = rig.ctx(vision_model=lambda *a, **k: (
        seen.append((len(k["image_paths"]), comfy_client._excl_users)), "Join 2: the cat vanishes")[1])
    out = H._handle_generate_video(ctx, {}, {"description": LONG, "use_current_images": False})
    n = len(_split(LONG))
    assert seen == [(2 * (n - 1), 0)], seen     # one call, the chain's card session closed
    assert "Join 2: the cat vanishes" in out and "do NOT call the clip seamless" in out


def test_a_chain_join_is_looked_at_where_the_stitch_cuts(tmp_path):
    # The raw last frame of a link beside the raw first frame of the next is ~2 s apart (a
    # trimmed tail, a protected head): every smooth join was reported as a jump (live 2026-10-25).
    rig = _Rig(tmp_path)
    taken = []
    rig.clips = Over(rig.clips, first_frame=lambda src, out, skip=0: (
        taken.append((src, skip)), open(out, "wb").write(b"f") > 0)[1])
    ctx = rig.ctx(vision_model=lambda *a, **k: "no jumps")
    H._handle_generate_video(ctx, {}, {"description": LONG, "use_current_images": False})
    head = int(V.HERRGOTT_CONTEXT)
    assert taken and all(skip in (head - 1, head) for _, skip in taken), taken
    assert taken[0][0] == taken[1][0]                # both sides from the new link's own frames


def test_a_long_continuation_of_a_chain_clip_is_still_cut_into_parts(tmp_path):
    # Bridged before the cut, the script was no longer found in the description and
    # «продолжение на 30 секунд» went out as one 15 s part (live 2026-10-13).
    rig, ctx, _ = _continued(tmp_path)
    H._handle_generate_video(ctx, {}, {"description": LONG, "seconds": 30})
    renders = rig.renders()
    n = len(_split(LONG))
    assert n > 1 and len(renders) == n, renders
    assert renders[0][0].startswith(V.CONTINUE_CTX_PREFIX + "BRIDGE. ")
    # the new links follow the clip's own: restarting at 2 overwrote its links
    assert [kw["chain"][1] for _, kw in renders] == list(range(3, 3 + n))
    assert rig.stitched == [("h3_continuous/chY", 2 + n)]


def test_voices_keep_the_old_path(tmp_path):
    rig = _Rig(tmp_path)
    voice = tmp_path / "v.wav"
    voice.write_bytes(b"w")
    H._handle_generate_video(rig.ctx(anim_voices=[str(voice)]), {},
                             {"description": "He says «Привет».", "use_current_images": False})
    assert rig.calls and not rig.calls[0][1].get("chain") and not rig.stitched


def test_a_prepared_part_calls_no_model():
    """Inside a chain the card is held and the chat model is out: generate_video must not
    reach for it (the Context-IR rewrite met an unloaded model every part)."""
    def boom(*a, **k):
        raise AssertionError("model call during a prepared render")

    def stop(*a, **k):
        raise V.VideoUnavailable("stop")
    ctx = types.SimpleNamespace(text_model=boom, vision_model=boom, video_engine=Over(
        V, engine_available=lambda ctx: (True, ""), prepare_prompt=boom, build_chain_part=stop))
    r = V.generate_video(ctx, "ready prompt", seconds=6.0, prepared=True, chain=("h3_continuous/chZ", 2))
    assert r["reason"] == "stop"


def test_photo_chain_rewrites_for_first_frame_conditioning_and_keeps_camera_motion(tmp_path):
    still = tmp_path / "balloon.png"
    still.write_bytes(b"image")
    rewritten = []

    def rewrite(ctx, prompt, **kw):
        rewritten.append((prompt, kw["mode"]))
        return prompt

    def stop(*a, **kw):
        raise V.VideoUnavailable("stop before render")

    ctx = types.SimpleNamespace(
        yes_no=lambda *a, **kw: True,
        video_engine=Over(V, engine_available=lambda ctx: (True, ""),
                          _context_ir_on=lambda: True, VIDEO_SPEECH_STRESS=False,
                          to_context_ir=rewrite, build_chain_part=stop))
    result = V.generate_video(ctx, "A slow camera push-in on the balloon.",
                              images=[str(still)], seconds=5.17, chain=("test", 1))
    assert result["reason"] == "stop before render"
    assert rewritten[0][1] == "i2va"
    assert "camera push-in" in rewritten[0][0]
    assert V._FRAMING_LOCK_SUFFIX not in rewritten[0][0]


def test_nonchain_references_keep_full_reference_prompt_family():
    modes = []
    ctx = types.SimpleNamespace(
        yes_no=lambda *a, **kw: True,
        video_engine=Over(V, _context_ir_on=lambda: True, VIDEO_SPEECH_STRESS=False,
                          to_context_ir=lambda ctx, d, **kw: modes.append(kw["mode"]) or d))
    V.prepare_prompt(ctx, "A slow camera push-in.", images=["one.png"], seconds=5.17)
    V.prepare_prompt(ctx, "Two people speak.", images=["one.png", "two.png"],
                     audios=["voice.wav"], seconds=5.17)
    assert modes == ["t2va", "ref2va"]


def test_multipart_chain_preparation_uses_the_native_conditioning_family(tmp_path):
    prepared = []
    rig = _Rig(tmp_path, prepare_prompt=lambda ctx, d, **kw: prepared.append(kw) or (d, 6.0),
               drawn_over=lambda *a: False)
    still = tmp_path / "balloon.png"
    still.write_bytes(b"image")
    result = H._handle_generate_video(rig.ctx(text_model=lambda *a, **kw: "", speech_ok=lambda *a: True),
                                      {"image_path": str(still)},
                                      {"description": LONG.replace("«", "").replace("»", ""), "use_current_images": True})
    assert len(prepared) > 1
    assert prepared[0].get("mode") == "i2va"
    assert all(p.get("mode") == "t2va" for p in prepared[1:])
    assert "Mode: image to video" in result


def test_voice_context_does_not_create_a_missing_picture_reference(tmp_path):
    prepared = []
    rig = _Rig(tmp_path, prepare_prompt=lambda ctx, d, **kw: prepared.append(d) or (d, 6.0),
               voices_of=lambda p: 'The man in <Picture 1> speaks in a low voice.',
               drawn_over=lambda *a: False)
    still = tmp_path / 'source.png'
    still.write_bytes(b'image')
    H._handle_generate_video(rig.ctx(text_model=lambda *a, **kw: '', speech_ok=lambda *a: True),
                             {'image_path': str(still)},
                             {'description': LONG.replace('«', '').replace('»', ''), 'use_current_images': True})
    assert len(prepared) > 1
    assert '<Picture 1>' not in prepared[1]
    assert 'the same person' in prepared[1]


def test_writer_added_picture_reference_is_bound_to_the_opening_frame(tmp_path):
    still = tmp_path / 'source.png'
    still.write_bytes(b'image')
    def prepare(ctx, description, **kw):
        return ('The person from <Picture 1> continues.' if kw.get('mode') == 't2va' else description, 6.0)
    rig = _Rig(tmp_path, prepare_prompt=prepare, guard_chain_layout=lambda ctx, src, prompt, part: prompt,
               drawn_over=lambda *a: False)
    ctx = rig.ctx(text_model=lambda *a, **kw: '', speech_ok=lambda *a: True,
                  video_plan_parts=['The person speaks.', 'The person continues.'])
    H._handle_generate_video(ctx, {'image_path': str(still), 'user_input_original': 'Animate the person.'},
                             {'description': 'The person speaks.', 'use_current_images': True})
    assert len(rig.renders()) >= 2
    assert '<Picture 1>' not in rig.renders()[1][0]
    assert 'same subjects from the previous shot' in rig.renders()[1][0]


def test_later_parts_see_the_hero_reference():
    """Parts 2-5 of the BAR clip ran with no picture and each had another man."""
    ctx = types.SimpleNamespace(video_engine=Over(V, _upload=lambda p, k="image": "hero.jpg"))
    kw = dict(chain="h3_continuous/chX", width=768, height=1024, seconds=6.0, seed=1, ctx=ctx)
    wf = V.build_chain_part("he stirs", idx=3, reference="hero.jpg", **kw)
    assert wf["cond"]["inputs"]["qwen_reference_1"] == ["ref", 0]
    assert wf["ref"]["inputs"]["image"] == "hero.jpg"
    assert "qwen_reference_1" not in V.build_chain_part("he stirs", idx=3, **kw)["cond"]["inputs"]


def test_a_spoken_line_stays_in_the_first_part_that_has_it():
    parts = ['He lifts the rifle and says «эта штука тяжёлая».',
             'He lowers it slowly, and says «эта штука тяжёлая».', 'He sighs.']
    out = V.dedupe_quotes(parts)
    assert out[0] == parts[0] and "тяжёлая" not in out[1] and out[1].startswith("He lowers it"), out
    only = ['He says «да».', 'He says «да».']
    assert V.dedupe_quotes(only)[1] == only[1]       # nothing else to render: kept


def test_the_speech_check_wants_the_lines_in_the_sound():
    assert not V.heard_lines(["эта штука слишком тяжёлая"], "ну эта штука слишком тяжелая блин")
    assert not V.heard_lines(["эта штука слишком тяжёлая"], "раз вас сына угут двойные")
    assert V.quoted_lines('He says «ладно, пойдёт» and nods.') == ["ладно, пойдёт"]


def test_the_judge_rewrites_bad_parts_and_never_a_quoted_line():
    parts = ['The man says «ну и тяжесть» and lifts the rifle.', "Sparks swirl in the dark.", "He sighs."]
    asked = []

    def model(ctx, system, user, **k):
        asked.append(system[:20])
        if system.startswith("You are a strict"):
            return '{"bad": [{"part": 2, "why": "only sparks"}]}' if len(asked) == 1 else '{"bad": []}'
        return "He lowers the rifle slowly and rests it on the table."
    ctx = types.SimpleNamespace(text_model=model)
    out = V.validate_script(ctx, parts)
    assert out[0] == parts[0] and out[2] == parts[2] and out[1].startswith("He lowers"), out
    # a rewrite that drops the quote is thrown away
    quoted = ['He says «да».']
    ctx2 = types.SimpleNamespace(text_model=lambda c, s, u, **k: (
        '{"bad": [{"part": 1, "why": "x"}]}' if s.startswith("You are a strict") else "He nods at the door."))
    assert V.validate_script(ctx2, quoted) == quoted
    # ...and so is one that pulls in another part's line: it was then said three times
    silent = ["He smiles.", 'He whispers «Не может быть…».']
    ctx3 = types.SimpleNamespace(text_model=lambda c, s, u, **k: (
        '{"bad": [{"part": 1, "why": "x"}]}' if s.startswith("You are a strict")
        else "He smiles warmly and whispers «Не может быть…»."))
    assert V.validate_script(ctx3, silent) == silent


def test_script_rewrite_cannot_invent_camera_cut():
    parts = ['The person on the left speaks.', 'The person on the right replies "Да, я помню".']
    def model(ctx, system, user, **kw):
        if system.startswith('You are a strict'):
            return '{"bad": [{"part": 2, "why": "unclear"}]}'
        return 'The camera cuts to a close-up. The person on the right replies "Да, я помню".'
    assert V.validate_script(types.SimpleNamespace(text_model=model), parts, rounds=1) == parts
    assert not V._changes_camera_work('The camera cuts to a close-up.', 'The camera cuts to a close-up.')
    assert V._changes_camera_work('The camera pans left.', 'The camera cuts to the man.')
    assert not V._changes_camera_work('He says "cut to the chase".', 'He says "cut to the chase".')
    assert V._changes_camera_work('The camera pans left.', 'The camera pans right.')
    assert V._changes_camera_work('The camera pans left as Alice waves.', 'Alice waves.')
    assert V._changes_camera_work('The camera slowly moves left.', '')
    assert V._changes_camera_work('The camera slowly pans across the room.', '')
    assert V._changes_camera_work('The camera cuts to Alice.', 'The camera cuts to Bob.')
    assert V._changes_camera_work('The camera cuts to Alice.', 'The camera cuts to Alice. The camera cuts to Bob.')


def test_judge_cannot_drop_requested_camera_move():
    parts = ['The camera pans left.', 'The man waves.']
    def model(ctx, system, user, **kw):
        return '{"bad": [{"part": 1, "why": "camera only", "drop": true}]}' if system.startswith('You are a strict') else 'The camera pans left.'
    assert V.validate_script(types.SimpleNamespace(text_model=model), parts, rounds=1) == parts


def test_two_story_steps_in_one_part_are_split_not_cut_to_one():
    """«He runs, launching the kite; the kite soars high» was rewritten to the run alone,
    and the soaring the user asked for was gone (2026-10-17)."""
    parts = ["A boy stands in a field.", "He runs and launches the kite; the kite soars high.",
             "The kite falls into the grass."]
    rounds = []

    def model(ctx, system, user, **k):
        if system.startswith("You are a strict"):
            rounds.append(1)
            return ('{"bad": [{"part": 2, "why": "two actions", "split": true}]}' if len(rounds) == 1
                    else '{"bad": []}')
        if system.startswith("Rewrite this part"):
            return "He runs and launches the kite.\nThe kite soars high into the sky."
        return "He runs across the field."
    moves = lambda q, text, **k: "stands" not in text and "sits" not in text
    ns = types.SimpleNamespace(text_model=model, yes_no=moves)
    out = V.validate_script(ns, parts, max_parts=5)
    assert out == [parts[0], "He runs and launches the kite.", "The kite soars high into the sky.",
                   parts[2]], out
    # a split line where nothing moves is no part of its own: «The man sits at a wooden table.»
    rounds.clear()
    split = model
    model = lambda c, s, u, **k: ("The boy sits in the grass.\nHe runs and launches the kite.\nThe kite soars."
                                  if s.startswith("Rewrite this part") else split(c, s, u, **k))
    out = V.validate_script(types.SimpleNamespace(text_model=model, yes_no=moves), parts, max_parts=5)
    assert out[1] == "The boy sits in the grass. He runs and launches the kite." and len(out) == 4, out
    model = split
    # no room for another part: the old rewrite stands
    rounds.clear()
    out = V.validate_script(ns, parts, max_parts=3)
    assert len(out) == 3 and out[1] == "He runs across the field.", out


def test_steps_past_the_part_limit_join_the_last_part_not_dropped():
    """Three camera steps, room for two: the third, «stops on the departure board», was cut off
    and the clip ended before the end the user asked for (2026-10-21)."""
    script = ["The camera flies along the train, then rises above its roof and stops on the departure board."]

    def model(ctx, system, user, **k):
        if system.startswith("You are a strict"):
            return '{"bad": [{"part": 1, "why": "three camera moves", "split": true}]}'
        if system.startswith("Rewrite this part"):
            return ("The camera flies along the train.\nThe camera rises above the roof of the train.\n"
                    "The camera stops on the departure board.")
        return "The camera flies along the train."
    out = V.validate_script(types.SimpleNamespace(text_model=model, yes_no=lambda q, t, **k: True),
                            script, rounds=1, max_parts=2)
    assert out == ["The camera flies along the train.",
                   "The camera rises above the roof of the train. The camera stops on the departure board."], out


def test_a_chain_part_runs_taomates_frame_adapter_at_its_three_steps():
    """FL2V Turbo at 6 steps came back soapy, so the chain took TaoMate's 3 steps -- but its
    text-to-video adapter, fed frames, drew flowers, anime faces and captions over them
    (2026-10-15). TaoMate's FL2VA adapter: as sharp, made for frames."""
    plain = V.build_workflow("a chef", mode="t2va", width=1344, height=768, frames=226, seed=1,
                             images=[], videos=[], audios=[], steps=V.VIDEO_STEPS, ctx=None)
    assert plain[V.N_LORA]["inputs"]["lora_name"] == V.VIDEO_T2VA_LORA_3STEP     # text only: T2AV
    for idx in (1, 2):
        part = V.build_chain_part("a chef", chain="h3_continuous/chX", idx=idx, width=1344,
                                  height=768, seconds=9.4, seed=1,
                                  first_frame="photo.png" if idx == 1 else "",
                                  ctx=_uploads("photo.png"))
        assert part[V.N_LORA]["inputs"]["lora_name"] == V.VIDEO_FL2VA_LORA_3STEP
        assert "fl2va" in V.VIDEO_FL2VA_LORA_3STEP
        assert part[V.N_SAMPLER]["inputs"]["steps"] == plain[V.N_SAMPLER]["inputs"]["steps"] == 3


def test_later_parts_know_the_scene_and_carry_no_example_action():
    # Part 2 of «naked women walk past the bag of money at Lakhta» came out as a hand
    # putting a tool on a counter and hitting a stone with a mallet -- the prefix's own example.
    assert "counter" not in V.CONTINUE_CTX_PREFIX and "tool" not in V.CONTINUE_CTX_PREFIX
    part = "They walk on toward the water."
    d = V.CONTINUE_CTX_PREFIX + V.with_scene(part, "A paper bag of money at Lakhta; women walk in.")
    assert "bag of money at Lakhta" in d and d.startswith(V.CONTINUE_CTX_PREFIX + part)
    assert _est(d) == _est(part)
    assert V.with_scene(part, "") == part


def test_a_small_flying_thing_gets_a_close_up_and_the_next_part_cuts_back():
    # Live 2026-10-11: on a wide shot the cork never left the bottle and the apple fell unseen.
    ctx = types.SimpleNamespace(text_model=lambda *a, **k: '{"close": [2]}')
    out = V.mark_inserts(ctx, ["A man twists the cork.", "The cork shoots out.", "Everyone laughs."])
    assert out[0] == "A man twists the cork."
    assert out[1].startswith(V.INSERT_HEAD) and out[2].startswith(V.INSERT_BACK)
    # the prefix names both cuts, in words nobody can mistake for a spoken line (no quotes)
    assert "a cut in close" in V.CONTINUE_CTX_PREFIX and "a cut back out" in V.CONTINUE_CTX_PREFIX
    assert "\"The camera cuts" not in V.CONTINUE_CTX_PREFIX


def test_marking_fails_soft():
    parts = ["a", "b"]
    assert V.mark_inserts(types.SimpleNamespace(text_model=lambda *a, **k: "no json here"), parts) == parts


def test_continue_in_words_goes_on_from_the_chats_last_clip(tmp_path):
    # «продолжи это видео» typed, no ▶️: a new clip from nothing, another style (live 2026-10-13).
    last = tmp_path / "delivered.mp4"
    last.write_bytes(b"s")
    rig = _Rig(tmp_path, chains={str(last): ("h3_continuous/chZ", 4)})
    ctx = rig.ctx(last_video_path=str(last), yes_no=lambda q, t, default=False, wait=True: "continue" in q or "MAKE a video clip" in q)
    H._handle_generate_video(ctx, {"user_input_original": "продолжи это видео: кот прыгает"},
                             {"description": "The cat jumps onto the table."})
    (d, kw), = rig.calls
    assert kw["chain"] == ("h3_continuous/chZ", 5) and kw["size_from"] == str(last)
    rig.calls.clear()
    H._handle_generate_video(rig.ctx(last_video_path=str(last)),
                             {"user_input_original": "сделай видео: кот прыгает"},
                             {"description": "The cat jumps onto the table."})
    assert not rig.calls[0][1].get("chain") or rig.calls[0][1]["chain"][1] == 1   # a new clip stays new


def test_attached_video_beats_its_newly_created_frame_sheet(tmp_path):
    last = tmp_path / "attached.mp4"
    last.write_bytes(b"source")
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"sheet")
    os.utime(last, (1, 1))
    rig = _Rig(tmp_path, chains={str(last): ("h3_continuous/chZ", 4)})
    ctx = rig.ctx(last_video_path=str(last), attached_video_path=str(last), voice_choice="",
                  yes_no=lambda q, t, default=False, wait=True: "continue" in q or "MAKE a video clip" in q,
                  deps=types.SimpleNamespace(_current_image_paths=lambda c, s: [str(sheet)],
                                             script_read=_fakes.script_model))
    from prompt_guard import wrap_quoted
    state = {"user_input_original": "продолжи это видео, он говорит «Привет»\n" +
             wrap_quoted("original video transcript", 'The old video says "unwanted words".')}
    H._handle_generate_video(ctx, state, {"description": "He says «Привет unwanted words».", "speakers": 1})
    assert not state.get("ask_voices")
    assert rig.calls[0][1]["chain"] == ("h3_continuous/chZ", 5)
    assert 'unwanted words' not in rig.calls[0][0]


def test_missing_attached_video_never_falls_back_to_an_old_clip(tmp_path):
    old = tmp_path / "old.mp4"
    old.write_bytes(b"old")
    rig = _Rig(tmp_path)
    ctx = rig.ctx(last_video_path=str(old), attached_video_path=str(tmp_path / "missing.mp4"))
    result = H._handle_generate_video(ctx, {"user_input_original": "продолжи это видео"},
                                      {"description": "Continue the video."})
    assert "attached video is no longer available" in result
    assert not rig.calls


def test_a_picture_sent_after_the_clip_is_animated_not_the_clip_continued(tmp_path):
    # «оживи на своё усмотрение» under a fresh meme went on from an old clip (live 2026-10-13).
    last = tmp_path / "delivered.mp4"
    last.write_bytes(b"s")
    pic = tmp_path / "meme.jpg"
    pic.write_bytes(b"j")
    os.utime(last, (1000, 1000))
    rig = _Rig(tmp_path, chains={str(last): ("h3_continuous/chZ", 4)})
    ctx = rig.ctx(last_video_path=str(last), yes_no=lambda q, t, default=False, wait=True: True,
                  deps=types.SimpleNamespace(_current_image_paths=lambda c, s: [str(pic)],
                                             script_read=_fakes.script_model))
    H._handle_generate_video(ctx, {"user_input_original": "оживи на своё усмотрение"},
                             {"description": "The cats charge."})
    assert not rig.calls[0][1].get("chain") or rig.calls[0][1]["chain"][1] == 1


def test_an_approved_plan_keeps_the_continuation_prefix_and_the_whole_first_part(tmp_path):
    # The approved first part replaced the description whole (prefix gone), and the bridge
    # then cut the prefix's length off the part: «What happens next: Camera: static…»
    # (live 2026-10-13).
    prepped = []
    first = "The baker sets the cat down on the floor and it runs off to the door."
    rig, ctx, _ = _continued(
        tmp_path, rig_kw={"prepare_prompt": lambda ctx, d, **kw: prepped.append(d) or (d, 6.0)},
        video_plan_parts=[first, "The baker waves after it and laughs."])
    H._handle_generate_video(ctx, {}, {"description": LONG})
    assert prepped and prepped[0].startswith(V.CONTINUE_CTX_PREFIX + "BRIDGE. " + first), prepped[:1]


def test_a_picture_script_is_rewritten_as_an_event_never_as_one_still_line():
    # One sentence back («The camera pans across the army») made one part: no plan, and a
    # render started with nothing happening (live 2026-10-13).
    one = types.SimpleNamespace(text_model=lambda *a, **k: "The camera pans across the army. Sound: wind.")
    assert V.action_script("Cats stand. A flag waves. Sound: wind.", one) == ""
    told = []
    event = ("The cats charge. The armies clash. One cat tears the enemy flag down. "
             "Sound: battle cries.")
    ok = types.SimpleNamespace(text_model=lambda c, sysp, user, **k: told.append(sysp) or event)
    assert V.action_script("Cats stand. A flag waves.", ok) == event
    assert "FIRST FRAME" in told[0] and "invent the obvious one" in told[0]


def test_translation_gloss_after_a_quote_is_cut():
    """The gloss «Море...» (The sea...) was spoken in English: only the quote may stay."""
    out = V.strip_translations(['He says «Море забирает печали» (The sea takes sorrows). Then "Идём" (Let us go) he adds.'])
    assert out == ['He says «Море забирает печали». Then "Идём" he adds.']


def test_a_clip_covered_over_at_its_edges_is_told_from_a_still_one(tmp_path):
    """Drawn flowers over a locked-off lake shot: the frame edges change wholesale."""
    import subprocess
    still, covered = str(tmp_path / "still.mp4"), str(tmp_path / "covered.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=teal:s=320x180:d=3",
                    "-pix_fmt", "yuv420p", still], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=teal:s=320x180:d=3",
                    "-vf", "drawbox=x=0:y=0:w=320:h=60:color=white:t=fill:enable='gte(t,1)'",
                    "-pix_fmt", "yuv420p", covered], check=True)
    assert V.overlay_drift(still) <= V.OVERLAY_DRIFT_MAX < V.overlay_drift(covered)
    assert V.overlay_drift(str(tmp_path / "missing.mp4")) is None


def test_later_parts_are_told_the_voices_part_one_chose():
    """Each part is written on its own: a hiker seen from behind whispered as a woman in part 1
    and spoke as a man in part 2. The voice wording carries over; the spoken words do not."""
    p1 = ("[Shot 1] The hiker stands still. <Subject 1> (S1) whispers in a woman's low, breathy voice, "
          "<d>[Russian] Наконец-то я здесь.</d> The wind whistles. A distant male voice calls out, "
          "<d>[Russian] Не забудь вернуться!</d> Camera: static.")
    v = V.voices_of(p1)
    assert "a woman's low, breathy voice" in v and "distant male voice" in v
    assert "Наконец" not in v and "wind" not in v
    assert V.voices_of("A man walks. Camera: static.") == ""


def test_light_drawings_over_a_photo_are_told_against_the_source(tmp_path):
    """Pale cartoon birds over a hazy sky barely moved the edges (24 < 35) and went out; a
    photo of real daisies reads as «drawn over» itself, so only a rise above the source counts."""
    import subprocess
    import torch
    clip = str(tmp_path / "c.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=teal:s=320x180:d=4",
                    "-pix_fmt", "yuv420p", clip], check=True)
    scores = []

    class Out:
        def __init__(self, n):
            p = torch.tensor(scores[:n]).clamp(1e-4, 1 - 1e-4)
            self.logits_per_image = torch.stack([torch.log(1 - p), torch.log(p)], 1)

    fake = [lambda **k: Out(len(k["images"])), lambda images, **k: {"images": images}]
    scores[:] = [0.1, 0.1, 0.99, 1.0, 0.99]           # a clean photo, then birds from the 2nd second
    assert V.drawn_over(clip, clip=fake) > V.DRAWN_OVER_MAX
    scores[:] = [0.95, 0.96, 0.95, 0.97, 0.94]        # real daisies all along
    assert V.drawn_over(clip, clip=fake) <= V.DRAWN_OVER_MAX
    assert V.drawn_over(str(tmp_path / "missing.mp4"), clip=fake) is None


def test_an_action_inside_a_camera_line_stays_in_its_own_part():
    """«…; the camera pans up as the kite catches a breeze and soars high» rode every part
    as a camera line, and the judge cut the climb out of each (2026-10-17)."""
    s = ("A boy stands in a field, holding a kite string. He runs forward, launching the kite; "
         "the camera pans up as the kite catches a breeze and soars high. The kite flutters and "
         "falls into the grass. The boy runs to it and laughs. Sound: wind.")
    parts = _split(s)
    assert sum("soars high" in p for p in parts) == 1, parts
    assert sum("camera pans up" in p for p in parts) == 1, parts   # a camera move happens once
    assert _split("The camera follows the cat as it walks.") == ["The camera follows the cat as it walks."]


def test_how_the_scene_opens_is_not_replayed_by_every_part():
    """«The scene begins with a close-up of the letter. The camera pans up to reveal the man»
    rode every part, and each part replayed the pan (2026-10-17); the lighting does belong
    to every part."""
    s = ("The scene begins with a close-up of an old letter on a table. The camera slowly pans up "
         "to reveal an old man. He reaches for the letter, picks it up and begins to read it. "
         "As he reads, his eyes widen and he whispers «Не может быть…» looking at the paper. "
         "Finally he looks up from the letter with a gentle smile. The lighting is soft and warm.")
    parts = _split(s)
    assert len(parts) > 1, parts
    assert sum("pans up" in p for p in parts) == 1 and "pans up" in parts[0], parts
    assert sum("scene begins" in p for p in parts) == 1, parts
    assert all("lighting is soft" in p for p in parts), parts


def test_a_line_inside_a_sound_line_is_said_once():
    """«Sound: paper rustling, a whispered «Не может быть…»» rode every part as a look line,
    and the line was said in three parts running (2026-10-17)."""
    s = ("His trembling hands unfold the yellowed letter on the table. Sound: paper rustling, heavy "
         "breathing, a whispered «Не может быть…». His eyes widen as he reads the letter slowly. "
         "He leans back in his chair and stares at the wall for a while. Finally a slow warm smile "
         "spreads across his face as he looks up.")
    parts = _split(s)
    assert len(parts) > 1, parts
    assert sum("Не может быть" in p for p in parts) == 1, parts


def test_a_talking_head_keeps_the_camera_still_whatever_the_agent_wrote():
    """It opens and ends on the still, so no camera move fits: the agent's own «the camera
    slowly zooms in» (the user asked for none) drifted all three takes off the still's eyes."""
    asked = []
    ctx = types.SimpleNamespace(
        yes_no=lambda q, text, *a, **k: asked.append(q) or True,     # «yes, it asks for a move»
        video_engine=Over(V, _context_ir_on=lambda: False, VIDEO_SPEECH_STRESS=False))
    said = 'The camera slowly zooms in on his face as he says: "Ах, вот и ты."'
    head, _ = V.prepare_prompt(ctx, said, images=["still.jpg"], seconds=5.0, head=True, keyframe=True)
    assert V._FRAMING_LOCK_SUFFIX in head
    assert not any("camera to move" in q for q in asked)
    shot, _ = V.prepare_prompt(ctx, said, images=["still.jpg"], seconds=5.0)
    assert V._FRAMING_LOCK_SUFFIX not in shot           # an animated photo still gets the zoom it asked for

def test_failed_chain_stitch_reports_partial_and_does_not_remember_full_chain(tmp_path):
    rig = _Rig(tmp_path, stitch_chain=lambda *args: None)
    ctx = rig.ctx()
    state = {}
    answer = H._handle_generate_video(ctx, state, {"description": LONG, "use_current_images": False})
    assert ctx.video_parts_incomplete
    assert "incomplete" in answer.lower()
    assert not rig.kept
    assert 'delivered video is incomplete' in answer
    assert 'unconfirmed' in ctx.last_video_prompt


def test_failed_continuation_join_reports_original_missing(tmp_path):
    src = tmp_path / 'original.mp4'
    tail = tmp_path / 'tail.mp4'
    src.write_bytes(b'source')
    tail.write_bytes(b'tail')
    rig = _Rig(tmp_path, herrgott_on=lambda: False, join_continuation=lambda *a, **kw: None)
    ctx = rig.ctx(continue_src=str(src), continue_tail=str(tail))
    answer = H._handle_generate_video(ctx, {}, {'description': 'The person waves.'})
    assert ctx.video_parts_incomplete
    assert 'incomplete' in answer.lower()

def test_failed_later_chain_render_is_reported_incomplete(tmp_path, monkeypatch):
    rig = _Rig(tmp_path)
    ctx = rig.ctx()
    monkeypatch.setattr(H, '_render_chain_parts', lambda ctx, path, parts, args, chain, idx, **kw: idx)
    answer = H._handle_generate_video(ctx, {}, {'description': LONG, 'use_current_images': False})
    assert ctx.video_parts_incomplete
    assert 'incomplete' in answer.lower()
    assert not rig.kept


def test_failed_later_approved_part_is_not_claimed_as_rendered(tmp_path, monkeypatch):
    from contextlib import nullcontext
    monkeypatch.setattr(comfy_client, 'card_session', lambda *a, **k: nullcontext())
    monkeypatch.setattr(comfy_client, 'this_thread_holds_card', lambda: True)
    rig = _Rig(tmp_path)
    ctx = rig.ctx(video_plan_parts=['First scene.', 'Second scene.', 'Last scene never rendered.'])
    memories = []
    checks = []
    ctx.deps = types.SimpleNamespace(script_read=_fakes.script_model, text_model=lambda *a, **kw: '',
                                     _remember=lambda *args: memories.append(args))
    monkeypatch.setattr(H, '_render_chain_parts', lambda ctx, path, parts, args, chain, idx, **kw: idx)
    monkeypatch.setattr(H, '_check_actions', lambda *args, **kw: checks.append(args) or 'Last scene never rendered.')
    answer = H._handle_generate_video(ctx, {}, {'description': LONG, 'use_current_images': False})
    assert ctx.video_parts_incomplete
    assert 'Last scene never rendered.' not in answer
    assert 'incomplete' in answer.lower()
    assert 'Say it is ready briefly' not in answer
    assert 'Last scene never rendered.' not in ctx.last_video_prompt
    assert 'unconfirmed' in ctx.last_video_prompt
    assert all('Last scene never rendered.' not in str(memory) for memory in memories)
    assert not checks


def test_video_planning_reports_preparation_stage(tmp_path, monkeypatch):
    from contextlib import nullcontext
    monkeypatch.setattr(comfy_client, 'card_session', lambda *a, **k: nullcontext())
    monkeypatch.setattr(comfy_client, 'this_thread_holds_card', lambda: True)
    rig = _Rig(tmp_path)
    stages = []
    ctx = rig.ctx(video_plan_parts=['First scene.', 'Second scene.'])
    ctx.set_stage = stages.append
    ctx.deps = types.SimpleNamespace(script_read=_fakes.script_model, text_model=lambda *a, **kw: '')
    monkeypatch.setattr(H, '_render_chain_parts', lambda ctx, path, parts, args, chain, idx, **kw: idx)
    H._handle_generate_video(ctx, {}, {'description': LONG, 'use_current_images': False})
    assert 'Preparing video' in stages
    import stages as stage_labels
    assert stage_labels.translate('Preparing video', 'ru') == 'Готовлю видео'

@pytest.mark.parametrize('spoken', [False, True])
def test_clip_speaker_is_checked_for_dialogue_not_quoted_sign(tmp_path, monkeypatch, spoken):
    monkeypatch.setitem(sys.modules, 'identity_metrics', types.SimpleNamespace(face_boxes=lambda _: [(0, 0, 1, 1)] * 2))
    seen = []
    rig, ctx, _ = _continued(tmp_path, rig_kw={'clip_speaker': lambda *a: seen.append(True) or 'the man'})
    seed = tmp_path / 'seed.jpg'
    seed.write_bytes(b'image')
    ctx.deps = types.SimpleNamespace(_current_image_paths=lambda c, s: [str(seed)], script_read=_fakes.script_model)
    checked = []
    def yes(q, t, default=False, **kw):
        if 'spoken dialogue' in q:
            checked.append(t)
            return spoken
        if 'explicitly identify the speaker' in q:
            return False
        return default
    ctx.yes_no = yes
    text = 'He says "Hello".' if spoken else 'Keep the sign "GOOD TALKS" unchanged.'
    H._handle_generate_video(ctx, {'user_input_original': text}, {'description': text})
    assert bool(seen) is spoken
    assert checked and all(item == text for item in checked)


def test_explicit_speaker_is_not_replaced_by_previous_clip_speaker(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'identity_metrics', types.SimpleNamespace(face_boxes=lambda _: [(0, 0, 1, 1)] * 2))
    seen = []
    rig, ctx, _ = _continued(tmp_path, rig_kw={
        'clip_speaker': lambda *a: seen.append(True) or 'the woman on the left',
        'dictated_speaker': lambda *a: 'the woman on the right'})
    seed = tmp_path / 'seed.jpg'
    seed.write_bytes(b'image')
    ctx.deps = types.SimpleNamespace(_current_image_paths=lambda c, s: [str(seed)], script_read=_fakes.script_model)
    ctx.yes_no = lambda q, t, default=False, **kw: True
    text = 'The woman on the right says "Hello".'
    H._handle_generate_video(ctx, {'user_input_original': text}, {'description': text})
    assert not seen


def test_uncertain_speaker_assignment_does_not_override_request(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'identity_metrics', types.SimpleNamespace(face_boxes=lambda _: [(0, 0, 1, 1)] * 2))
    seen = []
    rig, ctx, _ = _continued(tmp_path, rig_kw={
        'clip_speaker': lambda *a: seen.append(True) or 'the woman on the left',
        'dictated_speaker': lambda *a: ''})
    seed = tmp_path / 'seed.jpg'
    seed.write_bytes(b'image')
    ctx.deps = types.SimpleNamespace(_current_image_paths=lambda c, s: [str(seed)], script_read=_fakes.script_model)
    ctx.yes_no = lambda q, t, default=False, **kw: True if 'spoken dialogue' in q else default
    text = 'The woman on the right says "Hello".'
    H._handle_generate_video(ctx, {'user_input_original': text}, {'description': text})
    assert not seen

def test_silent_static_continuation_does_not_apply_voice(tmp_path):
    src, tail, seed = (tmp_path / name for name in ('original.mp4', 'tail.mp4', 'seed.jpg'))
    for p in (src, tail, seed):
        p.write_bytes(b'x')
    voices, checks, reads = [], [], []
    rig = _Rig(tmp_path, eye_drift=lambda *a, **k: checks.append("eyes") or 0,
               herrgott_on=lambda: False, static_continuation=lambda c, text: reads.append(text) or True,
               voice_sample=lambda p: voices.append(p) or str(src),
               join_continuation=lambda *a, **kw: str(tail))
    ctx = rig.ctx(continue_src=str(src), continue_tail=str(tail))
    ctx.deps = types.SimpleNamespace(_current_image_paths=lambda c, s: [str(seed)], script_read=_fakes.script_model)
    ctx.yes_no = lambda q, t, default=False, **kw: True if 'explicitly prohibit speech' in q else default
    ctx.speech_ok = lambda *a: checks.append('speech') or True
    H._handle_generate_video(ctx, {'user_input_original': 'Продолжи без речи, сохрани надпись «GOOD TALKS».'}, {'description': 'He waves silently beside the sign \"GOOD TALKS\".'})
    assert reads == ['Продолжи без речи, сохрани надпись «GOOD TALKS».']
    assert not voices and not checks
    assert ctx.video_static_continuation

def test_static_continuation_keeps_keyframes_without_freezing_face():
    ctx = types.SimpleNamespace(video_static_continuation=True, video_request='He turns his head and waves.',
                                yes_no=lambda *a, **k: False,
                                video_engine=Over(V, _context_ir_on=lambda: True,
                                                  to_context_ir=lambda ctx, description, **kw: description,
                                                  VIDEO_SPEECH_STRESS=False))
    prompt, _ = V.prepare_prompt(ctx, ctx.video_request, images=['still.jpg', 'still.jpg'],
                                 seconds=5, keyframe=True)
    assert V._FRAMING_LOCK_SUFFIX in prompt
    assert V._HEAD_LOCK_SUFFIX not in prompt

def test_preserving_existing_text_does_not_add_an_overlay(tmp_path, monkeypatch):
    clip = tmp_path / 'clip.mp4'
    clip.write_bytes(b'video')
    monkeypatch.setattr(V, '_onscreen_text_words', lambda *a: ['GOOD TALKS'])
    monkeypatch.setattr(V, '_overlay_text', lambda *a: pytest.fail('must not add a caption'))
    ctx = types.SimpleNamespace(video_request='Сохрани вывеску «GOOD TALKS».', video_text_overlay_allowed=False)
    assert V._ensure_onscreen_text(ctx, str(clip), ctx.video_request) == str(clip)

def test_preserving_source_text_still_uses_typography_adapter(tmp_path):
    old = tmp_path / 'old.png'
    old.write_bytes(b'image')
    rig = _Rig(tmp_path, drawn_over=lambda *a: False)
    ctx = rig.ctx(last_image_path=str(old))
    ctx.yes_no = lambda q, t, default=False, **kw: False if 'adding NEW visible text' in q else True
    H._handle_generate_video(ctx, {'user_input_original': 'Оживи и сохрани вывеску «GOOD TALKS».'},
                             {'description': 'Animate this image; keep the sign.', 'use_current_images': True})
    assert ctx.video_text_required and ctx.video_text_source
    assert ctx.video_text_overlay_allowed is False


def test_preserve_request_collects_ocr_guesses_for_prompt_guard(tmp_path, monkeypatch):
    import ocr_reader
    monkeypatch.setattr(ocr_reader, 'read_boxes', lambda path: [([], 'S0Z', 0.75), ([], 'ERRA', 0.98)])
    old = tmp_path / 'old.png'
    old.write_bytes(b'image')
    rig = _Rig(tmp_path, drawn_over=lambda *a: False)
    ctx = rig.ctx(last_image_path=str(old))
    ctx.yes_no = lambda q, t, default=False, **kw: 'adding NEW visible text' not in q
    H._handle_generate_video(ctx, {'user_input_original': 'Оживи и сохрани надписи.'},
                             {'description': 'Keep SOZ and ERRA.', 'use_current_images': True})
    assert ctx.video_untrusted_ocr == ('S0Z', 'ERRA')

def test_short_explicit_continuation_duration_survives_intent_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(H, 'intent_read', lambda ctx: lambda *a: {'song_seconds': 0})
    rig, ctx, _ = _continued(tmp_path, video_plan_ask=True)
    state = {'user_input_original': 'Продолжи видео на 5 секунд: он говорит «Добрый вечер, друзья».'}
    result = H._handle_generate_video(ctx, state, {'description': 'The man sits. He says "Добрый вечер, друзья".'})
    assert '[NOT MADE YET]' not in result
    assert rig.renders() and rig.renders()[0][1]['seconds'] == 5

def test_static_continuation_retries_camera_drift_and_keeps_better_take(tmp_path, monkeypatch):
    src, tail, seed = (tmp_path / name for name in ('original.mp4', 'tail.mp4', 'seed.jpg'))
    for p in (src, tail, seed): p.write_bytes(b'x')
    rig = _Rig(tmp_path, herrgott_on=lambda: False, static_continuation=lambda *a: True,
               voice_sample=lambda *a: '', new_bright_region=lambda *a: False,
               overlay_drift=lambda p: 64 if p.endswith('part1.mp4') else 5,
               OVERLAY_DRIFT_MAX=35, join_continuation=lambda *a, **kw: str(tail))
    ctx = rig.ctx(continue_src=str(src), continue_tail=str(tail))
    ctx.deps = types.SimpleNamespace(_current_image_paths=lambda c, s: [str(seed)], script_read=_fakes.script_model)
    ctx.yes_no = lambda q, t, default=False, **kw: False
    H._handle_generate_video(ctx, {'user_input_original': 'Камера неподвижна, человек машет.'},
                             {'description': 'The person waves, camera stationary.', 'seconds': 5})
    assert len(rig.renders()) == 2

def test_static_speech_check_ignores_quoted_source_text_not_requested(tmp_path):
    src, tail, seed = (tmp_path / name for name in ('original.mp4', 'tail.mp4', 'seed.jpg'))
    for p in (src, tail, seed): p.write_bytes(b'x')
    heard = []
    rig = _Rig(tmp_path, herrgott_on=lambda: False, static_continuation=lambda *a: True,
               voice_sample=lambda *a: '', new_bright_region=lambda *a: False,
               overlay_drift=lambda p: 0, join_continuation=lambda *a, **kw: str(tail))
    ctx = rig.ctx(continue_src=str(src), continue_tail=str(tail))
    ctx.deps = types.SimpleNamespace(_current_image_paths=lambda c, s: [str(seed)], script_read=_fakes.script_model)
    ctx.speech_ok = lambda c, p, lines: heard.append(lines) or True
    ctx.yes_no = lambda q, t, default=False, **kw: False
    H._handle_generate_video(ctx, {'user_input_original': 'Продолжи: он говорит «Добрый вечер, друзья».'},
                             {'description': 'The sign "СТРИМ НА ПАУЗЕ" stays. He says "Добрый вечер, друзья".', 'seconds': 5})
    assert heard and heard[0] == ['Добрый вечер, друзья']
