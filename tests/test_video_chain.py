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
    reference_images = []
    voice_choice = None
    continue_tail = continue_src = ""

    def set_stage(self, *_): pass
    def is_cancelled(self): return False
    def remember(self, *a): pass
    def memory_text(self): return ""


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
            self.calls.append((description, dict(kw, card=comfy_client.card_in_use())))
            return {"path": str(out), "status": "success", "seconds": 6.0}

        def prep(ctx, d, **kw):
            self.calls.append(("PREP", {"card": comfy_client.card_in_use()}))
            return d, 6.0

        self.engine = Over(V, **{
            "generate_video": gen,
            "engine_available": lambda ctx: (True, ""),
            "herrgott_on": lambda: True,
            "motion_context_on": lambda: False,
            "bridge_part": lambda ctx, f, part: "BRIDGE. " + part,
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
                          first_frame=lambda src, out: False)

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


def test_a_long_script_is_one_chain_stitched_once(tmp_path):
    asked = []
    rig = _Rig(tmp_path, bridge_part=_no_vision,
               bridge_script=lambda ctx, ps: asked.append(len(rig.calls)) or [ps[0]] + ["BRIDGE. " + p for p in ps[1:]])
    st = {}
    H._handle_generate_video(rig.ctx(), st, {"description": LONG, "use_current_images": False})
    calls = rig.calls
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


def test_the_joins_of_a_chain_are_looked_at_once_after_the_card_is_free(tmp_path):
    # The joins check lived only on the old join path; the live chain path never ran it.
    rig = _Rig(tmp_path)
    rig.clips = Over(rig.clips, first_frame=lambda src, out: open(out, "wb").write(b"f") > 0)
    seen = []
    ctx = rig.ctx(vision_model=lambda *a, **k: (
        seen.append((len(k["image_paths"]), comfy_client._excl_users)), "Join 2: the cat vanishes")[1])
    out = H._handle_generate_video(ctx, {}, {"description": LONG, "use_current_images": False})
    n = len(V.split_script(LONG))
    assert seen == [(2 * (n - 1), 0)], seen     # one call, the chain's card session closed
    assert "Join 2: the cat vanishes" in out and "do NOT call the clip seamless" in out


def test_a_long_continuation_of_a_chain_clip_is_still_cut_into_parts(tmp_path):
    # Bridged before the cut, the script was no longer found in the description and
    # «продолжение на 30 секунд» went out as one 15 s part (live 2026-10-13).
    rig, ctx, _ = _continued(tmp_path)
    H._handle_generate_video(ctx, {}, {"description": LONG, "seconds": 30})
    renders = rig.renders()
    n = len(V.split_script(LONG))
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
    assert V.heard_lines(["эта штука слишком тяжёлая"], "ну эта штука слишком тяжелая блин")
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


def test_a_chain_part_runs_taomates_frame_adapter_at_its_three_steps():
    """FL2V Turbo at 6 steps came back soapy, so the chain took TaoMate's 3 steps -- but its
    text-to-video adapter, fed frames, drew flowers, anime faces and captions over them
    (2026-10-15). TaoMate's FL2VA adapter: as sharp, made for frames."""
    plain = V.build_workflow("a chef", mode="t2va", width=1344, height=768, frames=226, seed=1,
                             images=[], videos=[], audios=[], steps=V.VIDEO_STEPS, ctx=None)
    assert plain[V.N_LORA]["inputs"]["lora_name"] == V.VIDEO_T2VA_LORA_3STEP     # text only: T2AV
    for idx in (1, 2):
        part = V.build_chain_part("a chef", chain="h3_continuous/chX", idx=idx, width=1344,
                                  height=768, seconds=9.4, seed=1)
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
    assert V.estimate_seconds(d) == V.estimate_seconds(part)
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
    ctx = rig.ctx(last_video_path=str(last), yes_no=lambda q, t, default=False, wait=True: "continue" in q)
    H._handle_generate_video(ctx, {"user_input_original": "продолжи это видео: кот прыгает"},
                             {"description": "The cat jumps onto the table."})
    (d, kw), = rig.calls
    assert kw["chain"] == ("h3_continuous/chZ", 5) and kw["size_from"] == str(last)
    rig.calls.clear()
    H._handle_generate_video(rig.ctx(last_video_path=str(last)),
                             {"user_input_original": "сделай видео: кот прыгает"},
                             {"description": "The cat jumps onto the table."})
    assert not rig.calls[0][1].get("chain") or rig.calls[0][1]["chain"][1] == 1   # a new clip stays new


def test_a_picture_sent_after_the_clip_is_animated_not_the_clip_continued(tmp_path):
    # «оживи на своё усмотрение» under a fresh meme went on from an old clip (live 2026-10-13).
    last = tmp_path / "delivered.mp4"
    last.write_bytes(b"s")
    pic = tmp_path / "meme.jpg"
    pic.write_bytes(b"j")
    os.utime(last, (1000, 1000))
    rig = _Rig(tmp_path, chains={str(last): ("h3_continuous/chZ", 4)})
    ctx = rig.ctx(last_video_path=str(last), yes_no=lambda q, t, default=False, wait=True: True,
                  deps=types.SimpleNamespace(_current_image_paths=lambda c, s: [str(pic)]))
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
    parts = V.split_script(s)
    assert sum("soars high" in p for p in parts) == 1, parts
    assert sum("camera pans up" in p for p in parts) == 1, parts   # a camera move happens once
    assert V.split_script("The camera follows the cat as it walks.") == ["The camera follows the cat as it walks."]


def test_how_the_scene_opens_is_not_replayed_by_every_part():
    """«The scene begins with a close-up of the letter. The camera pans up to reveal the man»
    rode every part, and each part replayed the pan (2026-10-17); the lighting does belong
    to every part."""
    s = ("The scene begins with a close-up of an old letter on a table. The camera slowly pans up "
         "to reveal an old man. He reaches for the letter, picks it up and begins to read it. "
         "As he reads, his eyes widen and he whispers «Не может быть…» looking at the paper. "
         "Finally he looks up from the letter with a gentle smile. The lighting is soft and warm.")
    parts = V.split_script(s)
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
    parts = V.split_script(s)
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
