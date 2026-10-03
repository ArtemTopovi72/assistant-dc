"""Music generation (MiniMax Music 3) — offline, with ComfyUI/LLM intercepted.

Nothing here touches a GPU, weights or a running ComfyUI: every external
boundary (submit/poll, the house LLM call) is replaced. What IS exercised is
the part that decides what gets rendered:

  * readiness — missing_weights() / engine_available() refuse honestly and
    never let generate_music() promise a song it cannot make.
  * the graph shape — lyrics/style/duration/seed reach the right nodes.
  * build_structured_caption — the LLM is called exactly once, the prompt
    threads `lang` correctly, and the JSON reply is parsed into
    {"lyrics", "style"}.
  * generate_music — happy path returns a path; MusicUnavailable is raised
    on missing weights or a ComfyUI failure, never a silent None.

Run: venv/Scripts/python.exe tests/test_music_generation.py
"""
import json
import os
import sys
import random
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import config as C
import music as M
M.MUSIC_ENGINE = "music3"   # these checks cover the dormant Music3 path; YuE2 is the default
import models

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


_TMP = tempfile.mkdtemp(prefix="music_gen_test_")
M.OUTPUT_DIR = _TMP


def _fake_audio(tag="song"):
    p = os.path.join(_TMP, f"{tag}.wav")
    with open(p, "wb") as fh:
        fh.write(b"RIFF" + b"\x00" * 4 + b"WAVEfmt " + b"\x00" * 4096)
    return p


def make_ctx():
    return models.Context(models=None, transcription_cache={}, cache_file=None,
                          asr_lock=threading.Lock(), tts_lock=threading.Lock())


print("=" * 70)
print("1. Config wiring matches what the graph reads")
print("=" * 70)

check("WORKFLOW_MUSIC_PATH is set and the file exists and is valid JSON",
      os.path.exists(C.WORKFLOW_MUSIC_PATH)
      and isinstance(json.load(open(C.WORKFLOW_MUSIC_PATH, encoding="utf-8")), dict))
check("MUSIC_JOB_TIMEOUT is a positive ceiling", C.MUSIC_JOB_TIMEOUT > 0)
check("MUSIC_DEFAULT_SECONDS is positive", C.MUSIC_DEFAULT_SECONDS > 0)
check("MUSIC_MAX_SECONDS is at least the default",
      C.MUSIC_MAX_SECONDS >= C.MUSIC_DEFAULT_SECONDS)

print()
print("=" * 70)
print("2. Weight presence")
print("=" * 70)

_real_models_dir = M._models_dir
_wdir = tempfile.mkdtemp(prefix="music_weights_test_")
M._models_dir = lambda: _wdir
try:
    miss = M.missing_weights()
    check("all three required files are reported missing on a bare dir",
          len(miss) == 3, miss)
    check("the missing paths name the right subfolders", all(
          any(sub in m for m in miss)
          for sub in ("diffusion_models", "text_encoders", "vae")), miss)

    # Now create them all and confirm missing_weights() clears.
    for sub, names in M.REQUIRED_FILES.items():
        d = os.path.join(_wdir, "models", sub)
        os.makedirs(d, exist_ok=True)
        for n in names:
            open(os.path.join(d, n), "wb").close()
    check("nothing is missing once all three files exist",
          M.missing_weights() == [], M.missing_weights())
finally:
    M._models_dir = _real_models_dir

print()
print("=" * 70)
print("3. engine_available refuses honestly")
print("=" * 70)

_real_missing = M.missing_weights
M.missing_weights = lambda preset=None: ["models/diffusion_models/minimax_music3_dit_fp32.safetensors"]
try:
    ok, why = M.engine_available(None)
    check("engine_available is False when weights are missing", not ok)
    check("...and names a missing file", "minimax_music3" in why, why)
finally:
    M.missing_weights = _real_missing

M.missing_weights = lambda preset=None: []
M._server_has_music3_nodes = lambda: False
try:
    ok, why = M.engine_available(None)
    check("engine_available is False when ComfyUI lacks the Music3 node", not ok)
    check("...and points at the docs for what's uncertain",
          "docs/music_generation.md" in why, why)
finally:
    M.missing_weights = _real_missing

M.missing_weights = lambda preset=None: []
M._server_has_music3_nodes = lambda: True
ok, why = M.engine_available(None)
check("engine_available is True once weights and the node are both present",
      ok and why == "", (ok, why))

print()
print("=" * 70)
print("4. The built graph wires lyrics/style/duration/seed into the right nodes")
print("=" * 70)

wf = M.build_workflow("[Verse]\nhello world", "upbeat pop, 120 BPM",
                      duration_s=90, seed=42)
check("the text-encode node carries lyrics, in canonical tag form",
      wf[M.N_TEXT]["inputs"]["lyrics"] == "[verse]\nhello world",
      wf[M.N_TEXT]["inputs"].get("lyrics"))
check("...and caption (the style description)",
      wf[M.N_TEXT]["inputs"]["caption"] == "upbeat pop, 120 BPM")
check("...and the requested max_duration in seconds",
      wf[M.N_TEXT]["inputs"]["max_duration"] == 90.0,
      wf[M.N_TEXT]["inputs"].get("max_duration"))
check("the seed reaches the text-encode node",
      wf[M.N_TEXT]["inputs"]["seed"] == 42)
check("the seed reaches the sampler", wf[M.N_SAMPLER]["inputs"]["seed"] == 42)
# 2026-09-20: when steps is not given, build_workflow now defaults to the
# Turbo LoRA path (measured 25% faster on a 60s song), not the plain
# MUSIC_STEPS/MUSIC_CFG config values -- those only apply when a caller
# explicitly asks for a step count (the GUI's Quality slider).
check("steps default to the turbo LoRA's step count when not given",
      wf[M.N_SAMPLER]["inputs"]["steps"] == C.MUSIC_STEPS_TURBO)
check("cfg defaults to the turbo LoRA's cfg when not given",
      wf[M.N_SAMPLER]["inputs"]["cfg"] == C.MUSIC_CFG_TURBO)
check("the turbo LoRA node is attached in the default (no-steps) path",
      any(n.get("class_type") == "LoraLoaderModelOnly" for n in wf.values()))

wf2 = M.build_workflow("x", "y", duration_s=30, seed=1, steps=50)
check("an explicit steps override wins over the turbo default",
      wf2[M.N_SAMPLER]["inputs"]["steps"] == 50)
check("an explicit steps override uses the plain MUSIC_CFG, not turbo cfg",
      wf2[M.N_SAMPLER]["inputs"]["cfg"] == C.MUSIC_CFG)
check("an explicit steps override skips the turbo LoRA node",
      not any(n.get("class_type") == "LoraLoaderModelOnly" for n in wf2.values()))

# Every node id the top-level constants name must exist in the loaded graph —
# a rename in workflow_music3.json that forgot to update music.py would
# silently point at a KeyError instead of a wrong value.
for nid in (M.N_UNET, M.N_CLIP, M.N_VAE, M.N_TEXT, M.N_SAMPLER, M.N_SAVE):
    check(f"node id {nid!r} exists in workflow_music3.json", nid in wf, list(wf))

print()
print("=" * 70)
print("4b. The lyric input contract is enforced in code, not hoped for")
print("=" * 70)
print("""
MiniMax Music 3 DISCARDS text that shares a line with a leading section tag:
"[verse] Morning light" sings nothing at all. A songwriter model that emits
that shape silently loses a line of the song, and neither the graph, the
render nor the delivered file gives any sign a line went missing -- so the
shape is repaired in code rather than left to the LLM's good behaviour.
""")

check("a tag sharing its line with words is split onto two lines",
      M.normalize_lyrics("[verse] Morning light") == "[verse]\nMorning light",
      repr(M.normalize_lyrics("[verse] Morning light")))
check("...and the words themselves survive the split",
      "Morning light" in M.normalize_lyrics("[verse] Morning light"))
check("known tags are canonicalised to the documented lowercase form",
      M.normalize_lyrics("[Verse]\nhi") == "[verse]\nhi",
      repr(M.normalize_lyrics("[Verse]\nhi")))
check("a numbered tag keeps its number",
      M.normalize_lyrics("[Verse 2]\nhi") == "[verse 2]\nhi",
      repr(M.normalize_lyrics("[Verse 2]\nhi")))
check("a hyphenated tag is recognised",
      M.normalize_lyrics("[Pre-Chorus]\nhi") == "[pre-chorus]\nhi",
      repr(M.normalize_lyrics("[Pre-Chorus]\nhi")))
# "[Guitar Solo x2]" is not a tag the model knows -- it would be SUNG. It is
# a direction about the music and is dropped (see rule 3 below); an unknown
# bracket token that is not about the music stays as words.
check("an unknown bracket token naming a sound is dropped, not sung",
      M.normalize_lyrics("[Guitar Solo x2]\nhi") == "hi",
      repr(M.normalize_lyrics("[Guitar Solo x2]\nhi")))
check("an unknown bracket token of ordinary words is left alone, not mangled",
      "[Hey you]" in M.normalize_lyrics("[Hey you]\nhi"),
      repr(M.normalize_lyrics("[Hey you]\nhi")))
# Only SECTION tags force a line break. A lyric is allowed brackets of its
# own, and splitting on those would shred one sung line into three.
check("a bracket inside a lyric line does NOT split the line",
      M.normalize_lyrics("[verse]\nI feel [so] alive") == "[verse]\nI feel [so] alive",
      repr(M.normalize_lyrics("[verse]\nI feel [so] alive")))
check("...even when the bracket sits at the start of the line",
      M.normalize_lyrics("[oh] what a night") == "[oh] what a night",
      repr(M.normalize_lyrics("[oh] what a night")))
check("a tag written with an underscore is still recognised",
      M.normalize_lyrics("[pre_chorus] hold on") == "[pre-chorus]\nhold on",
      repr(M.normalize_lyrics("[pre_chorus] hold on")))
check("normalising twice changes nothing the second time (idempotent)",
      M.normalize_lyrics(M.normalize_lyrics("[Verse] a\n\n\n[Chorus] b")) ==
      M.normalize_lyrics("[Verse] a\n\n\n[Chorus] b"))
# 3. Everything in `lyrics` is SUNG (live 2026-09-18 01:33: «(Low bass drone)»
# under [intro] came out as sung words, «часть слов типа хэви бас пропелась»).
_dir = M.normalize_lyrics("[intro]\n(Low bass drone)\n(Heavy beat kicks in)\n\n[verse]\nДобрый вечер!")
check("a whole-line sound direction under [intro] is dropped, not sung",
      "bass" not in _dir.lower() and "beat" not in _dir.lower() and "Добрый вечер!" in _dir, repr(_dir))
check("*asterisk* and [bracket] directions go the same way",
      M.normalize_lyrics("[verse]\n*тяжёлый гитарный рифф*\n[Guitar riff]\nслова") == "[verse]\nслова",
      repr(M.normalize_lyrics("[verse]\n*тяжёлый гитарный рифф*\n[Guitar riff]\nслова")))
check("a performance note inside a line is cut, the words stay",
      M.normalize_lyrics("[chorus]\nЯ люблю кредиты! (хором)\nещё раз (x2)") == "[chorus]\nЯ люблю кредиты!\nещё раз",
      repr(M.normalize_lyrics("[chorus]\nЯ люблю кредиты! (хором)\nещё раз (x2)")))
check("a backing-vocal line of ordinary words is kept (it is meant to be sung)",
      "(ooh, yeah)" in M.normalize_lyrics("[chorus]\n(ooh, yeah)\nwords"))
check("a Russian section name becomes the English tag",
      M.normalize_lyrics("Припев:\nслова\n[Куплет 2]\nещё") == "[chorus]\nслова\n[verse 2]\nещё",
      repr(M.normalize_lyrics("Припев:\nслова\n[Куплет 2]\nещё")))
check("the section tags themselves are never mistaken for directions",
      M.normalize_lyrics("[verse]\na\n[Chorus]\nb\n[solo]\n[instrumental]\n[outro]\nc")
      == "[verse]\na\n[chorus]\nb\n[solo]\n[instrumental]\n[outro]\nc")
_src = open(M.__file__, encoding="utf-8").read()
check("both songwriter briefs forbid stage directions in the lyrics",
      _src.count("stage directions") >= 2 and "(хором)" in _src)
check("runs of blank lines collapse to one",
      "\n\n\n" not in M.normalize_lyrics("[verse]\na\n\n\n\n[chorus]\nb"))
check("empty input does not explode", M.normalize_lyrics("") == "")
check("None does not explode", M.normalize_lyrics(None) == "")

# The graph is the boundary that matters: hand-written lyrics from the GUI
# tab or a script never went through the songwriter, so normalising only the
# LLM's output would leave exactly those callers broken.
wf_norm = M.build_workflow("[Verse] shared line", "style", duration_s=10, seed=1)
check("build_workflow normalises lyrics before they reach the graph",
      wf_norm[M.N_TEXT]["inputs"]["lyrics"] == "[verse]\nshared line",
      repr(wf_norm[M.N_TEXT]["inputs"]["lyrics"]))

# The weight variant is named in ONE place. It used to be written both in
# music.REQUIRED_FILES and in workflow_music3.json's loader nodes, so changing
# one left a graph pointing at a file the readiness check had already declared
# present -- engine_available() says yes, then ComfyUI rejects the graph.
_wf_v = M.build_workflow("[verse]\nx", "y", duration_s=10, seed=1)
check("the graph loads the DiT that REQUIRED_FILES checks for",
      _wf_v[M.N_UNET]["inputs"]["unet_name"] == M.REQUIRED_FILES["diffusion_models"][0],
      (_wf_v[M.N_UNET]["inputs"]["unet_name"], M.REQUIRED_FILES["diffusion_models"]))
check("...the text encoder too",
      _wf_v[M.N_CLIP]["inputs"]["clip_name"] == M.REQUIRED_FILES["text_encoders"][0],
      (_wf_v[M.N_CLIP]["inputs"]["clip_name"], M.REQUIRED_FILES["text_encoders"]))
check("...and the VAE",
      _wf_v[M.N_VAE]["inputs"]["vae_name"] == M.REQUIRED_FILES["vae"][0],
      (_wf_v[M.N_VAE]["inputs"]["vae_name"], M.REQUIRED_FILES["vae"]))

check("a caption with all three headings reads as structured",
      M.caption_is_structured("Global Metadata: x Vocal Details: y Arrangement: z"))
check("a caption missing a heading does not",
      not M.caption_is_structured("Global Metadata: x Arrangement: z"))
check("an empty caption does not", not M.caption_is_structured(""))

print()
print("=" * 70)
print("4bb. Three weight presets, fastest by default")
print("=" * 70)
print("""
Measured on one RTX 3090, 30s song: fast 11.14GB/40s, quality 18.75GB/59s,
max 28.52GB/240s. "max" is disproportionately slow because its 18.47GB encoder
does not fit in 24GB and streams -- the trade is the user's to make, so all
three are selectable and the fastest is the default.
""")

check("three presets exist", set(M.WEIGHT_PRESETS) == {"fast", "quality", "max"},
      list(M.WEIGHT_PRESETS))
check("the FASTEST one is the default", M.DEFAULT_PRESET == "fast", M.DEFAULT_PRESET)
check("every preset names a dit, clip and vae",
      all(all(k in s for k in ("dit", "clip", "vae")) for s in M.WEIGHT_PRESETS.values()))
check("every preset carries its measured size and time",
      all(s.get("gb", 0) > 0 and s.get("secs", 0) > 0 for s in M.WEIGHT_PRESETS.values()))
check("the presets are ordered fastest-first (the picker renders this order)",
      [s["secs"] for s in M.WEIGHT_PRESETS.values()] ==
      sorted(s["secs"] for s in M.WEIGHT_PRESETS.values()),
      [s["secs"] for s in M.WEIGHT_PRESETS.values()])

# Distinct files, or the presets would be three names for one thing.
_sets = [tuple(M.preset_files(p)) for p in M.WEIGHT_PRESETS]
check("the presets resolve to DIFFERENT file sets", len(set(_sets)) == 3, _sets)

# A stale/garbage preset comes from persisted per-chat state and must degrade,
# never raise -- a renamed preset must not take song generation down.
check("an unknown preset falls back to the default rather than raising",
      M.preset_files("no_such_preset") == M.preset_files(M.DEFAULT_PRESET))
check("None falls back too", M.preset_files(None) == M.preset_files(M.DEFAULT_PRESET))
for _bad in (123, [], {}):
    try:
        check(f"a non-string preset ({type(_bad).__name__}) does not crash",
              M.preset_files(_bad) == M.preset_files(M.DEFAULT_PRESET))
    except Exception as exc:
        check(f"a non-string preset ({type(_bad).__name__}) does not crash", False, exc)

# The graph must load the preset that was ASKED for, not the default.
for _p in M.WEIGHT_PRESETS:
    _wf = M.build_workflow("[verse]\nx", "y", duration_s=10, seed=1, preset=_p)
    _d, _c, _v = M.preset_files(_p)
    check(f"preset {_p!r} stamps its own dit into the graph",
          _wf[M.N_UNET]["inputs"]["unet_name"] == _d, _wf[M.N_UNET]["inputs"]["unet_name"])
    check(f"preset {_p!r} stamps its own text encoder",
          _wf[M.N_CLIP]["inputs"]["clip_name"] == _c, _wf[M.N_CLIP]["inputs"]["clip_name"])
    check(f"preset {_p!r} stamps its own vae",
          _wf[M.N_VAE]["inputs"]["vae_name"] == _v, _wf[M.N_VAE]["inputs"]["vae_name"])

# Readiness is PER PRESET: "max" can be a 28GB download away while "fast" is
# ready, and reporting the default's readiness would promise an unrenderable song.
# Section 3 left missing_weights stubbed to [] without restoring it, so the real
# one is put back explicitly here rather than silently measuring the stub.
M.missing_weights = _real_missing
_rmd = M._models_dir
_pdir = tempfile.mkdtemp(prefix="music_preset_test_")
M._models_dir = lambda: _pdir
try:
    for sub, n in zip(("diffusion_models", "text_encoders", "vae"),
                      M.preset_files("fast")):
        d = os.path.join(_pdir, "models", sub)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, n), "wb").close()
    check("'fast' reads ready when only its own files are present",
          M.missing_weights("fast") == [], M.missing_weights("fast"))
    check("...while 'max' correctly reports its own files missing",
          len(M.missing_weights("max")) > 0, M.missing_weights("max"))
    _ok_max, _why_max = M.engine_available(None, "max")
    check("engine_available('max') refuses when max's weights are absent",
          not _ok_max, (_ok_max, _why_max))
    M._server_has_music3_nodes = lambda: True
    check("engine_available('fast') accepts at the same moment",
          M.engine_available(None, "fast")[0] is True)
finally:
    M._models_dir = _rmd

print()
print("=" * 70)
print("4c. The song ends instead of being severed")
print("=" * 70)
print("""
Music3 renders up to max_duration and stops there whether or not the song is
finished. Measured across three real 30s renders: every one ended at 29.99s
with a last-250ms peak of 0.57-0.999 -- a hard cut at near-full volume, where
a piece that actually resolved would taper toward zero. Prompting for an
[outro] helps the model AIM at an ending; the taper here guarantees one.
""")

try:
    import numpy as _np, soundfile as _sf
    _HAVE_AUDIO = True
except Exception:
    _HAVE_AUDIO = False

if not _HAVE_AUDIO:
    print("SKIP  (numpy/soundfile unavailable)")
else:
    _fdir = tempfile.mkdtemp(prefix="music_fade_")
    _sr = 8000
    def _tone(seconds=6.0):
        p = os.path.join(_fdir, f"t{random.randint(1,10**9)}.wav")
        n = int(_sr * seconds)
        d = _np.ones((n, 2), dtype="float32")   # constant full-scale: any
        _sf.write(p, d, _sr)                    # taper is unmistakable
        return p

    p = _tone()
    before = _sf.read(p)[0]
    check("a flat full-scale clip starts out un-tapered",
          abs(float(_np.abs(before[-10:]).max()) - 1.0) < 1e-3)
    check("apply_fade reports it changed the file", M.apply_fade(p) is True)
    after, _ = _sf.read(p)
    check("...the END is now near silence",
          float(_np.abs(after[-10:]).max()) < 0.05,
          float(_np.abs(after[-10:]).max()))
    check("...the START is tapered too (no click)",
          float(_np.abs(after[:5]).max()) < 0.5, float(_np.abs(after[:5]).max()))
    check("...the MIDDLE is untouched",
          _np.allclose(before[len(before)//2], after[len(after)//2]),
          (before[len(before)//2], after[len(after)//2]))
    check("...and the song is not shortened", len(after) == len(before),
          (len(before), len(after)))

    # A 2s tail must not eat a 1s clip.
    p2 = _tone(seconds=1.0)
    M.apply_fade(p2)
    d2, _ = _sf.read(p2)
    check("a very short clip keeps most of itself un-faded (>= 1/3 intact)",
          float(_np.abs(d2[len(d2)//2 - 2:len(d2)//2 + 2]).max()) > 0.5,
          float(_np.abs(d2[len(d2)//2]).max()))

    # Disabled by config -> no rewrite at all.
    p3 = _tone()
    b3 = _sf.read(p3)[0]
    check("both fades at 0 disables it entirely",
          M.apply_fade(p3, fade_out_s=0, fade_in_s=0) is False)
    check("...and the file is byte-identical", _np.allclose(b3, _sf.read(p3)[0]))

    # Losing a 40s render over a post-processing detail would be far worse
    # than an abrupt ending, so a broken file must degrade, never raise.
    bad = os.path.join(_fdir, "not_audio.flac")
    with open(bad, "wb") as fh:
        fh.write(b"this is not a flac file")
    try:
        check("an unreadable file degrades to False instead of raising",
              M.apply_fade(bad) is False)
    except Exception as exc:
        check("an unreadable file degrades to False instead of raising", False, exc)
    check("a missing file does the same", M.apply_fade(
        os.path.join(_fdir, "nope.flac")) is False)

# The songwriter must AIM at an ending, not just rely on the taper.
# llm is imported locally: the module-level alias used by the later sections
# is not bound until section 5.
import llm as _L_outro
_orig_outro = _L_outro.call_llm_simple
_calls_o = []
_L_outro.call_llm_simple = lambda ctx, sys_p, user_p, **kw: (
    _calls_o.append(sys_p),
    json.dumps({"lyrics": "[verse]\nx\n[outro]\ny",
                "style": "Global Metadata: a Vocal Details: b Arrangement: c"}))[-1]
try:
    M.build_structured_caption(make_ctx(), "topic", "en", duration_s=60)
finally:
    _L_outro.call_llm_simple = _orig_outro
check("the prompt demands an [outro] that resolves the song",
      "[outro]" in _calls_o[0] and "resolves" in _calls_o[0], _calls_o[0][:300])
check("...and warns that overshooting the slot gets the song cut off",
      "cut off" in _calls_o[0].lower(), _calls_o[0][:300])
# The ask is a LINE COUNT now, not "roughly N seconds" -- the writer cannot
# know the renderer's singing rate, and guessed short every time it was asked
# to. See tests/test_music_length_budget.py for the measurements.
check("...and asks for a line count rather than a duration",
      "sung lines" in _calls_o[0] and str(M.line_budget(60)[0]) in _calls_o[0],
      _calls_o[0][-300:])

print()
print("=" * 70)
print("5. build_structured_caption calls the LLM once and threads lang")
print("=" * 70)

import llm as L

_calls = []
def _fake_llm_en(ctx, sys_p, user_p, **kw):
    _calls.append((sys_p, user_p, kw))
    return json.dumps({
        "lyrics": "[Verse]\nBuilding something new\n[Chorus]\nWatch it come alive",
        "style": "Global Metadata: upbeat pop. Vocal Details: female lead. "
                 "Arrangement: guitar and drums.",
    })

_real_call = L.call_llm_simple
L.call_llm_simple = _fake_llm_en
try:
    out = M.build_structured_caption(make_ctx(), "a song about building things", "en")
    check("exactly one LLM call was made", len(_calls) == 1, len(_calls))
    check("the topic reached the user prompt",
          "building things" in _calls[0][1], _calls[0][1])
    check("English is named in the system prompt for an 'en' request",
          "English" in _calls[0][0], _calls[0][0][:200])
    check("lyrics carry a verse tag, canonicalised to lowercase",
          "[verse]" in out["lyrics"], out)
    check("lyrics carry a chorus tag, canonicalised to lowercase",
          "[chorus]" in out["lyrics"], out)
    check("style covers the three required sections", all(
          s in out["style"] for s in ("Global Metadata", "Vocal Details", "Arrangement")),
          out["style"])
finally:
    L.call_llm_simple = _real_call

_calls.clear()
def _fake_llm_ru(ctx, sys_p, user_p, **kw):
    _calls.append((sys_p, user_p, kw))
    return json.dumps({
        "lyrics": "[Verse]\nСтроим новое\n[Chorus]\nСмотри, оно живёт",
        "style": "Global Metadata: upbeat pop. Vocal Details: female lead. "
                 "Arrangement: guitar and drums.",
    })

L.call_llm_simple = _fake_llm_ru
try:
    out_ru = M.build_structured_caption(make_ctx(), "песня о творчестве", "ru")
    check("Russian is named in the system prompt for a 'ru' request",
          "Russian" in _calls[0][0], _calls[0][0][:200])
    check("the tags stay in English even for a Russian song",
          "[verse]" in out_ru["lyrics"] and "[chorus]" in out_ru["lyrics"], out_ru)
    check("the lyric words themselves are Russian, not English",
          "Строим" in out_ru["lyrics"], out_ru["lyrics"])
finally:
    L.call_llm_simple = _real_call

print()
print("=" * 70)
print("6. build_structured_caption refuses on an unparseable / empty reply")
print("=" * 70)

L.call_llm_simple = lambda *a, **k: "not json at all, sorry"
try:
    try:
        M.build_structured_caption(make_ctx(), "topic", "en")
        check("garbage LLM output raises SongwritingFailed", False)
    except M.SongwritingFailed:
        check("garbage LLM output raises SongwritingFailed", True)
finally:
    L.call_llm_simple = _real_call

L.call_llm_simple = lambda *a, **k: json.dumps({"lyrics": "", "style": ""})
try:
    try:
        M.build_structured_caption(make_ctx(), "topic", "en")
        check("an empty lyrics/style pair raises SongwritingFailed", False)
    except M.SongwritingFailed:
        check("an empty lyrics/style pair raises SongwritingFailed", True)
finally:
    L.call_llm_simple = _real_call

# A writer failure must NOT masquerade as "the engine isn't installed": the
# live bug reported "song generation isn't set up yet" while the weights, the
# ComfyUI nodes and the server were all present and healthy, sending the user
# off to check an installation that was fine.
check("SongwritingFailed is NOT a MusicUnavailable (they mean different things)",
      not issubclass(M.SongwritingFailed, M.MusicUnavailable))

print()
print("=" * 70)
print("6b. The songwriter ESCALATES instead of giving up after one squeezed try")
print("=" * 70)
print("""
Live bug: one 2000-token attempt let the reasoning model spend its entire
budget thinking ("10194 chars of reasoning with no final answer,
finish_reason=length") and return nothing, so a healthy engine reported
itself as unconfigured. Same failure and same remedy as slides.py's deck
planner: a ladder whose later rungs drop force_think and vary the ask.
""")

_calls = []
def _fake_ladder(ctx, sys_p, user_p, **kw):
    _calls.append(kw)
    # Fail the first two rungs the way the real model did, succeed on the third.
    if len(_calls) < 3:
        return ""
    return json.dumps({"lyrics": "[Verse]\nline\n[Chorus]\nhook",
                       "style": "Global Metadata / Vocal Details / Arrangement"})

L.call_llm_simple = _fake_ladder
try:
    out = M.build_structured_caption(make_ctx(), "topic", "en")
    check("an early empty reply does not end the attempt -- it escalates",
          len(_calls) == 3, len(_calls))
    check("...and the recovered song is returned normally",
          out["lyrics"].startswith("[verse]"), out)
    # 2026-09-25, user: no reasoning anywhere by default (it cost minutes per call).
    check("no rung makes the model think (force_think off)",
          not any(c.get("force_think") for c in _calls), _calls)
    check("a later rung stops it thinking, so the budget holds the ANSWER",
          any(c.get("force_think") is False for c in _calls[1:]), _calls)
    check("the first rung's budget is far above the 2000 that failed live",
          _calls[0].get("max_tokens", 0) >= 5000, _calls[0])
finally:
    L.call_llm_simple = _real_call

_calls.clear()
L.call_llm_simple = lambda *a, **k: _calls.append(k) or ""
try:
    try:
        M.build_structured_caption(make_ctx(), "topic", "en")
    except M.SongwritingFailed:
        pass
    check("every rung is tried before giving up", len(_calls) == 3, len(_calls))
finally:
    L.call_llm_simple = _real_call

print()
print("=" * 70)
print("6c. An unstructured caption escalates, but never fails the song")
print("=" * 70)
print("""
The three headings are what makes the caption a timeline the model can follow
section by section; a flat sentence still renders, just vaguer. So a caption
without them is worth another rung -- and is still SHIPPED when the ladder
runs out, because refusing here would report a healthy engine as broken,
which is the exact bug the SongwritingFailed split was introduced to end.
""")

_calls.clear()
def _flat_then_structured(ctx, sys_p, user_p, **kw):
    _calls.append(kw)
    if len(_calls) == 1:
        return json.dumps({"lyrics": "[verse]\nline", "style": "a nice pop song"})
    return json.dumps({"lyrics": "[verse]\nline",
                       "style": "Global Metadata: pop. Vocal Details: female. "
                                "Arrangement: guitar."})

L.call_llm_simple = _flat_then_structured
try:
    out = M.build_structured_caption(make_ctx(), "topic", "en")
    check("a flat caption on rung 1 spends another rung", len(_calls) == 2, len(_calls))
    check("...and the structured caption is what comes back",
          M.caption_is_structured(out["style"]), out["style"])
finally:
    L.call_llm_simple = _real_call

_calls.clear()
L.call_llm_simple = lambda *a, **k: _calls.append(k) or json.dumps(
    {"lyrics": "[verse]\nline", "style": "just a flat sentence"})
try:
    out = M.build_structured_caption(make_ctx(), "topic", "en")
    check("a caption that is flat on EVERY rung is still shipped, not failed",
          out["lyrics"] and out["style"], out)
    check("...after spending the whole ladder trying for structure",
          len(_calls) == 3, len(_calls))
finally:
    L.call_llm_simple = _real_call

print()
print("=" * 70)
print("7. Nothing is promised when the engine is not available")
print("=" * 70)

M.missing_weights = lambda preset=None: ["models/diffusion_models/minimax_music3_dit_fp32.safetensors"]
try:
    try:
        M.generate_music(make_ctx(), "[Verse]\nhi", "pop")
        check("generate_music refuses rather than pretending", False)
    except M.MusicUnavailable as exc:
        check("generate_music refuses rather than pretending", True)
        check("...and the reason mentions the missing weights",
              "music3" in str(exc).lower(), str(exc))
finally:
    M.missing_weights = _real_missing

print()
print("=" * 70)
print("8. A render that produces no file raises, and a real file is adopted")
print("=" * 70)

M.missing_weights = lambda preset=None: []
M._server_has_music3_nodes = lambda: True
import comfy_client as _cc
_real_submit = _cc._submit_and_poll
try:
    _cc._submit_and_poll = lambda *a, **k: None
    try:
        M.generate_music(make_ctx(), "[Verse]\nhi", "pop")
        check("no file -> MusicUnavailable", False)
    except M.MusicUnavailable:
        check("no file -> MusicUnavailable", True)

    made = _fake_audio("result")
    _cc._submit_and_poll = lambda *a, **k: made
    path = M.generate_music(make_ctx(), "[Verse]\nhi", "pop", duration_s=45, seed=7)
    check("a real file -> a path is returned", bool(path) and os.path.exists(path), path)
    check("the file was adopted into OUTPUT_DIR",
          os.path.dirname(os.path.abspath(path)) == os.path.abspath(_TMP), path)

    ctx = make_ctx()
    M.generate_music(ctx, "[Verse]\nhi", "pop")
    check("ctx.last_music_path is set so 'that song' can resolve later",
          bool(getattr(ctx, "last_music_path", None)))
finally:
    _cc._submit_and_poll = _real_submit
    M.missing_weights = _real_missing

print()
print("=" * 70)
print("9. Duration is clamped to the configured range")
print("=" * 70)

_cc._submit_and_poll = lambda *a, **k: _fake_audio("dur")
M.missing_weights = lambda preset=None: []
M._server_has_music3_nodes = lambda: True
try:
    seen = {}
    _orig_build = M.build_workflow
    def _spy(lyrics, style, *, duration_s, seed, steps=None, preset=None):
        seen["duration_s"] = duration_s
        return _orig_build(lyrics, style, duration_s=duration_s, seed=seed, steps=steps)
    M.build_workflow = _spy
    M.generate_music(make_ctx(), "[Verse]\nhi", "pop", duration_s=C.MUSIC_MAX_SECONDS + 999)
    check("an over-long request is clamped to MUSIC_MAX_SECONDS, then given the render headroom",
          seen["duration_s"] == M.render_ceiling(C.MUSIC_MAX_SECONDS), seen)
    check("the headroom: ~20% + 10 s over the ask, never past MUSIC_CEILING_SECONDS, never below the ask",
          M.render_ceiling(60) == 82 and M.render_ceiling(180) == min(C.MUSIC_CEILING_SECONDS, 226)
          and M.render_ceiling(1) == 11 and M.render_ceiling(C.MUSIC_CEILING_SECONDS) == C.MUSIC_CEILING_SECONDS)
    M.generate_music(make_ctx(), "[Verse]\nhi", "pop", duration_s=-5)
    check("a non-positive request does not crash and stays positive",
          seen["duration_s"] >= 1, seen)
finally:
    M.build_workflow = _orig_build
    _cc._submit_and_poll = _real_submit
    M.missing_weights = _real_missing

print()
print(f"{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
