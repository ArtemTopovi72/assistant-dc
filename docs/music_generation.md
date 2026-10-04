# Music generation (MiniMax Music 3)

`music.py` mirrors `video.py`'s shape: readiness checks against real weight
files and a real ComfyUI node, a `build_workflow()` that is pure graph
construction (testable with no GPU), and `generate_music()` that submits
through `comfy_client._submit_and_poll` and adopts the result into
`config.OUTPUT_DIR`.

## Model files

Downloaded into `%USERPROFILE%\Documents\ComfyUI\models\`:

- `diffusion_models/minimax_music3_dit_fp32.safetensors` — the DiT transformer,
  full fp32 (not fp16/int8 — the highest-quality variant per the task).
- `text_encoders/minimax_music3_text_encoder_bf16.safetensors` — full bf16
  (not the `_pruned` variant).
- `vae/minimax_music3_dav.safetensors` — the audio VAE.

`music.REQUIRED_FILES` checks exactly these three paths; `missing_weights()`
and `engine_available()` follow the same contract as their `video.py`
counterparts.

## CONFIRMED live, end-to-end

The graph below was verified against a real ComfyUI and produced real audio
— this is no longer a guess. Verified 2026-08-16 against a fresh clone of
`comfyanonymous/ComfyUI` at `main` HEAD (commit `37ac9ff`, just past the
`v0.32.0` tag — `comfy_extras/nodes_minimax_music.py` is where
`MiniMaxMusic3TextEncode`/`EmptyMiniMaxMusic3LatentAudio` land; **neither the
`v0.30.0` nor `v0.32.0` tagged releases have these nodes yet**, so a stock
install needs a newer-than-release checkout until the next tag ships it),
run standalone on port 8001 against the shared
`%USERPROFILE%\Documents\ComfyUI` model/base directory. Cross-checked
against the official `Comfy-Org/workflow_templates`
`templates/audio_minimax_music_3.json` example, which uses the identical
node shape.

| id | class_type | inputs that matter |
|----|------------|---------------------|
| 1 | `UNETLoader` | `unet_name="minimax_music3_dit_fp32.safetensors"`, `weight_dtype="default"` |
| 2 | `CLIPLoader` | `clip_name="minimax_music3_text_encoder_bf16.safetensors"`, `type="minimax"` (confirmed real enum value, not a guess) |
| 3 | `VAELoader` | `vae_name="minimax_music3_dav.safetensors"` |
| 4 | `MiniMaxMusic3TextEncode` | `clip`, `caption` (the style/production-brief text), `lyrics`, `seed`, `max_duration` (seconds) — outputs `CONDITIONING` **and** a clamped `seconds` float used to size the latent |
| 5 | `ConditioningZeroOut` | `conditioning` (node 4's output) — the negative conditioning; no separate params |
| 6 | `EmptyMiniMaxMusic3LatentAudio` | `seconds` (wired **from** node 4's second output, not set independently — the model itself decides the exact frame count for a given `max_duration`), `batch_size=1` |
| 7 | `KSampler` | standard; `positive`=node 4, `negative`=node 5, `latent_image`=node 6 |
| 8 | `VAEDecodeAudio` | `samples`, `vae` |
| 9 | `SaveAudio` | `audio`, `filename_prefix` — writes a `.flac` |

Two bugs were found and fixed while getting this working end-to-end:

1. **`music._server_has_music3_nodes()`** checked the wrong node class name
   (a guessed `MiniMaxMusic3TextToAudio` that never existed) against
   `GET /object_info/<name>` — and separately, that per-node endpoint turns
   out to return an empty `{}` body (still HTTP 200) for real nodes too, on
   this ComfyUI build. Fixed to check the real name
   (`MiniMaxMusic3TextEncode`) against membership in the **full**
   `/object_info` listing instead.
2. **`comfy_client.py`'s `_poll_history`** only ever looked for a
   `node_out["images"]` key when scanning a completed job's outputs.
   `SaveAudio`/`SaveAudioAdvanced` report their file under `node_out["audio"]`
   — same `{filename, subfolder}` shape, different key — so a genuinely
   successful audio render was read as "job completed with no output" and
   silently discarded. This is shared transport code used by every ComfyUI
   caller in the project (image.py, video.py, music.py), not a music-specific
   bug — fixed generically (`node_out.get("images") or node_out.get("audio")`)
   with a regression test in `tests/test_image_polling.py::test_saved_audio_success`.

Two live end-to-end generations (30s requested each) confirmed the fix:
an EN and RU song, both real 44.1kHz stereo FLAC output from
`music.generate_music()` through the actual ComfyUI graph.

Cfg/steps defaults (`MUSIC_CFG=3.0`, `MUSIC_STEPS=30` in `config.py`) are
untuned placeholders, not confirmed-optimal — Music3 is not CFG-distilled
the way H3 is (it does take real negative conditioning via
`ConditioningZeroOut`), so there is room to tune quality/speed further, but
that is a taste question, not a correctness one.

## Public contract

See `music.py` docstrings. In short:

```python
missing_weights() -> list[str]
engine_available(ctx=None) -> tuple[bool, str]
normalize_lyrics(text: str) -> str                            # enforces the tag contract
caption_is_structured(style: str) -> bool                     # the three headings
build_structured_caption(ctx, topic, lang, *, duration_s=None) -> dict  # {"lyrics", "style"}
generate_music(ctx, lyrics, style, *, duration_s=60, seed=None) -> str  # path
```

`build_structured_caption` calls the house LLM (`llm.call_llm_simple`,
the same primitive `tg_accounts._weather_correct_fn` uses) with a prompt
that asks for a JSON object `{"lyrics": ..., "style": ...}`. Lyrics are
written in the requested language (`lang` — `'en'` or `'ru'`); the section
tags stay in English in both cases, since that is MiniMax's required
lyric-format vocabulary, not something to localize. The reply is parsed with
`utils.safe_json_from_llm` (the project's one hardened LLM-JSON parser — see
`docs/llm-json-extraction` note in memory), falling back to a bare
`json.loads`.

## The prompt contract

Music3 takes two texts that do different jobs: `lyrics` carries the sung
words, `caption` (our `style`) carries **all** the musical control, applied
over time rather than as one global tag. Two rules from MiniMax's prompting
guide are input contracts rather than style advice, so both are enforced in
code (`normalize_lyrics`, called from `build_workflow` — the last boundary
before the graph — so hand-written lyrics from the GUI tab or a script get
the same guarantee as LLM-written ones):

1. **A section tag must sit alone on its line.** `[verse] Morning light` does
   not sing "Morning light" — the model discards text sharing a line with a
   leading tag. Nothing downstream can detect this: the graph, the render and
   the delivered file all look perfectly healthy with a line missing.
2. **The tag vocabulary is nine lowercase tags** — `[intro] [verse]
   [pre-chorus] [chorus] [post-chorus] [bridge] [instrumental] [solo]
   [outro]`. Known tags are canonicalised to lowercase (numbered variants like
   `[Verse 2]` keep their number); unrecognised bracket tokens are left alone.

The caption is written under exactly three headings, in order — **Global
Metadata**, **Vocal Details**, **Arrangement** — at roughly 250–450 words.
`caption_is_structured()` checks for them: a caption missing them spends
another rung of the ladder, but is still shipped if every rung comes back
flat, because an unstructured caption is a vaguer song and not a broken one.
Stating the singer's gender and timbre explicitly is the single biggest
guard against the model drifting into an unwanted instrumental.

`duration_s` only shapes how much song to write. The requested duration is an
**upper bound** — Music3 stops when the words run out, so short lyrics make a
short song regardless of what the render asks for.

### The ladder's budget grows, and that is deliberate

The three rungs escalate `max_tokens` (12000 → 16000 → 16000) rather than
shrinking it. On the house Gemma model there is no reasoning switch at all —
`llm.py` force-clears `no_think` and the `<think></think>` prefill, since
they are Qwen levers Gemma's template treats as literal junk — so the token
budget is the *only* lever these rungs actually pull there. The previous
ladder shrank it (6000 → 3500 → 3000) and thereby starved the exact failure
it existed to recover from: the model reasons ~4k tokens about a detailed
brief, hits the ceiling before writing a word of the answer, and each rung
gets *less* room than the one that just failed. Measured live against
`gemma-4-26b-a4b-qat`: the full brief needs ~12k to think **and** answer, and
lands a structured caption on the first attempt at that budget (~63s).
`force_think` still drops away on later rungs for the Qwen families, where it
is a real switch and a large budget simply goes unused.

## Sample songs

Two hand-written examples of exactly what `build_structured_caption` should
produce — one English, one Russian — are in `docs/sample_songs.md`, for
manual testing against a real ComfyUI once the Music3 node package and
weights are both confirmed in place.

## Installing YuE2 (the default song engine)

`setup` does not install YuE2; `venv\Scripts\python scripts\install_yue2.py` does.
It makes the venv with Python 3.12 from uv (installed by setup; `--python "py -3.12"`
to use another interpreter), installs CUDA torch first and then the package, and
downloads both models. `--cuda cu130` picks another torch build; a second run skips
what is done; `scripts/healthcheck.py` reports the songs part. Songs stay off until
these exist (`media/music.py: yue2_available()` checks them):

| What | Where |
|---|---|
| a venv with the official `yue2-infer` package ([multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE), Python 3.12) | `venv_yue2/` |
| `m-a-p/YuE2-3B` (`config.json` + `*.safetensors`) | `models_ext/YuE2-3B/` |
| `m-a-p/YuE2-Vae` | `models_ext/YuE2-Vae/` (a local folder: the hub cache needs symlinks, which Windows refuses without developer mode) |

`scripts/yue2_render.py` patches `GraphAR.__init__(attention_backend=)` and
`CachedNAR.__init__(model, chunk, attention, query_chunk_size)`; these match
yue2-infer 0.1.6 (commit `1dc1c50`). Install torch 2.10.0 from the CUDA index
before the package, or the resolver takes PyPI's CPU wheel on Windows.

## Lyrics: writing and polishing (`media/lyrics_craft.py`)

Used by ✍️ Write lyrics and ✨ Improve lyrics in the Telegram Music menu, and by
🎵 Songs before singing (a wish becomes a written lyric; ready words are polished on
request).

- **Write**: a plan first (the hook line, what each verse shows, concrete images, the
  metre and rhyme scheme), then `LYRICS_DRAFTS` drafts (3). Each is scored by the rule
  check and the model critic; when the two best are close, the model compares them head
  to head, asked both ways round (a judge leans to one position).
- **Rule check** (`analyse`): a rhyme on the stressed vowel (stress from the voice's
  accentor, RUAccent with silero-stress for homographs, read for the whole line; ё is
  always stressed), lazy pairs (same word, one word inside the other, worn pairs, two
  verbs in one form), line length within a pair, rhythm, lines over 13 syllables, a
  foreign word in a Russian lyric, song form (verse shape, chorus drift, a short hook
  line, the same end word twice). A repeated chorus is judged once. Rich and
  two-syllable rhymes earn a little back.
- **Polish** (`polish`): up to 3 rounds. A non-rhyming pair first gets versions of its
  weaker line (the first that passes the check wins, in every repeat of a chorus), then
  the model rewrites what is still named; the best-scoring version is kept, never a
  worse rewrite. Creative calls use `min_p` sampling (`LYRICS_MIN_P`, 0.05).
- The user sees two messages: what was changed and what is left (in their language),
  then the lyric alone to copy.

## YuE2: how long a lyric sings (measured 2026-10-04, `bench/yue2_seconds_per_line.py`)
One render per size, same style/seed (Russian pop, 100 BPM, female, piano); lines counted with the `[outro]`:

| sung lines | audio | s/line | wall |
|---|---|---|---|
| 9 | 60.0 s | 6.7 | 37 s |
| 17 | 109.2 s | 6.4 | 50 s |
| 25 | 147.6 s | 5.9 | 64 s |
| 37 | 200.6 s | 5.4 | 85 s |

Song ≈ **15 s of intro/outro + 5.0–5.5 s per sung line**. `music.line_budget` was Music3's (3.8 s/line, 180 s → 47 lines ≈ 250 s of song); it is now `SONG_LEAD_SECONDS=15`, `SECONDS_PER_SUNG_LINE=5.2`, rates 4.8–5.6, `MIN_SUNG_LINES=4` (30 s ask → 4 lines ≈ 36–40 s, the shortest YuE2 goes). The model card says the model decides the length; the duration button only sizes the lyric.
