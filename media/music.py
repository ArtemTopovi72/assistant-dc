"""Music generation through MiniMax Music 3 on a local ComfyUI.

Structurally a sibling of video.py: one DiT checkpoint takes a lyrics string
(with [Verse]/[Chorus]/[Bridge]... section tags, MiniMax's own lyric format)
and a style description (genre/BPM/key/vocal/arrangement) and produces a
mixed song as a single audio latent, decoded through a dedicated VAE. There
is no separate vocal/instrumental pass to orchestrate and no reference
images/video/audio to route between modes — Music3 is a single-mode
text-to-audio graph, which is why this module is much smaller than video.py.

Node class names in workflow_music3.json were confirmed live against a
ComfyUI built from comfyanonymous/ComfyUI main (the node landed after the
v0.32.0 tag, in comfy_extras/nodes_minimax_music.py — v0.30.0/v0.32.0 do not
have it, so a stock install needs a newer-than-release checkout until the
next tagged version ships it). The graph is exactly the shape of the
official Comfy-Org workflow_templates/templates/audio_minimax_music_3.json
example: UNETLoader -> CLIPLoader(type="minimax") -> VAELoader feed
MiniMaxMusic3TextEncode(clip, caption, lyrics, seed, max_duration), whose
CONDITIONING output goes to KSampler as positive and through
ConditioningZeroOut as negative, latent comes from
EmptyMiniMaxMusic3LatentAudio(seconds=<the encode node's own clamped
"seconds" output>), decoded with VAEDecodeAudio and written with SaveAudio.
See docs/music_generation.md for the object_info census that confirmed this.
"""
from __future__ import annotations

import json
import logging
import os
import config as _cfg_env   # env_int/env_float: a bad .env value falls back, never crashes the import
import random
import threading
import time
from pathlib import Path
import re
from typing import Callable, Optional

import config as _config
import comfy_client
from config import (
    COMFY_URL, OUTPUT_DIR, WORKFLOW_MUSIC_PATH,
    MUSIC_JOB_TIMEOUT, MUSIC_DEFAULT_SECONDS, MUSIC_MAX_SECONDS, MUSIC_CEILING_SECONDS,
    MUSIC_STEPS, MUSIC_CFG,
    MUSIC_TURBO_LORA, MUSIC_TURBO_STRENGTH, MUSIC_STEPS_TURBO, MUSIC_CFG_TURBO,
)

logger = logging.getLogger("assistant.music")

# Node ids inside workflow_music3.json (see that file), matching the real
# node graph 1:1: 1=UNETLoader, 2=CLIPLoader, 3=VAELoader,
# 4=MiniMaxMusic3TextEncode (caption+lyrics live here), 5=ConditioningZeroOut
# (negative, no params to set), 6=EmptyMiniMaxMusic3LatentAudio (its
# "seconds" input is wired FROM node 4's clamped seconds output, not set
# here), 7=KSampler, 8=VAEDecodeAudio, 9=SaveAudio.
N_UNET, N_CLIP, N_VAE, N_TEXT = "1", "2", "3", "4"
N_SAMPLER, N_SAVE = "7", "9"

# Why the last generation produced nothing, for an honest message to the user.
# Mirrors video._VIDEO_FAILURE / image._GENERATE_FAILURE.
_MUSIC_FAILURE: dict = {"reason": "server_error", "detail": ""}


# --------------------------------------------------------------------------- #
# The prompt contract (MiniMax Music 3)
# --------------------------------------------------------------------------- #
# Music3 takes exactly two texts and treats them very differently:
#
#   lyrics  — the sung WORDS, laid out with bracketed section tags.
#   caption — the MUSIC description. All musical control lives here, and the
#             model follows it OVER TIME rather than as one global tag, which
#             is why it is written as a per-section timeline.
#
# Two rules out of MiniMax's prompting guide are input CONTRACTS rather than
# style advice, so they are enforced in code instead of being left to whatever
# the songwriter model felt like emitting that turn:
#
#   1. A section tag must sit ALONE on its line. "[verse] Morning light" does
#      not sing "Morning light" — the model discards text that shares a line
#      with a leading tag. A songwriter that emits that shape silently loses a
#      line of the song, with nothing anywhere to say a line went missing.
#   2. The tag vocabulary is the nine tags below. The official prompt library
#      writes them lowercase everywhere, so they are canonicalised to that
#      form rather than trusting the case the LLM happened to produce.
SECTION_TAGS = ("intro", "verse", "pre-chorus", "chorus", "post-chorus",
                "bridge", "instrumental", "solo", "outro")

# Split a line at a KNOWN section tag, keeping the tag (capturing group).
# Deliberately built from SECTION_TAGS rather than matching any "[...]": a
# lyric is allowed to contain brackets of its own ("I feel [so] alive"), and
# splitting on those would shred a real line into three. Hyphenated tags are
# matched with any of hyphen/underscore/space, and a trailing number
# ("[Verse 2]") is part of the tag.
_TAG_ALT = "|".join(re.escape(t).replace(r"\-", r"[\-_ ]?")
                    for t in sorted(SECTION_TAGS, key=len, reverse=True))
_TAG_SPLIT_RE = re.compile(rf"(\[\s*(?:{_TAG_ALT})\s*\d*\s*\])", re.IGNORECASE)
# The inside of a tag: a name, plus an optional trailing number ("Verse 2").
_TAG_BODY_RE = re.compile(r"^\s*([A-Za-z][A-Za-z _\-]*?)\s*(\d*)\s*$")

# The caption's three required headings, in the order Music3 expects them.
CAPTION_HEADINGS = ("Global Metadata", "Vocal Details", "Arrangement")


# 3. Everything in `lyrics` is SUNG. Music3 has no notion of a stage
#    direction: "(Low bass drone)" on its own line under [intro] is sung as
#    "low bass drone", a Russian "[Припев]" is an unknown tag and is sung as
#    a word, "(хором)" after a line is sung too (live 2026-09-18 01:33: «часть
#    слов типа хэви бас пропелась»). The songwriter is told so, and the
#    normaliser removes what it wrote anyway: whole-line directions that name
#    a sound or an instrument, performance notes in brackets inside a line,
#    and section names written in Russian (mapped to the English tag).
_RU_TAGS = {
    "вступление": "intro", "интро": "intro",
    "куплет": "verse",
    "предприпев": "pre-chorus", "пред-припев": "pre-chorus",
    "припев": "chorus",
    "постприпев": "post-chorus", "пост-припев": "post-chorus",
    "бридж": "bridge", "мост": "bridge",
    "проигрыш": "instrumental", "инструментал": "instrumental",
    "соло": "solo",
    "концовка": "outro", "аутро": "outro", "финал": "outro", "кода": "outro",
}
_EN_LABEL_RE = re.compile(r"^\s*(intro|verse|pre-chorus|chorus|bridge|outro|hook)\s*(\d*)\s*(?:\([^)]*\))?\s*:\s*$", re.IGNORECASE)
_RU_TAG_RE = re.compile(
    r"^\s*[\[\(]?\s*(" + "|".join(sorted(_RU_TAGS, key=len, reverse=True))
    + r")\s*(\d*)\s*[\]\)]?\s*(?:\([^)]*\))?\s*:?\s*$", re.IGNORECASE)
# A whole line that is a direction about the MUSIC or the DELIVERY rather than
# words to sing: in brackets/parentheses/asterisks, or bare, naming a sound.
_DIRECTION_WORDS = (
    r"bass|beat|drum|guitar|riff|synth|sound|drone|kick|fade|silen|piano|string|"
    r"pad|808|hi-?hat|snare|percussion|whisper|hum|breath|scream|shout|"
    r"instrumental|solo|break|drop|build|tempo|slow|fast|loud|quiet|softly|"
    r"repeat|x\s*\d|\d\s*x|chorus|verse|music|melody|"
    r"бас|бит|барабан|гитар|рифф|синт|звук|шум|тиш|фортепиан|пианин|струн|"
    r"шёпот|шепот|крик|дыхан|вздох|хором|хор\b|повтор|раз[аы]?\b|тише|громче|"
    r"медленн|быстр|музык|мелоди|проигрыш|соло|пауза|инструмент|нараст|затиха")
_DIRECTION_RE = re.compile(r"(?:" + _DIRECTION_WORDS + r")", re.IGNORECASE)
_WRAPPED_LINE_RE = re.compile(r"^\s*(?:\((.*)\)|\[(.*)\]|\*(.*)\*|\{(.*)\})\s*$")
# A performance note INSIDE a line: "(хором)", "(x2)", "[whispered]" -- only
# when it names a delivery/repeat, so a sung "(ooh yeah)" backing line stays.
_INLINE_NOTE_RE = re.compile(
    r"\s*[\(\[\{]\s*(?:" + _DIRECTION_WORDS + r")[^\)\]\}]{0,40}[\)\]\}]", re.IGNORECASE)


def is_stage_direction(line: str) -> bool:
    """A whole line that describes the music/delivery instead of being sung."""
    s = (line or "").strip()
    if not s or _TAG_SPLIT_RE.fullmatch(s):
        return False                      # a section tag is structure, kept
    m = _WRAPPED_LINE_RE.match(s)
    inner = next((g for g in m.groups() if g is not None), None) if m else None
    if inner is not None:
        # Wrapped, and about a sound/delivery: a direction. A wrapped line
        # with ordinary words ("(ooh, yeah)") is a backing vocal and is kept.
        return bool(_DIRECTION_RE.search(inner))
    return False


def _canonical_tag(token: str) -> str:
    """'[Verse 2]' -> '[verse 2]' for known tags; anything else untouched."""
    m = _TAG_BODY_RE.match(token[1:-1])
    if not m:
        return token
    name = m.group(1).strip().lower().replace("_", "-").replace(" ", "-")
    if name not in SECTION_TAGS:
        return token                      # not our vocabulary — leave it alone
    return f"[{name} {m.group(2)}]" if m.group(2) else f"[{name}]"


def normalize_lyrics(text: str) -> str:
    """Enforce the two lyric-input contracts above. Idempotent.

    Splits any line that shares itself with a tag so the tag stands alone
    (the words after it would otherwise never be sung), canonicalises known
    tags to lowercase, and collapses runs of blank lines to a single one.
    """
    out: list[str] = []
    dropped: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            if out and out[-1]:
                out.append("")            # keep ONE blank line between sections
            continue
        # A Russian section name ("Припев:", "[Куплет 2]") is the English tag.
        m_en = _EN_LABEL_RE.match(line)      # "Verse 2:", "Bridge (heavy):"
        if m_en:
            line = f"[{m_en.group(1).lower()} {m_en.group(2)}]" if m_en.group(2) else f"[{m_en.group(1).lower()}]"
        m_ru = _RU_TAG_RE.match(line)
        if m_ru:
            tag = _RU_TAGS[m_ru.group(1).lower()]
            line = f"[{tag} {m_ru.group(2)}]" if m_ru.group(2) else f"[{tag}]"
        for part in _TAG_SPLIT_RE.split(line):
            part = part.strip()
            if not part:
                continue
            if _TAG_SPLIT_RE.fullmatch(part):
                out.append(_canonical_tag(part))
                continue
            # Words, not a tag: a direction is dropped, a note inside the
            # line is cut out of it.
            if is_stage_direction(part):
                dropped.append(part)
                continue
            cleaned = _INLINE_NOTE_RE.sub("", part).strip()
            if cleaned != part:
                dropped.append(part)
                if not cleaned:
                    continue
                part = cleaned
            out.append(part)
    while out and not out[-1]:
        out.pop()
    if dropped:
        logger.info("lyrics: dropped %d stage direction(s) that would have been sung: %s",
                    len(dropped), " | ".join(d[:40] for d in dropped[:6]))
    return "\n".join(out)


def caption_is_structured(style: str) -> bool:
    """Does this caption carry all three required headings?"""
    low = (style or "").lower()
    return all(h.lower() in low for h in CAPTION_HEADINGS)


# --------------------------------------------------------------------------- #
# The settings vocabulary
# --------------------------------------------------------------------------- #
# What a user is allowed to pin, and the ENGLISH phrase each choice becomes in
# the caption brief. This lives HERE, not in a surface module, because it is a
# property of the model rather than of any one UI: the Telegram picker
# (tg_music.py) and the desktop tab (gui_music_tab.py) both render it, and two
# copies would drift into two different sets of genres offered by two screens
# that claim to configure the same engine. Surfaces own their own LABELS —
# translated for Telegram, English for the desktop tab — never the phrases.
#
# Genres are named the way MiniMax's own prompt library names its 18 style
# families, so each phrase lands on vocabulary the model has actually seen
# rather than an invented synonym. This is a curated subset that fits a
# keyboard; leaving the setting unchosen lets the songwriter reach anything.
GENRES: dict = {
    "auto":       "",                       # let the topic decide
    "pop":        "Pop / contemporary pop",
    "rock":       "Pop rock / alternative rock",
    "edm":        "EDM / house / club electronic",
    "synth":      "Synth-pop / electropop / dream pop",
    "hiphop":     "Hip-hop / rap",
    "rnb":        "Contemporary R&B / neo-soul",
    "soul":       "Soul / blues / gospel",
    "jazz":       "Vocal jazz / jazz ballad / swing",
    "folk":       "Indie folk / singer-songwriter / modern acoustic",
    "country":    "Country / Americana",
    "metal":      "Hard rock / metal",
    "cinematic":  "Cinematic orchestral / epic soundtrack",
    "lofi":       "Lo-fi hip hop / ambient chill",
}

# Tempo is offered as BANDS rather than a free BPM number on purpose: the
# guide's own rule is not to fabricate precision — an exact BPM is for when you
# truly want it pinned, and a range leaves the model room to be musical. Each
# band still names its approximate BPM so the choice means something concrete.
TEMPOS: dict = {
    "auto":     "",
    "slow":     "a slow, unhurried tempo of roughly 65-75 BPM",
    "medium":   "a moderate tempo of roughly 95-105 BPM",
    "fast":     "a driving, up-tempo feel of roughly 125-135 BPM",
    "veryfast": "a high-energy tempo of roughly 145-160 BPM",
}
TEMPO_BPM: dict = {
    "auto": "", "slow": "~70", "medium": "~100", "fast": "~130", "veryfast": "~150",
}
# A typed value, within reason: a tempo key "bpm:118" pins the number exactly
# (the guide's "exact BPM is for when you truly want it pinned"), a duration is
# any number of seconds in DURATION_RANGE. The ranges are what the engine can
# actually deliver, not what a text field can hold.
BPM_RANGE: tuple = (50, 200)
DURATION_RANGE: tuple = (20, 180)
STEPS_RANGE: tuple = (20, 50)         # DiT sampler steps offered in Quality
STEPS_CHOICES: tuple = (20, 25, 30, 40, 50)


def clamp_steps(n) -> int:
    try:
        n = int(n or 0)
    except (TypeError, ValueError):
        return MUSIC_STEPS
    lo, hi = STEPS_RANGE
    return n if lo <= n <= hi else MUSIC_STEPS


def custom_bpm(tempo_key: str) -> int:
    """The pinned BPM behind a "bpm:N" tempo key, or 0."""
    k = (tempo_key or "").strip().lower()
    if not k.startswith("bpm:"):
        return 0
    try:
        n = int(k[4:])
    except ValueError:
        return 0
    return n if BPM_RANGE[0] <= n <= BPM_RANGE[1] else 0


def render_ceiling(duration_s: int) -> int:
    """The max_duration handed to the render for a song ASKED at `duration_s`.

    The model card: the duration is an upper bound and generation ends when
    the model emits its end-of-audio token. So the ask gets ~20% + 10 s of
    headroom and the lyric is sized for the ask itself (line_budget): the
    song reaches its [outro] and stops on its own, instead of being severed
    mid-word at exactly the number on the button.
    """
    d = max(1, int(duration_s or MUSIC_DEFAULT_SECONDS))
    return max(d, min(MUSIC_CEILING_SECONDS, int(d * 1.2) + 10))


VOCALS: dict = {
    "auto":         "",
    "female":       "a FEMALE lead vocal",
    "male":         "a MALE lead vocal",
    "duet":         "a male/female duet trading lines",
    "instrumental": "INSTRUMENTAL",      # a different SHAPE — see prefs_from
}

# The round numbers people actually pick. Clamped to MUSIC_MAX_SECONDS at
# render time regardless.
#
# 240 was offered and could NOT be delivered. Measured end to end
# (bench/music_duration_e2e.py), same slot, same style, one render each:
#
#     240s ask, 73 lines (the budget's own target) -> 149.5s   (62%)
#     240s ask, 110 lines                          -> 175.6s   (73%)
#
# Half again as many words bought 17% more song: the planner compresses as the
# lyric grows -- 2.05 s per line at 73, 1.60 s at 110 -- so the slot cannot be
# filled by writing more. 180 was verified at 179.9s and 30 at 30.0s, both with
# the budget's own lyric, so the picker now stops where the engine does. An
# option that delivers 62% of what it promises is worse than one that is not
# offered.
DURATIONS: tuple = (30, 60, 120, 180)


def prefs_from(genre: str = "", tempo: str = "", vocal: str = "") -> dict:
    """Turn chosen setting KEYS into the prefs dict the brief understands.

    Unknown or "auto" values are dropped rather than passed through, so an
    unset setting stays absent instead of becoming an empty instruction.
    """
    out: dict = {}
    for field, key, table in (("genre", genre, GENRES), ("tempo", tempo, TEMPOS),
                              ("vocal", vocal, VOCALS)):
        k = (key or "").strip().lower() if isinstance(key, str) else ""
        if field == "tempo" and custom_bpm(k):
            out[field] = f"exactly {custom_bpm(k)} BPM"
            continue
        if k and k != "auto" and table.get(k):
            out[field] = table[k]
    # Instrumental is a different SHAPE of request, not a kind of voice: it
    # changes what `lyrics` must CONTAIN, not just how the caption describes
    # the singer. Emitting both keys would leave the brief holding two facts
    # that contradict each other ("the lead vocal is INSTRUMENTAL").
    if (vocal or "").strip().lower() == "instrumental":
        out.pop("vocal", None)
        out["instrumental"] = True
    return out


def choose_auto_params(ctx, topic: str, lang: str, *, genre: bool = False,
                       tempo: bool = False, vocal: bool = False,
                       duration: bool = False, fixed: dict | None = None) -> dict:
    """The bot's own pick for every setting left on Auto, as one short LLM
    call BEFORE the songwriting: {"genre": key, "bpm": int, "vocal": key,
    "seconds": int, "why": sentence} -- only the fields asked for, each
    validated against the tables/ranges, so the caption brief gets them as
    explicit requirements and the user can be told what was chosen and why.
    Empty dict when nothing is on Auto or the model fumbles (the songwriter
    then decides inside the caption, as before).

    `fixed` is what the user DID choose (the prefs_from() dict and/or a
    duration) -- it goes into the ask as given conditions, so the picks fit
    both the topic and the user's part: a tempo for a country song, a length
    for a duet, not a guess made blind.
    """
    want = [k for k, on in (("genre", genre), ("bpm", tempo), ("vocal", vocal),
                            ("seconds", duration)) if on]
    if not want:
        return {}
    import llm as _llm
    import utils as _utils
    lang_word = "Russian" if (lang or "").lower().startswith("ru") else "English"
    spec = {
        "genre": "genre: one of " + ", ".join(k for k in GENRES if k != "auto"),
        "bpm": f"bpm: an integer {BPM_RANGE[0]}-{BPM_RANGE[1]}",
        "vocal": "vocal: one of female, male, duet",
        "seconds": f"seconds: an integer {DURATION_RANGE[0]}-{DURATION_RANGE[1]} -- a jingle or a joke "
                   "is short, a ballad or an anthem is long; 60-120 fits most songs",
    }
    sys_p = ("You are a music producer choosing the production parameters that suit a song "
             "topic best. Reply with ONLY a JSON object, no commentary, with exactly these keys: "
             + "; ".join(spec[k] for k in want)
             + f"; why: one short sentence in {lang_word} explaining the choice to the listener.")
    given = "; ".join(f"{k}: {v}" for k, v in (fixed or {}).items() if v)
    user_p = f"Song topic: {topic}"
    if given:
        user_p += chr(10) + f"Already fixed by the user (do not change, pick the rest to fit these): {given}"
    try:
        raw = _llm.call_llm_simple(ctx, sys_p, user_p, temperature=0.4,
                                   max_tokens=200, prefill="<think></think>")
        data = _utils.safe_json_from_llm(raw or "") or {}
    except Exception:
        logger.warning("auto params: the model fumbled", exc_info=True)
        return {}
    out: dict = {}
    # "lo-fi", "hip-hop", "r&b" -- and live, "lifi" for a lullaby.
    import difflib
    g = next(iter(difflib.get_close_matches(re.sub(r"[^a-z]", "", str(data.get("genre", "")).lower()),
                                            list(GENRES), n=1, cutoff=0.75)), "")
    if genre and g in GENRES and g != "auto":
        out["genre"] = g
    try:
        b = int(data.get("bpm", 0))
        if tempo and BPM_RANGE[0] <= b <= BPM_RANGE[1]:
            out["bpm"] = b
    except (TypeError, ValueError):
        pass
    v = str(data.get("vocal", "")).strip().lower()
    if vocal and v in ("female", "male", "duet"):
        out["vocal"] = v
    if vocal and _no_vocals(topic):
        out["vocal"] = "instrumental"          # «джингл без вокала» was reported as a male vocal
    try:
        sec = int(data.get("seconds", 0))
        if duration and sec:
            # «джингл на 15 секунд» came back 15 and was dropped: the nearest length the engine can do.
            out["seconds"] = max(DURATION_RANGE[0], min(sec, DURATION_RANGE[1]))
    except (TypeError, ValueError):
        pass
    if out:
        out["why"] = str(data.get("why", "")).strip()[:300]
    return out


class MusicUnavailable(RuntimeError):
    """The engine cannot run at all (missing weights, ComfyUI unreachable)."""


class SongwritingFailed(RuntimeError):
    """The LLM could not write usable lyrics/style — the ENGINE is fine.

    Deliberately NOT a MusicUnavailable subclass. Conflating the two told
    users "song generation isn't set up yet" when the weights, the nodes and
    ComfyUI were all present and working perfectly and it was only the
    songwriter model that fumbled a turn — a message that sends someone off
    reinstalling things instead of just trying again.
    """


# --------------------------------------------------------------------------- #
# Readiness
# --------------------------------------------------------------------------- #
def _models_dir() -> str:
    return os.getenv("COMFY_BASE_DIR", os.path.join(os.path.expanduser("~"), "Documents", "ComfyUI"))


# Which weight variant this install runs, measured rather than assumed. All
# times are a 30s song on the one RTX 3090, same lyrics/caption, fresh seeds:
#
#   fp32 DiT + bf16 encoder      28.5GB   685s   (streams: does not fit 24GB)
#   fp16 DiT + int8 encoder      14.3GB   230s   on torch cu128
#   fp16 DiT + int8 encoder      14.3GB    49s   on torch cu130
#   w4a8 everything (3rd party)   7.9GB    59s   on cu130 (crashed on cu128)
#   w4a8 DiT + int8 encoder      11.1GB    40s   <- this, smallest AND fastest
#
# The cu130 column is the whole story: comfy/quant_ops.py disables the
# comfy_kitchen CUDA backend outright when torch reports CUDA < 13, so every
# quantized weight fell back to backends/eager/ — which is not merely slow, it
# ABORTED the process on w4a8 (Fatal Python error in
# _dequant_int4_grouped_to_int8). With cu130 the native kernels run and the
# same graph is 4.7x faster.
#
# The split matters because the two halves do different work: ~87% of a render
# is the text encoder's 751-step AR decode, which int8_convrot does in 16s
# against the third-party w4a8 encoder's 35s — so the encoder stays int8 even
# though a smaller one exists. The DiT only drives the 30 diffusion steps, so
# it can be the tiny w4a8 build for free.
#
# Overridable by env so a variant can be swapped without editing code. The
# names are the single source of truth: build_workflow stamps them into the
# loader nodes, so workflow_music3.json cannot drift out of agreement with
# what missing_weights() checks for — the previous shape had the filenames
# written in BOTH places, where changing one silently left a graph pointing at
# a file the readiness check had already declared present.
#
# NOTE: the DiT and VAE here are third-party (dummy9996) rather than
# Comfy-Org. safetensors, so no code execution, but the provenance is worth
# knowing. The official fp16 DiT is one env var away if quality ever looks off.
# Three selectable presets rather than one compiled-in choice, because the
# speed/quality trade is a taste question the user should own per song. Times
# are the measured 30s renders above.
#
# The threshold that shapes this: pruned_bf16 is 16.71GB and FITS in 24GB, so
# it runs its AR decode in 25s; the unpruned bf16 is 18.47GB, does NOT fit, and
# streams at 196s. "max" is therefore 4x slower than "quality" for an unpruned
# encoder and an fp32 DiT — kept because it is the true ceiling and the choice
# is the user's, not because it is a good default.
WEIGHT_PRESETS: dict = {
    # ~40s — quantized throughout, everything resident.
    "fast": {
        "dit":  "minimax_music3_dit_w4a8.safetensors",
        "clip": "minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
        "vae":  "minimax_music3_dav-bf16.safetensors",
        "gb": 11.14, "secs": 40,
    },
    # ~59s — full-precision (unquantized) encoder that still fits in VRAM.
    "quality": {
        "dit":  "minimax_music3_dit_w4a8.safetensors",
        "clip": "minimax_music3_text_encoder_pruned_bf16.safetensors",
        "vae":  "minimax_music3_dav.safetensors",
        "gb": 18.75, "secs": 59,
    },
    # ~240s — fp32 DiT + the unpruned encoder. Exceeds VRAM and streams.
    "max": {
        "dit":  "minimax_music3_dit_fp32.safetensors",
        "clip": "minimax_music3_text_encoder_bf16.safetensors",
        "vae":  "minimax_music3_dav.safetensors",
        "gb": 28.52, "secs": 240,
    },
}
DEFAULT_PRESET = os.getenv("MUSIC_PRESET", "fast")
# ── measured render times ────────────────────────────────────────────────────
# The "secs" in WEIGHT_PRESETS were one 30-second measurement each; the menu
# quoted them for every length, so «⚡ ~40с» sat next to a 180-second song
# that takes five and a half minutes. The ETA is now the MEDIAN of completed
# renders of that preset, as seconds of wall time per second of song asked,
# seeded from the app log (start line -> "faded song" line, whole renders,
# weight load included), and every real render appends its own.
ETA_QUOTE_SECONDS = 180          # the length the settings menu quotes for
_ETA_SEEDS: dict = {             # (asked seconds, wall seconds, steps) from assistant_app.log
    "fast":    [(180, 255, 30), (180, 347, 30), (180, 321, 30), (180, 336, 30), (180, 572, 30)],   # 09-10 .. 09-14
    "quality": [(120, 356, 30)],                                               # 09-14 01:45
    "max":     [(180, 2077, 30), (180, 1911, 30)],                             # 09-14 00:38, 08-21 20:54
}
_ETA_FILE = "music_render_times.json"
# A render is an autoregressive stage (fixed per second of song) plus the DiT
# stage (grows with the step count). Measured 15.09 on ⚡ for an 82-second
# song: AR 45 s, DiT 52 s at 30 steps -> the DiT share at 30 steps is ~½.
# 🐘 streams its fp32 DiT over PCIe, so there the DiT is nearly everything.
_DIT_SHARE_AT_30: dict = {"fast": 0.5, "quality": 0.5, "max": 0.9}
_STEPS_REF = 30


def _steps_factor(preset: str, steps: int) -> float:
    """wall(steps) / wall(30 steps) for this preset."""
    share = _DIT_SHARE_AT_30.get(preset, 0.5)
    return (1.0 - share) + share * (max(1, int(steps)) / float(_STEPS_REF))


_ETA_LOCK = threading.Lock()


def _eta_path():
    return Path(OUTPUT_DIR) / _ETA_FILE


def _eta_runs() -> dict:
    """{preset: [(asked, wall), ...]} -- the seeds plus what was recorded."""
    runs = {k: list(v) for k, v in _ETA_SEEDS.items()}
    try:
        import json
        with open(_eta_path(), "r", encoding="utf-8") as fh:
            rows = json.load(fh)
    except Exception:
        return runs
    for row in rows if isinstance(rows, list) else []:
        try:    # one malformed row must not drop every row after it
            runs.setdefault(str(row["preset"]), []).append(
                (int(row["asked"]), float(row["wall"]), int(row.get("steps", _STEPS_REF))))
        except Exception:
            continue
    return runs


def record_render_time(preset: str, asked_s: int, wall_s: float, steps: int = _STEPS_REF) -> None:
    """Append one completed render (only whole, successful ones are called in)."""
    if not preset or asked_s <= 0 or wall_s <= 0:
        return
    try:
        import json
        import time as _time
        with _ETA_LOCK:     # two songs finishing together must not drop a row
            path = _eta_path()
            rows = []
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    rows = json.load(fh)
            except FileNotFoundError:
                pass
            except ValueError:
                # a torn file used to fail every later call, so render times
                # stopped being learned for good; start the log over instead
                logger.warning("render-time log %s was unreadable; starting it over", path)
            if not isinstance(rows, list):
                rows = []
            rows.append({"preset": preset, "asked": int(asked_s), "wall": round(float(wall_s), 1),
                         "steps": int(steps or _STEPS_REF), "at": int(_time.time())})
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(rows[-200:], fh)
            os.replace(tmp, path)
    except Exception:
        logger.warning("could not record the render time", exc_info=True)


def eta_seconds(preset: str, asked_s: int = ETA_QUOTE_SECONDS, steps: Optional[int] = None) -> int:
    """Expected wall seconds for a song ASKED at `asked_s` on `preset` with
    `steps` DiT steps: every measured run is normalised to 30 steps
    (_steps_factor), the median rate (wall / asked) is taken, then scaled
    back to the steps asked. Falls back to the preset's 30-second figure
    when nothing was ever measured."""
    import statistics
    steps = clamp_steps(steps if steps is not None else MUSIC_STEPS)
    runs = [(a, w, st) for a, w, st in _eta_runs().get(preset, []) if a > 0 and w > 0]
    if runs:
        rate30 = statistics.median(w / a / _steps_factor(preset, st) for a, w, st in runs)
    else:
        spec = WEIGHT_PRESETS.get(preset) or {}
        rate30 = float(spec.get("secs", 60)) / 30.0
    rate = rate30 * _steps_factor(preset, steps)
    return max(1, int(round(rate * max(1, int(asked_s or ETA_QUOTE_SECONDS)))))


def eta_label(seconds: int, lang: str = "ru") -> str:
    """«~5 мин» / «~33 мин» / «~45 с» -- minutes once past 90 s."""
    ru = (lang or "").lower().startswith("ru")
    if seconds >= 90:
        m = int(round(seconds / 60.0))
        return f"~{m} {'мин' if ru else 'min'}"
    return f"~{int(seconds)} {'с' if ru else 's'}"


if DEFAULT_PRESET not in WEIGHT_PRESETS:
    logger.warning("unknown MUSIC_PRESET %r — falling back to 'fast'", DEFAULT_PRESET)
    DEFAULT_PRESET = "fast"


def preset_files(preset: Optional[str] = None) -> tuple:
    """(dit, clip, vae) filenames for a preset name.

    An unknown name resolves to the default rather than raising: this is fed
    from persisted per-chat state, and a stale value left by a renamed preset
    must not take song generation down with it.
    """
    p = (preset or DEFAULT_PRESET)
    p = p if isinstance(p, str) and p in WEIGHT_PRESETS else DEFAULT_PRESET
    spec = WEIGHT_PRESETS[p]
    # The single-file env overrides still win, so one variant can be swapped
    # for a one-off test without touching the preset table.
    return (os.getenv("MUSIC_DIT",  spec["dit"]),
            os.getenv("MUSIC_CLIP", spec["clip"]),
            os.getenv("MUSIC_VAE",  spec["vae"]))


# Kept as module attributes for the default preset: existing callers and the
# suites read these, and the docs refer to them.
MUSIC_DIT, MUSIC_CLIP, MUSIC_VAE = preset_files()

REQUIRED_FILES = {
    "diffusion_models": [MUSIC_DIT],
    "text_encoders": [MUSIC_CLIP],
    "vae": [MUSIC_VAE],
}


def missing_weights(preset: Optional[str] = None) -> list[str]:
    """Which model files this preset needs but does not have (empty = ready).

    Preset-aware because the presets do not share files: "max" can be a 28GB
    download away while "fast" is ready, and reporting the default preset's
    readiness for a request that asked for another one would promise a song
    that cannot be rendered.
    """
    dit, clip, vae = preset_files(preset)
    base = os.path.join(_models_dir(), "models")
    out = []
    for sub, n in (("diffusion_models", dit), ("text_encoders", clip), ("vae", vae)):
        if not os.path.exists(os.path.join(base, sub, n)):
            out.append(f"models/{sub}/{n}")
    return out


MUSIC_ENGINE = os.getenv("MUSIC_ENGINE", "yue2")


def engine_available(ctx=None, preset: Optional[str] = None) -> tuple[bool, str]:
    """Can we generate a song right now? Returns (ok, human reason if not).

    Checked BEFORE the agent promises anything, same reasoning as
    video.engine_available: a missing node means the graph is rejected after
    the user was already told a song is coming, and a missing multi-GB
    checkpoint means a download, not a retry. `preset` is checked rather than
    the default, since the presets do not share weight files.
    """
    # YuE2 is the one engine since 2026-09-25 (user: Music3 removed). The
    # Music3 checks below stay only for MUSIC_ENGINE=music3 with weights restored.
    if MUSIC_ENGINE == "yue2":
        return (True, "") if (yue2_cpp_available() or yue2_available()) else (False, "YuE2 is not installed")
    miss = missing_weights(preset)
    if miss:
        return False, ("the Music3 weights are not downloaded yet (missing: "
                       + ", ".join(miss[:3]) + (" …" if len(miss) > 3 else "")
                       + ")")
    if not _server_has_music3_nodes():
        return False, ("this ComfyUI does not have the MiniMax Music3 node — "
                       "see docs/music_generation.md for the node package this "
                       "needs")
    return True, ""


_NODE_CACHE: dict = {}


def _server_has_music3_nodes() -> bool:
    """Ask the live ComfyUI whether it knows the Music3 text-encode node.

    Cached per URL, and fails OPEN on a request error (a server that is
    merely down may come back) exactly like video._server_has_h3_nodes.

    Deliberately hits the FULL /object_info listing and checks membership,
    not /object_info/<name>: this ComfyUI build returns an empty {} body
    (200 OK) from the single-node endpoint even for nodes that genuinely
    exist, confirmed live against a real server — a per-node GET here would
    always read as "missing" and refuse every request.
    """
    if COMFY_URL in _NODE_CACHE:
        return _NODE_CACHE[COMFY_URL]
    ok = False
    try:
        import requests
        r = requests.get(f"{COMFY_URL}/object_info", timeout=15)
        ok = r.status_code == 200 and "MiniMaxMusic3TextEncode" in (r.json() or {})
    except Exception as exc:
        logger.warning("could not ask ComfyUI about the Music3 node: %s", exc)
        return False        # not cached: a server that is merely down may come back
    _NODE_CACHE[COMFY_URL] = ok
    return ok


def _valid_audio_file(path: str) -> bool:
    """A real, non-empty audio file — not a truncated write or a stub."""
    try:
        if not (path and os.path.exists(path)):
            return False
        # A minute of any lossy/lossless codec is well over a few KB; anything
        # tiny is a failed write, not a very short song.
        if os.path.getsize(path) < 2 * 1024:
            logger.error("music file is implausibly small: %s (%d bytes)",
                         path, os.path.getsize(path))
            return False
        return os.path.splitext(path)[1].lower() in (
            ".wav", ".flac", ".mp3", ".ogg", ".m4a")
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Building the graph
# --------------------------------------------------------------------------- #
def _load_workflow() -> dict:
    path = WORKFLOW_MUSIC_PATH
    if not os.path.exists(path):
        raise MusicUnavailable(f"music workflow missing: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_workflow(lyrics: str, style: str, *, duration_s: int, seed: int,
                   steps: Optional[int] = None,
                   preset: Optional[str] = None) -> dict:
    """The ComfyUI API graph for one song.

    Split out from `generate_music` so the shape of the graph can be
    asserted in tests without a ComfyUI, a GPU or any weights in sight —
    mirrors video.build_workflow.
    """
    wf = _load_workflow()
    # Stamp the chosen preset's files into the loaders, so the graph always
    # names exactly what missing_weights(preset) checked for.
    dit, clip, vae = preset_files(preset)
    wf[N_UNET]["inputs"]["unet_name"] = dit
    wf[N_CLIP]["inputs"]["clip_name"] = clip
    wf[N_VAE]["inputs"]["vae_name"] = vae
    txt = wf[N_TEXT]["inputs"]
    # Normalised HERE, at the last boundary before the graph, rather than only
    # where the songwriter returns: a caller passing hand-written lyrics (the
    # GUI tab, a script, a test) gets the same tag-on-its-own-line guarantee,
    # and the LLM path simply normalises an already-normalised string.
    txt["lyrics"] = normalize_lyrics(lyrics)
    txt["caption"] = style
    txt["seed"] = int(seed)
    txt["max_duration"] = float(duration_s)

    # An explicit `steps` (the GUI's Quality slider, STEPS_RANGE 20-50) means
    # the caller wants the plain, non-distilled model at that exact step
    # count -- the Turbo LoRA below is distilled for 8 NFE specifically, and
    # sampling it at 20+ steps is out of its trained regime, not just slower.
    # Only when nobody asked for a step count does the fast default apply
    # (2026-09-20: 25% faster wall-clock on a 60s song, only ~7% on 20s).
    if steps is None:
        new_id = "lora_" + N_UNET
        for node in wf.values():
            if node is wf.get(new_id):
                continue
            for k, v in list((node.get("inputs") or {}).items()):
                if isinstance(v, list) and len(v) == 2 and str(v[0]) == N_UNET:
                    node["inputs"][k] = [new_id, 0]
        wf[new_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": [N_UNET, 0], "lora_name": MUSIC_TURBO_LORA,
                       "strength_model": MUSIC_TURBO_STRENGTH},
        }
        steps = MUSIC_STEPS_TURBO
        cfg = MUSIC_CFG_TURBO
    else:
        cfg = MUSIC_CFG

    smp = wf[N_SAMPLER]["inputs"]
    smp["seed"] = int(seed)
    smp["steps"] = int(steps)
    smp["cfg"] = float(cfg)
    return wf


# --------------------------------------------------------------------------- #
# LLM-authored song structure
# --------------------------------------------------------------------------- #
def _requirements_block(prefs: Optional[dict]) -> str:
    """The user's chosen settings, phrased as top-precedence requirements.

    MiniMax's precedence order is: explicit requirements first, then
    section-tag directives, then implications of the description, then genre
    defaults. So settings the user actually picked are stated as requirements
    that must survive the whole caption — not folded into the topic, where the
    songwriter is free to reinterpret them as a mood and write something else.

    Returns "" when nothing was chosen, which leaves the brief exactly as it
    was: an unset setting must not silently become an instruction.
    """
    if not prefs:
        return ""
    lines = []
    if prefs.get("genre"):
        lines.append(f"  * The genre IS: {prefs['genre']}. Do not drift to another.")
    if prefs.get("tempo"):
        lines.append(f"  * The tempo IS {prefs['tempo']}. State it in Basic Attributes.")
    if prefs.get("instrumental"):
        # The guide's own instruction for instrumental work: say so explicitly
        # and name the instrument carrying the lead melodic line. Lyrics are
        # dropped to bare tags rather than left as words the model would sing
        # over a caption that just told it not to sing.
        lines.append("  * This song is INSTRUMENTAL — there is no singer. Say so "
                     "explicitly in Vocal Details and name the instrument that "
                     "carries the lead melodic line in its place. `lyrics` must "
                     "contain ONLY section tags, one per line, with NO words "
                     "underneath any of them.")
    elif prefs.get("vocal"):
        lines.append(f"  * The lead vocal IS {prefs['vocal']}. Say so explicitly "
                     "under Vocal Gender & Timbre.")
    if not lines:
        return ""
    return ("\n\nHARD REQUIREMENTS — these outrank everything else in this "
            "brief, including anything the topic implies, and must hold for "
            "the WHOLE song:\n" + "\n".join(lines))


# ── how long a lyric actually sings for ──────────────────────────────────────
# Measured on the renderer's own acoustic planner rather than guessed
# (bench/music_duration.py, which reads MiniMaxMusic3TextEncode's `seconds`
# output without rendering anything):
#
#   ceiling 60s,  9 / 17 / 33 / 65 sung lines -> planned 60.0s every time.
#       Even the shortest lyric wanted MORE than the slot, so 60s says nothing
#       about the rate; it only proves the ceiling is a real ceiling.
#   ceiling 180s, 17 sung lines               -> planned 131.2s, stopped early.
#       ~7.7s per sung line at 108 BPM, and the planner stops when the words
#       run out. This is the whole of "I asked for three minutes and got one
#       and a half".
#
# Whisper on three real songs gives the other end of the range: a fast Russian
# lyric sings at ~3.3s per line. So the rate is a property of the TEMPO, which
# our own style caption sets, and a single constant cannot be right for both.
# The working estimate is the middle; the band is what gets enforced, so a song
# lands inside its slot instead of at half of it.
# Rates measured on RENDERED songs, seconds of audio per sung line:
#   3.3  a fast Russian lyric (Whisper, song #1: ~30 lines in 100s)
#   4.3  another Russian song (Whisper, song #3)
#   4.6  a Russian song rendered end-to-end for a 180s slot: 30 lines -> 138.3s
#   7.7  an English pop lyric at 108 BPM (the planner probe)
# The English point is a different style and is NOT what this house writes; the
# Russian band is 3.3-4.6, so the slow end used for the floor is 4.6.
#
# The TARGET comes from the FAST end deliberately. Undershooting cannot be
# repaired -- the song simply stops early, which is the reported bug -- while
# overshooting is cut at the ceiling and tapered by apply_fade(). Asking for
# enough words to fill the slot even at the quickest delivery is what makes the
# duration button mean something.
# With render_ceiling() giving the slot headroom, the target moved from the
# fast end (3.3, which overran the slot on purpose and relied on the chop +
# fade) to just under the middle of the measured band: the words run out
# near the ask and the [outro] lands inside the headroom.
SECONDS_PER_SUNG_LINE = 3.8      # near the middle of the measured band
_RATE_FAST = 3.3                 # fastest measured -> how many lines to ask for
_RATE_SLOW = 4.6                 # slowest measured in Russian -> the floor
MIN_SUNG_LINES = 8
MAX_SUNG_LINES = 80


def count_sung_lines(lyrics: str) -> int:
    """Lines that are actually SUNG: not blank, not a [section] tag.

    The tags are structure, not words, and counting them would let a lyric of
    eight tags and four lines pass as a full song.
    """
    n = 0
    for line in (lyrics or "").splitlines():
        s = line.strip()
        if s and not s.startswith("["):
            n += 1
    return n


def lines_for_duration(duration_s) -> int:
    """How many sung lines a `duration_s` slot wants."""
    try:
        secs = float(duration_s or 0)
    except (TypeError, ValueError):
        secs = 0.0
    if secs <= 0:
        return 0
    want = int(round(secs / SECONDS_PER_SUNG_LINE))
    return max(MIN_SUNG_LINES, min(want, MAX_SUNG_LINES))


def line_budget(duration_s) -> tuple:
    """(target, floor, ceiling) sung lines for a slot.

    The FLOOR is what the slow end needs: fewer lines than this and the song
    ends early even when sung unhurriedly -- the reported bug. Measured live
    before this was tightened: a 180s slot accepted 30 lines and produced
    138.3s, because the floor had been computed from an English pop rate of
    7.7s per line while the song actually sang at 4.6.

    The TARGET is the fast end, so the words fill the slot however briskly they
    are delivered. The CEILING is just the hard cap: a lyric longer than the
    slot is cut and faded, which is a far smaller failure than one that stops
    forty seconds early.
    """
    target = lines_for_duration(duration_s)
    if not target:
        return 0, 0, 0
    secs = float(duration_s)
    lo = max(MIN_SUNG_LINES, int(secs / _RATE_SLOW))
    return target, min(lo, target), MAX_SUNG_LINES


def _no_vocals(topic: str) -> bool:
    """The song is to have no singing (an instrumental, «без слов»). The model's yes/no."""
    import intent
    return intent.ask_yes("A user asked for this song: {text}\n\nDoes the request ask for NO "
                          "singing -- an instrumental, no vocals, no words?", topic)


def _own_form(topic: str) -> bool:
    """The request fixes the song's own shape (exact verse/line counts, no chorus)."""
    import intent
    return intent.ask_yes("A user asked for this song: {text}\n\nDoes the request fix the "
                          "song's exact structure -- how many verses or lines, or no chorus?", topic)


def build_structured_caption(ctx, topic: str, lang: str, *,
                             duration_s: Optional[int] = None,
                             prefs: Optional[dict] = None) -> dict:
    """Expand a short topic/wish into a full song via the house LLM.

    Returns {"lyrics": str, "style": str} — the two inputs Music3 takes, in
    the shape its prompting guide specifies (see the prompt-contract section
    at the top of this module). `lyrics` is tagged section text written in
    `lang` ('en' or 'ru') with the tags themselves always English, since that
    is the vocabulary the model recognises rather than a translated one.
    `style` is the structured caption: prose under exactly three headings —
    Global Metadata, Vocal Details, Arrangement — read by the model as a
    timeline rather than one global label.

    `duration_s` only shapes how MUCH song to write. The requested duration
    is an upper bound: Music3 stops when the words run out, so short lyrics
    make a short song no matter what duration the render asks for.
    """
    import llm as _llm

    lang_name = "Russian" if (lang or "").strip().lower().startswith("ru") else "English"
    length_hint = ""
    want_lines, min_lines, max_lines = line_budget(duration_s)
    if not (prefs or {}).get("instrumental") and _no_vocals(topic):
        # «эмбиент без слов и без вокала» was sung as «Ммм / Ооо».
        prefs = {k: v for k, v in (prefs or {}).items() if k != "vocal"} | {"instrumental": True}
    if (prefs or {}).get("instrumental") or _own_form(topic):
        # «ровно три куплета, без припева» came back with a 15-line outro
        # padding it up to the slot's line count (live 2026-09-29).
        want_lines = min_lines = 0
    if want_lines:
        # A COUNT, not a duration. The writer has no idea how many seconds a
        # line takes -- asking it for "roughly 180 seconds of music" is asking
        # it to guess at the renderer's singing rate, and it guessed short every
        # time. The count is computed from a measured rate instead; see
        # SECONDS_PER_SUNG_LINE.
        length_hint = (
            f" LENGTH: write {want_lines} sung lines — no fewer than "
            f"{min_lines} and no more than {max_lines}, counting every line "
            "that has words on it and not counting the [section] tags. This "
            f"fills a {int(duration_s)}-second slot: the model stops singing "
            "when the words run out, so a short lyric ends the song early, and "
            "the render is cut off at the slot, so an over-long one is severed "
            "mid-phrase. Add or drop whole sections to hit the count.")
    sys_p = (
        "You are a songwriter and music producer preparing the two inputs the "
        "MiniMax Music 3 text-to-music model takes. Given a short topic or "
        "wish, write a complete song for it. Answer with ONLY a JSON object, "
        "no commentary, no code fences, shaped exactly like: "
        '{"lyrics": "...", "style": "..."}.\n\n'
        "`lyrics` — the sung words, and nothing else.\n"
        "  * Lay the song out with these lowercase section tags: [intro] "
        "[verse] [pre-chorus] [chorus] [post-chorus] [bridge] [instrumental] "
        "[solo] [outro].\n"
        "  * EVERY tag sits alone on its own line, with that section's words "
        "on the lines BELOW it. Text sharing a line with a tag is thrown away "
        "by the model and never sung.\n"
        "  * The tags are ALWAYS English (that is the model's required "
        f"format); the words of the song itself are written in {lang_name}.\n"
        "  * EVERYTHING in `lyrics` is sung out loud, word for word. Never put "
        "stage directions, sound descriptions or performance notes there — no "
        "'(Low bass drone)', '(heavy beat kicks in)', '(хором)', '(x2)', "
        "'(whispered)': the singer would sing those words. Instruments, "
        "sounds and delivery belong in `style`; a repeat is written out in "
        "full. Parentheses are ONLY for backing-vocal words that are meant to "
        "be sung.\n"
        "  * ALWAYS finish with an [outro] section that resolves the song — a "
        "short closing line or a repeat of the hook that lands as an ending, "
        "not a fresh idea. Without one the music simply stops mid-phrase.\n"
        f"  * Use at least two verses and a repeated chorus.{length_hint}\n\n"
        "`style` — the MUSIC description, never the words. Write it under "
        "exactly these three headings, in this order, 250-450 words total:\n\n"
        "  Global Metadata\n"
        "    - Basic Attributes: genre and subgenres, tempo, and key/scale "
        "ONLY if you genuinely want them pinned (otherwise give a qualitative "
        "tempo such as 'driving' or 'unhurried').\n"
        "    - Global Emotional Progression: the arc from opening through the "
        "peak to how it resolves.\n"
        "    - Application Scenarios & Imagery: a concrete scene the song "
        "belongs to — this anchors mood far better than adjectives.\n"
        "    - Sonics & Production Profile: stereo width, frequency balance, "
        "dynamics.\n\n"
        "  Vocal Details\n"
        "    - Vocal Gender & Timbre: ALWAYS state the singer's gender and "
        "timbre explicitly. Leaving this unsaid is the single biggest cause "
        "of the model drifting into an unwanted instrumental.\n"
        "    - Vocal Style: delivery and dynamics, section by section.\n"
        "    - Harmony/Backing Vocals: doubles, stacked harmonies, "
        "call-and-response, and where they appear.\n"
        "    - Vocal FX: reverb, delay, saturation — restrained, and where "
        "each applies.\n\n"
        "  Arrangement\n"
        "    - Instrument Lifecycle (Primary/Secondary): what anchors the song "
        "start to finish, and what enters, exits or transforms along the way.\n"
        "    - Groove & Foundation Progression: the rhythmic foundation and "
        "how its density evolves from section to section.\n"
        "    - Embellishments, Textures & Spatial FX: risers, sweeps, ear "
        "candy, reverb tails — only where they matter.\n\n"
        "Write `style` in English whatever language the lyrics are in, since "
        "it is a production brief and not a lyric. Read the Arrangement as a "
        "TIMELINE: for each section say what enters, exits, changes or "
        "intensifies. Never quote lyric lines inside the caption, never "
        "invent a BPM or key you do not actually want, and never contradict "
        "earlier requirements later in the caption."
        + _requirements_block(prefs)
    )
    # The short ask for the last rung: the same job stripped of the craft
    # guidance, so a model that keeps over-thinking the full brief has less to
    # chew on. The three headings, the own-line tags and the explicit vocal
    # gender survive the trim — those are input contracts, not craft.
    sys_minimal = (
        "Write a song about the user's topic. Reply with ONLY this JSON, no "
        'commentary and no code fences: {"lyrics": "...", "style": "..."}. '
        "In `lyrics`, use lowercase [intro] / [verse] / [chorus] / [bridge] / "
        "[outro] tags (always spelled in English), EACH ALONE ON ITS OWN LINE "
        f"with the sung words on the lines below it, written in {lang_name}. "
        "End with an [outro] that closes the song rather than starting a new "
        "idea. Every word in `lyrics` is sung: no stage directions or sound "
        "notes like '(heavy beat)' or '(хором)' — describe the music in `style`. "
        "In `style`, write English prose under exactly three headings: "
        "'Global Metadata' (genre, tempo, emotional arc), 'Vocal Details' "
        "(state the singer's gender and timbre explicitly), and 'Arrangement' "
        "(the instruments, and how they change from section to section)."
        # The requirements ride every rung: they are the user's explicit
        # choices, and a rung that quietly dropped them would hand back a
        # song in the wrong genre with no indication anything was ignored.
        + _requirements_block(prefs)
    )
    user_p = f"Topic / wish: {topic}"

    # A ladder, not a single shot — the same failure and the same remedy as
    # slides.py's deck planner (see the comment there). Live bug: one 2000-token
    # attempt let Gemma spend the WHOLE budget in its reasoning channel and
    # return nothing at all ("10194 chars of reasoning with no final answer,
    # finish_reason=length"), which surfaced to the user as "song generation
    # isn't set up yet" even though the engine was perfectly healthy.
    #
    # The budget GROWS down the ladder, which looks backwards until you read
    # llm.py: on the house Gemma model there is no reasoning switch at all —
    # `no_think` and the <think></think> prefill are both force-cleared, since
    # they are Qwen levers that Gemma's template treats as literal junk. So on
    # Gemma the ONLY lever any of these rungs actually pulls is the token
    # budget, and a ladder that shrinks it (6000 -> 3500 -> 3000, as this one
    # used to) starves the exact failure it is trying to recover from: the
    # model reasons ~4k tokens about a detailed brief, hits the ceiling before
    # writing a word of the answer, and each rung leaves it LESS room than the
    # one that just failed. Measured live against gemma-4-26b: the full brief
    # needs ~12k to think and answer, and lands a structured caption first try
    # at that budget (63s). force_think still drops away on the later rungs for
    # the Qwen families, where it is a real switch and a big budget is simply
    # unused; the shorter ASK on the last rung gives a model that keeps
    # over-thinking the full brief less to chew on.
    attempts = (
        dict(prompt=sys_p,       temperature=0.9, max_tokens=12000, force_think=False),   # 2026-09-25: no reasoning anywhere
        dict(prompt=sys_p,       temperature=0.7, max_tokens=16000, force_think=False),
        dict(prompt=sys_minimal, temperature=0.5, max_tokens=16000, force_think=False),
    )
    for i, cfg in enumerate(attempts, start=1):
        raw = _llm.call_llm_simple(ctx, cfg["prompt"], user_p,
                                   temperature=cfg["temperature"],
                                   max_tokens=cfg["max_tokens"],
                                   force_think=cfg["force_think"]) or ""
        lyrics, style = _parse_song_json(raw)
        lyrics = normalize_lyrics(lyrics)
        if lyrics and style:
            # A caption missing the three headings still renders, but renders
            # a vaguer song — so spend a remaining rung trying for the real
            # shape, and only settle for prose when the ladder is out of rungs.
            # Never fail outright over it: an unstructured caption is a worse
            # song, not a broken one, and refusing here would resurrect the
            # "song generation isn't set up yet" lie on a healthy engine.
            if not caption_is_structured(style) and i < len(attempts):
                logger.warning("songwriter attempt %d returned a caption with no "
                               "Global Metadata/Vocal Details/Arrangement "
                               "headings — escalating", i)
                continue
            # The prompt asks for the count; this checks it. On this house's
            # models an unchecked instruction is a suggestion -- the same
            # lesson as the deck planner's bullet floor.
            got_lines = count_sung_lines(lyrics)
            if min_lines and got_lines < min_lines:
                if i < len(attempts):
                    logger.warning("songwriter attempt %d wrote %d sung lines for a "
                                   "%ss slot (needs %d) — escalating",
                                   i, got_lines, duration_s, min_lines)
                    user_p = (f"Topic / wish: {topic}\n\nYour previous lyric had only "
                              f"{got_lines} sung lines, which ends the song less than "
                              f"halfway through its {int(duration_s)}-second slot. "
                              f"Write {want_lines} sung lines this time — add whole "
                              "verses and repeat the chorus rather than padding "
                              "existing lines.")
                    continue
                logger.warning("shipping a %d-line lyric for a %ss slot after %d "
                               "attempts — the song will end early",
                               got_lines, duration_s, i)
            if i > 1:
                logger.info("songwriter recovered on attempt %d", i)
            if not caption_is_structured(style):
                logger.warning("shipping an unstructured caption after %d "
                               "attempts — the song will be vaguer than asked", i)
            return {"lyrics": lyrics, "style": style}
        logger.warning("songwriter attempt %d produced no usable lyrics/style "
                       "(%d chars back)", i, len(raw))
        if ctx is not None and getattr(ctx, "is_cancelled", None) and ctx.is_cancelled():
            break

    raise SongwritingFailed("the songwriting model did not return a usable "
                            "lyrics/style pair after %d attempts" % len(attempts))


def _parse_song_json(raw: Optional[str]) -> tuple[str, str]:
    """Best-effort extraction of {"lyrics", "style"} from the LLM's reply.

    Uses the project's hardened JSON extractor (utils.safe_json_from_llm)
    when available, since raw LLM output routinely comes wrapped in code
    fences or trailing commentary; falls back to a bare json.loads.
    """
    if not raw:
        return "", ""
    data = None
    try:
        import utils as _utils
        data = _utils.safe_json_from_llm(raw)
    except Exception:
        data = None
    if not isinstance(data, dict):
        try:
            data = json.loads(raw)
        except Exception:
            data = None
    if not isinstance(data, dict):
        return "", ""
    lyrics = str(data.get("lyrics") or "").strip()
    style = str(data.get("style") or "").strip()
    return lyrics, style


# --------------------------------------------------------------------------- #
# Generating
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# YuE2 (second engine, picked per user). Its model card wants a different
# input shape than Music3:
#   * section tags in Title case from a small set ([Intro] [Verse] [Pre-Chorus]
#     [Chorus] [Bridge] [Interlude] [Outro]), no numbers, a blank line between;
#   * the style as a SHORT comma-separated tag list ("City Pop, upbeat, groovy
#     bass, electric guitar, synth"), not Music3's three-heading prose caption.
# Length follows the lyric volume, as with Music3, so the line budget holds.
# --------------------------------------------------------------------------- #
_YUE2_TAG = {"intro": "Intro", "verse": "Verse", "pre-chorus": "Pre-Chorus",
             "chorus": "Chorus", "post-chorus": "Chorus", "bridge": "Bridge",
             "instrumental": "Interlude", "solo": "Interlude", "outro": "Outro"}
_TAG_LINE_RE = re.compile(r"^\[([a-z-]+)(?:\s+\d+)?\]$")


def yue2_lyrics(lyrics: str) -> str:
    """Our normalized lyric -> YuE2's section vocabulary."""
    out, intro = [], False
    for line in normalize_lyrics(lyrics).splitlines():
        m = _TAG_LINE_RE.match(line.strip())
        if m and m.group(1) in _YUE2_TAG:
            if out and out[-1]:
                out.append("")
            out.append(f"[{_YUE2_TAG[m.group(1)]}]")
            intro = _YUE2_TAG[m.group(1)] == "Intro"
        elif not intro:
            # [Intro] stays empty: the model card's rule, words under it are
            # crammed into the instrumental opening
            out.append(line)
    return "\n".join(out).strip()


_CAPTION_VERBS = "starts|builds|ends|moves|adds|drops|opens|closes|becomes|shifts|features|uses"


def yue2_style(style: str, max_chars: int = 300) -> str:
    """Music3's caption -> a short tag list: the short fragments of each sentence."""
    text = re.sub(r"(?i)\b(%s)\s*[:,]?" % "|".join(map(re.escape, CAPTION_HEADINGS)), ". ", style or "")
    # Any other "Label:" the songwriter invents (Basic Attributes:, Application
    # Scenarios & Imagery:) is a heading too, not a tag (live 2026-09-25).
    text = re.sub(r"(^|[.;,\n])\s*[A-Z][\w&/ -]{2,40}:", ". ", text)
    tags, seen = [], set()
    for frag in re.split(r"[.;,\n]+|\bwith\b|\band\b", text):
        frag = re.sub(r"^((the|a|an|it|%s)(\s+|$))+" % _CAPTION_VERBS, "", frag.strip(), flags=re.I).strip(" -")
        if not frag or len(frag.split()) > 6 or frag.lower() in seen:
            continue
        seen.add(frag.lower())
        tags.append(frag)
    out = ", ".join(tags)
    return out if len(out) <= max_chars else out[:max_chars].rsplit(",", 1)[0]


YUE2_PYTHON = _config.venv_python(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                               "venv_yue2"))
YUE2_ODE_STEPS = _cfg_env.env_int("YUE2_ODE_STEPS", 16)
YUE2_TIMEOUT = 1500          # measured 235-810 s for one song; long lyrics run longer


def _has_weights(folder: str) -> bool:
    """config.json and a weight file: one model.safetensors or the shards
    (model-00001-of-0000N.safetensors) a hub download may bring instead."""
    import glob
    return (os.path.isfile(os.path.join(folder, "config.json"))
            and any(os.path.getsize(p) > 0 for p in glob.glob(os.path.join(folder, "*.safetensors"))))


def yue2_available() -> bool:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return (os.path.isfile(YUE2_PYTHON)
            and _has_weights(os.path.join(root, "models_ext", "YuE2-3B"))
            and _has_weights(os.path.join(root, "models_ext", "YuE2-Vae")))


def run_gpu_worker(ctx, python: str, script: str, job: dict, label: str, timeout: int,
                   cmd=None, env=None) -> tuple:
    """Run one render script in its own venv with the card to itself.

    The job goes in as a JSON file; the worker's output goes to a temp LOG FILE,
    not a pipe -- an unread pipe fills on progress bars and hangs the render.
    Returns (finished, log_tail). Raises MusicUnavailable("cancelled") on Stop.
    """
    import subprocess
    import tempfile
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False)
    log_path = f.name[:-5] + ".log"
    finished = False
    try:
        with comfy_client._gpu_slot(exclusive=True, label=label), open(log_path, "wb") as log:
            argv = [a.replace("{job}", f.name) for a in cmd] if cmd else [python, os.path.join(root, "scripts", script), f.name]
            proc = subprocess.Popen(argv, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            # From launch, not from the call: a second song queued behind the
            # first spent its whole timeout waiting for the card and was killed
            # 100 s into its own render (live 2026-09-27 17:35).
            t0 = time.time()
            while proc.poll() is None:
                if ctx is not None and getattr(ctx, "is_cancelled", None) and ctx.is_cancelled():
                    proc.kill()
                    # reaped before the GPU slot is released: a worker still
                    # dying holds its VRAM, and the next job would start on it
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
                    _MUSIC_FAILURE.update({"reason": "cancelled", "detail": "cancelled"})
                    raise MusicUnavailable("cancelled")
                if time.time() - t0 > timeout:
                    logger.error("%s: timed out after %ds, killed", label, timeout)
                    proc.kill()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
                    break
                time.sleep(1)
            finished = proc.returncode == 0
            if not finished:
                logger.error("%s worker exit code %s after %.0fs", label, proc.returncode, time.time() - t0)
        with open(log_path, "rb") as fh:
            # 20 KB, not 800 chars: the AR stage's "(truncated)" line sits
            # mid-log and the caller reads it (_generate_yue2)
            tail = fh.read()[-20000:].decode("utf-8", "replace")
    finally:
        for tmp in (f.name, log_path):
            try:
                os.unlink(tmp)          # our own temp files
            except OSError:
                pass
    return finished, tail


_CPP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models_ext", "wsl_src")
YUE2_CPP_EXE = os.path.join(_CPP_DIR, "yue2.cpp-master", "build", "yue-synth.exe")
YUE2_CPP_MODEL = os.path.join(_CPP_DIR, "gguf", "YuE2-3B-BF16.gguf")
YUE2_CPP_VAE = os.path.join(_CPP_DIR, "gguf", "YuE2-Vae-F32.gguf")


def yue2_cpp_available() -> bool:
    return all(os.path.isfile(p) for p in (YUE2_CPP_EXE, YUE2_CPP_MODEL, YUE2_CPP_VAE))


# Russian stress: YuE2 sings its learned stress (гнИлое) whatever we send.
# Tried 2026-09-28, all by ear, none moved it: acute accent, capital vowel,
# an LLM for homographs (worse), a long note on the stressed syllable, the
# stressed syllable on the downbeat (ABC edited, yue-plan/"abc" request), and
# akanye phonetic spelling (the Suno trick). Research: bench/song_stress.py.


def _drop_last_section(yue_lyrics: str) -> str:
    """YuE2 lyrics without their last section; "" when two or fewer are left."""
    parts = [p for p in yue_lyrics.split("\n\n") if p.strip()]
    return "\n\n".join(parts[:-1]) if len(parts) > 2 else ""


def _generate_yue2(ctx, lyrics: str, style: str, seed: int) -> str:
    """One YuE2 render with the card to itself.

    yue2.cpp BF16 first (2026-09-27: user picked it by ear over our PyTorch
    path, 7-21 s faster per song); the PyTorch venv is the fallback.
    """
    cpp = yue2_cpp_available()
    if not cpp and not yue2_available():
        _MUSIC_FAILURE.update({"reason": "unavailable", "detail": "YuE2 is not installed"})
        raise MusicUnavailable("YuE2 is not installed")
    ext = "mp3" if cpp else "flac"
    out = str(OUTPUT_DIR / f"song_yue2_{int(time.time() * 1000)}.{ext}")
    job = {"lyrics": yue2_lyrics(lyrics), "style": yue2_style(style), "seed": int(seed), "out": out}
    logger.info("YuE2 (%s): seed %d, style: %s", "cpp bf16" if cpp else "torch", seed, job["style"])
    t0 = time.time()
    for attempt in (1, 2):
        tail = _render_yue2_once(ctx, cpp, job, seed)
        # The token budget (24576) ran out before the end: the song stops
        # mid-line. Rendering the same words again repeats it (the plan is the
        # same), so the retry sings one section less.
        if "(truncated)" not in tail and "TRUNCATED" not in tail:
            break
        shorter = _drop_last_section(job["lyrics"])
        if attempt == 2 or not shorter:
            logger.warning("YuE2: truncated, delivering as is")
            break
        logger.warning("YuE2: truncated, rendering again without the last section")
        job["lyrics"] = shorter
    if not _valid_audio_file(out):
        logger.error("YuE2 produced no file: %s", tail[-800:])
        _MUSIC_FAILURE.update({"reason": "server_error", "detail": "YuE2 produced no file"})
        raise MusicUnavailable("YuE2 produced no file")
    _master(out, fade=("(truncated)" in tail or "TRUNCATED" in tail))
    logger.info("YuE2: %s in %.0fs", os.path.basename(out), time.time() - t0)
    return out


def _master(path: str, fade: bool = False) -> None:
    """In place: loudness to -14 LUFS with true peak under -1 dBTP (YuE2's raw
    output level wanders song to song), and a 2 s fade-out on a song the token
    budget cut off mid-line. No ffmpeg or a failed pass -> the file as it was."""
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        return
    af = "loudnorm=I=-14:TP=-1:LRA=11"
    if fade:
        try:
            import soundfile as sf
            dur = sf.info(path).duration
            af += f",afade=t=out:st={max(0.0, dur - 2):.2f}:d=2"
        except Exception:
            pass
    root, ext = os.path.splitext(path)
    tmp = f"{root}.master{ext}"
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path,
                            "-af", af, "-ar", "44100", tmp],
                           capture_output=True, timeout=120,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode == 0 and _valid_audio_file(tmp):
            os.replace(tmp, path)
        else:
            logger.warning("YuE2 mastering skipped: %s", r.stderr[-300:])
    except Exception as exc:
        logger.warning("YuE2 mastering skipped: %s", exc)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)          # our own temp file


def _render_yue2_once(ctx, cpp: bool, job: dict, seed: int) -> str:
    out = job["out"]
    if cpp:
        # 16 ODE steps, not the reference 32: NAR was 50 of 99 s on a 3.6-min
        # song (bench 2026-10-02, $TEMP/yueab); yue2.cpp's own default is 16.
        job.update({"cot": "full", "lm_seed": int(seed), "steps": YUE2_ODE_STEPS})
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join([os.path.dirname(YUE2_CPP_EXE),
                                       os.path.join(os.environ.get("CUDA_PATH", ""), "bin"), env["PATH"]])
        cmd = [YUE2_CPP_EXE, "--model", YUE2_CPP_MODEL, "--vae", YUE2_CPP_VAE,
               "--request", "{job}", "--out", out]
        _, tail = run_gpu_worker(ctx, "", "", job, "YuE2", YUE2_TIMEOUT, cmd=cmd, env=env)
    else:
        _, tail = run_gpu_worker(ctx, YUE2_PYTHON, "yue2_render.py", job, "YuE2", YUE2_TIMEOUT)
    return tail


def generate_music(ctx, lyrics: str, style: str, *, duration_s: int = 60,
                   seed: Optional[int] = None,
                   preset: Optional[str] = None,
                   on_progress: Optional[Callable] = None, steps: Optional[int] = None,
                   engine: Optional[str] = None) -> str:
    """Render one song. Returns the local path to the resulting audio file.

    `engine="yue2"` renders with YuE2 instead of Music3 (same lyric and caption).

    Raises MusicUnavailable if the weights are missing, ComfyUI lacks the
    node, or the render produces no usable file.
    """
    _MUSIC_FAILURE.update({"reason": "server_error", "detail": ""})

    if (engine or MUSIC_ENGINE) == "yue2":
        if not (lyrics or "").strip() and not (style or "").strip():
            _MUSIC_FAILURE.update({"reason": "empty_prompt", "detail": "no lyrics or style were given"})
            raise MusicUnavailable("no lyrics or style were given")
        if ctx is not None and hasattr(ctx, "set_stage"):
            ctx.set_stage("Composing a song")
        final = _generate_yue2(ctx, lyrics, style,
                               seed if seed and int(seed) > 0 else random.randint(1, 2**31 - 1))
        apply_fade(final)
        if ctx is not None:
            ctx.last_music_path = final
        return final

    ok, why = engine_available(ctx, preset)
    if not ok:
        _MUSIC_FAILURE.update({"reason": "unavailable", "detail": why})
        logger.error("music engine unavailable: %s", why)
        raise MusicUnavailable(why)

    if not (lyrics or "").strip() and not (style or "").strip():
        _MUSIC_FAILURE.update({"reason": "empty_prompt",
                               "detail": "no lyrics or style were given"})
        raise MusicUnavailable("no lyrics or style were given")

    duration_s = max(1, min(int(duration_s or MUSIC_DEFAULT_SECONDS), MUSIC_MAX_SECONDS))
    ceiling_s = render_ceiling(duration_s)
    if seed is None or int(seed) < 1:
        seed = random.randint(1, 2**31 - 1)

    _p = preset if (preset in WEIGHT_PRESETS) else DEFAULT_PRESET
    logger.info("Music3[%s]: asked %ds (ceiling %ds), seed %d, lyrics %d chars, style %d chars",
                _p, duration_s, ceiling_s, seed, len(lyrics or ""), len(style or ""))
    # The words themselves, so a song that comes back as "мяу-мяу" can be
    # checked against what was actually asked of the singer.
    logger.info("Music3 lyrics: %s", " / ".join((lyrics or "").splitlines())[:600])
    logger.info("Music3 style: %s", (style or "")[:400])
    if ctx is not None and hasattr(ctx, "set_stage"):
        ctx.set_stage("Composing a song")

    wf = build_workflow(lyrics, style, duration_s=ceiling_s, seed=seed,
                        preset=preset, steps=steps)

    _t_render = time.time()
    out = comfy_client._submit_and_poll(ctx, wf, timeout=MUSIC_JOB_TIMEOUT,
                                        label="Music3", on_progress=on_progress,
                                        validate=_valid_audio_file, exclusive=True,
                                        job_timeout=MUSIC_JOB_TIMEOUT)
    if out:
        # The turbo-default path (steps was None) stamped its real 8-step
        # count into the graph inside build_workflow -- read it back rather
        # than clamp_steps(None or MUSIC_STEPS), which would log a false 20
        # and corrupt eta_seconds()'s rate history with an 8-step render's
        # wall time mislabeled as a 20-step one.
        _actual_steps = int(wf[N_SAMPLER]["inputs"]["steps"])
        record_render_time(_p, duration_s, time.time() - _t_render, _actual_steps)
    if not out:
        if ctx is not None and getattr(ctx, "is_cancelled", None) and ctx.is_cancelled():
            _MUSIC_FAILURE.update({"reason": "cancelled", "detail": "cancelled"})
            raise MusicUnavailable("cancelled")
        _MUSIC_FAILURE.update({"reason": "server_error",
                               "detail": "the render produced no file"})
        raise MusicUnavailable("the render produced no file")

    final = _adopt_output(out)
    # Faded AFTER adopting, so the taper is written to OUR copy and ComfyUI's
    # own output tree is left byte-for-byte as the engine produced it.
    apply_fade(final)
    if ctx is not None:
        ctx.last_music_path = final
    return final


def apply_fade(path: str, fade_out_s: Optional[float] = None,
               fade_in_s: Optional[float] = None) -> bool:
    """Taper the start/end of a rendered song in place. True if it was changed.

    Music3 renders up to `max_duration` and stops there whether or not the
    song has finished, so a song whose words outlast its slot is severed
    mid-phrase at full volume — measured across three 30s renders, every one
    ended at 29.99s with a last-250ms peak of 0.57-0.999, where a piece that
    actually resolved would taper toward zero. Prompting for an [outro] helps
    the model AIM at an ending but cannot guarantee it lands inside the slot,
    so the taper is applied deterministically here as well.

    Raised-cosine rather than linear: a linear ramp is audible as a mechanical
    "turn the volume down", while the cosine's soft shoulders read as a
    natural decay.

    Never raises — a song that failed to fade is still a song, and losing a
    40s render over a post-processing detail would be a much worse outcome
    than an abrupt ending.
    """
    fo = _config.MUSIC_FADE_OUT_S if fade_out_s is None else fade_out_s
    fi = _config.MUSIC_FADE_IN_S if fade_in_s is None else fade_in_s
    if fo <= 0 and fi <= 0:
        return False
    try:
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(path, always_2d=True)
        n = len(data)
        if n == 0:
            return False
        # Never fade more than a third of the song from either end: on a very
        # short clip a 2s tail would eat most of the audio.
        n_out = min(int(fo * sr), n // 3)
        n_in = min(int(fi * sr), n // 3)
        if n_out <= 0 and n_in <= 0:
            return False
        if n_out > 0:
            ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, n_out)))
            data[-n_out:] *= ramp[:, None]
        if n_in > 0:
            ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_in)))
            data[:n_in] *= ramp[:, None]
        sf.write(path, data, sr)
        logger.info("faded song: %.2fs in / %.2fs out (%s)",
                    n_in / sr, n_out / sr, os.path.basename(path))
        return True
    except Exception as exc:
        logger.warning("could not fade %s (%s) — delivering it unfaded",
                       os.path.basename(path), exc)
        return False


def _adopt_output(path: str) -> str:
    """Copy the song out of ComfyUI's output tree into ours.

    The body used to live here, byte-identical to video._adopt_output — and only
    video's copy carried the explanation of why its filename-collision guard
    exists, so this one read like padding that could be tidied away. Both now
    call comfy_client.adopt_output, which owns the logic and the reasoning.

    Kept as a named wrapper rather than inlining the call: it is the seam this
    module's behaviour is patched at, and it pins the artifact label.

    OUTPUT_DIR is passed explicitly and read HERE, at call time. The suites
    rebind `music.OUTPUT_DIR` to a temp directory; a delegation that let the
    helper read its own copy instead ignored that and wrote fixture songs into
    the user's real output tree (caught by test_music_generation's "the file was
    adopted into OUTPUT_DIR" check).
    """
    return comfy_client.adopt_output(path, "song", OUTPUT_DIR)
