import base64
import copy
import hashlib
import json
import logging
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Sequence

logger = logging.getLogger("assistant.utils")


def free_process_memory() -> None:
    """Release process memory held by our own Python side. Used by clear-context so
    wiping the conversation actually returns RAM/VRAM instead of just dropping
    references. Best-effort: runs a GC pass, empties the Torch CUDA allocator cache
    (if Torch is loaded), and clears our in-memory image caches. The LM Studio model
    weights live in a separate process and are not touched here."""
    import gc
    import sys
    gc.collect()
    try:
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:
        logger.debug("CUDA cache clear skipped: %s", exc)
    try:
        image_mod = sys.modules.get("image")
        if image_mod is not None:
            for cache_name in ("_INVENTORY_CACHE",):
                cache = getattr(image_mod, cache_name, None)
                if isinstance(cache, dict):
                    cache.clear()
    except Exception as exc:
        logger.debug("Image cache clear skipped: %s", exc)
    gc.collect()


# Transient files the pipeline drops into OUTPUT_DIR every turn/edit. They are
# never referenced again after the session, but accumulate forever (the dir is
# also the project's tests/ folder). `_ref_*.wav` is deliberately NOT listed —
# it is a content-digest cache for converted custom reference voices.
_TRANSIENT_PATTERNS = (
    "_redraw_mask_*.png", "_capture_*.jpg", "_pasted_*.png",
    "_working_input_*.jpg", "_assistant_reply_*.wav",
    "_debug_camera_*.wav", "seg_*.wav",
)


def cleanup_runtime_artifacts(directory, max_age_days: float = 7.0) -> int:
    """Delete transient pipeline artifacts in `directory` older than
    `max_age_days`. Returns how many files were removed. Never raises."""
    removed = 0
    try:
        cutoff = time.time() - max_age_days * 86400
        # The image pipeline's scratch markers (crop tiles, QA cutouts, FireRed
        # tiles -- never delivered, see image.is_intermediate_artifact) were
        # missing here, so they piled up: 8.9k files / 2.4 GB in runtime/.
        # A user upload is kept whatever its format, not only .jpg.
        try:
            from image import _SCRATCH_MARKERS
        except Exception:
            _SCRATCH_MARKERS = ()
        patterns = _TRANSIENT_PATTERNS + ("_working_input_*",) + tuple(
            f"*{m}*" for m in _SCRATCH_MARKERS)
        for pattern in patterns:
            for f in Path(directory).glob(pattern):
                try:
                    if f.is_file() and f.stat().st_mtime < cutoff:
                        f.unlink()
                        removed += 1
                except OSError:
                    continue
    except Exception as exc:
        logger.warning("Artifact cleanup failed: %s", exc)
    if removed:
        logger.info("Artifact cleanup: removed %d old file(s)", removed)
    return removed


# ── model control-token grammar ──────────────────────────────────────────────
# Chat models leak their channel/control tokens into `content` when a template
# mismatches or a stream is cut. The original code only knew ONE spelling,
# `<|channel>thought…<channel|>` (Gemma 4), so every other variant reached the
# user verbatim — e.g. an answer that began "<|channel>>Here is a list of…".
# Worse, the harmony spelling puts REASONING in a named channel, so a narrow
# rule leaks the model's private thinking, not merely an ugly token.
#
# Known spellings:
#   Gemma 4  <|channel>thought …reasoning… <channel|> answer <channel|>
#   harmony  <|start|>assistant<|channel>analysis<|message|> …reasoning…
#                              <|channel>final<|message|> answer <|return|>
#   degenerate residue: a bare "<|channel>", "<|channel>>", "assistantfinal"
_REASONING_CHANNEL = r"(?:thought|thinking|analysis|commentary|reflection|critic)"

# A reasoning channel runs until the next channel/control marker or end-of-text.
# A tool call ENDS the reasoning block too — without it in the lookahead, a
# reasoning block with no closing marker swallows the tool call that follows it.
_CHANNEL_BLOCK_RE = re.compile(
    rf"<\|channel>\s*{_REASONING_CHANNEL}\b[\s\S]*?"
    rf"(?=<\|channel>|<channel\|>|<\|start\|>|<\|end\|>|<\|return\|>|<\|tool_call>|$)",
    re.IGNORECASE,
)
# Muse Glimmer through LM Studio (runtime 2.44.0, 2026-09-24): the reasoning is
# NOT split into reasoning_content. It arrives inline as
#   " to=self<|message|> …echo of the question + reasoning… <|eom|>answer"
# and the residual-token rule below deleted only the markers, so the bot sent
# "to=selfUser says "привет". Need respond in Russian…Привет!" to a user. A
# block with no <|eom|> ran out of budget while thinking: all of it is reasoning.
_TO_SELF_BLOCK_RE = re.compile(
    r"(?:^|(?<=\s))to=self\s*<\|message\|>[\s\S]*?(?:<\|eom\|>|<\|end\|>|$)",
    re.IGNORECASE)
# Everything up to and including the LAST "final" marker is preamble, not answer.
_CHANNEL_FINAL_RE = re.compile(
    r"[\s\S]*<\|channel>\s*final\s*(?:<\|message\|>)?", re.IGNORECASE)
# Residual control tokens in either spelling, plus any ">" left dangling by a
# "<|channel>>" (the exact shape that reached the user).
#
# tool_call is DELIBERATELY excluded. It is not decoration: <|tool_call> and
# <tool_call|> delimit a machine-readable call that llm.extract_gemma4_tool_calls
# still has to parse, and strip_textual_tool_calls needs to remove as a whole
# block. Deleting the delimiters here left the bare body `call:search{query:
# <|"|>…<|"|>}` behind — unparseable as a call, invisible to the leak stripper,
# and sent to the user verbatim instead of the search actually running.
_CONTROL_TOKEN_RE = re.compile(
    r"<\|(?!tool_call\b)[a-z_]{1,24}\|?>+"   # <|channel> <|message|> <|end|> <|channel>>
    r"|<(?!tool_call\b)[a-z_]{1,24}\|>",     # <channel|>
    re.IGNORECASE,
)
_ASSISTANT_FINAL_RE = re.compile(r"^\s*assistant\s*final\b\s*", re.IGNORECASE)

# Salvage for an UNTERMINATED reasoning block. Gemma 4 sometimes opens
# `<|channel>thought` and never emits a closing marker, so the block rule runs to
# end-of-string and takes a finished document down with it (observed: a 13.8k
# reply whose Markdown report was deleted wholesale). A top-level Markdown
# heading, a bold heading, or a numbered/bulleted list that then continues for
# several lines is document structure, not thinking — reasoning prose does not
# open with "# ". Only consulted when stripping left NOTHING, so it can never
# resurrect reasoning that sat alongside a real answer.
#
# HEADINGS ONLY. This first also accepted numbered and bulleted lists, and that
# was a serious mistake: reasoning is FULL of lists ("- the user wants X", "1.
# check the sources"), so the salvage happily shipped 13k characters of the
# model's private thinking as a research report — English headings, first-person
# deliberation, the lot. A "# " at the start of a line is document structure;
# a dash is not. If the tail has no heading, returning empty is correct — the
# caller's retry ladder then gets a chance to produce a real answer.
_SALVAGE_START_RE = re.compile(r"^#{1,3} \S", re.MULTILINE)
_SALVAGE_MIN_CHARS = 400


def _salvage_unterminated(block: str) -> str:
    """Recover a document that a runaway reasoning block swallowed, or ""."""
    best = ""
    for m in _SALVAGE_START_RE.finditer(block):
        cand = block[m.start():].strip()
        if len(cand) >= _SALVAGE_MIN_CHARS:
            best = cand
            break
    if not best:
        return ""
    logger.warning(
        "Salvaged %d chars of answer from an unterminated reasoning block "
        "(model opened <|channel>thought and never closed it)", len(best))
    return best


def strip_control_tokens(text: str) -> str:
    """Remove model channel/control-token grammar, keeping only the answer.

    Fails SAFE: if the rules would consume everything, fall back to deleting the
    tokens alone, so a real answer is never turned into an empty reply.
    """
    if not text or "<|" not in text and "|>" not in text and "assistantfinal" not in text.lower():
        return text
    original = text
    # An unterminated block (one that runs to end-of-string) is the only case
    # that can swallow a finished answer, so remember it for salvage below.
    _runaway = ""
    for _m in _CHANNEL_BLOCK_RE.finditer(text):
        if _m.end() >= len(text):
            _runaway = _m.group(0)
    text, had_reasoning = _CHANNEL_BLOCK_RE.subn("", text)
    text, _self_n = _TO_SELF_BLOCK_RE.subn("", text)
    had_reasoning += _self_n
    if _CHANNEL_FINAL_RE.search(text):
        text = _CHANNEL_FINAL_RE.sub("", text)
    text = _CONTROL_TOKEN_RE.sub("", text)
    text = _ASSISTANT_FINAL_RE.sub("", text)
    if not text.strip() and _runaway:
        salvaged = _salvage_unterminated(_runaway)
        if salvaged:
            return _ASSISTANT_FINAL_RE.sub("", _CONTROL_TOKEN_RE.sub("", salvaged))
    if not text.strip() and not had_reasoning:
        # Nothing but scaffolding, and no reasoning block was identified — keep
        # whatever survives token removal rather than returning an empty answer.
        # When a reasoning block WAS removed, empty is the correct result: the
        # model only thought (matching <think> handling), and restoring it here
        # would show the user reasoning that was meant to stay hidden.
        text = _ASSISTANT_FINAL_RE.sub("", _CONTROL_TOKEN_RE.sub("", original))
    return text


def strip_think_tags(text: str) -> str:
    if not text:
        return text
    # Channel/control grammar first: the closing marker also acts as an
    # end-of-turn token appended to the real answer, so the block must go before
    # the tokens — stripping tokens alone would leave the reasoning behind.
    text = strip_control_tokens(text)
    # Qwen/generic <think>…</think> and Gemma <thought>…</thought> blocks
    text = re.sub(r'<\s*(?:think|thought)[^>]*>.*?<\s*/\s*(?:think|thought)\s*>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<\s*(?:think|thought)[^>]*>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    # orphan closing </think> or </thought> (opening was a prefill): drop leading reasoning
    text = re.sub(r'^.*?<\s*/\s*(?:think|thought)\s*>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\n\s*\n', '\n\n', text).strip()
    return text


# Qwen-style textual tool-call markup. When the tool round budget is exhausted
# (tools no longer offered), a tool-happy model may write its next "call" as
# plain text — that must never reach the user as the final answer.
_TEXTUAL_TOOL_CALL_RE = re.compile(
    r"<\s*tool_call\s*>.*?(?:<\s*/\s*tool_call\s*>|\Z)"
    r"|<\|tool_call\>.*?(?:<tool_call\|>|\Z)"
    r"|\[TOOL_REQUEST\].*?(?:\[END_TOOL_REQUEST\]|\Z)"
    r"|\[TOOL_CALLS?\].*?(?:\[/?(?:END_)?TOOL_CALLS?\]|\Z)"
    r"|<function=.*?(?:</function>|\Z)"
    # An UNWRAPPED call body. The <|"|> string delimiter is what makes this safe
    # to match: it is model markup that never occurs in prose, so this cannot eat
    # a sentence that merely contains the word "call:".
    r"|call:\w+\{[^{}]*<\|\"\|>[\s\S]*?\}"
    # Orphan delimiter left by a truncated stream (the opener/closer pair above
    # needs both halves; _CONTROL_TOKEN_RE no longer sweeps these up).
    r"|<\|tool_call\>|<tool_call\|>",
    re.DOTALL | re.IGNORECASE,
)


# A ReAct-style call written as a bare JSON object. Live, 2026-09-12: after
# the picture was already drawn the model's closing message was
#   { "action": "generate_image", "action_input": "{'description': ...}" }
#   Вот твой синий слон на пляже!
# and the voice note read the JSON aloud ("плюс экшен, плюс инпут…"). Only
# objects whose FIRST key is one of the call-shaped keys are swept, so a
# JSON answer the user asked for is left alone.
_JSON_CALL_KEYS = ("action", "tool", "tool_name", "function", "name", "tool_call", "function_call")
_JSON_CALL_RE = re.compile(
    r"\{\s*\"(?:%s)\"\s*:\s*\"[\w.\-]+\"\s*,\s*\"(?:action_input|arguments|args|input|parameters|params|tool_input)\"\s*:"
    % "|".join(_JSON_CALL_KEYS), re.IGNORECASE)


def _strip_json_calls(text: str) -> str:
    out, i = [], 0
    while True:
        m = _JSON_CALL_RE.search(text, i)
        if not m:
            out.append(text[i:]); break
        # Walk to the matching closing brace of the object that starts at m.start().
        depth, j, in_str, esc = 0, m.start(), False, False
        while j < len(text):
            ch = text[j]
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_str = False
            elif ch == '"': in_str = True
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    j += 1; break
            j += 1
        out.append(text[i:m.start()]); i = j
    return "".join(out)


def strip_textual_tool_calls(text: str) -> str:
    """Remove leaked plain-text tool-call markup from a final answer."""
    if not text or ("<" not in text and "{" not in text and "[" not in text):
        return text
    cleaned = _TEXTUAL_TOOL_CALL_RE.sub("", text)
    if "{" in cleaned:
        cleaned = _strip_json_calls(cleaned)
    cleaned = cleaned.strip()
    if cleaned != text.strip():
        logger.warning("Stripped a textual tool-call leak from the final answer")
    return cleaned


# Headers some fine-tunes (e.g. Qwen3.6-35B "Aggressive") emit when they leak
# reasoning as plain, untagged prose that strip_think_tags cannot catch.
_REASONING_LEAK_HEADERS = (
    "thinking process:",
    "here's a thinking process",
    "here is a thinking process",
    "here is my thinking",
    "let me think",
)
_ANSWER_MARKER_RE = re.compile(
    r'(?:\*+\s*)?(?:final answer|answer)\s*[:\-]\s*\**\s*',
    re.IGNORECASE,
)


def strip_reasoning_leak(text: str) -> str:
    """Best-effort cleanup of *untagged* reasoning leaks for final answers.

    Conservative by design: only rewrites when the text clearly starts with a
    known reasoning header AND contains an explicit answer marker. In every other
    case the (tag-stripped) text is returned unchanged, so a genuine answer is
    never accidentally discarded. The real fix for leak-prone models is model
    choice (see config.MODEL_NAME); this is a safety net.
    """
    text = strip_think_tags(text)
    if not text:
        return text
    head = text.lstrip()[:40].lower()
    if not head.startswith(_REASONING_LEAK_HEADERS):
        return text
    # The model reasoned out loud — try to recover an explicit final answer.
    matches = list(_ANSWER_MARKER_RE.finditer(text))
    if matches:
        recovered = text[matches[-1].end():].strip()
        if recovered:
            logger.info("Recovered final answer from a reasoning leak")
            return recovered
    logger.warning("Reasoning leak detected with no recoverable answer marker")
    return text


# Sentence boundary that also splits "…Shemyakin.I could not…" — the observed
# leak had NO whitespace between the two copies, so \s+ alone would miss it.
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?…])\s*')
_MIN_REPEAT_CHARS = 25


def _norm_for_repeat(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip().lower()


def collapse_verbatim_repetition(text: str) -> str:
    """Drop a sentence/line that repeats the one immediately before it verbatim.

    A model that stalls near the end of a short answer emits it twice — observed
    live as "I could not find any information about X.I could not find any
    information about X.". The streaming repetition guard cannot catch this: it
    samples every LLM_REPEAT_CHECK_EVERY chars, and a doubled one-liner is far
    below that.

    Deliberately narrow, because repetition is sometimes intentional (chorus,
    emphasis, a list with a repeated stem):
      · only ADJACENT duplicates, never a sentence echoed later on;
      · only units of >= 25 characters;
      · code fences are left completely alone.
    """
    if not text or "```" in text:
        return text

    out_lines = []
    prev_line_key = None
    for line in text.split("\n"):
        parts = [p for p in _SENTENCE_SPLIT_RE.split(line) if p]
        kept: list = []
        for part in parts:
            key = _norm_for_repeat(part)
            if (kept and len(key) >= _MIN_REPEAT_CHARS
                    and key == _norm_for_repeat(kept[-1])):
                logger.warning("Collapsed a verbatim repeated sentence in the answer")
                continue
            kept.append(part)
        rebuilt = " ".join(p.strip() for p in kept) if len(kept) != len(parts) else line

        key = _norm_for_repeat(rebuilt)
        if (prev_line_key is not None and key == prev_line_key
                and len(key) >= _MIN_REPEAT_CHARS):
            logger.warning("Collapsed a verbatim repeated line in the answer")
            continue
        prev_line_key = key if key else None
        out_lines.append(rebuilt)

    return "\n".join(out_lines)


def load_transcription_cache(cache_file: Path) -> dict:
    if not cache_file.exists():
        return {}
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception as exc:
        logger.warning("Failed to load transcription cache: %s", exc)
    return {}


def audio_hash_from_path(audio_path: str) -> str:
    file_size = os.path.getsize(audio_path)
    with open(audio_path, "rb") as f:
        if file_size <= 2 * 1024 * 1024:
            data = f.read()
        else:
            data = f.read(1024 * 1024)
            f.seek(max(file_size - 1024 * 1024, 0), os.SEEK_SET)
            data += f.read(1024 * 1024)
    # Mix in the file size: for >2MB files we hash only the head+tail, so two
    # clips sharing those bytes but differing in the middle (or in length) would
    # otherwise collide and serve each other's cached transcription.
    h = hashlib.md5(data)
    h.update(str(file_size).encode("ascii"))
    return h.hexdigest()


def safe_empty_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def throttle_external_calls(ctx) -> None:
    """Space out outbound calls made through `ctx`.

    Not every caller hands us a full `models.Context`. Test harnesses and the
    lighter-weight contexts built by some tools carry only the handful of
    attributes they need, and a context with no throttling state has nothing to
    throttle *against* — pacing is per-context, so there is no shared clock to
    respect. Treat that as a no-op instead of raising.

    This used to be an AttributeError, and the failure was invisible: the only
    hot caller (`llm.py`) catches pre-dispatch errors, logs them and returns
    None, so a context missing `api_lock` silently turned every model call in
    that code path into "no answer" while the suite still exited 0.
    """
    if ctx is None:
        return
    lock = getattr(ctx, "api_lock", None)
    if lock is None:
        return
    with lock:
        now = time.time()
        interval = getattr(ctx, "api_min_interval", 0.0) or 0.0
        wait_for = interval - (now - getattr(ctx, "last_api_call_time", 0.0))
        if wait_for > 0:
            time.sleep(wait_for)
        ctx.last_api_call_time = time.time()


def ensure_required_files() -> None:
    from config import WEIGHTS_PATH, VOCAB_PATH, DC_REF_WAV, WORKFLOW_FIRERED_EDIT_PATH, OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = []
    for path, label in [
        (WEIGHTS_PATH, "weights"),
        (VOCAB_PATH, "vocab"),
        (DC_REF_WAV, "reference wav"),
        (WORKFLOW_FIRERED_EDIT_PATH, "workflow"),
    ]:
        if not path.exists():
            missing.append(f"{label}: {path}")
    if missing:
        # a fresh clone has neither (both are private, gitignored): say what to do,
        # not just what is missing (live 10-03: a friend's clone died on it twice)
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing) + "\n\n"
            "weights: put a Russian F5-TTS checkpoint at that path (see README).\n"
            "reference wav: a clean 5-15 s recording of the voice to speak with, at that\n"
            "path or set ASSISTANT_REF_WAV=C:\\path\\to\\voice.wav")


def truncate_text(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (truncated)"


def sanitize_history(messages: List[dict]) -> List[dict]:
    """Strip tool-call scaffolding from a *persisted* history before it is reused.

    Tool calls and their results are one-shot within the turn that issued them.
    Keeping them in long-lived history is fragile: windowed trimming can drop an
    assistant ``tool_calls`` message while keeping its ``tool`` result (or vice
    versa), and the OpenAI-compatible API rejects a ``tool`` message that does not
    immediately follow the matching ``tool_calls``. We therefore keep only system
    and user turns plus assistant turns that carry real text and no tool calls.
    """
    out: List[dict] = []
    for m in messages:
        role = m.get("role")
        if role in ("system", "user"):
            out.append(m)
        elif role == "assistant" and not m.get("tool_calls"):
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                out.append(m)
        # dropped: role == "tool", and assistant messages that only carry tool_calls
    return out


def trim_messages(messages: List[dict], max_non_system_messages: int = 12,
                  max_chars: int = 0, min_keep: int = 8) -> List[dict]:
    if not messages:
        return []
    head = messages[:1] if messages[0].get("role") == "system" else []
    tail = messages[len(head):][-max_non_system_messages:]
    if max_chars:
        # A count alone dropped «кошка Муся» after 17 one-line turns that fit easily.
        total = sum(len(str(m.get("content") or "")) for m in tail)
        while len(tail) > min_keep and total > max_chars:
            total -= len(str(tail[0].get("content") or ""))
            tail = tail[1:]
    return copy.deepcopy(head + tail)


def image_bytes_to_data_url(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def downscale_image_bytes(image_bytes: bytes, max_edge: int = 1536) -> bytes:
    """Shrink an over-large sent image so its longest edge is <= max_edge.
    A 12MP phone photo otherwise becomes a multi-MB vision payload and an
    OOM-prone full-res CLIPSeg/VAE encode in the inpaint path, with no quality
    benefit (the vision model and the old model both work well under ~1.5K px).
    Returns the original bytes unchanged on any failure, or if already small AND
    already in a format the vision endpoint accepts.

    The format check is not cosmetic. This function feeds the analysis proxy, and
    the proxy is encoded as a data URL whose mime defaults to image/jpeg — so a
    SMALL .webp used to short-circuit here and get sent as webp bytes labelled
    JPEG. The vision server cannot decode that, the call fails, and the retry
    ladder re-sends the same undecodable payload: the ".webp retry storm" in the
    logs. Large webp never showed the bug because the resize path already
    re-encoded to JPEG."""
    # Formats the OpenAI-style vision payload is safe to carry as-is.
    _SAFE = {"JPEG", "PNG"}
    try:
        from PIL import Image
        import io as _io
        with Image.open(_io.BytesIO(image_bytes)) as im:
            w, h = im.size
            fmt = (im.format or "").upper()
            if max(w, h) <= max_edge and fmt in _SAFE:
                return image_bytes
            if max(w, h) <= max_edge:
                # Small but in an unsafe format: transcode only, keep the pixels.
                new_size = (w, h)
                im = im.convert("RGB")
                logger.info("Transcoded sent image %s -> JPEG for the vision payload", fmt or "?")
            else:
                scale = max_edge / float(max(w, h))
                new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
                im = im.convert("RGB").resize(new_size, Image.LANCZOS)
                logger.info("Downscaled sent image %dx%d -> %dx%d", w, h, *new_size)
            buf = _io.BytesIO()
            im.save(buf, "JPEG", quality=92)
            return buf.getvalue()
    except Exception as exc:
        logger.warning("Image downscale failed (using original): %s", exc)
    return image_bytes


def file_to_data_url(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/png"
    with open(image_path, "rb") as f:
        data = f.read()
    return image_bytes_to_data_url(data, mime_type=mime_type)


def iter_json_objects(raw: str):
    """Yield every complete top-level {...} object in `raw`, in order.

    Brace matching is string-aware: a brace inside a string value ("a sign
    reading {OPEN") must not open or close an object, or the scan runs past the
    end and yields nothing at all.
    """
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(raw or ""):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(raw[start:i + 1], strict=False)
                except Exception:
                    obj = None
                if isinstance(obj, dict):
                    yield obj
                start = -1
            elif depth < 0:
                depth = 0


def safe_json_from_llm(text: str, required_keys: Optional[Sequence[str]] = None
                       ) -> Optional[dict]:
    """Pull the JSON object out of an LLM reply that may be fenced or chatty.

    This used to slice from the FIRST '{' to the LAST '}' and parse that. Any
    reply with a brace after the object — trailing prose ("use {} if unsure"), or
    the duplicate ```json block these models like to append — spanned two objects
    and failed to parse, so a perfectly good answer was thrown away and the caller
    fell back. Measured on real replies, not hypothetical.

    Now: scan every complete object and prefer the one carrying the most of
    `required_keys`. Pass `required_keys` wherever the caller knows the shape it
    wants — without it, a stray `{}` earlier in the reply can win.
    """
    if not text:
        return None
    text = strip_think_tags(text)
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    keys = tuple(required_keys or ())
    candidates = []
    try:
        whole = json.loads(cleaned, strict=False)
    except Exception:
        pass
    else:
        if isinstance(whole, dict):
            candidates.append(whole)
    candidates.extend(iter_json_objects(cleaned))
    if not candidates:
        return None
    if not keys:
        # No shape to go on: the first non-empty object beats a leading `{}`.
        for c in candidates:
            if c:
                return c
        return candidates[0]
    best = max(candidates, key=lambda c: sum(k in c for k in keys))
    return best if any(k in best for k in keys) else None


def explicit_lang(text: str) -> str:
    """"en"/"ru" when the request names the output language («на английском»),
    else "" -- the model's read (agent/intent.py), not a word list."""
    if not (text or "").strip():
        return ""
    import intent
    return intent.read(None, text)["reply_language"]
