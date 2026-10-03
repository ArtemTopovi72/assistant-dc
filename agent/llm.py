import json
import os
import config as _cfg_env   # env_int/env_float: a bad .env value falls back, never crashes the import
import tempfile
import json as _json
import logging
import threading
import time
from typing import List, Optional

import requests

from config import (LM_STUDIO_URL, LM_STUDIO_BASE, MODEL_NAME, LLM_MAX_RETRIES, LLM_RETRY_BASE_DELAY,
                    LLM_CONNECT_TIMEOUT, LLM_STREAM_STALL_TIMEOUT,
                    LLM_STREAM_MAX_SECONDS, LLM_STREAM_MAX_CHARS,
                    LLM_REPEAT_CHECK_EVERY, LLM_REPEAT_TAIL_CHARS, LLM_REPEAT_CYCLE_MAX,
                    LLM_REPEAT_MIN_SPAN, LLM_CARD_WAIT_S)
from utils import (strip_think_tags, strip_control_tokens, file_to_data_url,
                   image_bytes_to_data_url, throttle_external_calls)

logger = logging.getLogger("assistant.llm")

# Qwen3 soft switch that suppresses the reasoning channel.
_NO_THINK_MARKER = "/no_think"

_VALID_EFFORT = ("low", "medium", "high")


def _is_gpt_oss(model_name: str) -> bool:
    """True for OpenAI gpt-oss models (gpt-oss-20b / gpt-oss-120b / 'GPT-120').
    These reason via a `reasoning_effort` LEVEL, not the Qwen on/off switch.
    """
    m = (model_name or "").lower().replace("_", "-")
    return ("gpt-oss" in m or "gptoss" in m or "gpt-120" in m or "gpt-20b" in m)


GLIMMER_THINK_HEADROOM = _cfg_env.env_int("GLIMMER_THINK_HEADROOM", 768)
GLIMMER_THINK_HEADROOM_HIGH = _cfg_env.env_int("GLIMMER_THINK_HEADROOM_HIGH", 4096)


def _is_glimmer(model_name: str) -> bool:
    """Meta Muse Glimmer (2026-08). Its card: reasoning is set by a line
    `Reasoning strength: low|medium|high|xhigh` in the system prompt; none of
    the other families' switches apply."""
    return "glimmer" in (model_name or "").lower()


def _glimmer_strength(messages: List[dict], strength: str) -> List[dict]:
    """Copy of `messages` with the strength line at the head of the system turn."""
    line = f"Reasoning strength: {strength}"
    out = [dict(m) for m in messages]
    if out and out[0].get("role") == "system":
        body = str(out[0].get("content") or "")
        if not body.startswith("Reasoning strength:"):
            out[0]["content"] = line + "\n\n" + body
    else:
        out.insert(0, {"role": "system", "content": line})
    return out


def _is_gemma4(model_name: str) -> bool:
    """True for Gemma 4 models, which reason by a different contract again.

    Not a reasoning LEVEL (gpt-oss) and not an on/off switch (Qwen): Gemma 4
    thinks on every turn and the only documented control is a `<|think|>` token
    at the head of the system prompt. Measured 2026-07-29 on gemma-4-12b-qat:
    `reasoning_effort`, `reasoning: "off"` and `enable_thinking` are all no-ops —
    five configurations returned byte-identical output. Nothing here tries to
    turn its reasoning up; it is already up. This flag exists to stop us sending
    the OTHER families' switches at it and to give it room to finish.
    """
    return "gemma" in (model_name or "").lower()


# Gemma 4 spends 300-800 tokens thinking before it answers, scaling with the
# question (~90 on "capital of France", ~800 on a multi-step word problem), and
# the thinking is billed against max_tokens. Our chat budgets are 700-1500, so a
# hard question plus a real answer runs out mid-sentence — which reads exactly
# like the model being stupid, and is the single most likely cause of it. This
# is a FLOOR, not a target: the model stops when it is done and nothing is spent
# on the easy turns.
GEMMA_MIN_TOKENS = 3000

# Gemma 4's "thinking disabled" shape, sent as an assistant prefill: the thought
# block is already open AND closed, so the model's first token is the answer.
# There is no request knob for this (see _is_gemma4), but the prefill is.
# Measured 2026-09-23 on hauhaucs gemma4-26b-a4b q5_k_m, the forced closing
# reply after a failed tool chain, 3 runs each: without it 377-652 tokens of
# reasoning before the answer (and live, 2026-09-22, 3 x 3000 tokens of pure
# reasoning with NO answer -- the user got a sentence cut off mid-word); with
# it 157-158 tokens, ~2 s, a direct honest answer every time. Only for turns
# that summarise work already done -- it removes the thinking, not just the
# budget.
GEMMA_NO_THINK_PREFILL = "<|channel>thought\n<channel|>"


# Every Gemma 4 call gets that prefill (send_to_lm_studio: reasoning is off
# for every call, with no switch to turn it back on).

# The sampling profile Google's card specifies for Gemma 4 "across all use
# cases". NOT APPLIED — measured against our own temperature on 45 questions x 2
# repeats and it made no difference at all (89/90 both ways). Documented here so
# the next person does not have to re-derive it, and so the decision not to use
# it is visible. If a harder eval ever separates them, this is the profile to
# try — and note that any structured-output caller (layout JSON, critique
# verdicts) asks for <= 0.2 deliberately and must keep its own temperature.
GEMMA_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}


def _reasoning_effort(ctx) -> str:
    eff = (getattr(ctx, "reasoning_effort", "high") or "high").lower()
    return eff if eff in _VALID_EFFORT else "high"


import re as _re

# Gemma 4 channel format (from model card google/gemma-4-31B-it-qat-q4_0-unquantized):
#   thinking enabled:  <|channel>thought\n[reasoning]<channel|>[answer]<channel|>
#   thinking disabled: <|channel>thought\n<channel|>[answer]<channel|>
# Strip full thinking blocks (not just tokens) from stored history so the
# reasoning never gets replayed to other models or corrupts compaction summaries.


def strip_channel_tokens(text: str) -> str:
    """Remove model thinking blocks and channel/control tokens.

    Delegates to utils.strip_control_tokens so there is ONE definition of the
    grammar: this used to carry its own narrower copy that knew only the
    `<|channel>thought` spelling, and the two drifted apart.
    """
    if not text:
        return text
    return strip_control_tokens(text).strip()


# ── Gemma 4 textual tool-call formats ────────────────────────────────────────
# Gemma 4 emits tool calls inline in content in (at least) two formats:
#   Format A:  <|tool_call>call:NAME{KEY:<|"|>VAL<|"|>}<tool_call|>
#   Format B:  [TOOL_REQUEST]\n{"name":"NAME","arguments":{...}}\n[END_TOOL_REQUEST]
# Both are detected and converted to the standard OpenAI tool_calls array.
_G4_CALL_RE = _re.compile(
    r'<\|tool_call\>call:(\w+)\{(.*?)\}<tool_call\|>',
    _re.DOTALL,
)
# Format B: bracket-delimited JSON block
_G4_BRACKET_RE = _re.compile(
    r'\[TOOL_REQUEST\]\s*(.*?)\s*\[END_TOOL_REQUEST\]',
    _re.DOTALL | _re.IGNORECASE,
)
# Format C: the same body with NO delimiters — `call:NAME{...}`. Seen in the wild
# when an upstream stripper removed the <|tool_call> wrapper (the call then
# reached the user as text instead of running), and emitted bare by some
# fine-tunes. Gated on the <|"|> string delimiter, which never occurs in prose,
# so an ordinary sentence containing "call:" cannot match.
_G4_BARE_RE = _re.compile(r'call:(\w+)\{([^{}]*<\|"\|>[\s\S]*?)\}')
# Format D: CLOSING delimiter only, plain-quoted args —
#   call:calculate{expression: "17 * 23 - 91"}<tool_call|>
# Measured live on gemma-4-12b-qat 2026-07-29: this is what it emits when the
# system prompt describes tools. It has no `<|tool_call>` opener (so Format A
# misses it) and plain quotes rather than <|"|> (so Format C misses it), which
# meant the call was neither executed NOR stripped — the raw markup reached the
# user in place of an answer, and the tool never ran. Gated on the closing
# delimiter, which never occurs in prose.
_G4_CLOSED_RE = _re.compile(r'call:(\w+)\{([^{}]*)\}\s*<tool_call\|>')
# String value (delimited by <|"|>…<|"|>) — used by format A
_G4_STR_RE  = _re.compile(r'(\w+):<\|"\|>(.*?)<\|"\|>', _re.DOTALL)
# Numeric or boolean value after strings are removed — format A
# `\s*` after the colon: Format D writes `max_results: 3` with a space, and
# without this the key parsed as absent — the tool then ran with a silently
# missing argument, which is worse than not parsing the call at all.
_G4_NUM_RE  = _re.compile(r'(\w+)\s*:\s*(-?\d+(?:\.\d+)?)\b')
_G4_BOOL_RE = _re.compile(r'(\w+)\s*:\s*(true|false)\b', _re.IGNORECASE)


# Plain double-quoted value: `expression: "17 * 23 - 91"`. Only consulted after
# the <|"|>-delimited form, which stays authoritative where both could match.
_G4_PLAIN_STR_RE = _re.compile(r'(\w+)\s*:\s*"([^"]*)"')


def _parse_gemma4_args(args_str: str) -> dict:
    args: dict = {}
    for m in _G4_STR_RE.finditer(args_str):
        args[m.group(1)] = m.group(2)
    rest = _G4_STR_RE.sub("", args_str)
    for m in _G4_PLAIN_STR_RE.finditer(rest):
        args.setdefault(m.group(1), m.group(2))
    rest = _G4_PLAIN_STR_RE.sub("", rest)
    for m in _G4_BOOL_RE.finditer(rest):
        args.setdefault(m.group(1), m.group(2).lower() == "true")
    for m in _G4_NUM_RE.finditer(rest):
        if m.group(1) not in args:
            v = m.group(2)
            args[m.group(1)] = float(v) if "." in v else int(v)
    return args


def extract_gemma4_tool_calls(content: str):
    """Parse Gemma 4 textual tool-call markup out of content (all known formats).

    Returns (tool_calls_list, remaining_text). If no markup is found,
    returns ([], original content).
    """
    import json as _json
    tc_list = []
    idx = 0

    # Format A: <|tool_call>call:NAME{...}<tool_call|>
    for m in _G4_CALL_RE.finditer(content):
        name = m.group(1)
        args = _parse_gemma4_args(m.group(2))
        tc_list.append({
            "id": f"call_g4_{idx}",
            "type": "function",
            "function": {"name": name, "arguments": _json.dumps(args, ensure_ascii=False)},
        })
        idx += 1
    remaining = _G4_CALL_RE.sub("", content).strip()

    # Format B: [TOOL_REQUEST] {"name":..., "arguments":{...}} [END_TOOL_REQUEST]
    for m in _G4_BRACKET_RE.finditer(remaining):
        try:
            obj = _json.loads(m.group(1))
            name = obj.get("name") or obj.get("function") or ""
            args = obj.get("arguments") or obj.get("parameters") or {}
            if name and isinstance(args, dict):
                tc_list.append({
                    "id": f"call_g4_{idx}",
                    "type": "function",
                    "function": {"name": name, "arguments": _json.dumps(args, ensure_ascii=False)},
                })
                idx += 1
        except Exception:
            pass
    remaining = _G4_BRACKET_RE.sub("", remaining).strip()

    # Format D: `call:NAME{...}<tool_call|>` — closing delimiter, plain quotes.
    # Runs BEFORE the bare Format C so the closing tag is consumed with its call
    # instead of being left behind as a stray token in the visible text.
    for m in _G4_CLOSED_RE.finditer(remaining):
        args = _parse_gemma4_args(m.group(2))
        tc_list.append({
            "id": f"call_g4_{idx}",
            "type": "function",
            "function": {"name": m.group(1),
                         "arguments": _json.dumps(args, ensure_ascii=False)},
        })
        idx += 1
    remaining = _G4_CLOSED_RE.sub("", remaining).strip()

    # Format C: bare `call:NAME{...}` (wrapper lost or never emitted)
    for m in _G4_BARE_RE.finditer(remaining):
        args = _parse_gemma4_args(m.group(2))
        tc_list.append({
            "id": f"call_g4_{idx}",
            "type": "function",
            "function": {"name": m.group(1),
                         "arguments": _json.dumps(args, ensure_ascii=False)},
        })
        idx += 1
    remaining = _G4_BARE_RE.sub("", remaining).strip()

    # Format F: `[TOOL_CALL]generate_image(description='…')` — a Python call.
    # Live 2026-09-29, 1 draw in 3 on the house model: the markup was the reply.
    for name, args, span in _py_calls_in(remaining):
        tc_list.append({
            "id": f"call_g4_{idx}",
            "type": "function",
            "function": {"name": name, "arguments": _json.dumps(args, ensure_ascii=False)},
        })
        idx += 1
        remaining = remaining.replace(span, "")
    remaining = remaining.strip()

    # Format E: a JSON call in prose — `<tool_call>{"name": "generate_image",
    # "arguments": {...}}</tool_call>` or the bare object. Live 2026-09-13
    # (mega run 3, steps 72/75/77): three drawing turns in a row came back
    # this way, utils.strip_textual_tool_calls deleted the block as a leak,
    # and the user got "Вот ваш логотип" with no picture. A call the model
    # wrote out is still a call: parse it, run it.
    for name, args, span in _json_calls_in(remaining):
        tc_list.append({
            "id": f"call_g4_{idx}",
            "type": "function",
            "function": {"name": name,
                         "arguments": _json.dumps(args, ensure_ascii=False)},
        })
        idx += 1
    if tc_list:
        remaining = _JSON_CALL_WRAP_RE.sub("", _strip_json_call_objects(remaining)).strip()

    return tc_list, remaining


_PY_CALL_START_RE = _re.compile(r"\[TOOL_CALLS?\]\s*(\w+)\(", _re.IGNORECASE)
_PY_CALL_END_RE = _re.compile(r"\s*\[/?(?:END_)?TOOL_CALLS?\]", _re.IGNORECASE)


def _py_calls_in(text: str):
    """(name, kwargs, matched span) for each `[TOOL_CALL]name(k='v', …)`.
    The closing paren is the first one after which the call parses."""
    import ast
    out = []
    for m in _PY_CALL_START_RE.finditer(text):
        for j in (k for k, ch in enumerate(text) if ch == ")" and k >= m.end()):
            try:
                call = ast.parse(text[m.start(1):j + 1], mode="eval").body
            except SyntaxError:
                continue
            if not isinstance(call, ast.Call):
                break
            try:
                kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg}
            except ValueError:
                break
            end = _PY_CALL_END_RE.match(text, j + 1)
            out.append((m.group(1), kwargs, text[m.start():end.end() if end else j + 1]))
            break
    return out


_JSON_CALL_START_RE = _re.compile(
    r'\{\s*"(?:name|tool|function|tool_name|function_name)"\s*:\s*"([\w.\-]+)"\s*,\s*'
    r'"(?:arguments|args|parameters|params|input|tool_input|action_input)"\s*:',
    _re.IGNORECASE)
_JSON_CALL_WRAP_RE = _re.compile(r"<\s*/?\s*tool_call\s*>|```(?:json)?|```", _re.IGNORECASE)


def _json_object_end(text: str, start: int) -> int:
    """Index just past the object that opens at text[start] ('{'), or -1."""
    depth, j, in_str, esc = 0, start, False, False
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
                return j + 1
        j += 1
    return -1


def _json_calls_in(text: str):
    """Yield (name, args, (start, end)) for every JSON tool call in `text`."""
    import json as _json
    i = 0
    while True:
        m = _JSON_CALL_START_RE.search(text, i)
        if not m:
            return
        end = _json_object_end(text, m.start())
        if end < 0:
            return
        i = end
        try:
            obj = _json.loads(text[m.start():end])
        except Exception:
            continue
        name = obj.get("name") or obj.get("tool") or obj.get("function") or \
            obj.get("tool_name") or obj.get("function_name") or ""
        args = None
        for k in ("arguments", "args", "parameters", "params", "input", "tool_input", "action_input"):
            if k in obj:
                args = obj[k]; break
        if isinstance(args, str):
            try:
                args = _json.loads(args)
            except Exception:
                args = {"input": args}
        if name and isinstance(args, dict):
            yield name, args, (m.start(), end)


def _strip_json_call_objects(text: str) -> str:
    out, i = [], 0
    for _n, _a, (a, b) in list(_json_calls_in(text)):
        out.append(text[i:a]); i = b
    out.append(text[i:])
    return "".join(out)


def _strip_model_artifacts(messages: List[dict]) -> List[dict]:
    """Remove Gemma 4 thinking blocks, channel markers, and stray <thought> tags from outbound messages.

    One documented exception: an assistant turn that CARRIES A TOOL CALL keeps
    its thinking. The Gemma 4 card is explicit — thoughts from previous turns
    must not be replayed, "with the exception of tool call turns where thinking
    content should be preserved". That reasoning is the model's record of why it
    called the tool; stripping it leaves the next round looking at a bare call
    with no rationale, which is how a multi-step tool sequence turns incoherent.
    """
    out = []
    for m in messages:
        content = m.get("content")
        if m.get("tool_calls"):
            out.append(m)
            continue
        if isinstance(content, str) and (
                "<|" in content or "|>" in content or "<thought" in content.lower()
                or "assistantfinal" in content.lower()):
            m = dict(m)
            m["content"] = strip_think_tags(strip_channel_tokens(content))
        out.append(m)
    return out


def _apply_no_think(messages: List[dict]) -> List[dict]:
    """Return a copy of messages with the Qwen3 `/no_think` switch in a system turn.

    Does not mutate the caller's list. If a system message exists, the marker is
    appended to it; otherwise a minimal system message is prepended.
    """
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "system":
            content = m.get("content")
            if isinstance(content, str) and _NO_THINK_MARKER not in content:
                m["content"] = content.rstrip() + " " + _NO_THINK_MARKER
            return out
    out.insert(0, {"role": "system", "content": _NO_THINK_MARKER})
    return out


# How many LLM streams are open right now. A render that wants the whole card
# must not pull the model out from under one of them: live, 2026-09-12, two
# users drew at once, the first claim unloaded the model while the second
# user's layout planner was still streaming ("Model unloaded"), the revive
# then loaded 20 GB back INTO the render's card, and both the render and
# every LLM call after it crawled for twenty minutes.
_inflight = 0
_inflight_cv = threading.Condition()


def _inflight_enter() -> None:
    global _inflight
    with _inflight_cv:
        _inflight += 1


def _inflight_exit() -> None:
    global _inflight
    with _inflight_cv:
        _inflight = max(0, _inflight - 1)
        _inflight_cv.notify_all()


def inflight() -> int:
    with _inflight_cv:
        return _inflight


def wait_no_inflight(timeout: float) -> bool:
    """Block until no LLM stream is open, or `timeout` seconds pass."""
    deadline = time.monotonic() + max(0.0, timeout)
    with _inflight_cv:
        while _inflight > 0:
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            _inflight_cv.wait(min(left, 1.0))
        return True


def _wait_for_free_card(ctx) -> None:
    """Wait out a render that has been given the whole card.

    A render evicts the chat model (comfy_client._gpu_slot, exclusive mode).
    Calling LM Studio now would JIT-load 20 GB into what the render left over:
    the load itself crawls, and it steals the card back from the very job the
    caller is waiting on. So wait for it, and SAY so -- an unexplained pause
    reads as a hung bot, which is how this looked before the stage existed.

    A thread that is itself holding the render slot never waits: it would be
    waiting for itself.
    """
    try:
        import comfy_client as _cc
        if _cc.this_thread_holds_card():
            return
        busy = _cc.card_is_exclusive()
        if not busy:
            return
        if ctx is not None and hasattr(ctx, "set_stage"):
            ctx.set_stage("Waiting for the graphics card")
        logger.info("chat turn waiting: the card is rendering %s", busy)
        if not _cc.wait_for_card(LLM_CARD_WAIT_S):
            logger.warning("waited %ds for the card and gave up - calling anyway",
                           LLM_CARD_WAIT_S)
    except Exception:
        # Never let VRAM etiquette swallow a turn.
        logger.exception("card wait failed - proceeding with the call")


def send_to_lm_studio(
    ctx,
    messages: List[dict],
    tools: Optional[List[dict]] = None,
    tool_choice: str = "auto",
    temperature: float = 0.5,
    max_tokens: int = 1500,
    prefill: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    force_think: bool = False,
    json_schema: Optional[dict] = None,
) -> Optional[dict]:
    """
    Sends messages to LM Studio. Returns the full message dict from the assistant
    (may include 'content' and/or 'tool_calls').
    Returns None on error.

    prefill: optional assistant-turn prefix. Passing "<think></think>" reliably
    suppresses the reasoning channel on reasoning-prone fine-tunes that ignore the
    /no_think switch (the closed think block forces generation to start after it).
    Do not combine with tools. IGNORED for gpt-oss (harmony uses channels, not
    <think> tags — a literal "<think></think>" assistant turn is junk there).

    reasoning_effort: per-call override of ctx.reasoning_effort for gpt-oss
    (low/medium/high). Use "low" for big writing/summarization calls so the model
    spends its token budget on the OUTPUT instead of burning it all on the analysis
    channel and returning empty content.
    """
    _wait_for_free_card(ctx)
    model_name = getattr(ctx, "model_name", "") or MODEL_NAME
    # Reasoning is OFF for every call, here, in the one function every model
    # call goes through -- not per caller, not by the thinking switch, not by
    # force_think. Live 2026-10-01 16:51: with the switch on, a 7-second кружок
    # spent 2813 tokens and 26.7 s on six storyboard lines and each zoomed tile
    # 300-750 tokens against a 350 budget; a per-call opt-out had let it through.
    no_think = True
    force_think = False
    is_oss = _is_gpt_oss(model_name)
    is_gemma = _is_gemma4(model_name)
    is_glimmer = _is_glimmer(model_name)

    # force_think: deliberately let a reasoning fine-tune think (long-form synthesis
    # writes far better WITH chain-of-thought). Suppressing it on these models (the
    # /no_think marker + reasoning:"off" + a <think></think> prefill, all at once) makes
    # them emit an UNCLOSED <think> block that strip_think_tags reduces to empty — the
    # exact cause of empty report sections. In force_think mode we send a clean prompt
    # (no marker, no prefill, reasoning ON) and rely on strip_think_tags to remove the
    # CLOSED think block, keeping the real answer. Caller must budget tokens for both.
    if force_think and not is_oss:
        no_think = False
        prefill = None

    if is_gemma:
        # None of the other families' switches mean anything here, and the
        # <think></think> prefill is a Qwen lever — on Gemma it is literal junk
        # prepended to an assistant turn, inside a template that emits its own
        # `<|channel>thought` block regardless. Drop them all rather than send
        # noise, and give the model room to finish thinking AND answer.
        # The ONE exception is Gemma's own empty thought block: see
        # GEMMA_NO_THINK_PREFILL. With it there is no thinking to budget for.
        # Thinking OFF everywhere (see the top of this function). The coding
        # gap that once justified an exception (13/13 vs 11/13) was the
        # harness, not the reasoning: no-think delivered 6/6 on the bench.
        no_think = False
        # A forced tool call (tool_choice="required") is grammar-constrained,
        # and the grammar rejects the empty thought block as its first piece:
        # live 2026-09-23 the video button failed 6/6 with "Failed to
        # initialize samplers: Unexpected empty grammar stack after accepting
        # piece: <|channel>" and the user got "Я задумался и не выдал ответ".
        # Such a call is a tool call and nothing else, so there is no
        # thinking to suppress: send it without the prefill.
        if tools and tool_choice == "required":
            # ANY prefill: the caller's Qwen "<think></think>" dies the same
            # way ("...after accepting piece: think"). That one had broken every
            # forced calculate/deck turn on the tool bench since long before
            # 2026-09-23 -- 10 of Gemma's 30 lost points.
            prefill = None
        else:
            # 2026-09-23, user: "так было всегда и везде". Every call answers
            # at once in the model card's thinking-disabled shape; a plain JSON
            # step had been burning 2000+ reasoning tokens before its answer.
            prefill = GEMMA_NO_THINK_PREFILL
            # The grammar rejects that block as its first piece just like a
            # forced tool call does ("Unexpected empty grammar stack after
            # accepting piece: <|channel>", 51 times in the live log by
            # 2026-10-02: every json_schema call failed all 3 attempts). The
            # answer still comes without thinking; callers parse it with
            # utils.safe_json_from_llm.
            json_schema = None

    # Strip Gemma 4 <channel|> artifacts that may have been saved to history.
    messages = _strip_model_artifacts(messages)
    # gpt-oss ignores the Qwen `/no_think` marker — don't pollute its messages with it.
    outbound = messages if is_oss else (_apply_no_think(messages) if no_think else messages)
    if is_glimmer:
        # Same policy as everywhere: no reasoning unless the user or the caller
        # asks for it. The Qwen marker and prefill mean nothing to it.
        _wants = False
        outbound = _glimmer_strength(messages, "high" if _wants else "low")
        prefill = None
        # It still thinks at "low" (~100-250 tokens measured), and that is billed
        # against max_tokens: the vision self-test's 80-token budget came back
        # as "Yes, there is a". Headroom on top, so a short budget still means a
        # short ANSWER.
        max_tokens = max_tokens + (GLIMMER_THINK_HEADROOM_HIGH if _wants
                                   else GLIMMER_THINK_HEADROOM)

    # The <think></think> prefill is a Qwen lever; for gpt-oss it injects literal junk
    # into the harmony stream, so skip it there.
    if prefill is not None and not is_oss:
        outbound = list(outbound) + [{"role": "assistant", "content": prefill}]

    payload = {
        "model": model_name,
        "messages": outbound,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        # The last chunk then carries token usage, for runtime/turns (turn_trace).
        "stream_options": {"include_usage": True},
    }
    if json_schema:
        # Grammar-constrained output (LM Studio applies it after the think block):
        # "```json" fences and missing keys become impossible (measured 3/3 vs 0/3 bare).
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "answer", "strict": True, "schema": json_schema}}

    if is_glimmer:
        pass    # the system-prompt line above is its only reasoning control
    elif is_gemma:
        # No reasoning key at all: every one of them measured as a byte-identical
        # no-op on this model, and `reasoning: "off"` in particular is a request
        # it cannot honour.
        #
        # Google's card also specifies temperature 1.0 / top_p 0.95 / top_k 64
        # "across all use cases", and we deliberately do NOT apply it. A/B on 45
        # questions x 2 repeats, sampling the only variable: 89/90 either way,
        # +0.0 points, median reasoning 324 vs 316 tokens. The honest reading is
        # that the set saturates (98.9%) and cannot resolve a difference — not
        # that none exists — so this stays on the caller's temperature rather
        # than changing behaviour everywhere on vendor faith. GEMMA_SAMPLING is
        # kept below as the documented profile to try if a harder eval ever
        # shows a gap. GEMMA_SAMPLING=1 switches it on for exactly that A/B
        # (sandbox_hard is the harder eval).
        if os.getenv("GEMMA_SAMPLING") == "1":
            payload.update(GEMMA_SAMPLING)
    elif is_oss:
        # gpt-oss: control reasoning DEPTH via the harmony `reasoning_effort` level
        # (low/medium/high). LM Studio injects it into the model's system channel.
        # Do NOT send `reasoning: "off"` / enable_thinking here — those are Qwen-isms
        # and would fight the harmony template.
        payload["reasoning_effort"] = (
            "low"   # gpt-oss cannot switch reasoning off; the lowest it has
        )
    elif force_think:
        # Let the reasoning model think; enable the template's thinking path explicitly.
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    else:
        payload["reasoning"] = "off"
        if no_think:
            # Honored by models whose chat template supports it; harmless otherwise.
            payload["chat_template_kwargs"] = {"enable_thinking": False}

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    for attempt in range(LLM_MAX_RETRIES):
        try:
            if attempt > 0:
                # Make the retry visible: without this the GUI keeps showing the
                # previous stage for the whole backoff+re-stream, which reads as
                # a hang rather than a recovery in progress.
                _set_stage = getattr(ctx, "set_stage", None)
                if _set_stage is not None:
                    _set_stage(f"Retrying the model call ({attempt + 1}/{LLM_MAX_RETRIES})")
            try:
                throttle_external_calls(ctx)
            except Exception as exc:
                # This runs BEFORE any request leaves the process, so a failure
                # here is a defect on our side (a context missing api_lock /
                # api_min_interval, say) — never a transient server condition.
                # Retrying multiplies one bug into LLM_MAX_RETRIES identical
                # failures separated by backoff sleeps, which is how a single
                # bad call turned into a "retry storm" in the logs.
                logger.error("LLM call aborted before dispatch — not retryable: "
                             "%s: %s", type(exc).__name__, exc, exc_info=True)
                return None
            # Re-checked per attempt: a render may have claimed the card
            # between two attempts, and the retry must wait like a fresh call.
            if attempt:
                _wait_for_free_card(ctx)
            _inflight_enter()
            try:
                message = _stream_chat(ctx, payload)
            finally:
                _inflight_exit()
            if message is _CANCELLED:
                logger.info("LM Studio stream cancelled by user")
                return None

            # The prompt did not fit. Retrying it unchanged is guaranteed to
            # fail again -- measured on the chaos bench, where a long tool chain
            # burned all three attempts on identical rejections and the round
            # came back None, losing the turn. Make it fit instead.
            if message is _SERVER_ERROR:
                # A bare 500 ("Internal Server Error", no diagnosis) came
                # nine times in a row on 2026-09-15 19:11 after a tool call
                # was cut mid-arguments: the broken call sat in history and
                # the template choked on it every time. Mend what can be
                # mended -- unparsable tool arguments, then size -- and try
                # once more; an untouched payload is simply retried.
                if _sanitize_tool_calls(payload) or _shrink_payload(payload):
                    logger.warning("LM Studio 500 — mended the payload and retrying "
                                   "(attempt %d/%d)", attempt + 1, LLM_MAX_RETRIES)
                    continue
                message = None
            if message is _NO_VISION:
                logger.error("The loaded model %s has no vision; an image request "
                             "cannot succeed -- not retried.", payload.get("model"))
                if ctx is not None:
                    try:
                        ctx.last_llm_error = "no_vision"
                    except Exception:
                        pass
                return None
            if message is _CTX_OVERFLOW:
                if _shrink_payload(payload):
                    logger.warning(
                        "Prompt did not fit the model's context — trimmed it "
                        "and retrying (attempt %d/%d)",
                        attempt + 1, LLM_MAX_RETRIES)
                    continue
                logger.error(
                    "Prompt does not fit the model's context and there is "
                    "nothing left to trim — reload the model with a larger "
                    "context length.")
                return None

            if message is not None:
                content = str(message.get("content", "") or "").strip()

                from_reasoning = False
                if not content and not is_oss:
                    # Some Qwen finetunes inline the whole answer into reasoning_content
                    # with empty content; recover it there. NEVER for gpt-oss — its
                    # reasoning_content is the raw harmony ANALYSIS channel (chain of
                    # thought). Dumping that as the answer produced "reports" that were
                    # literally the model thinking out loud about how to format briefs.
                    content = str(message.get("reasoning_content", "") or "").strip()
                    from_reasoning = bool(content)

                _pre_strip_len = len(content or "")
                content = strip_think_tags(content)
                # The model spent the whole reply thinking and never wrote an
                # answer. Silently returning "" made this look like a dead
                # backend; callers that can retry need to be able to SEE it.
                if _pre_strip_len > 200 and not content.strip():
                    # Name the CALLER. Without it these warnings are unattributable
                    # — a 55-minute research run produced three of them and there
                    # was no way to tell which stage (briefing? verification?
                    # synthesis?) was burning the time. The system prompt's first
                    # line identifies the stage well enough and costs nothing.
                    _who = ""
                    for _m in (outbound or []):
                        if _m.get("role") == "system":
                            _who = " ".join(str(_m.get("content", ""))
                                            .strip().splitlines()[:1])[:70]
                            break
                    logger.warning(
                        "reply was %d chars of reasoning with no final answer "
                        "(model=%s, prompt=%r, budget=%d, finish_reason=%s) — "
                        "returning empty",
                        _pre_strip_len, getattr(ctx, "model_name", "?"),
                        _who, payload.get("max_tokens", 0),
                        message.get("finish_reason") or "?")
                    # Tell the CALLER, not just the log. "Empty" and "the budget
                    # was too small to hold the thinking AND the answer" need
                    # completely different responses: a caller that retries the
                    # same squeeze burns the same minute again for the same
                    # nothing. Callers that can widen the ask look at this.
                    message["reasoning_only"] = True
                    message["reasoning_chars"] = _pre_strip_len

                # Gemma 4 puts tool calls in content as textual markup rather
                # than the standard tool_calls array — convert them on the way out.
                # Two known formats: <|tool_call>...<tool_call|> and
                # [TOOL_REQUEST]{...}[END_TOOL_REQUEST]
                _has_g4_markup = (
                    "<|tool_call>" in content or
                    "[TOOL_REQUEST]" in content.upper() or
                    ("call:" in content and '<|"|>' in content) or
                    # Format D: the closing delimiter is the whole signal — there
                    # is no opener and the args use plain quotes.
                    ("call:" in content and "<tool_call|>" in content) or
                    # Format E: a JSON call written into the prose.
                    bool(_JSON_CALL_START_RE.search(content)) or
                    bool(_PY_CALL_START_RE.search(content))        # Format F
                )
                if _has_g4_markup and from_reasoning:
                    # Markup found in text recovered from the REASONING channel, not
                    # from content. A model deliberating "I should call:search{...}"
                    # has not called anything — Qwen's own parser refuses to read
                    # tool tags before </think> for exactly this reason. Executing it
                    # would run a search/generate the user never got, off a thought
                    # the model may have argued itself out of two lines later.
                    # Logged, not silent: a genuine call lost here would otherwise
                    # look like the model simply doing nothing.
                    logger.warning(
                        "tool-call markup found in reasoning_content — not executed "
                        "(a thought about calling is not a call); %d chars", len(content))
                elif _has_g4_markup and not message.get("tool_calls"):
                    tc_list, remaining = extract_gemma4_tool_calls(content)
                    if tc_list:
                        message["tool_calls"] = tc_list
                        content = remaining

                message["content"] = content
                return message

            logger.error(
                "LM Studio stream produced no message (attempt %d/%d)",
                attempt + 1,
                LLM_MAX_RETRIES,
            )
        except Exception as exc:
            logger.error(
                "LM Studio request failed (attempt %d/%d): %s",
                attempt + 1,
                LLM_MAX_RETRIES,
                exc,
            )

        if _is_cancelled(ctx):
            return None

        if attempt < LLM_MAX_RETRIES - 1:
            time.sleep(LLM_RETRY_BASE_DELAY * (2 ** attempt))

    return None


# Sentinel returned by _stream_chat when the user cancelled mid-stream.
_CANCELLED = object()


_N_CTX_RE = _re.compile(r"n_keep:\s*(\d+)\s*>=\s*n_ctx:\s*(\d+)", _re.IGNORECASE)


# LM Studio rejects an over-long prompt in two different phrasings, and only
# one of them carries numbers. The llama.cpp backend says
# "n_keep: 9001 >= n_ctx: 8192"; the newer path says just "Context size has
# been exceeded." with no figures at all. Only the first was recognised, so on
# the second the shortfall parsed as 0, the self-heal declined to act, the
# hint logged nothing, and the retry loop re-sent the SAME oversized payload
# three times before giving up — which is exactly how a long tool chain lost
# its round to "LM Studio returned None on round 2".
_CTX_ERR_RE = _re.compile(
    r"n_keep:\s*\d+\s*>=\s*n_ctx:|context\s\w+ (?:has been )?exceed"
    r"|exceeds? the (?:model|maximum) context"
    r"|context (?:length|window|size) (?:limit )?(?:exceeded|too small)"
    r"|prompt is too long",
    _re.IGNORECASE,
)

# Returned instead of None when the call failed specifically because the prompt
# did not fit. The caller can then make it FIT, rather than retrying a payload
# that is guaranteed to be rejected again.
_CTX_OVERFLOW = object()
# Returned by _stream_chat on an HTTP 500 with no recognisable diagnosis.
_SERVER_ERROR = object()
# The served model has no vision and the request carries an image. Deterministic:
# retrying sent the same rejection three times in three seconds (log, 2026-09-24).
_NO_VISION = object()
_GRAMMAR_REJECT_RE = _re.compile(r"empty grammar stack|Failed to initialize samplers", _re.I)
_NO_VISION_RE = _re.compile(r"does not support image|image input is not supported", _re.I)


def _sanitize_tool_calls(payload: dict) -> bool:
    """Replace tool-call arguments that are not valid JSON with '{}'.

    A call cut off at the token ceiling leaves half a JSON string in the
    assistant message; the chat template cannot render it and the server
    answers 500 to every later request. True when something was mended."""
    changed = False
    for m in payload.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    json.loads(args)
                    continue
                except Exception:
                    pass
            elif isinstance(args, dict):
                continue
            fn["arguments"] = '{"_truncated": true}'
            changed = True
    return changed


# How much of one tool result to keep when the prompt has to be cut down. A
# tool result is the bulk of a long chain and the least load-bearing part of
# it: the model needs to know what came back, not every byte of it.
_CTX_TOOL_RESULT_CAP = 1200
_TRUNC_NOTE = "\n… [truncated to fit the model's context]"


def _looks_like_context_overflow(body: str) -> bool:
    return bool(_CTX_ERR_RE.search(body or ""))


def _shrink_payload(payload: dict) -> bool:
    """Cut the prompt down so an over-long call can be retried. True if it changed.

    Two passes, gentlest first, and only one pass per call so each retry gives
    up as little as possible:

      1. truncate over-long tool results, oldest first;
      2. drop the oldest exchange outright.

    An assistant message carrying tool_calls and the tool messages answering it
    are dropped TOGETHER. Splitting them leaves a tool result with nothing it
    replies to, which the API rejects outright -- trading a context error for a
    400 is not a repair.
    """
    msgs = payload.get("messages") or []
    if len(msgs) < 4:
        return False

    # Pass 1 — squeeze the tool results, oldest first, sparing the most recent
    # one: that is what the model is reasoning over right now, and it is the
    # one result that has to survive verbatim.
    # Position in the list is the wrong test. Guarding "the last three
    # messages" spared the second tool result of a two-call chain purely
    # because a user message sat behind it, so only ONE result could ever be
    # trimmed and a long single-turn chain -- the case actually observed --
    # had nothing to give.
    _tool_ix = [i for i, m in enumerate(msgs) if m.get("role") == "tool"]
    _spare = set(_tool_ix[-1:])
    for i, m in enumerate(msgs):
        if i not in _spare and m.get("role") == "tool":
            c = str(m.get("content") or "")
            # Already trimmed. Without this the truncated text -- cap plus the
            # note -- is still longer than the cap, so it is cut again, and
            # again: _shrink_payload never reports exhaustion and the caller
            # retries until it runs out of attempts.
            if c.endswith(_TRUNC_NOTE):
                continue
            if len(c) > _CTX_TOOL_RESULT_CAP:
                m["content"] = c[:_CTX_TOOL_RESULT_CAP] + _TRUNC_NOTE
                return True

    # Pass 2 — drop the oldest exchange. The system prompt stays (it carries the
    # persona and the rules) and so do the last four messages.
    start = 1 if msgs and msgs[0].get("role") == "system" else 0
    if len(msgs) - start <= 4:
        return False
    # Drop a whole EXCHANGE -- the oldest user turn and everything answering it
    # -- stopping at the next user message. Dropping single messages orphans a
    # tool result from the assistant turn that called it, and the API rejects
    # that outright: a 400 in place of a context error is not a repair.
    # The exchange boundary is found FIRST, with no clamp: stopping the scan
    # early to protect the tail is what orphaned a tool result -- the block was
    # cut in half, taking the assistant turn and leaving the tool reply behind.
    end = start + 1
    while end < len(msgs) and msgs[end].get("role") != "user":
        end += 1
    # Only then decide whether the whole block can be spared.
    if len(msgs) - (end - start) < start + 2:
        return False
    del msgs[start:end]
    return True


def _context_shortfall(body: str) -> int:
    """Tokens the prompt needed, if `body` is a context rejection; else 0."""
    m = _N_CTX_RE.search(body or "")
    return int(m.group(1)) if m else 0


def _try_heal_context(body: str, payload: dict) -> bool:
    """Reload the model bigger and report whether the call is worth retrying.

    LM Studio JIT-loads a fresh instance at the model's DEFAULT context whenever
    it decides the model is not loaded, so a one-time reload at startup does not
    hold — an 8192 instance reappeared mid-session and silently disabled every
    tool. Repair it where the failure is actually observed.
    """
    need = _context_shortfall(body)
    if not need:
        return False
    try:
        import lmstudio as _lms
        ok, _msg = _lms.heal_context(payload.get("model", ""), need)
        return bool(ok)
    except Exception:
        logger.debug("context self-heal unavailable", exc_info=True)
        return False


_MODEL_GONE = _re.compile(r"No models loaded|model has crashed|model is not loaded|Model reloaded", _re.I)
# The request that killed the model gets one of these; the ones queued behind
# it get "No models loaded".
_MODEL_CRASHED = _re.compile(r"model has crashed|Model reloaded", _re.I)
_revive_lock = __import__("threading").Lock()
_revive_last = 0.0
_revive_ok = False
LLM_REVIVE_MIN_INTERVAL_S = 60.0


def _try_revive_model(body: str, payload: dict) -> bool:
    """Bring the chat model back after LM Studio lost it, and say whether to retry.

    On 2026-09-11 the model crashed mid-session (a vision call on a 1728x2304
    image, loaded into a card ComfyUI had not yet vacated). Nothing was loaded
    afterwards, and every turn for the next hour answered "No models loaded" --
    which the user saw as "I got stuck thinking" on a plain "hello". The
    server itself is fine in this state; only the model is gone, and the app
    already knows how to put it back. Rate-limited so a model that crashes on
    load cannot turn into a reload storm.
    """
    global _revive_last
    if not _MODEL_GONE.search(body or ""):
        return False
    try:
        import comfy_client as _cc
        if _cc.card_is_exclusive():
            # Not gone -- lent to a render, which brings it back when the
            # queue drains. Loading it now would put 20 GB into the render's
            # card; the caller's retry waits for the card instead.
            logger.info("model is unloaded for a render (%s) - not reviving, waiting",
                        _cc.card_is_exclusive())
            return False
        if not _cc._excl_free.is_set():
            # A render is handing the card back and is already reloading the
            # model; a second load raced it (live 2026-09-28, 5-min hang).
            logger.info("render is giving the card back - waiting for its reload")
            if _cc.wait_for_card(600) and _model_served(payload.get("model", "") or MODEL_NAME):
                return True
    except Exception:
        pass
    model = payload.get("model", "") or MODEL_NAME
    global _revive_ok
    # The lock is held for the whole reload: a second caller that hits "No
    # models loaded" while the first is reloading BLOCKS here, then finds a
    # fresh successful revive and simply retries. It used to return False
    # inside the cooldown and lose its turn (live, 2026-09-12: the turn right
    # after a crash always died).
    with _revive_lock:
        now = time.time()
        if now - _revive_last < LLM_REVIVE_MIN_INTERVAL_S:
            # The cooldown is for a model that dies ON LOAD. One that loaded
            # fine and was then killed by a request (live 2026-09-13: a tall
            # photo tripped llama.cpp's n_ubatch assert twice in 20 s) is a
            # new incident -- the cached "ok" would let every turn for the
            # next minute die on "No models loaded".
            if not (_revive_ok and not _model_served(model)):
                return bool(_revive_ok)
            logger.warning("model %s died again inside the revive cooldown - reviving", model)
        _revive_last = now
        _revive_ok = False
        _revive_last = now
        _revive_ok = False
        try:
            import lmstudio as _lms
            logger.warning("LM Studio has no model loaded - bringing %s back", model)
            ok, msg = _lms.ensure_exclusive(LM_STUDIO_BASE, model)
            # lms prints a TUI spinner (ESC sequences + braille) into its
            # output; keep only the last meaningful line for the log.
            msg_txt = _re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]|[⠀-⣿]", "", str(msg))
            lines = [ln.strip() for ln in _re.split(r"[\r\n]+", msg_txt) if ln.strip()]
            logger.warning("model revive %s: %s", "ok" if ok else "FAILED", (lines[-1] if lines else "")[:200])
            _revive_ok = bool(ok)
            return _revive_ok
        except Exception:
            logger.exception("model revive failed")
            return False


def _model_served(model: str) -> bool:
    """Is `model` in the server's loaded list right now? Errors count as served
    so a flaky probe cannot trigger a reload."""
    try:
        import lmstudio as _lms
        ids = _lms.loaded_model_ids(LM_STUDIO_BASE)
        return any(str(i).lower() == str(model).lower() for i in ids)
    except Exception:
        return True


def _halve_payload_images(payload: dict) -> bool:
    """Shrink every image in the payload to half its longest side. True if any.

    A picture whose tokens do not fit one llama.cpp micro-batch crashes the
    model (GGML_ASSERT n_ubatch >= n_tokens, non-causal image attention);
    re-sending it unchanged after the revive killed the model a second time.
    Pixel bounds do not bound tokens for a tiled encoder, so the retry sends
    a smaller picture rather than the same one.
    """
    changed = False
    for msg in payload.get("messages") or []:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not (isinstance(part, dict) and part.get("type") == "image_url"):
                continue
            url = ((part.get("image_url") or {}).get("url") or "")
            if not url.startswith("data:") or "," not in url:
                continue
            try:
                import base64
                import io
                from PIL import Image
                raw = base64.b64decode(url.split(",", 1)[1])
                with Image.open(io.BytesIO(raw)) as im:
                    w, h = im.size
                    if max(w, h) < 64:
                        continue
                    small = im.convert("RGB").resize((max(1, w // 2), max(1, h // 2)), Image.LANCZOS)
                    buf = io.BytesIO(); small.save(buf, "JPEG", quality=90)
                part["image_url"]["url"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
                logger.warning("vision: model crashed on a %dx%d image - retrying at %dx%d", w, h, *small.size)
                changed = True
            except Exception:
                logger.debug("could not shrink a payload image", exc_info=True)
    return changed


def _log_context_hint(body: str, payload: dict) -> None:
    """Turn a context-length rejection into an instruction the user can act on.

    This failure mode is silent and total: with the tool schemas attached the
    prompt no longer fits the loaded context, LM Studio rejects EVERY tool-bearing
    call, and the agent — seeing empty rounds — ends up telling the user the tools
    are unavailable. The number it needs is right there in the error.
    """
    m = _N_CTX_RE.search(body or "")
    if not m:
        # The numberless phrasing ("Context size has been exceeded."). It says
        # nothing about sizes, so this used to log NOTHING at all -- the one
        # message whose whole job is to name the number the user has to raise
        # stayed silent on the phrasing that carries no number. Estimate it
        # instead: ~3 characters per token is close enough to point at the
        # right order of magnitude, which is all the reader needs.
        if _looks_like_context_overflow(body):
            _chars = len(_json.dumps(payload.get("messages") or [],
                                     ensure_ascii=False))
            _chars += len(_json.dumps(payload.get("tools") or [],
                                      ensure_ascii=False))
            logger.error(
                "The prompt does not fit the loaded context (the server did "
                "not say by how much). It is roughly %d tokens, %d of which "
                "are the %d tool schema(s). Reload '%s' in LM Studio with a "
                "larger context length — check `lms ps`, and note that every "
                "tool-bearing call will fail this way until you do.",
                _chars // 3,
                len(_json.dumps(payload.get("tools") or [],
                                ensure_ascii=False)) // 3,
                len(payload.get("tools") or []),
                payload.get("model", "?"))
        return
    need, have = int(m.group(1)), int(m.group(2))
    logger.error(
        "The prompt does not fit the loaded context: needs ~%d tokens, the model "
        "is loaded with %d. Every call carrying the %d tool schema(s) will fail "
        "this way, so the agent cannot use ANY tool. Reload '%s' in LM Studio with "
        "a context length of at least %d (double it for headroom).",
        need, have, len(payload.get("tools") or []),
        payload.get("model", "?"), max(need + 2048, have * 2))


def _is_cancelled(ctx) -> bool:
    fn = getattr(ctx, "is_cancelled", None)
    try:
        return bool(fn and fn())
    except Exception:
        return False


def _looks_degenerate(tail: str) -> tuple:
    """Detect a stuck generation loop from the tail of a stream.

    Returns ``(reason, signature)`` — reason is "" when the text looks fine. The
    signature identifies WHICH repetition was seen, so the caller can require the
    same loop to survive two consecutive checks before aborting (see below).

    Two cheap checks, both on a bounded tail so cost stays flat regardless of
    output length:

    1. Cycle repetition — the tail is the same block repeated back to back.
    2. Line starvation — many lines, almost none distinct (a bullet or heading the
       model re-emits forever), for loops whose period exceeds the cycle search.

    FALSE-POSITIVE GUARDS, learned the hard way — the first version aborted six
    legitimate-looking deep-research briefs in one run:
    * A minimum repeated SPAN (not just a repeat count). Without it, cycle length
      k=1 flags a markdown rule "------------", an ellipsis, or a number like
      1200000 as a loop.
    * The caller requires the SAME signature twice. Source pages legitimately
      contain duplicated entries (afisha/tour listings repeat the same line), and
      a faithful extraction of them is bounded; a genuinely stuck model keeps
      repeating past the next checkpoint. That difference is the real signal —
      repeat count alone cannot separate the two.
    """
    if len(tail) < 400:
        return "", ""
    stripped = tail.strip()
    if not stripped:
        return "", ""
    # 1) exact cycle at the tail
    for k in range(1, min(LLM_REPEAT_CYCLE_MAX, len(stripped) // 4) + 1):
        block = stripped[-k:]
        if not block.strip():
            continue
        reps = 1
        pos = len(stripped) - k
        while pos - k >= 0 and stripped[pos - k:pos] == block:
            reps += 1
            pos -= k
        if reps >= 4 and k * reps >= LLM_REPEAT_MIN_SPAN:
            return (f"{k}-char block repeated {reps}x", f"cycle:{k}:{block[:40]!r}")
    # 2) line starvation
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if len(lines) >= 12:
        distinct = len(set(lines))
        if distinct <= max(2, len(lines) // 10):
            common = max(set(lines), key=lines.count)
            return (f"{len(lines)} lines, only {distinct} distinct",
                    f"lines:{common[:40]!r}")
    return "", ""


def _decode_line(raw) -> str:
    """Decode a raw SSE line safely as UTF-8."""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _stream_chat(ctx, payload: dict):
    """One model request, through our own queue (llm_gate) -- a slot is held
    only while the request runs, and a cancelled turn stops waiting at once."""
    import llm_gate
    import config as _cfg
    if LM_STUDIO_URL != _cfg.LM_STUDIO_URL:
        # Pointed at something else (a suite's fake server): not our queue.
        return _stream_chat_raw(ctx, payload)
    try:
        with llm_gate.slot(ctx):
            return _stream_chat_raw(ctx, payload)
    except llm_gate.Cancelled:
        logger.info("turn cancelled while waiting for a model slot")
        return _CANCELLED


def _stream_chat_raw(ctx, payload: dict):
    """POST a streaming chat completion and assemble the assistant message.

    Reads raw bytes and decodes them manually as UTF-8 to avoid mojibake caused by
    iter_lines(decode_unicode=True) on chunked SSE responses.
    Returns the message dict, None on HTTP/parse error, or the _CANCELLED sentinel
    if the user cancelled.
    """
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    # tool_calls accumulate by index across delta chunks (OpenAI streaming shape).
    tool_acc: dict = {}
    streamed_chars = 0
    last_repeat_check = 0
    # One signature PER CHANNEL. The two-strike rule (same loop still running at
    # the next checkpoint) is per channel too, so a clean content stream cannot
    # reset a loop that is running in the reasoning channel, or vice versa.
    loop_sigs = {"content": "", "reasoning": ""}
    finish_reason = ""
    usage: dict = {}
    _started = time.perf_counter()

    headers = {"Accept": "text/event-stream"}

    # timeout=(connect, read): the READ value is the per-chunk stall timeout — if the
    # server stops emitting tokens for this long mid-stream, requests raises (retriable)
    # instead of the old ~32-minute hang. The wall-clock + char caps below stop a model
    # that keeps trickling/looping tokens forever (where the read timeout never fires).
    deadline = time.monotonic() + LLM_STREAM_MAX_SECONDS
    with requests.post(
        LM_STUDIO_URL,
        json=payload,
        timeout=(LLM_CONNECT_TIMEOUT, LLM_STREAM_STALL_TIMEOUT),
        stream=True,
        headers=headers,
    ) as resp:
        if resp.status_code != 200:
            # The BODY is the diagnosis and used to be thrown away. LM Studio
            # answers a request that does not fit the loaded context with
            # "n_keep: 9905 >= n_ctx: 8192" — logging only "HTTP 400" turned a
            # precise, actionable error into a silent empty turn, which the agent
            # then reported to the user as "I cannot draw, the tools are
            # unavailable". Never discard it again.
            try:
                body = resp.text[:600]
            except Exception:
                body = "<unreadable>"
            logger.error("LM Studio HTTP %d: %s", resp.status_code, body)
            _log_context_hint(body, payload)
            _try_heal_context(body, payload)
            _try_revive_model(body, payload)
            if _looks_like_context_overflow(body):
                return _CTX_OVERFLOW
            if _NO_VISION_RE.search(body):
                return _NO_VISION
            return _SERVER_ERROR if resp.status_code >= 500 else None

        try:
            for raw in resp.iter_lines(decode_unicode=False):
                if _is_cancelled(ctx):
                    resp.close()
                    return _CANCELLED

                # Wall-clock guard: a model stuck in a token loop streams forever — the
                # per-read stall timeout never fires because bytes keep arriving. Cut it off
                # and keep whatever was produced so the turn still completes.
                if time.monotonic() > deadline:
                    logger.warning("LM Studio stream exceeded %.0fs wall-clock cap — aborting "
                                   "(model likely stuck); using %d chars so far",
                                   LLM_STREAM_MAX_SECONDS, streamed_chars)
                    resp.close()
                    break

                if not raw:
                    continue

                line = _decode_line(raw).strip()
                if not line:
                    continue

                # SSE can send comments like ": keep-alive"
                if line.startswith(":"):
                    continue

                if line.startswith("data:"):
                    line = line.split(":", 1)[1].lstrip()

                if line == "[DONE]":
                    break

                try:
                    chunk = json.loads(line)
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
                # An error can arrive INSIDE the stream as {"error": ...} with no
                # "choices" key. That used to fall into the generic `continue` below
                # and the call returned an empty message — indistinguishable from a
                # model that simply said nothing.
                if isinstance(chunk, dict) and chunk.get("usage"):
                    usage = chunk["usage"]
                if isinstance(chunk, dict) and chunk.get("error"):
                    err = chunk["error"]
                    logger.error("LM Studio streamed an error: %s", str(err)[:600])
                    _log_context_hint(str(err), payload)
                    # Fix it and let the caller's retry loop take another run at it,
                    # rather than failing the user's turn over a reloadable server.
                    _try_heal_context(str(err), payload)
                    _try_revive_model(str(err), payload)
                    if _MODEL_CRASHED.search(str(err)):
                        _halve_payload_images(payload)
                    if (_GRAMMAR_REJECT_RE.search(str(err))
                            and payload.get("tool_choice") == "required"):
                        # Forced call + Gemma opening its <|channel> reasoning token:
                        # the tool grammar rejects the very first piece, every retry
                        # (lost TG turn, 2026-09-23 20:34, 5 of 186 forced calls).
                        # Keep the single schema, drop the "required" -- the model
                        # still has exactly one tool to call.
                        payload["tool_choice"] = "auto"
                        logger.warning("forced tool call rejected by the grammar "
                                       "(<|channel>) -- retrying with tool_choice=auto")
                    # Tell the caller WHY, so it can make the prompt fit instead of
                    # re-sending a payload that cannot possibly be accepted.
                    return (_CTX_OVERFLOW if _looks_like_context_overflow(str(err))
                            else None)
                try:
                    _choice = chunk["choices"][0]
                    delta = _choice.get("delta", {}) or {}
                except (KeyError, IndexError, TypeError):
                    continue
                # Why the model stopped. Without it, "the model thought and never
                # answered" is unattributable: a budget it ran out of ("length") and
                # a turn it chose to end ("stop") look identical from here and need
                # opposite fixes.
                if _choice.get("finish_reason"):
                    finish_reason = str(_choice["finish_reason"])

                content = delta.get("content")
                if content:
                    content_parts.append(str(content))
                    streamed_chars += len(content)

                reasoning = delta.get("reasoning_content")
                if reasoning:
                    reasoning_parts.append(str(reasoning))
                    streamed_chars += len(reasoning)

                # Runaway guard: a degenerate repetition loop ("the the the…") streams fast,
                # so neither the read-stall nor wall-clock guard may trip in time. Cap total
                # streamed characters and stop reading — the accumulated text is returned.
                if streamed_chars > LLM_STREAM_MAX_CHARS:
                    logger.warning("LM Studio stream exceeded %d-char cap — aborting (model "
                                   "appears stuck in a generation loop)", LLM_STREAM_MAX_CHARS)
                    resp.close()
                    break

                # Repetition guard: the char/wall-clock caps above are last-resort — they
                # let a stuck model burn the full 600s and 200k chars before giving up
                # (observed: 3 consecutive 600s stalls in one deep-research run). Sample
                # the tail periodically and bail as soon as the output is provably looping,
                # which costs seconds instead of minutes. Checked on a bounded tail every
                # LLM_REPEAT_CHECK_EVERY chars, so the cost does not grow with output size.
                #
                # BOTH channels are sampled. `streamed_chars` has always counted
                # content AND reasoning, but the tail used to be built from
                # content_parts alone — so a model looping inside its REASONING
                # channel ticked the checkpoint over and over against an empty tail
                # and was never caught. That is the house model's channel: Gemma 4
                # spends most of its budget thinking (which is why max_tokens is
                # floored to GEMMA_MIN_TOKENS), so the guard was blind exactly where
                # it was needed. Measured on an identical 47-char loop: a content
                # loop was cut after 2.4k chars, the same loop in reasoning ran to
                # 144k and would have continued to the 200k/600s last-resort caps.
                if (LLM_REPEAT_CHECK_EVERY > 0
                        and streamed_chars - last_repeat_check >= LLM_REPEAT_CHECK_EVERY):
                    last_repeat_check = streamed_chars
                    _looping = ""
                    for _chan, _parts in (("content", content_parts),
                                          ("reasoning", reasoning_parts)):
                        tail = "".join(_parts[-400:])[-LLM_REPEAT_TAIL_CHARS:]
                        reason, sig = _looks_degenerate(tail)
                        if reason and sig == loop_sigs[_chan]:
                            # Same loop still running one checkpoint later — it is not
                            # duplicated source content being faithfully transcribed.
                            _looping = f"{_chan}: {reason}"
                        elif reason:
                            logger.info("LM Studio %s looks repetitive at %d chars (%s) — "
                                        "watching one more checkpoint",
                                        _chan, streamed_chars, reason)
                        loop_sigs[_chan] = sig
                    if _looping:
                        logger.warning("LM Studio stream aborted after %d chars — output is "
                                       "still looping (%s); keeping what was produced",
                                       streamed_chars, _looping)
                        resp.close()
                        break

                for tcd in (delta.get("tool_calls") or []):
                    idx = tcd.get("index", 0)
                    slot = tool_acc.setdefault(
                        idx,
                        {
                            "id": None,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if tcd.get("id"):
                        slot["id"] = tcd["id"]
                    fn = tcd.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] += str(fn["name"])
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += str(fn["arguments"])
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as exc:
            # Mid-stream stall/drop (the per-read timeout fired). Keep partial output if we
            # got any — discarding it to retry just re-streams from scratch and risks
            # stalling again; only re-raise when nothing usable arrived so the caller's
            # retry loop can handle a connection that produced nothing.
            if not (content_parts or reasoning_parts or tool_acc):
                raise
            logger.warning("LM Studio stream stalled mid-message (%s) — keeping %d partial "
                           "chars instead of discarding", type(exc).__name__, streamed_chars)

    _content = "".join(content_parts)
    if "to=self" in _content:
        # Glimmer's inline reasoning (see utils._TO_SELF_BLOCK_RE): move it to
        # the reasoning channel here, at the source, so no caller -- JSON
        # parsers included -- ever reads it as the answer.
        from utils import _TO_SELF_BLOCK_RE
        for _m in _TO_SELF_BLOCK_RE.finditer(_content):
            reasoning_parts.append(_m.group(0))
        _content = _TO_SELF_BLOCK_RE.sub("", _content).lstrip()
    message: dict = {
        "role": "assistant",
        "content": _content,
        "finish_reason": finish_reason,
    }
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_acc:
        message["tool_calls"] = [tool_acc[i] for i in sorted(tool_acc)]
    try:
        import turn_trace
        _sys = next((m.get("content") for m in payload.get("messages") or []
                     if m.get("role") == "system"), "") or ""
        turn_trace.llm(_sys if isinstance(_sys, str) else str(_sys), usage,
                       time.perf_counter() - _started, finish_reason, streamed_chars,
                       messages=payload.get("messages"), reply=message)
    except Exception:
        pass
    return message


def call_llm_simple(
    ctx,
    system_prompt: str,
    user_message: str,
    history: Optional[List[dict]] = None,
    temperature: float = 0.5,
    max_tokens: int = 1500,
    prefill: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    force_think: bool = False,
    json_schema: Optional[dict] = None,
) -> Optional[str]:
    """Simple LLM call without tool calling. Returns text content only.

    force_think: let a reasoning fine-tune think freely (no /no_think, no prefill).
    Use for long-form synthesis — quality is much higher WITH chain-of-thought, and
    suppression on these models backfires into empty output. Budget tokens for the
    think block PLUS the answer.
    """
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    result = send_to_lm_studio(
        ctx,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        prefill=prefill,
        reasoning_effort=reasoning_effort,
        force_think=force_think,
        json_schema=json_schema,
    )
    if result is None:
        return None
    return result.get("content") or None


# The crashes on big images (1400x2100, 1920x1088, live 2026-09-12) were the
# llama.cpp micro-batch: an image's tokens must fit one ubatch, default 512.
# lmstudio._rest_load sets it to 2048 and Gemma 4 spends at most 1120 tokens
# on a picture (measured: 1024 px 288, 1280 px 441, 2048 px 1068, 3000 px
# 1136), so 2048 px is the encoder's full resolution. At 1024 px a projector
# on wallpaper was «телевизор»; at full size the wall shows through.
VISION_MAX_SIDE = _cfg_env.env_int("VISION_MAX_SIDE", 2048)


def _vision_sized(image_path: str) -> str:
    """`image_path`, or a bounded JPEG/PNG copy of it when it is too big."""
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            w, h = im.size
            if max(w, h) <= VISION_MAX_SIDE:
                return image_path
            scale = VISION_MAX_SIDE / float(max(w, h))
            small = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
            small = small.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
            alpha = small.mode == "RGBA"
            out = os.path.join(tempfile.gettempdir(),
                               f"_vision_{abs(hash(image_path)) % 10**9}_{os.path.getmtime(image_path):.0f}"
                               + (".png" if alpha else ".jpg"))
            if not os.path.exists(out):
                small.save(out, "PNG" if alpha else "JPEG", **({} if alpha else {"quality": 92}))
            logger.info("vision: %dx%d image sent as %dx%d", w, h, *small.size)
            return out
    except Exception:
        logger.debug("vision downscale failed; sending the original", exc_info=True)
        return image_path


def _vision_sized_bytes(image_bytes: bytes) -> tuple:
    """(bytes, mime) bounded to VISION_MAX_SIDE, like _vision_sized for paths.

    The chat-turn photo arrives as BYTES through graph.vision_agent_node's
    1536-px proxy and skipped the 1024 bound that every path-based call gets;
    a 933x1400 upload crashed the model 4 times out of 4 (live 2026-09-13,
    "The model has crashed without additional information"), each time costing
    a revive and the turn. Same picture at 682x1024 never did.
    """
    try:
        import io
        from PIL import Image
        with Image.open(io.BytesIO(image_bytes)) as im:
            w, h = im.size
            fmt = (im.format or "").upper()
            if max(w, h) <= VISION_MAX_SIDE and fmt in ("JPEG", "PNG"):
                return image_bytes, ("image/png" if fmt == "PNG" else "image/jpeg")
            scale = min(1.0, VISION_MAX_SIDE / float(max(w, h)))
            small = im.convert("RGB").resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
            buf = io.BytesIO(); small.save(buf, "JPEG", quality=92)
            if scale < 1.0:
                logger.info("vision: %dx%d image bytes sent as %dx%d", w, h, *small.size)
            return buf.getvalue(), "image/jpeg"
    except Exception:
        logger.debug("vision bytes downscale failed; sending the original", exc_info=True)
        return image_bytes, "image/jpeg"


def analyze_image_with_llm(
    ctx,
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    user_text: str = "",
    system_prompt: str = "",
    temperature: float = 0.2,
    max_tokens: int = 1900,
    prefill: Optional[str] = None,
    image_paths: Optional[list] = None,
) -> Optional[str]:
    """Vision call: analyze an image with optional text question.

    image_paths: [(label, path)] -- several pictures in ONE call, each at its
    own full resolution and preceded by its label (frames of a video: the
    model sees every frame sharp AND how they follow each other).

    prefill: optional assistant-turn prefix (e.g. "<think></think>") to suppress
    the reasoning channel on reasoning-prone fine-tunes — useful when the caller
    needs structured/JSON output.
    """
    if image_paths:
        content = []
        for label, p in image_paths:
            try:
                url = file_to_data_url(_vision_sized(p))
            except Exception as exc:
                logger.warning("analyze_image_with_llm: could not read image %r: %s", p, exc)
                continue
            content += [{"type": "text", "text": str(label)},
                        {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}]
        if not content:
            return None
        if user_text.strip():
            content.append({"type": "text", "text": user_text})
        return _vision_call(ctx, system_prompt, content, temperature, max_tokens, prefill)
    if image_path is None and image_bytes is None:
        return None

    # Build the data URL defensively: a caller may pass a path that was valid a moment
    # ago but is now missing/unreadable (a temp render deleted between steps, a race with
    # cleanup). file_to_data_url open()s it directly, so an unguarded call raises
    # FileNotFoundError/OSError and crashes the whole turn at hot call sites that do NOT
    # wrap it (tools.py inspect-image QA, image.py evaluate_image). Treat an unusable
    # image as "no image to analyze" and return None — the documented contract.
    try:
        if image_path is not None:
            data_url = file_to_data_url(_vision_sized(image_path))
        else:
            sized, mime = _vision_sized_bytes(image_bytes or b"")
            data_url = image_bytes_to_data_url(sized, mime_type=mime)
    except Exception as exc:
        logger.warning("analyze_image_with_llm: could not read image %r: %s",
                       image_path or "<bytes>", exc)
        return None

    # Picture first, then the words: Gemma 4 was trained with image content
    # before the text (Google's Gemma 4 guide / HF launch notes).
    content = [{"type": "image_url", "image_url": {"url": data_url, "detail": "auto"}}]
    if user_text.strip():
        content.append({"type": "text", "text": user_text})

    return _vision_call(ctx, system_prompt, content, temperature, max_tokens, prefill)


def _vision_call(ctx, system_prompt, content, temperature, max_tokens, prefill):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    # A single retry covers a TRANSIENT empty (e.g. LM Studio returning no message
    # while it is mid model-swap). The forensic investigation
    # (docs/vision_pipeline_forensics.md) proved that neither the model nor the parser
    # is at fault for empty vision responses: both the 9B and the 35B return a proper
    # answer in `content` that strip_think_tags preserves intact. Empty/garbled
    # responses came entirely from LM Studio failing to SERVE the requested model
    # (unknown id → silent substitution; oversized model → "Model unloaded"; mid-swap
    # → transient empty). So we do NOT reroute to another model — we retry once and let
    # the caller's fail-safe handle a persistent miss.
    content_out = None
    for _ in range(2):
        result = send_to_lm_studio(
            ctx, messages, temperature=temperature, max_tokens=max_tokens,
            prefill=prefill,
        )
        content_out = (result or {}).get("content") if result else None
        if content_out:
            break
    return content_out
