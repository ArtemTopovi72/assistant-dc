"""The tool control plane: what each tool PRODUCES, what it REQUIRES, and what
its arguments have to MEAN before it is allowed to run.

Two mechanisms live here, and both exist because the alternative is a failure
that surfaces far away from its cause.

1. A producer-consumer dependency graph. Every image-editing tool needs a
   picture to already exist; transfer_image needs two; the web tools need the
   internet switch to be on. Until now each handler discovered that for itself,
   several hundred lines deep, and the ones that forgot discovered it inside
   ComfyUI. Declaring the edges in one table means the precondition is checked
   BEFORE the handler runs, the message names the missing thing AND the tool
   that produces it, and a tool added later cannot quietly forget the check.

2. Semantic argument validation. Pydantic proves an argument has the right
   TYPE; it cannot prove it has a sensible VALUE. `description=" . "` is a
   perfectly valid non-empty string and it is also this project's oldest
   landmine — an effectively empty positive prompt renders a default person
   instead of failing (see the old model default-woman note). `expression="how
   many people live in Tokyo"` is a valid string and not arithmetic. Those
   checks live here, next to the graph, because both answer the same question:
   can this call succeed at all, or should the model be told to do something
   else first?

Everything returned is a "[TOOL ERROR] ..." string, because the agent loop
already treats that prefix as a failed step and replans. The messages are
written to be ACTIONABLE — they say which tool to call first, not merely that
something was absent.

This module is a leaf: it imports nothing from the pipeline, so tools.py can
gate on it without a cycle and the suites can exercise it with plain dicts.
"""
from __future__ import annotations

import ast
import math
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional


# --------------------------------------------------------------------------- #
# Probes: how we tell, from the live ctx/state, whether a resource exists.
# These deliberately mirror what the handlers themselves look at, so the gate
# and the handler can never disagree about whether there is an image.
# --------------------------------------------------------------------------- #

def image_paths(ctx, state) -> list:
    """Every picture the current turn could work on, oldest first, de-duped and
    filtered to what is actually still on disk.

    Same sources the handlers read: the multi-image registers the Telegram and
    GUI paths fill in, the session's loaded reference images, and the single
    working image. A path that no longer exists is not an image.
    """
    state = state if isinstance(state, dict) else {}
    cands = []
    for src in (state.get("image_paths"),
                getattr(ctx, "recent_image_paths", None),
                getattr(ctx, "reference_images", None)):
        if isinstance(src, (list, tuple)):
            cands.extend(src)
    for one in (state.get("image_path"), getattr(ctx, "last_image_path", None)):
        if one:
            cands.append(one)
    seen, out = set(), []
    for p in cands:
        if not p or p in seen:
            continue
        try:
            if not os.path.exists(p):
                continue
        except (TypeError, ValueError, OSError):
            continue
        seen.add(p)
        out.append(p)
    return out


def _has_image(ctx, state) -> bool:
    return bool(image_paths(ctx, state))


def _has_two_images(ctx, state) -> bool:
    return len(image_paths(ctx, state)) >= 2


def _has_internet(ctx, state) -> bool:
    return bool(getattr(ctx, "web_search_enabled", True))


# --------------------------------------------------------------------------- #
# The graph itself.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Resource:
    """A precondition some tools produce and others consume."""
    name: str
    label: str                       # human phrase used in the error message
    probe: Callable[[object, object], bool]
    producers: tuple = ()            # tools that CREATE this resource
    advice: str = ""                 # what to do when no producer applies


RESOURCES: dict = {
    "image": Resource(
        name="image",
        label="a picture to work on",
        probe=_has_image,
        producers=("generate_image", "find_photo"),
        advice="or ask the user to send/paste a picture first",
    ),
    "two_images": Resource(
        name="two_images",
        label="TWO pictures loaded (a target plus at least one reference)",
        probe=_has_two_images,
        producers=(),
        advice=("ask the user to send the second picture (the reference to take "
                "the item from), or use inpaint_image if the change can be "
                "described in words instead of copied from another photo"),
    ),
    "internet": Resource(
        name="internet",
        label="internet access",
        probe=_has_internet,
        producers=(),
        advice=("internet access is switched off, so answer from your own "
                "knowledge and tell the user plainly if you cannot"),
    ),
}


# tool -> the resources it creates. Used for the "which tool produces it"
# half of the error message, and readable as documentation of the data flow.
PRODUCES: dict = {
    "generate_image": ("image",),
    "find_photo": ("image",),
    "transfer_image": ("image",),
    "redraw_image": ("image",),
    "inpaint_image": ("image",),
    "fix_hands": ("image",),
    "fix_artifact": ("image",),
}

# tool -> the resources that must already exist before it can run.
REQUIRES: dict = {
    "redraw_image": ("image",),
    "inpaint_image": ("image",),
    "inspect_image": ("image",),
    "fix_hands": ("image",),
    "fix_artifact": ("image",),
    "transfer_image": ("two_images",),
    "search": ("internet",),
    "deep_research": ("internet",),
    "find_photo": ("internet",),
}


def missing_prerequisite(ctx, state, tool: str) -> Optional[str]:
    """Return an actionable [TOOL ERROR] if `tool` cannot run right now.

    The message names the missing prerequisite AND the tool that produces it,
    because the loop's only lever after a failed step is to call something
    else — telling it *what* is the difference between a replan and a retry
    of the same doomed call.
    """
    for res_name in REQUIRES.get(tool, ()):
        res = RESOURCES.get(res_name)
        if res is None or res.probe(ctx, state):
            continue
        if res.producers:
            fix = ("Call " + " or ".join(res.producers) + " first"
                   + (f", {res.advice}" if res.advice else "") + ".")
        else:
            fix = (res.advice[:1].upper() + res.advice[1:] + ".") if res.advice else ""
        return (f"[TOOL ERROR] {tool} cannot run yet — it needs {res.label}, and "
                f"there is none. {fix} Do NOT call {tool} again until that is done.")
    return None


# --------------------------------------------------------------------------- #
# Semantic argument validation: right type, wrong meaning.
# --------------------------------------------------------------------------- #

# A character that carries meaning in any script we serve (latin, cyrillic,
# CJK) or a digit. Punctuation, emoji and whitespace do not count.
_MEANINGFUL = re.compile(r"[0-9A-Za-zÀ-ɏЀ-ӿ一-鿿]")

# Values a small model emits when it has nothing to say but the schema demands
# a string. Each of these used to sail through `min_length=1` and reach the
# renderer as an effectively empty positive prompt.
_PLACEHOLDERS = {
    "", "none", "null", "nil", "n/a", "na", "nan", "empty", "undefined",
    "unknown", "todo", "tbd", "string", "text", "prompt", "description",
    "image", "picture", "-", "--", "...", "…", ".", "?", "нет", "пусто",
}

# Fields whose value is a natural-language instruction that MUST carry meaning.
# An empty-in-effect value here is the old model default-person landmine: the
# render silently succeeds and returns a picture nobody asked for.
MEANINGFUL_TEXT_FIELDS: dict = {
    "generate_image": ("description",),
    "generate_video": ("description",),
    "inpaint_image": ("instructions", "region"),
    "transfer_image": ("instructions",),
    "fix_artifact": ("region",),
    "inspect_image": ("check",),
    "search": ("query",),
    "deep_research": ("topic",),
    "find_photo": ("query",),
    "remember_fact": ("fact",),
    "create_presentation": ("topic",),
}

# The redraw modes that actually exist. Kept as a module constant because
# tool_args' Literal, the description prose and this gate must agree; a mode
# outside this set is silently coerced to None by pydantic, which turns a
# requested "upscale" into a quality pass that ignores the request.
# 'enhance'/'upscale'/'restore'/'outpaint' used to be distinct pipelines, all
# built on the old model checkpoint that was removed from the product along with
# its files — 'redraw' (FireRed) is the only mode left with an engine behind it.
REDRAW_MODES = ("redraw",)

# The reference roles transfer_image understands. An unknown role is dropped
# downstream, so the transfer runs with the model's intent thrown away.
TRANSFER_ROLES = (
    "object_source", "clothing_source", "face_reference", "hairstyle_reference",
    "pose_reference", "style_reference", "identity_reference", "scene_reference",
    "lighting_reference",
)

_MATH_NAMES = frozenset(
    [k for k in dir(math) if not k.startswith("_")] + ["abs", "round", "int", "float"]
    # tools._calendar_ns: blocking these made "сколько дней до 1 января" fall
    # back to mental math and answer 94 instead of 95 (live 2026-09-28).
    + ["today", "date", "days_between", "years_between", "weekday"])


def is_meaningful_text(value) -> bool:
    """True when a text argument says something a pipeline can act on."""
    s = str(value or "").strip()
    if s.lower().strip(" .!?-_'\"") in _PLACEHOLDERS:
        return False
    return len(_MEANINGFUL.findall(s)) >= 2


def _expression_problem(expr: str) -> Optional[str]:
    """Why `expr` is not arithmetic, or None if it is.

    Prose reaches calculate constantly ("how many people live in Tokyo"). It
    parses as a NameError deep in eval and comes back as a cryptic failure; the
    model then retries the same prose. Naming the actual problem lets it either
    write the formula or drop the tool.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return ("that is prose, not arithmetic. calculate evaluates a single math "
                "formula, e.g. '(15*1.2)/3' or 'sqrt(144)'")
    except (ValueError, MemoryError, RecursionError):
        return "the expression could not be parsed as a math formula"
    unknown = sorted({n.id for n in ast.walk(tree)
                      if isinstance(n, ast.Name) and n.id not in _MATH_NAMES})
    if unknown:
        return (f"it refers to {', '.join(repr(u) for u in unknown[:4])}, which is not "
                "a number or a math function. Substitute the actual numbers, or "
                "answer without the calculator")
    if not any(isinstance(n, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Compare, ast.Constant))
               for n in ast.walk(tree)):
        return "it is not a computable expression"
    return None


def invalid_arguments(tool: str, args: dict) -> Optional[str]:
    """Semantic (not type) validation of a tool call's arguments.

    Runs on the raw, synonym-normalised args BEFORE pydantic, because several
    of these values are silently repaired by pydantic's coercing validators —
    an invalid redraw mode becomes None, and None then becomes a different mode
    than the user asked for. Catching it here is the only place the model can
    still be told it named a mode that does not exist.
    """
    if not isinstance(args, dict):
        return None

    for field in MEANINGFUL_TEXT_FIELDS.get(tool, ()):
        if field not in args:
            continue                       # absent is pydantic's business
        if not is_meaningful_text(args.get(field)):
            return (f"[TOOL ERROR] {tool} was called with an empty or placeholder "
                    f"'{field}' ({args.get(field)!r}). That would produce a result "
                    "nobody asked for, so nothing was run. Re-call it with a real, "
                    f"specific '{field}' describing what the user actually wants, or "
                    "answer them without this tool.")

    if tool == "redraw_image":
        mode = args.get("mode")
        if mode not in (None, "") and str(mode).strip().lower() not in REDRAW_MODES:
            return (f"[TOOL ERROR] redraw_image has no mode {str(mode)!r}. The real "
                    f"modes are: {', '.join(REDRAW_MODES)}. Re-call it with 'redraw' "
                    "and instructions describing the change (or leave instructions "
                    "empty for a general quality pass).")

    if tool == "transfer_image":
        roles = args.get("roles")
        if isinstance(roles, (list, tuple)):
            bad = [r for r in roles
                   if str(r).strip().lower() not in TRANSFER_ROLES]
            if bad:
                return (f"[TOOL ERROR] transfer_image got unknown role(s) "
                        f"{', '.join(repr(str(b)) for b in bad[:4])}. Valid roles are: "
                        f"{', '.join(TRANSFER_ROLES)}. Re-call with roles from that "
                        "list, or omit 'roles' entirely to auto-infer them.")

    if tool == "calculate":
        expr = str(args.get("expression") or "").strip()
        if expr and len(expr) <= 500 and "__" not in expr:
            problem = _expression_problem(expr)
            if problem:
                return (f"[TOOL ERROR] calculate cannot evaluate {expr!r} — {problem}.")

    return None


# --------------------------------------------------------------------------- #
# An accepted render is final for the turn.
# --------------------------------------------------------------------------- #
# generate_image already judges its output with the vision model. When that
# judgement is a pass, a further inspect/edit by the agent in the SAME turn is
# not verification, it is a second, unreliable critic with the power to redraw:
# on 2026-09-11 a 10/10 render was "fixed" for a shadow that was not there,
# lost its subject to the removal, and went through six more renders before
# the model crashed. The user is the only critic who gets to send it back, and
# they do that in their own words, in their own turn -- which arrives with a
# fresh state and no `fresh_render` in it.
ACCEPTED_SCORE = 7
_EDIT_TOOLS = frozenset({"inpaint_image", "redraw_image", "fix_hands", "fix_artifact"})


def self_edit_of_accepted_render(ctx, state, tool: str, args: dict) -> Optional[str]:
    if tool not in _EDIT_TOOLS:
        return None
    fresh = (state or {}).get("fresh_render") or {}
    if not fresh.get("accepted"):
        return None
    # The pencil button is the user asking for an edit in this very turn.
    if (state or {}).get("edit_intent"):
        return None
    return (f"[TOOL ERROR] {tool} refused: the picture you just generated "
            f"({fresh.get('score')}/10) was checked and ACCEPTED. Do not edit or "
            "redraw your own fresh render -- your second look invents defects the "
            "picture does not have. Answer the user now and deliver it as it is; "
            "if they want a change they will ask for it.")


def check_tool_call(ctx, state, tool: str, args: dict) -> Optional[str]:
    """The whole pre-execution gate: preconditions first, then argument meaning.

    Preconditions come first on purpose — when there is no picture at all, the
    honest complaint is 'there is nothing to edit', not 'your region argument
    is vague'.
    """
    return (picture_question_edit(state, tool)
            or missing_prerequisite(ctx, state, tool)
            or self_edit_of_accepted_render(ctx, state, tool, args)
            or invalid_arguments(tool, args))


_PICTURE_TOOLS = _EDIT_TOOLS | frozenset({"generate_image", "upscale_image", "transfer_image"})


def picture_question_edit(state, tool: str) -> Optional[str]:
    """A question about the picture is answered in words; an image tool on
    such a turn is refused before it spends the card (graph.is_picture_question)."""
    if tool in _PICTURE_TOOLS and (state or {}).get("picture_question"):
        return ("[TOOL ERROR] The user asked a QUESTION about the picture, not for a "
                "change. Do not edit or generate anything: answer in words from the "
                "image description (inspect_image is allowed if you need a closer look).")
    return None
