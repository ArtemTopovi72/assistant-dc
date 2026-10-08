import logging
import re
from typing import Optional
import time
from pathlib import Path

from langgraph.graph import StateGraph, END, START

from config import ASSISTANT_ACTOR, OUTPUT_DIR, TTS_SPEECH_ADAPT
from models import AgentState, Context
from prompts import VISION_PROMPT, SPEECH_ADAPT_PROMPT
# Per-turn composition (sampling settings + the user message), extracted to
# graph_compose. Re-exported: the suites reach both through graph.  # noqa: F401
from graph_compose import _GenSettings, _compose_user_message
from llm import send_to_lm_studio, analyze_image_with_llm, call_llm_simple
from audio import post_process_answer, stress_plus, synth_single_segment, play_audio_file
from tools import TOOL_SCHEMAS, execute_tool
from utils import downscale_image_bytes
# The personality node (the tool-calling loop) and its two satellites, plus
# everything only they read. Re-exported by name: send_to_lm_studio,
# execute_tool, TOOL_SCHEMAS and _TOOL_TRIGGER_RE stay bound HERE (this module
# owns them, imported above and defined below) because the suites patch all
# four directly on `graph` — graph_personality reaches them back through
# `import graph as _g` at call time rather than by value.  # noqa: F401
from graph_personality import (
    MAX_TOOL_CALLS_PER_ROUND, _FAST_PATH_CAPABILITY_DENIAL_RE, _INTENT_TOOL_MAP, _intent_tools,
    _ACTION_CLAIM_RE, _DECK_PREFIX, _fast_path_allowed,
    _fast_path_reply, _finalize_answer, personality_node,
)

logger = logging.getLogger("assistant.graph")


# Rolling history compaction lives in graph_history. Re-exported by name so the
# graph.compact_history_if_needed / graph.HISTORY_COMPACT_EVERY references in
# call sites and suites keep resolving here.
from graph_history import (  # noqa: F401,E402
    HISTORY_COMPACT_EVERY, _HISTORY_KEEP_TURNS, _HISTORY_SUMMARY_MARKER,
    _OUTCOME_BY_TOOL, _outcome_sentence, _render_turns_for_summary,
    compact_history_if_needed,
)




# The English-first language boundary lives in graph_language. Re-exported by
# name so graph._translate_to_english / graph._match_reply_language keep
# resolving. Neither is ever patched by the suites (only read directly, e.g.
# G._match_reply_language(...) in test_reply_finalization), so translate_node
# staying here and _finalize_answer moving to graph_personality both reach
# them the same simple way: imported by value, no indirection needed.
from graph_language import (
    _TRANSLATE_OUT_PROMPT, _QUOTED_LITERAL_RE, _protect_literals, _restore_literals,
    _foreign_ratio, _match_reply_language, _translate_to_english,
)


# The prefix the ✏️ "describe what to change" button puts in front of the user's
# text. Defined here (not only in tg_bot) because the guard that keeps such a
# turn off generate_image lives on this side of the boundary.
_EDIT_INTENT_PREFIX = "edit the image:"



# --------------------------------------------------------------------------- #
# The graph nodes.
#
# These were closures inside build_graph, which is why build_graph was 938 lines
# long. They only ever captured `ctx`, so they are ordinary module-level
# functions taking it as their first argument; build_graph binds it when it
# wires the graph. Module level also makes them individually patchable and
# individually callable from a probe.
# --------------------------------------------------------------------------- #


def translate_node(ctx: Context, state: AgentState) -> AgentState:
    original = state.get("user_input", "") or ""
    # The previous turn's accepted render must not be "fresh" in this one: the
    # user's request for a change arrives here, and it has to be allowed.
    state.pop("fresh_render", None)
    # Record the EDIT intent from the raw text, before translation rewrites it.
    # The ✏️ button prefixes the message with "edit the image: "; that is a
    # deterministic statement of what the user pressed, and it must not depend
    # on the model reading the system prompt correctly. Live failure it fixes:
    # the prefix arrived intact and the agent still called generate_image,
    # answering "Я создал изображение банки меда" instead of adding the
    # lettering the user asked for — the previous picture was replaced by a
    # brand-new one.
    if original.strip().lower().startswith(_EDIT_INTENT_PREFIX):
        state["edit_intent"] = True
    # The previous reply disambiguates a short follow-up: «что посмотреть в
    # первый день?» after a Kazan trip plan became "what to watch" (films)
    # and the answer lost the city (live 2026-09-28).
    _prev = next((str(m.get("content") or "") for m in reversed(state.get("messages") or [])
                  if m.get("role") == "assistant" and m.get("content")), "")
    try:
        english = _translate_to_english(ctx, original, context=_prev[-400:])
    except TypeError:                     # a patched translator without `context`
        english = _translate_to_english(ctx, original)
    # Set even when untranslated: an Italian line left as-is had no original,
    # and the reply-language guard skipped its English answer.
    state["user_input_original"] = original
    state["user_input"] = english
    return state


def _add_earlier_pictures(ctx, state, summary: str) -> str:
    """«что общего у двух фото?»: the overview covers the current picture only,
    and the answer was "you sent just one photo" (live 2026-10-08). When the
    user relates several pictures, the chat's earlier ones are read with the
    same question and put next to it."""
    earlier = [p for p in (getattr(ctx, "image_undo", None) or [])
               if p and p != getattr(ctx, "last_image_path", None) and Path(p).exists()][-2:]
    text = state.get("user_input_original") or state.get("user_input") or ""
    if not earlier or not summary or "Failed to analyze" in summary or not text.strip():
        return summary
    import intent
    if not intent.ask_yes("The user wrote: {text}. Do they ask about several of the pictures "
                          "they sent together (compare them, what they share, which is "
                          "better, both of them)?", text, default=False):
        return summary
    parts = ["Latest picture:" + chr(10) + summary]
    for i, path in enumerate(earlier, 1):
        try:
            with open(path, "rb") as fh:
                got = analyze_image_with_llm(ctx=ctx, image_bytes=downscale_image_bytes(fh.read()),
                                             user_text=text, system_prompt=VISION_PROMPT)
        except Exception:
            logger.warning("earlier picture could not be read", exc_info=True)
            continue
        if got:
            parts.append("Earlier picture %d of %d sent in this chat:" % (i, len(earlier)) + chr(10) + got)
    return (chr(10) * 2).join(parts) if len(parts) > 1 else summary


def vision_agent_node(ctx: Context, state: AgentState) -> AgentState:
    image_bytes = state.get("image_data")
    if not image_bytes:
        # No new photo this turn — but a question that plainly needs a
        # fresh look at the CURRENT working image (see
        # needs_relook) must not be left to whatever stale,
        # multi-photo summary happens to sit in session memory.
        last_path = getattr(ctx, "last_image_path", None)
        user_text = state.get("user_input", "") or ""
        if needs_relook(ctx, state):
            try:
                with open(last_path, "rb") as fh:
                    current_bytes = fh.read()
                proxy_bytes = downscale_image_bytes(current_bytes)
                ctx.set_stage("Looking at the video again" if is_video_sheet(last_path)
                              else "Looking at the image")
                if is_video_sheet(last_path):
                    user_text = ("This sheet holds the timestamped keyframes of the video the "
                                 "user sent. Look at the frames closely for what the request "
                                 "needs (surfaces, joints, defects, details), citing timestamps. "
                                 "Request: " + user_text)
                response = analyze_image_with_llm(
                    ctx=ctx, image_bytes=proxy_bytes,
                    user_text=user_text, system_prompt=VISION_PROMPT)
                _pq = is_picture_question(ctx, state)
                if response and _pq and not is_video_sheet(last_path):
                    # A question deserves the small details too (vision_close).
                    response = _close_look(ctx, current_bytes, user_text, response)
                vision_summary = response if response else "Failed to analyze image."
                vision_summary = _add_earlier_pictures(ctx, state, vision_summary)
                state["vision_summary"] = vision_summary
                if "Failed to analyze" not in vision_summary:
                    ctx.remember("vision", vision_summary, {"has_image": True})
                    state["session_memory_text"] = ctx.memory_text()
                if _pq:
                    state["picture_question"] = True
            except Exception:
                logger.warning("re-look at the current image failed", exc_info=True)
        return state

    original_bytes = image_bytes

    # PROXY (longest edge ~1536, JPEG) — used ONLY for analysis: the VLM, and
    # any downstream grounding/detection/captioning. The vision model gains
    # nothing from full res and full-res would bloat the payload. The proxy is
    # NEVER the edit source.
    proxy_bytes = downscale_image_bytes(original_bytes)
    state["image_data"] = proxy_bytes

    # WORKING / EDIT IMAGE = the ORIGINAL upload at full native resolution,
    # written verbatim (no downscale, no recompression). inpaint_image /
    # redraw_image edit exactly what was sent so the result keeps the native
    # resolution. The crop-based editor builds and maps its OWN Florence proxy
    # internally (proxy bbox -> source coords), so handing it the full-res
    # original is correct and needs no coordinate mapping here.
    ext = "jpg"
    try:
        import io as _io
        from PIL import Image as _Image
        with _Image.open(_io.BytesIO(original_bytes)) as _im:
            fmt = (_im.format or "").lower()
        ext = {"jpeg": "jpg", "png": "png", "webp": "webp",
               "bmp": "bmp", "tiff": "tiff", "gif": "png"}.get(fmt, "jpg")
    except Exception:
        pass
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        work_path = OUTPUT_DIR / f"_working_input_{int(time.time() * 1000)}.{ext}"
        with open(work_path, "wb") as fh:
            fh.write(original_bytes)  # full-resolution original, byte-for-byte
        ctx.last_image_path = str(work_path)
        # Also override state["image_path"]: tool handlers resolve their source as
        # state.get("image_path") or ctx.last_image_path, so a stale path from an
        # image generated in an EARLIER turn would otherwise shadow the photo the
        # user just sent and the edit would land on the wrong image.
        state["image_path"] = str(work_path)
    except Exception as exc:
        logger.warning("Could not persist sent image as working image: %s", exc)

    ctx.set_stage("Looking at the image")
    response = analyze_image_with_llm(
        ctx=ctx,
        image_bytes=proxy_bytes,  # analysis uses the proxy, not the original
        user_text=state.get("user_input", ""),
        system_prompt=VISION_PROMPT,
    )
    _pq = is_picture_question(ctx, state)
    if response and _pq:
        # A question about the photo: the zoomed parts are read too, so the
        # label, the number, the thing in the hand make it into the answer.
        response = _close_look(ctx, original_bytes, state.get("user_input", ""), response)
    vision_summary = response if response else "Failed to analyze image."
    vision_summary = _add_earlier_pictures(ctx, state, vision_summary)
    state["vision_summary"] = vision_summary
    if ctx.last_image_path and vision_summary and "Failed to analyze" not in vision_summary:
        # Used as the prompt seed if the user later asks to redraw the whole thing.
        # Skip the failure sentinel so a non-vision model can't poison the seed.
        ctx.last_image_prompt = vision_summary

    ctx.remember("vision", vision_summary, {"has_image": True})
    state["session_memory_text"] = ctx.memory_text()
    if _pq:
        state["picture_question"] = True
    return state


# What makes a text "written": a label colon, a dash, a list line, markdown, arrows.
_WRITTEN_RE = re.compile(r":\s|\s[—–-]\s|^\s*(?:[-*•]|\d+[.)])\s|[*_`#→;/()]|\n", re.M)
_DIGITS_RE = re.compile(r"\d+")


def speakable(ctx: Context, text: str) -> str:
    """The answer as it would be SAID: lists, «тезис: утверждение» and dashes
    become sentences (SPEECH_ADAPT_PROMPT). Falls back to the text as written
    when the rewrite fails or drops a number -- a voice that loses «14:30» is
    worse than one that reads a colon."""
    if not TTS_SPEECH_ADAPT or not _WRITTEN_RE.search(text or ""):
        return text
    try:
        out = (call_llm_simple(ctx, SPEECH_ADAPT_PROMPT, text, temperature=0.3,
                               max_tokens=min(3000, len(text) + 300),
                               prefill="<think></think>") or "").strip()
    except Exception:
        logger.exception("speech adaptation failed; voicing the text as written")
        return text
    # A URL is not read out and a list's own numbering becomes «во-вторых».
    # The numbering may sit inside markdown: «**1. Уровень…**», «### 2.» -- live
    # 13:27 a three-level plan was rejected for "losing" 1, 2, 3 and read as a list.
    _facts = re.sub(r"(?m)^[\s#*_>]*\d+[.)]\s", "", re.sub(r"https?://\S+", "", text))
    lost = [d for d in _DIGITS_RE.findall(_facts)
            if d not in out]
    if not out or lost or not 0.5 <= len(out) / max(1, len(text)) <= 2.0:
        logger.info("speech adaptation rejected (lost numbers %s, %d -> %d chars)",
                    lost[:5], len(text), len(out))
        return text
    return out.replace("`", "")


def spoken_text(ctx: Context, text: str) -> str:
    """Everything a reply goes through before a voice says it: retold as speech,
    then numbers, dates and lists spelled for the engine. One function for every
    path that voices a reply -- the Telegram fallback used to read the raw text."""
    return post_process_answer(speakable(ctx, text))


def tts_node(ctx: Context, state: AgentState) -> AgentState:
    if getattr(ctx, "tts_disabled", False):
        return state  # voice output is muted by user
    text = state.get("final_answer", "")
    if not text:
        return state

    ctx.set_stage("Speaking")
    # A TTS failure must NEVER destroy the turn: the text answer is already
    # in state["final_answer"], and an exception here would propagate out of
    # graph.invoke and lose it. Speak if possible; stay silent otherwise.
    try:
        clean = spoken_text(ctx, text)
        # Stress here, but DON'T pad here: synth_single_segment ->
        # preprocess_text_for_synthesis appends TTS_END_PADDING exactly once for
        # every synthesis path. Padding here too would double it (path-dependent).
        stressed = stress_plus(ctx, clean).rstrip()
        temp_wav = OUTPUT_DIR / f"_assistant_reply_{int(time.time() * 1000)}.wav"
        out_stem = str(temp_wav.with_suffix(""))

        result_path = synth_single_segment(
            ctx=ctx,
            idx=-1,
            actor=ASSISTANT_ACTOR,
            raw_text=stressed,
            out_stem=out_stem,
            use_censoring=False,
            apply_stress=False,
        )

        if result_path and Path(result_path).exists():
            state["tts_path"] = result_path
            if not getattr(ctx, "gui_mode", False):
                play_audio_file(result_path)  # GUI plays it inline instead
        else:
            logger.warning("Speech synthesis failed")
    except Exception:
        logger.exception("Speech synthesis crashed — delivering text only")

    return state


# A user upload is persisted by vision_agent_node under this stem; a picture
# the assistant drew never is. That is how a follow-up can tell "the photo the
# user sent" from "the render on screen" without a register lookup.
_USER_UPLOAD_STEM = "_working_input_"


def is_video_sheet(path) -> bool:
    try:
        return Path(path).parent.name.startswith("video_") and Path(path).name == "sheet.jpg"
    except Exception:
        return False




def _close_look(ctx, image_bytes: bytes, question: str, overview: str) -> str:
    """The overview, enriched with the small details from zoomed parts
    (vision_close.look_closely); the overview alone when there is nothing
    to zoom into or the close look fails."""
    try:
        import vision_close
        closer = vision_close.look_closely(ctx, image_bytes, question, overview)
    except Exception:
        logger.warning("close look failed; the overview stands", exc_info=True)
        closer = None
    return closer or overview


def _picture_read(ctx, state) -> dict:
    """The model's read of this turn (agent/intent.py), shared with the loop:
    the same text and inputs graph_personality reads, so ONE call per turn.
    It replaced three word lists (a «describe/что нарисовано» list, an edit
    verb list, a question-word list) -- «Что нарисовано?» was answered with an
    enhance pass, «Скажи а» went to the vision model."""
    from graph_personality import _turn_intent
    return _turn_intent(ctx, state, state.get("user_input_original") or state.get("user_input", ""))


def _changes_picture(read: dict) -> bool:
    from graph_fastpath import _IMAGE_ACTION_TOOLS
    return bool(set(read.get("wants") or ()) & _IMAGE_ACTION_TOOLS)


def is_picture_question(ctx, state) -> bool:
    """A question ABOUT the picture, with no order to change it."""
    r = _picture_read(ctx, state)
    return bool((r["is_question"] or r["about_picture"]) and not _changes_picture(r))


def _file_newer_than(ctx, image_path) -> bool:
    """A file landed in the working folder after this picture: «в чём там ошибка?»
    is about the file (live 10-03: crash logs forwarded, the question re-read the
    last picture and critiqued an apple)."""
    root = getattr(getattr(ctx, "sandbox", None), "root", None)
    if root is None:
        return False
    try:
        t = Path(image_path).stat().st_mtime
        return any(p.is_file() and p.stat().st_mtime > t for p in Path(root).iterdir())
    except OSError:
        return False


def needs_relook(ctx: Optional[Context], state: AgentState) -> bool:
    """Should this turn look at the CURRENT working image again?

    Two cases. The model reads the message as about the picture (intent
    about_picture), or the user pointed at it.
    And ANY question when the working image is a photo the user sent: live
    (journey 2), a receipt was photographed and read, then "which item is the
    most expensive?" was answered with "I need to see the receipt" -- the first
    look had summarised the items and dropped the prices, and nothing routed
    the follow-up back to the photo. One extra vision call is cheap; a wrong
    answer about the user's own document is not.
    """
    if state.get("image_data"):
        return False
    last_path = getattr(ctx, "last_image_path", None) if ctx is not None else None
    if not last_path or not Path(last_path).exists():
        return False
    # Someone else's forwarded words are not the user's question: «За кого
    # проходит?» in a forwarded chat re-read the last photo four tiles deep
    # (live 2026-09-29) -- the read sees the user's own words only.
    if _file_newer_than(ctx, last_path):
        return False
    r = _picture_read(ctx, state)
    if _changes_picture(r):
        return False          # an order to change it: the editing path
    # The user POINTED at this picture (replied to it, pressed ❓ under it):
    # whatever they say is about it -- «обсудим цитату» is no question (16:57).
    if r["about_picture"] or getattr(ctx, "image_pointed_at", False):
        return True
    # Any question about a photo the user sent or a video they sent: a receipt
    # read once, then "which item is the most expensive?" (journey 2).
    return bool(r["is_question"]) and (is_video_sheet(last_path)
                                      or Path(last_path).name.startswith(_USER_UPLOAD_STEM))


def entry_router(state: AgentState, ctx: Optional[Context] = None) -> str:
    if state.get("image_data"):
        return "vision_agent"
    # The re-look branch of vision_agent_node was unreachable: this router
    # only ever sent a turn there WITH a new photo, so every follow-up about
    # the picture on screen went straight to the model and its memory.
    if needs_relook(ctx, state):
        return "vision_agent"
    return "personality"


def build_graph(ctx: Context):
    """Wire the nodes into the runnable graph. Wiring only — the behaviour lives
    in the node functions above.

    Each node is bound to `ctx` here rather than captured, and looked up on this
    MODULE at call time, so a suite that patches e.g. graph.personality_node
    after build_graph has already run still sees its patch take effect.
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("translate", lambda s: translate_node(ctx, s))
    workflow.add_node("vision_agent", lambda s: vision_agent_node(ctx, s))
    workflow.add_node("personality", lambda s: personality_node(ctx, s))
    workflow.add_node("tts", lambda s: tts_node(ctx, s))

    workflow.add_edge(START, "translate")
    workflow.add_conditional_edges(
        "translate",
        lambda s: entry_router(s, ctx),
        {"vision_agent": "vision_agent", "personality": "personality"},
    )
    workflow.add_edge("vision_agent", "personality")
    workflow.add_edge("personality", "tts")
    workflow.add_edge("tts", END)

    return workflow.compile()
