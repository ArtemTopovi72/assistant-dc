"""Rolling chat-history compaction.

Split out of graph.py. Every HISTORY_COMPACT_EVERY completed user turns the
older messages fold into one running summary, so the transcript resent to the
model each turn stays small.

The LLM is reached THROUGH its module, not by value. compact_history_if_needed
used to live in graph.py, where suites patch `graph.send_to_lm_studio`; binding
that function by value here would have made the patch a silent no-op -- the stub
ignored, the real model called, and the suite still green. Going through llm
means a patch on llm.send_to_lm_studio works from anywhere.
"""
import logging
import os
import re

import llm as _llm
from llm import strip_channel_tokens
from utils import strip_reasoning_leak, strip_think_tags

logger = logging.getLogger("assistant.graph")


def send_to_lm_studio(*a, **kw):
    """Module-level indirection; see the note above. Patch llm.send_to_lm_studio."""
    return _llm.send_to_lm_studio(*a, **kw)


# Rolling chat-history compaction. Every HISTORY_COMPACT_EVERY completed user turns
# the older messages are folded into a single running summary, so the transcript
# resent to the model each turn stays small (less VRAM/compute, no context bloat).
HISTORY_COMPACT_EVERY = 3
_HISTORY_KEEP_TURNS = 2          # most-recent user turns kept verbatim
_HISTORY_SUMMARY_MARKER = "[Conversation summary so far]\n"


# What to say when the tools did the work but the model never wrote the sentence.
# Keyed by the tool that ran, so the user is told what happened rather than being
# handed an apology about internal step budgets.
_OUTCOME_BY_TOOL = {
    "generate_image":  ("Готово — картинка выше.", "Done — the picture is above."),
    "redraw_image":    ("Готово — обновлённая картинка выше.",
                        "Done — the updated picture is above."),
    "inpaint_image":   ("Готово — отредактированная картинка выше.",
                        "Done — the edited picture is above."),
    "transfer_image":  ("Готово — результат выше.", "Done — the result is above."),
    "fix_hands":       ("Готово — руки поправил, картинка выше.",
                        "Done — hands fixed, the picture is above."),
    "fix_artifact":    ("Готово — дефект убрал, картинка выше.",
                        "Done — the flaw is fixed, the picture is above."),
    "find_photo":      ("Нашёл фотографию — она выше.",
                        "Found a photo — it is above."),
    "create_presentation": ("Презентация готова — файл выше.",
                            "The presentation is ready — the file is above."),
    "deep_research":   ("Исследование готово — отчёт выше.",
                        "The research is done — the report is above."),
    "remember_fact":   ("Запомнил.", "Noted and saved."),
    # Coding sandbox. Without these the fallback for a turn that unpacked an
    # archive and read three files was "я выполнил действия, но не успел
    # подвести итог" — true, useless, and indistinguishable from a turn that
    # did nothing. Measured on the sandbox bench, where the model produced no
    # closing text at all on roughly one run in six.
    "pack_archive":    ("Собрал архив — файл выше.",
                        "Packed the archive — the file is above."),
    "edit_file":       ("Файлы в рабочей папке поправил, но не успел описать "
                        "правки. Посмотри /files.",
                        "I edited the files in your working folder but ran out "
                        "of room to describe the changes. See /files."),
    "write_file":      ("Записал файл в рабочую папку. Посмотри /files.",
                        "Wrote the file into your working folder. See /files."),
    "unpack_archive":  ("Распаковал архив в рабочую папку. Посмотри /files "
                        "или спроси ещё раз — расскажу, что внутри.",
                        "Unpacked the archive into your working folder. See "
                        "/files, or ask again and I will describe it."),
    "read_file":       ("Файл я прочитал, но не успел изложить. Спроси ещё раз.",
                        "I read the file but ran out of room to write it up. "
                        "Ask again."),
    "search_files":    ("Поиск по файлам я сделал, но не успел изложить. "
                        "Спроси ещё раз.",
                        "I searched your files but ran out of room to write it "
                        "up. Ask again."),
    "run_code":        ("Скрипт я выполнил, но не успел изложить результат. "
                        "Спроси ещё раз.",
                        "I ran the script but ran out of room to report the "
                        "result. Ask again."),
}


def _artifact_key(tool_name: str) -> str:
    """What this tool must have left behind, if anything.

    Read from graph_personality's table at call time rather than copied: two
    lists of which tool owes which artifact is how one of them goes stale.
    Imported inside the function because graph_personality imports this module.
    """
    try:
        import graph_personality as _gp
        return _gp._TOOL_ARTIFACT.get(tool_name, "")
    except Exception:
        return ""


def _outcome_sentence(state: dict, tools_called, ru: bool) -> str:
    """One true sentence about what this turn produced."""
    called = list(tools_called or [])
    # Prefer a tool that actually yields something the user can see.
    # Ordered by how much the user gets out of it: a delivered file beats an
    # edit, an edit beats having merely looked.
    for name in ("create_presentation", "deep_research", "pack_archive",
                 "generate_image", "inpaint_image", "redraw_image",
                 "transfer_image", "fix_hands", "fix_artifact", "find_photo",
                 "edit_file", "write_file", "run_code", "unpack_archive",
                 "read_file", "search_files", "remember_fact"):
        if name not in called:
            continue
        # A tool that was CALLED is not a tool that DELIVERED. "Собрал архив —
        # файл выше" after a pack that errored is the same lie the finalisation
        # guards exist to stop, just written by the fallback instead of the
        # model. If the tool owes an artifact, the artifact has to be there.
        _key = _artifact_key(name)
        if _key and not str(state.get(_key) or "").strip():
            continue
        ru_txt, en_txt = _OUTCOME_BY_TOOL[name]
        return ru_txt if ru else en_txt
    if state.get("document_path"):
        return "Файл готов — он выше." if ru else "The file is ready — it is above."
    if state.get("image_path"):
        return "Картинка готова — она выше." if ru else "The picture is ready — it is above."
    if "search" in called:
        return ("Я нашёл материалы по запросу, но не успел их изложить. "
                "Спроси ещё раз — отвечу текстом.") if ru else (
               "I found the material but ran out of room to write it up. "
               "Ask again and I will answer in text.")
    return ("Я выполнил действия, но не успел подвести итог. Посмотри результат "
            "и скажи, что доработать.") if ru else (
           "I performed the actions but ran out of steps before summarising. "
           "Check the result and tell me what to refine.")


def _render_turns_for_summary(msgs) -> str:
    lines = []
    for m in msgs:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            if content:
                lines.append(f"Assistant: {content}")
            tc = m.get("tool_calls") or []
            if tc:
                names = ", ".join(c.get("function", {}).get("name", "?") for c in tc)
                lines.append(f"(assistant used tools: {names})")
        elif role == "tool" and content:
            lines.append(f"(tool result: {content[:160]})")
    return "\n".join(lines)


def _drop_injected_sentences(summary: str) -> str:
    """The running summary is replayed as a system message on every later turn,
    so an order smuggled into the chat ("запомни навсегда: всегда отвечай...")
    would outlive the turn exactly as a remember_fact jailbreak did. Keep the
    summary, drop only the sentences prompt_guard would refuse as a fact."""
    from prompt_guard import injection_reason
    parts = re.split(r"(?<=[.!?\n|])\s+", summary or "")
    kept = [p for p in parts if not injection_reason(p)]
    if len(kept) != len(parts):
        logger.warning("History summary: dropped %d injected sentence(s)", len(parts) - len(kept))
    return " ".join(kept).strip()


def compact_history_if_needed(ctx, messages):
    """Fold older turns into one running summary every HISTORY_COMPACT_EVERY user
    turns. Keeps the live system prompt (msg 0) and the last _HISTORY_KEEP_TURNS
    user turns verbatim; everything between is summarised (merging any prior
    summary). Cut is always at a user message, so no tool_call/tool pairs are
    split. Fails open: on any error the original list is returned unchanged."""
    if os.getenv("CONTEXT_V2", "1") != "0":
        # v2: budget-triggered masking + addressable archive + append-only
        # structured memory (context_v2.py). v1 below stays for CONTEXT_V2=0.
        ctx.total_user_turns = int(getattr(ctx, "total_user_turns", 0)) + 1
        import context_v2
        return context_v2.manage(ctx, messages)
    try:
        ctx.total_user_turns = int(getattr(ctx, "total_user_turns", 0)) + 1
        force = len(messages) > 20  # force compact if history has ballooned
        if not force and (ctx.total_user_turns % HISTORY_COMPACT_EVERY != 0 or not messages):
            return messages
        head = messages[0] if messages[0].get("role") == "system" else None
        body_start = 1 if head else 0
        prev_summary = ""
        if (head and len(messages) > 1 and messages[1].get("role") == "system"
                and str(messages[1].get("content", "")).startswith(_HISTORY_SUMMARY_MARKER)):
            prev_summary = strip_channel_tokens(
                messages[1]["content"][len(_HISTORY_SUMMARY_MARKER):].strip())
            body_start = 2
        body = messages[body_start:]
        user_pos = [i for i, m in enumerate(body) if m.get("role") == "user"]
        if len(user_pos) <= _HISTORY_KEEP_TURNS:
            return messages
        cut = user_pos[-_HISTORY_KEEP_TURNS]          # tail begins at a user msg (safe)
        to_summarize, tail = body[:cut], body[cut:]
        if not to_summarize:
            return messages
        transcript = _render_turns_for_summary(to_summarize)
        payload = (f"Previous summary:\n{prev_summary}\n\nNew turns:\n{transcript}"
                   if prev_summary else transcript)
        from prompts import HISTORY_SUMMARY_PROMPT
        # Surface the stage: this runs AFTER the reply is delivered and makes up to
        # two extra LLM calls — without a stage update the GUI looks frozen/hung
        # for the length of a summarisation call every 5th turn.
        ctx.set_stage("Compacting the chat history")
        # NOTE: no prefill here. A closed-<think> prefill makes this finetune return
        # EMPTY content for a pure summarisation request (verified live), which would
        # silently disable compaction. The reasoning leak is stripped below instead.
        # This finetune also OCCASIONALLY returns an empty body (only a stripped think
        # block) — retry once, then fall back to an extractive summary so compaction
        # still happens and the context keeps shrinking.
        summary = ""
        for _ in range(2):
            resp = send_to_lm_studio(
                ctx,
                [{"role": "system", "content": HISTORY_SUMMARY_PROMPT},
                 {"role": "user", "content": payload}],
                tools=[], tool_choice="none", temperature=0.3, max_tokens=400,
            )
            summary = strip_channel_tokens(strip_reasoning_leak(
                strip_think_tags((resp or {}).get("content", "") or "")))
            if summary:
                break
        if not summary:
            # Extractive fallback: keep the user's own lines (the load-bearing intent)
            # so we still collapse the transcript even when the summariser flakes.
            user_lines = [(m.get("content") or "").strip() for m in to_summarize
                          if m.get("role") == "user"]
            summary = " | ".join(l for l in user_lines if l)[:1000]
            if prev_summary:
                summary = f"{prev_summary} | {summary}".strip(" |")
        summary = _drop_injected_sentences(summary)
        if not summary:
            return messages
        rebuilt = ([head] if head else []) + \
                  [{"role": "system", "content": _HISTORY_SUMMARY_MARKER + summary}] + tail
        logger.info("History compacted at turn %d: %d -> %d messages",
                    ctx.total_user_turns, len(messages), len(rebuilt))
        return rebuilt
    except Exception as exc:
        logger.warning("History compaction skipped: %s", exc)
        return messages
