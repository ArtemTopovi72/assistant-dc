"""The room's brain: what a character says, and who talks next.

Split out of gui_madhouse_tab.py. Nothing in here touches Qt, a widget or the
event loop -- it is the part of the Madhouse that could run headless, and the
part worth reading on its own.

The reply-generation seam documented on generate_madhouse_reply is unchanged:
swap the body for any other backend and the tab keeps working, because
MadhouseReplyWorker only requires a plain string back.

Both functions are re-exported from gui_madhouse_tab, and MadhouseReplyWorker
still resolves them through that module's globals -- so a suite that patches
`gui_madhouse_tab.generate_madhouse_reply` steers the worker exactly as before.
"""
import logging
import random
import re

logger = logging.getLogger("assistant.gui")   # same channel as gui.py


def generate_madhouse_reply(ctx, character: dict, history: list) -> str:
    """Produce the next line for `character`.

    ---- THIS IS THE REPLY-GENERATION SEAM -------------------------------------
    Swap the body for any other backend (OpenAI, a local pipeline, a canned
    script) and the rest of the tab keeps working — it only requires that this
    returns a plain string and may raise. It runs on a worker QThread, so a slow
    network call is fine and never blocks the UI.

    `character` is {"id", "name", "voice", "prompt"}; `history` is the transcript
    as [{"name", "text", ...}, ...], oldest first.
    ---------------------------------------------------------------------------
    """
    if ctx is None:
        # No model loaded yet — keep the room alive with a placeholder line.
        return f"{character['name']} says: This is an automatic response"
    import llm

    def _text(res):
        out = ((res or {}).get("content") or "").strip()
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
        # Models like to prefix their own name even when told not to.
        return re.sub(rf"^{re.escape(character['name'])}\s*[:：-]\s*", "", out).strip()

    # Who else is in the room, including the human — so a character can address
    # people by name instead of talking into the void.
    others = [c["name"] for c in (character.get("_cast") or []) if c["id"] != character["id"]]
    human = (character.get("_human") or "").strip()
    roster = ", ".join(others) or "nobody yet"
    system = (
        f"{character['prompt'].strip()}\n\n"
        f"You are {character['name']}, one voice in a live group conversation. "
        f"In the room with you: {roster}."
        + (f" {human} is the real human in the room — a person, not a character; "
           "address them by name and take what they say seriously."
           if human else "")
        + " Stay in character. Reply with ONE short spoken turn (1-2 sentences), "
          "reacting to what was just said. If someone addressed you by name, answer "
          "THEM directly; address others by name when you speak to them. Plain "
          "speech only — no narration, no stage directions, no name prefix, no quotes."
    )
    # SpeakerMem: what has already scrolled out of the window below, filed per
    # speaker (madhouse_memory). Off switch: MADHOUSE_SPEAKER_MEMORY=0.
    import os
    if os.getenv("MADHOUSE_SPEAKER_MEMORY", "1") != "0":
        import madhouse_memory
        room = madhouse_memory.distil(ctx, history, window=12)
        block = madhouse_memory.prompt_block(
            room, character["name"], others + ([human] if human else []))
        if block:
            system += "\n\n" + block
    turns = []
    for m in history[-12:]:
        role = "assistant" if m.get("character_id") == character["id"] else "user"
        turns.append({"role": role, "content": f"{m['name']}: {m['text']}"})
    # The window must not OPEN on this character's own line. Once the transcript is
    # longer than the window, the slice can start with an "assistant" turn — system
    # then assistant, with nothing to answer — and the model returns empty every
    # time. With a two-character cast the roles alternate perfectly, so that shape
    # arrives on the third turn: "each says one phrase, then an error" (reproduced
    # 2026-07-22, call#15 roles=sauauau... -> EMPTY on all retries). Dropping the
    # leading own-turns costs a line or two of already-truncated context.
    while turns and turns[0]["role"] == "assistant":
        turns.pop(0)
    msgs = [{"role": "system", "content": system}] + turns
    if len(msgs) == 1:
        msgs.append({"role": "user", "content": "Open the conversation."})
    # Retry ladder. The <think></think> prefill is what makes the reasoning
    # fine-tunes answer in character at all: measured on the 35B with a real room
    # prompt, WITH the prefill 12/12 attempts returned a line, WITHOUT it 6/6 came
    # back empty (the model opens an unclosed <think> that strip_think_tags reduces
    # to nothing). So a retry must KEEP the prefill and give it more room — dropping
    # it, as this used to, turned a flaky first attempt into a guaranteed failure.
    # The bare attempt stays last, only for models the prefill silences entirely.
    attempts = (
        {"max_tokens": 200, "prefill": "<think></think>"},
        {"max_tokens": 500, "prefill": "<think></think>"},
        {"max_tokens": 1200},
    )
    text = ""
    for i, kwargs in enumerate(attempts, 1):
        res = llm.send_to_lm_studio(ctx, msgs, tools=[], tool_choice="none",
                                    temperature=0.9, **kwargs)
        text = _text(res)
        if text:
            break
        logger.warning(
            "Madhouse reply attempt %d/%d produced nothing for %s (model=%s, "
            "prefill=%s, raw content=%r)", i, len(attempts), character["name"],
            getattr(ctx, "model_name", "?"), "prefill" in kwargs,
            (res or {}).get("content"))
    if not text:
        # NEVER fall back to the placeholder here: a silent fake line makes a broken
        # model call look like a working room. Raise so the tab prints the reason.
        # Cancellation is by far the most common cause and looks nothing like a
        # model fault, so name it explicitly rather than blaming LM Studio.
        try:
            cancelled = bool(ctx.is_cancelled())
        except Exception:
            cancelled = False
        raise RuntimeError(
            "the turn was cancelled before the model answered" if cancelled else
            f"the model returned no text (is LM Studio still serving "
            f"'{getattr(ctx, 'model_name', '?')}'?)")
    return text


def _name_is_addressed(name: str, text: str) -> bool:
    """Was `name` spoken to in `text`? Tolerates Russian case endings (Лёха → Лёх/Лёхе)
    and diminutives by matching a stem, not the exact word."""
    low = text.lower()
    if re.search(rf"(?<!\w){re.escape(name.lower())}(?!\w)", low):
        return True
    stem = name.lower()[:-1] if len(name) > 3 else name.lower()
    return len(stem) >= 3 and bool(re.search(rf"(?<!\w){re.escape(stem)}\w{{0,3}}(?!\w)", low))


def choose_next_speaker(ctx, characters, history, last_id, use_router=True, human_name=""):
    """Decide who talks next — the room's turn-taking brain.

    Three tiers, cheapest first:
      1. Direct address — if the last line names someone, that person answers. No
         model call: being called by name is not a judgement call.
      2. LLM router — asks the model who would naturally speak next, given the cast
         and the transcript. This is what makes it feel like people rather than a
         round-robin.
      3. Random non-repeating pick — the fallback whenever the router is off,
         unavailable, or returns something unrecognizable.
    """
    pool = [c for c in characters if c["id"] != last_id] or list(characters)
    if not pool:
        return None
    if not history:
        return random.choice(pool)

    last = history[-1]
    addressed = [c for c in pool if _name_is_addressed(c["name"], last["text"])]
    if addressed:
        return random.choice(addressed)

    if not use_router or ctx is None:
        return random.choice(pool)

    try:
        import llm
        roster = "\n".join(f"- {c['name']}: {(c['prompt'] or '').strip()[:140]}" for c in pool)
        convo = "\n".join(f"{m['name']}: {m['text']}" for m in history[-8:])
        msgs = [
            {"role": "system",
             "content": "You direct a live group conversation. Given the cast and the "
                        "transcript, decide who would most naturally speak NEXT — the "
                        "person addressed, challenged, asked a question, or the one with "
                        "the strongest reason to react. Answer with ONE name and nothing "
                        "else."},
            {"role": "user",
             "content": (f"Cast (choose one of these names):\n{roster}\n\n"
                         + (f"The human in the room is {human_name}.\n\n" if human_name else "")
                         + f"Transcript:\n{convo}\n\nWho speaks next?")},
        ]
        res = llm.send_to_lm_studio(ctx, msgs, tools=[], tool_choice="none",
                                    temperature=0.3, max_tokens=24,
                                    prefill="<think></think>")
        pick = ((res or {}).get("content") or "").strip()
        pick = re.sub(r"<think>.*?</think>", "", pick, flags=re.S).strip().strip('"\'.,:!? ')
        if pick:
            low = pick.lower()
            for c in pool:                      # exact, then contained
                if c["name"].lower() == low:
                    return c
            for c in pool:
                if c["name"].lower() in low or _name_is_addressed(c["name"], pick):
                    return c
            logger.info("Madhouse router returned an unknown name: %r", pick)
    except Exception:
        logger.exception("Madhouse router failed; falling back to a random speaker")
    return random.choice(pool)
