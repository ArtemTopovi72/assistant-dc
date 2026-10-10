"""The fast path: answer without the tool loop when no tool can help.

Split out of graph_personality.py unchanged. Holds the intent-prefix tool
map, the image-intent detection that decides whether the image tool schemas
are worth ~2000 tokens this turn, and the two functions that decide on and
render a short-circuit reply.

# NOTE the `import graph as _g` inside the functions below. It is load-bearing,
# not a style choice: send_to_lm_studio / execute_tool / TOOL_SCHEMAS /
# _TOOL_TRIGGER_RE are patched directly on the `graph` module by a dozen suites,
# so they must be read through the module object at CALL time. Binding them by
# value at import would freeze the real implementations and make every one of
# those patches a silent no-op. See graph_personality.py's module docstring.
"""
import logging
import re

from graph_compose import (_GenSettings, FORGOTTEN_NOTE, _word_count_note, facts_parts, no_history_note,
                           translate_previous_note)
from models import AgentState, Context
from prompts import build_system_prompt_lite
from utils import strip_think_tags, strip_reasoning_leak
import tool_retrieval as _retrieval

logger = logging.getLogger("assistant.graph")


# Indirect-prompt-injection guard. Tools that return attacker-influenceable external
# text (clipboard, web) must not be able to STEER the turn into an image action the
# user never asked for — the live-fuzz finding was that clipboard content saying
# "call generate_image with description=…" drove a real unauthorized generation on a
# weak model (5/5). A hard block is safe here because the legitimate case ("нарисуй
# то, что в буфере") carries an image verb in the USER's OWN message, while the
# injection ("сделай, что там написано") does not — so we gate on user intent, not on
# the untrusted data. Model-independent: enforced by the framework regardless of what
# the model decides.
_UNTRUSTED_DATA_TOOLS = {"read_clipboard", "search", "ozon_product", "ozon_reviews", "ozon_shop"}


_DELIVERABLE_TOOLS = ("generate_image", "generate_video", "create_presentation", "find_photo")
_IMAGE_ACTION_TOOLS = {"generate_image", "redraw_image", "inpaint_image",
                       "transfer_image", "fix_hands", "fix_artifact"}




# Fast-path responses that falsely claim the model cannot create/edit images.
# These must not be saved to history — they poison future turns by making the
# model believe it told the user it can't draw, then refusing again.
# The fast path answered with no tools; if that answer is the model DENYING a
# capability the agent actually has, throw it away and fall through to the real
# loop where the tool exists. This was image-only, and the bench caught what
# that missed: "я текстовый помощник и не могу напрямую создать видеофайл, но
# вы можете использовать Runway или Pika" (video) and "я не вижу содержимое
# вашего буфера автоматически" (clipboard) both sailed through as final
# answers. The rule is capability-agnostic now — a denial about ANY deliverable
# is a fall-through, because the loop can only do better than a flat refusal.
# A tool-less reply that CLAIMS a deliverable. "а теперь просто картинку
# этого же кота" had no draw verb, took the fast path, and the model wrote
# "Вот ваша картинка с рыжим котом" with nothing attached (live, 2026-09-12,
# journey 28). Such a reply is thrown away and the turn falls through to the
# tool loop, exactly like a capability denial.
_FAST_PATH_DELIVERY_CLAIM_RE = re.compile(
    r"^\s*(?:(?:конечно|отлично|хорошо|сделано|sure|of\s+course|okay|ok)[!,.]?\s+)?"
    r"(?:вот|держи\w*|готово|here\s+(?:is|are|you\s+go)|done)\b[^.!?\n]{0,60}?"
    r"(?:картин\w*|изображен\w*|рисун\w*|фото\w*|видео|ролик\w*|клип\w*|презентац\w*|"
    r"файл\w*|документ\w*|песн\w*|трек\w*|picture|image|photo|drawing|video|clip|"
    r"presentation|deck|file|song|track)|"
    r"(?:картинк\w*|изображени\w*|видео|презентац\w*|файл\w*)\s+(?:готов\w*|сохранен\w*|отправлен\w*|создан\w*)|"
    # the whole reply is «Вот так выглядит Эйфелева башня ночью.» with nothing shown
    r"^\s*(?:вот|here'?s?)\s+(?:так\s+|как\s+|what\s+)?[^.!?\n]{0,60}?(?:выгляд\w*|looks?\s+like)[^.!?\n]*[.!]?\s*$|"
    r"(?:I(?:'ve| have)\s+)?(?:created|generated|drawn|attached|sent|made)\s+(?:the|a|your)\s+"
    r"(?:picture|image|photo|video|clip|presentation|deck|file|song)",
    re.IGNORECASE)


# The fast path has NO tools, so any answer that reports or promises an action
# is invented. Routing bench 2026-09-24, paraphrases with no cue word: "Хорошо,
# я забыл все факты" (forget_facts never ran), "я запустил глубокое
# исследование", `{"action": "dalle.text2im", ...}` written as text, prompts
# offered instead of a picture, "Этот код выведет 499999500000" (never run).
# Each one falls through to the full loop, which has the tools.
_FAST_PATH_ACTION_CLAIM_RE = re.compile(
    # a tool call written out as text
    r"\"action(?:_input)?\"\s*:|<tool_call>|\{\s*\"name\"\s*:\s*\"\w+\"\s*,\s*\"(?:arguments|parameters)\"|"
    # first-person side effects in the past tense
    r"\bя\s+(?:\w+\s+){0,2}?(?:забыл|удалил|стёр|стер|запомнил|записал|сохранил|запустил|"
    r"отправил|добавил|сгенерировал|нарисовал|создал|очистил|поставил|установил)\w*|"
    r"^\s*(?:хорошо|готово|ок|окей|сделано)[,.!]?\s+(?:\w+\s+){0,2}?"
    r"(?:забыл|удалил|стёр|стер|запомнил|записал|сохранил|запустил)\w*|"
    # a promise to act right now
    r"\b(?:запускаю|начинаю|приступаю|генерирую|рисую|создаю|напомню|отправляю|"
    r"звоню|заказываю|оформляю|бронирую|оплачиваю)\b|"
    # passive "done" claims: «Будильник на 7 утра отменен» with no tool (live)
    r"(?:напоминани|будильник)\w*[^.!?\n]{0,30}\b(?:отмен[её]н|удал[её]н|снят|установлен|поставлен)\w*|"
    r"\bя\s+(?:\w+\s+){0,2}?отменил\w*|"
    # «Запомнил, 47 -- интересное число» with no remember_fact (live)
    r"(?:^|[.!?]\s+)(?:запомнил|записал|сохранил|отменил|удалил|поставил|установил|"
    r"сделал|добавил|убрал|изменил|переделал|сократил|исправил|обновил|заменил)\w*\b|"
    r"\b(?:разбужу|напомню|поставлю\s+(?:таймер|будильник))\b|"
    r"\bсейчас\s+(?:я\s+)?(?:сделаю|нарисую|сгенерирую|создам|запущу|найду|подготовлю|поищу)\b|"
    # «Хорошо, я сделаю фон светлым и белым для всех слайдов.» — and nothing was done
    r"^\s*(?:(?:хорошо|ок|окей|конечно|отлично|понял|договорились)[,!.]?\s+)+(?:я\s+)?"
    r"(?:сделаю|изменю|поменяю|переделаю|добавлю|уберу|нарисую|создам|отправлю|сгенерирую|исправлю)\b|"
    # doing the tool's job in prose instead
    r"промпт\w*\s+для\s+(?:генерац|нейросет|midjourney|stable)|"
    r"(?:код|скрипт|программа)\s+(?:выведет|вернёт|вернет|напечатает)|"
    r"\bя\s+могу\s+(?:\w+\s+){0,2}?(?:созда|сгенер|нарис|сдела|найти|запуст)\w*[^.!?]{0,80}\b(?:но|если)\b|"
    r"\bI(?:'ve| have)\s+(?:forgotten|deleted|removed|saved|started|launched)\b|"
    r"\bI(?:'m| am)\s+(?:now\s+)?(?:generating|creating|drawing|searching|running)\b",
    re.IGNORECASE | re.MULTILINE)


_FAST_PATH_CAPABILITY_DENIAL_RE = re.compile(
    # Russian: "не могу/не умею/не имею возможности" + any deliverable verb.
    r"не\s+(?:могу|умею|способ\w+|имею\s+возможности)\s+"
    r"(?:\w+\s+){0,3}?(?:создав?|созда|нарис|генер|рисов|сделать|сформир|"
    r"постро|записать|прочитать|получить|отправ|показать)|"
    # "я не вижу …", "у меня нет доступа", "я текстовый помощник/модель".
    r"я\s+не\s+вижу|"
    r"(?:у\s+меня\s+)?нет\s+(?:прямого\s+)?доступа|"
    r"я\s+(?:—\s*|-\s*)?(?:просто\s+)?текстов\w+\s+(?:помощник|модель|ассистент)|"
    # Deflection to a competitor product instead of using our own tool.
    r"вы\s+можете\s+использовать\s+(?:нейросет|сервис|Runway|Pika|Midjourney)|"
    # Asking the user to paste what a tool could have fetched.
    r"вставьте\s+(?:текст|содержимое)|"
    # English mirrors of all of the above.
    r"cannot\s+(?:create|generate|draw|produce|access|read|build|make)|"
    r"can\'t\s+(?:create|generate|draw|produce|access|read|build|make)|"
    r"unable\s+to\s+(?:create|generate|draw|produce|access|read)|"
    r"I\s+don\'t\s+have\s+(?:the\s+ability|access|permission)|"
    r"I\s+am\s+(?:just\s+)?a\s+text[- ]based",
    re.IGNORECASE,
)


# When the user taps a keyboard button, their intent is unambiguous — only the
# relevant tool(s) are needed. Checked against the start of user_input (which
# carries the pending_prefix prepended by the Telegram handler).
_INTENT_TOOL_MAP: list[tuple[str, frozenset]] = [
    ("generate an image of:",          frozenset({"generate_image"})),
    ("regenerate the image",           frozenset({"generate_image"})),
    ("edit the image:",                frozenset({"redraw_image", "inpaint_image",
                                                   "fix_hands", "fix_artifact",
                                                   "transfer_image", "inspect_image"})),
    # tg_callbacks._cb_change_clothes' free-text ask-flow prefix (2026-09-19).
    # Without an entry here this button shipped the FULL ~31-tool list on
    # every press — caught by test_night_batch.py's own "narrows the tool
    # list" check, which _CB_CMDS["change_clothes"]'s bare text never
    # satisfied either (no prefix here matched it, before or after that
    # button gained its ask-flow).
    ("change the outfit to:",          frozenset({"redraw_image", "inpaint_image",
                                                   "inspect_image"})),
    # 🧽 Remove object (button under a picture / in the Draw menu).
    ("remove from the image:",         frozenset({"inpaint_image", "inspect_image"})),
    # 🔤 Remove lettering: inpaint_image's OCR-mask removal path (is_lettering_removal).
    ("remove all the lettering",       frozenset({"inpaint_image", "inspect_image"})),
    # _CB_CMDS["change_clothes"]'s own fallback text/shape, kept narrowed too
    # in case anything still enqueues it directly instead of going through
    # the ask-flow above.
    ("change the person's outfit to",  frozenset({"redraw_image", "inpaint_image",
                                                   "inspect_image"})),
    ("search the web for:",            frozenset({"search"})),
    ("do a deep research on:",         frozenset({"deep_research"})),
    ("remember this:",                 frozenset({"remember_fact"})),
    ("describe and analyze this image", frozenset({"inspect_image"})),
    # The inline buttons under every picture. These were MISSING, so pressing one
    # button — the most unambiguous input the product has — shipped the entire tool
    # list on every round of the loop. The payloads are machine-generated and start
    # with a literal tag, so matching them is exact, not a guess.
    ("[upscale]",                      frozenset({"redraw_image"})),
    ("[enhance]",                      frozenset({"redraw_image"})),
    ("[restore]",                      frozenset({"redraw_image"})),
    ("outpaint the current image",     frozenset({"redraw_image"})),
    ("find a photo of:",               frozenset({"find_photo"})),
    ("create a presentation about:",   frozenset({"create_presentation"})),
    ("summarize this transcript",      frozenset()),   # pure text work, no tools
]


def _intent_tools(user_input: str):
    """Return the frozenset of tool names for a button-driven intent prefix,
    or None if the message has no recognized prefix (full tool set applies)."""
    lower = (user_input or "").lower().lstrip()
    for prefix, names in _INTENT_TOOL_MAP:
        if lower.startswith(prefix):
            return names
    return None


# Image-manipulation tools — omitted entirely when no image context exists and the
# message has no image-creation/editing keywords. Saves ~2000 tokens per call.
_IMAGE_TOOL_NAMES = frozenset({
    "generate_image", "redraw_image", "inpaint_image", "inspect_image",
    "transfer_image", "fix_hands", "fix_artifact", "find_photo",
})




def _sandbox_has_files(ctx) -> bool:
    """Does this turn have a working folder with anything in it?

    Cheap on purpose -- one directory listing, stopped at the first entry --
    because it runs on every turn. Any failure answers False: an unreadable
    sandbox is a reason to take the ordinary path, never a reason to fail a
    greeting.
    """
    box = getattr(ctx, "sandbox", None)
    if box is None:
        return False
    try:
        return any(box.root.iterdir())
    except Exception:
        return False


def _previous_reply(state) -> str:
    """The last assistant line before the current message: context for the
    intent read of a short follow-up («а в евро?»)."""
    msgs = list(state.get("messages") or [])
    if msgs and msgs[-1].get("role") == "user":
        msgs = msgs[:-1]
    for m in reversed(msgs):
        if m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"].strip():
            return m["content"]
    return ""


def _intent(ctx, state, user_input: str, original_input: str) -> dict:
    """The model's read of what this message asks for (agent/intent.py)."""
    # Same inputs as graph_personality's reads, so the turn costs ONE model call.
    from graph_personality import _turn_intent
    return _turn_intent(ctx, state, original_input or user_input)


def _is_photo_intake(ctx, state: AgentState, user_input: str, original_input: str) -> bool:
    """A photo that just arrived with no request attached: no caption at all
    ("image"), or a caption that only names or introduces it («Вот Френк
    Харриган»). Such a turn only needs the vision description turned into
    "this is X -- what should I do with it?" (live 2026-09-22: the full loop
    deliberated for over a minute and never answered)."""
    if not state.get("image_data") or not (state.get("vision_summary") or "").strip():
        return False
    if user_input.strip() == "image":
        return True
    got = _intent(ctx, state, user_input, original_input)
    return got["presents_photo"] and not got["needs_tool"]


def _followup_of_tool_turn(state, text: str, ctx=None) -> bool:
    """A message right after a turn that used a tool, which the model reads as
    needing a tool too: «а в Нью-Йорке?» after local_time (live 2026-09-28,
    the fast path guessed the clock). It gets the previous turn's tools."""
    msgs = list(state.get("messages", []))
    if msgs and msgs[-1].get("role") == "user":
        msgs = msgs[:-1]                     # the current message, if already in
    users = [i for i, m in enumerate(msgs) if m.get("role") == "user"]
    if not users or not any(m.get("role") == "tool" for m in msgs[users[-1]:]):
        return False
    import intent
    return intent.read(ctx, text, _previous_reply(state))["needs_tool"]


def _fast_path_allowed(ctx: Context, state: AgentState, user_input: str,
                       original_input: str) -> bool:
    """Whether this turn may skip the tool loop entirely.

    What the message asks for is read by the model (agent/intent.py), not by
    keyword regexes: «добавили» held «оба», paraphrases with no cue word took
    this path and got invented answers. The checks left here are about the
    STATE (a picture, a deck, a video, files in play), not the words.
    """
    if ctx.is_cancelled():
        return False
    if _is_photo_intake(ctx, state, user_input, original_input):
        return True
    if _sandbox_has_files(ctx):
        return False
    got = _intent(ctx, state, user_input, original_input)
    if got["bare_ack"] and not got["needs_tool"] and not state.get("image_data"):
        return True
    return (
        not got["needs_tool"]
        and not state.get("image_data")
        and not getattr(ctx, "last_image_path", None)
        # «сделай её короче» after a deck: "Сделал ещё короче" with nothing done.
        and not getattr(ctx, "last_deck", None)
        and not getattr(ctx, "last_video_path", None)
        and not ctx.is_cancelled()
    )


def _fast_path_reply(ctx: Context, messages: list, gen: "_GenSettings",
                     user_input: str, effective_input: str,
                     original_input: str, vision_summary: str = ""):
    """One tool-less LLM call for a short, tool-free turn.

    Returns the assistant message to append, or None to fall through to the
    full loop (no content, or the model claimed it cannot do something). It
    mutates nothing: the caller owns `messages` and `state`.
    """
    import graph as _g
    # Use a tiny system prompt (no tool docs) — keeps the fast path under
    # ~300 tokens even with facts included; tool schemas are the expensive
    # part (2-3k tokens), not the couple of lines a fact store holds.
    #
    # Saved facts ARE still included below. Live 2026-09-18, journey 9: this
    # path used to skip them entirely, so "как меня зовут и что ты обо мне
    # знаешь?" (short, no tool trigger -> fast path) got a confident "you
    # haven't told me your name yet" three turns after remember_fact saved
    # it -- the model never saw the block that would have answered it.
    lite_system = build_system_prompt_lite(
        ctx.custom_personality_text, concise=gen.concise, length=gen.length,
        reply_lang=getattr(ctx, "reply_lang", "ru") or "ru",
        tz=getattr(ctx, "user_tz", "") or "")
    lite_user_parts = []
    facts_text = ctx.facts_text(original_input or user_input)
    if facts_text:
        lite_user_parts.extend(facts_parts(facts_text))
    elif getattr(ctx, "facts_forgotten", False):
        lite_user_parts.append(FORGOTTEN_NOTE)
    _tr_prev = translate_previous_note(original_input or user_input, messages[1:-1])
    if original_input and not _tr_prev:
        # A bare path or a number has no language: live 10-10 «C:/Program Files/Git/start»
        # in a Russian chat was answered in Spanish.
        _chat_lang = {"ru": "Russian", "en": "English"}.get(getattr(ctx, "reply_lang", "ru") or "ru", "Russian")
        lite_user_parts.append(
            f"[Reply in the same language as: «{original_input}» -- if it has no clear "
            f"language of its own (a path, a number, a link), reply in {_chat_lang}]")
    prefill = gen.prefill
    if vision_summary:
        # Photo intake (_is_photo_intake): describe it, then ask. The description
        # is already here, so there is nothing to deliberate about -- on Gemma
        # the empty thought block makes it answer at once.
        import llm
        lite_user_parts.append(f"Image description:\n{vision_summary}")
        if user_input.strip() != "image":
            effective_input = (
                f"{effective_input}\n\n[The user sent this photo with the message "
                "above. Briefly say what is in it, using the image description (one "
                "or two sentences, the concrete subject/scene, tied to what they "
                "said), then ask what they would like to do with it. Do not start "
                "editing or drawing anything.]")
        if llm._is_gemma4(getattr(ctx, "model_name", "") or llm.MODEL_NAME):
            prefill = llm.GEMMA_NO_THINK_PREFILL
    lite_user_parts.append(effective_input)
    # After the question: placed before it, the model ignored it once history existed (4/4 wrong).
    lite_user_parts.extend(p for p in [_word_count_note(original_input or user_input)
                                       or no_history_note(original_input or user_input,
                                                          messages[1:-1])
                                       or _tr_prev] if p)
    lite_user_msg = {"role": "user", "content": "\n\n".join(lite_user_parts)}

    fast_msgs = [{"role": "system", "content": lite_system}]
    # Include the last user→assistant pair so Jinja templates see proper
    # role alternation (system→assistant→user fails on Qwen's template).
    non_sys = [m for m in messages[1:-1]
               if m.get("role") in ("user", "assistant") and m.get("content")]
    # Up to 30 pairs within ~12000 chars: a fact from ten turns back must still be seen.
    pairs, _chars = [], 0
    i = len(non_sys) - 1
    while i >= 1 and len(pairs) < 60:
        if non_sys[i].get("role") != "assistant" or non_sys[i - 1].get("role") != "user":
            break
        _chars += len(str(non_sys[i].get("content"))) + len(str(non_sys[i - 1].get("content")))
        if _chars > 12000 and pairs:
            break
        pairs[:0] = [non_sys[i - 1], non_sys[i]]
        i -= 2
    fast_msgs.extend(pairs)
    fast_msgs.append(lite_user_msg)
    ctx.set_stage("Writing a response")
    fast_resp = _g.send_to_lm_studio(
        ctx, fast_msgs,
        tools=[], tool_choice="none",
        temperature=gen.temperature,
        max_tokens=min(gen.max_tokens, 600),
        prefill=prefill,
    )
    fast_content = strip_think_tags(
        strip_reasoning_leak((fast_resp or {}).get("content", "") or ""))
    if (fast_resp or {}).get("finish_reason") == "length":
        # Live: a 60-row table stopped at row 25 mid-cell and was delivered.
        logger.info("Fast path hit its token cap — falling through to the full loop")
        return None
    if fast_content and _FAST_PATH_DELIVERY_CLAIM_RE.search(fast_content):
        logger.info("Fast path claimed a deliverable with no tool — falling through to the full loop")
        return None
    if fast_content and _FAST_PATH_ACTION_CLAIM_RE.search(fast_content):
        logger.info("Fast path claimed or promised an action with no tool — falling through to the full loop")
        return None
    if fast_content and not _FAST_PATH_CAPABILITY_DENIAL_RE.search(fast_content):
        # The fast path returns before graph_finalize, so the drift guard
        # never saw its answers: "а теперь расскажи её по-русски" on an
        # English session came back in English (live, 2026-09-12, journey 26).
        from graph_language import _match_reply_language, _unparrot, _ru_thousands, _fix_totals, _prompt_leak_guard
        fast_content = _match_reply_language(ctx, fast_content, original_input)
        fast_content = _unparrot(ctx, fast_content, original_input, messages[1:])
        fast_content = _prompt_leak_guard(ctx, _ru_thousands(fast_content))
        fast_content = _fix_totals(fast_content)
        # Return the CLEANED answer, not the raw response. Storing the raw
        # content carried reasoning blocks and channel/control tokens into
        # every later context window (and into whatever renders the history),
        # which is the same family as the "<|channel>>…" leak. tool_calls are
        # dropped for the reason the forced-close path drops them below: this
        # call passes tool_choice="none", so any tool_calls a model emits
        # anyway would sit in history unanswered and invalidate a re-send.
        fast_msg = dict(fast_resp or {})
        fast_msg.pop("tool_calls", None)
        fast_msg["content"] = fast_content
        logger.debug("Fast path: %d words, ~%d msgs sent",
                     len(user_input.split()), len(fast_msgs))
        return fast_msg
    if fast_content:
        logger.debug("Fast path produced a capability denial — falling through to full loop")
    else:
        logger.debug("Fast path produced no content — falling through to full loop")
    return None
