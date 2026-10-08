"""Turning whatever the tool loop ended on into the one answer the user gets.

Split out of graph_personality.py unchanged. _finalize_answer is the retry
ladder plus the last-resort text; the module also owns the fabrication guard
(_ACTION_CLAIM_RE / _pop_fabrication_draft) that stops a draft claiming an
action the agent never took from reaching the user.

# NOTE the `import graph as _g` inside the functions below. It is load-bearing,
# not a style choice: send_to_lm_studio / execute_tool / TOOL_SCHEMAS /
# _TOOL_TRIGGER_RE are patched directly on the `graph` module by a dozen suites,
# so they must be read through the module object at CALL time. Binding them by
# value at import would freeze the real implementations and make every one of
# those patches a silent no-op. See graph_personality.py's module docstring.
"""
import logging
import os
import re

from graph_compose import _GenSettings
from graph_history import _outcome_sentence
from graph_language import _match_reply_language, _unparrot, _ru_thousands, _fix_totals, _prompt_leak_guard
from models import AgentState, Context
from utils import (strip_reasoning_leak, strip_textual_tool_calls,
                   collapse_verbatim_repetition)

logger = logging.getLogger("assistant.graph")


# Russian action verbs an answer uses when claiming an edit/creation either
# happened ("я добавил очки", "я перерисовал сцену") OR is about to happen right
# now ("сейчас перерисую", "запущу рисовку"). Used by the anti-fabrication guard:
# such a claim with ZERO tool calls in the turn means nothing actually ran — and
# the future-tense promise is the worse case, because the model keeps re-promising
# ("рисуй" -> "сейчас перерисую" -> ...) without ever calling the tool.
_ACTION_CLAIM_RE = re.compile(
    r"\b(?:"
    # past tense — claims a completed action
    r"добавил|заменил|перерисовал|нарисовал|сгенерировал|изменил|изменена|"
    r"перекрасил|убрал|удалил|исправил|отредактировал|поменял|сделал|"
    # future/present 1st person — promises to act now without acting
    r"перерисую|перерисуем|нарисую|нарисуем|рисую|рисуем|"
    r"сгенерирую|сгенерируем|генерирую|генерируем|"
    r"создам|создаю|создадим|изменю|изменим|перекрашу|перекрасим|"
    r"добавлю|добавим|заменю|заменим|убер[уеё]\w*|удал[юи]\w*|"
    r"исправлю|исправим|отредактирую|отредактируем|поменяю|поменяем|"
    r"сделаю|сделаем|запущу|запускаю"
    r")\w*\b"
    # English equivalents — the guard was Russian-only, so an English-language
    # session could fabricate "I've added the glasses" unchecked. Anchored to a
    # first-person pronoun so a bare mention of the verb doesn't trip it.
    r"|\bi\b[^.!?\n]{0,24}?\b(?:added|removed|replaced|changed|redrew|redrawn|"
    r"drew|generated|created|edited|erased|repainted|recolou?red|inpainted|"
    r"upscaled)\b"
    r"|\bi(?:['’]ll|\s+will|['’]?m\s+going\s+to)\s+(?:now\s+)?"
    r"(?:draw|redraw|generate|create|edit|add|remove|replace|fix|repaint|"
    r"inpaint|upscale)\b",
    re.IGNORECASE,
)


# The same promise, about FILES. Live 2026-09-14, twice in a row on a 112 MB
# archive: «Сначала я распакую архив ... Начинаю поиск.» / «сначала распакую
# его, а затем просканирую ... Начинаю исследование.» — and the turn ended with
# no tool call. A reply that announces file work with zero tool calls while
# the working folder has something in it is a promise, not an answer.
_FILE_PROMISE_RE = re.compile(
    r"\b(?:"
    r"распаку[юем]\w*|разархивиру[юем]\w*|извлеку|извлечём|извлечем|"
    r"просканиру[юем]\w*|проверю|проверим|изучу|изучим|исследую|исследуем|"
    r"открою|откроем|прочитаю|прочитаем|посмотрю|посмотрим|найду|найдём|найдем|"
    r"начинаю|начну|приступаю|приступлю|"
    # Live 2026-09-14, «заново» after a dedupe ask: «применю ... проведу ...
    # сформирую ... коллаж будет готов» -- a plan in the future tense with
    # zero tool calls slipped past the verbs above.
    r"применю|применим|провед[уё]м?|сформиру[юе]м?|собер[уё]м?|пересобер[уё]м?|"
    r"отбер[уё]м?|отфильтру[юе]м?|обработа[юе]м?|подготовл[юи]м?|напиш[уе]м?|"
    r"сравн[юи]м?|сгруппиру[юе]м?"
    r")\w*\b"
    r"|\bбуд[еу]т\s+готов\w*"
    r"|\bi(?:[\'’]ll|\s+will|[\'’]?m\s+going\s+to|[\'’]?m\s+starting\s+to)\s+(?:now\s+)?"
    r"(?:unpack|unzip|extract|open|read|scan|inspect|examine|look|search|explore|start|"
    r"rebuild|build|assemble|apply|run|write|filter|compare|group|prepare|process)\b"
    r"|\bwill\s+be\s+ready\b"
    r"|\b(?:starting|beginning)\s+(?:the\s+)?(?:search|scan|investigation|extraction|analysis)\b",
    re.IGNORECASE,
)


# «Я уже выполнил эту задачу!» to a request the user has just REPEATED (live
# 2026-09-14: the same dedupe ask sent twice, then «заново»). A repeated
# request means the last result did not satisfy them; an answer that points
# at the previous turn, with no tool called now, is a refusal dressed as
# diligence -- and after a restart the "previous turn" may be a lie the
# history preserved.
_ALREADY_DONE_RE = re.compile(
    r"\bуже\s+(?:выполн|сдела|пересобра|собра|удали|отправ|присла|готов|обработа)\w*"
    r"|\bв\s+(?:предыдущ|прошл)\w+\s+(?:шаге|ответе|сообщении|ходе|раз)"
    r"|\b(?:i(?:[\'’]ve)?|it(?:[\'’]s)?|this\s+(?:was|is))\s+already\s+"
    r"(?:did|done|completed|rebuilt|built|sent|removed|handled|finished)\b"
    r"|\bin\s+(?:the|my)\s+(?:previous|last)\s+(?:step|reply|message|turn)\b",
    re.IGNORECASE,
)

def _user_read(text: str) -> dict:
    """The model's read of the user's words (agent/intent.py)."""
    import intent
    return intent.read(None, text or "")


def _norm_request(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def _is_redo_request(user_input: str, messages) -> bool:
    """The user asked again: read as a redo, or the same text as an earlier turn."""
    if (user_input or "").strip() and _user_read(user_input)["redo"]:
        return True
    cur = _norm_request(user_input)
    if len(cur) < 12:
        return False
    seen = 0
    for m in messages or ():
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str) and cur in _norm_request(c):
            seen += 1
    return seen >= 2          # the current turn plus at least one earlier


# Imperative remember-requests in the user's input ("запомни: ...", "запиши...",
# "не забудь..."). When one is present and the turn produced an answer without a
# remember_fact call, the model is claiming to remember while saving nothing —
# same fabrication class as the image guard above. (The trailing \b keeps the
# imperative "запомни" from matching the model's own past tense "запомнил".)
# The 📊 Presentation button's literal prefix. Machine-generated, so matching it
# is exact rather than a guess about intent.
_DECK_PREFIX = "create a presentation about:"




def _fact_of_remember_request(text: str) -> str:
    """The fact itself, when the graph has to save the user's sentence verbatim
    ("запомни: меня зовут Артём" -> "меня зовут Артём"). The model drops the
    request words; its answer must be a piece of the user's own text, else the
    whole sentence is kept -- a polished paraphrase is not a verbatim save."""
    text = (text or "").strip()
    if not text or (os.getenv("F5_TEST_RUN") and not os.getenv("INTENT_LIVE")):
        return text
    from llm import call_llm_simple
    try:
        out = (call_llm_simple(None, "Copy text exactly; no comments.",
                               "Copy this message without its request words (remember, "
                               "запомни, write down, не забудь...), keeping the rest "
                               "exactly as written:\n" + text[:1000],
                               temperature=0.0, max_tokens=300) or "").strip().strip("«»\"")
    except Exception:
        return text
    return out if len(out) >= 3 and out.lower() in text.lower() else text


# The two "I produced nothing" texts. A turn that ends in one of them has NOT
# answered the question, and callers that keep a chat history must not store
# it as if it had: live (2026-09-12, journey 5) two such turns in a row left
# two unanswered questions in the transcript, and the model answered the
# OLDER of them when the user asked a third, different one.
_STUCK_RU = ("Я задумался и не выдал ответ — это сбой на моей "
             "стороне, а не отказ. Повтори вопрос, пожалуйста.")
_STUCK_EN = ("I got stuck thinking and produced no answer — that is a fault on "
             "my side, not a refusal. Please ask me again.")


def is_failure_placeholder(text) -> bool:
    """True when `text` is the no-answer placeholder (or empty)."""
    t = (text or "").strip()
    return not t or t in (_STUCK_RU, _STUCK_EN)


def _pop_fabrication_draft(messages: list, note: str) -> list:
    """Shared shape of a fabrication-guard corrective round: pull the model's
    unsupported claim back out of persisted history and hand it back on the next
    call together with a corrective system nudge. Returns the `outbound_extra`
    list for that one call; the caller still owns resetting last_message/
    guard_popped and its own per-guard `continue`."""
    claiming_draft = messages.pop()  # keep the lie out of history
    return [claiming_draft, {"role": "system", "content": note}]


# A markdown image embed the MODEL wrote. Pictures reach the user through the
# delivery layer (state['image_path'] -> tg_bot / the GUI), never through a
# URL in the text, so an ![...](http...) in an answer is always invented.
# Measured on the chaos bench: when find_photo failed, the model answered
# "Вот реальное фото Эйфелевой башни" followed by a fabricated link to
# upload.wikimedia.org -- a plausible URL for a picture it never found, which
# is worse than an empty answer because it looks like it worked.
_FABRICATED_IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def _strip_fabricated_images(text: str) -> str:
    """Remove model-written image embeds; keep ordinary links."""
    out = _FABRICATED_IMAGE_MD_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


# An answer that states a computed value. Paired with the knowledge that
# `calculate` failed and returned no value this turn, it can only be invented.
# Measured over 140 chaos runs, route_calc was the single biggest liar (13 of
# 27): with the calculator broken the model still answered "Результат
# вычисления 18432 * 977 + 15% равен 20 965 334.4" -- a different fabricated
# number every run, against a true value of 20 709 273.6.
#
# Handled here rather than with another corrective round because the two
# mechanisms measured very differently: the deterministic scrub of fabricated
# image links left ZERO through in 140 runs, while a corrective round is
# advice the model is free to ignore -- and did.
_STATED_RESULT_RE = re.compile(
    r"результат\s+вычислен|результат\s*:\s*[-+(]?\d"
    r"|равен\s*[-+(]?\d|\bравно\s*[-+(]?\d"
    r"|==\s*\d|(?<![<>=!])\s=\s*[-+(]?\d"
    r"|\b(?:equals|is)\s+[-+(]?\d{3,}",
    re.IGNORECASE,
)

# What a failed tool was supposed to hand over, in the words the user will
# understand. Used to write an honest sentence in place of a claim that the
# thing exists.
_PROMISE_NOUN = {
    "generate_image": "картинку",
    "find_photo": "фотографию",
    "redraw_image": "изменённую картинку",
    "inpaint_image": "изменённую картинку",
    "transfer_image": "изменённую картинку",
    "fix_hands": "исправленную картинку",
    "fix_artifact": "исправленную картинку",
    "generate_video": "видео",
    "create_presentation": "презентацию",
    "pack_archive": "готовый архив",
}


def _tool_said_unsupported(messages) -> bool:
    """The last tool result of the turn says the assistant cannot do this at all
    (an engine that was removed): «попробуй ещё раз» would send the user in
    circles (live 2026-10-08, «расширь картинку со всех сторон»)."""
    for m in reversed(list(messages or [])):
        if m.get("role") == "user":
            break
        if m.get("role") == "tool":
            return "[TOOL ERROR]" in str(m.get("content") or "") and \
                "not something this assistant can do" in str(m.get("content") or "")
    return False


def _honest_failure_line(failed_promises, unsupported: bool = False) -> str:
    nouns = [_PROMISE_NOUN[t] for t in sorted(failed_promises)
             if t in _PROMISE_NOUN]
    if not nouns:
        return ""
    if unsupported:
        return ("Такое я сделать не умею, поэтому ничего не менял. Повторять просьбу "
                "бесполезно; выдавать несделанное за готовое я не стану.")
    what = nouns[0] if len(nouns) == 1 else " и ".join((", ".join(nouns[:-1]),
                                                        nouns[-1]))
    return (f"Не получилось сделать {what} — инструмент вернул ошибку и ничего "
            f"не создал. Попробуй ещё раз, пожалуйста; выдавать несделанное за "
            f"готовое я не стану.")


# The user has been told nothing at all. Measured as the VAGUE bucket on the
# chaos bench: 12 of 140 runs where a tool had hard-failed and the answer
# neither claimed success nor admitted the failure -- not a lie, but the user
# walks away not knowing the thing they asked for was never made. An admission
# already in the draft must not be doubled, hence the counter-regex.
# A claim that a FILE now exists, as opposed to a claim about a picture or a
# number. Measured end to end on the sandbox bench: after editing a modpack the
# answer was "запаковал всё обратно, ваш обновленный архив готов" while
# pack_archive had never been called and document_path was empty -- the user
# would have been told the work was finished and handed nothing.
#
# No existing guard could fire: nothing errored (so failed_promises is empty)
# and no verifier spoke. The missing invariant is simpler than either -- if the
# answer says a file is ready, a file has to exist.
_FILE_READY_RE = re.compile(
    r"(?:архив\w*|файл\w*|презентац\w*|документ\w*|отч[её]т\w*|zip|jar)"
    r"[^.!?]{0,60}(?:готов|собран|запакован|прикреп|отправ|высла)"
    r"|(?:запаковал|упаковал|собрал|сформировал|приложил)[^.!?]{0,40}"
    r"(?:архив|файл|zip|jar|презентац|документ)"
    r"|(?:packed|zipped|attached)[^.!?]{0,40}(?:archive|file|zip|jar)",
    re.IGNORECASE,
)

_ARTIFACT_KEYS = ("document_path", "image_path", "video_path")

# The user's own words asking for a file. Needed only when forwarded material
# is in the turn: «презентация готова» there is usually the forwarded person's
# news being retold, not the bot's promise.

_NO_FILE_ANSWER = (
    "Готового файла у меня нет — собрать его я не успел, отдавать пока нечего. "
    "Скажи, продолжить?"
)


_ADMISSION_RE = re.compile(
    r"не\s+(?:получилось|удалось|смог\w*|вышло)|не\s+сработал"
    r"|ошибк\w*|сбой|не\s+сраб|техническ\w+\s+пробл"
    r"|\bfailed\b|\bcould\s?n[o']t\b|\berror\b",
    re.IGNORECASE,
)

# The verifier looked at the result and said the requested change is NOT there.
# Different from a failed tool: a file DOES exist and IS delivered, so the
# honest correction is "here it is, but the edit did not land", never "nothing
# was made". This is the deterministic half of a guard that until now existed
# only as a corrective round -- advice the model ignored on 9 of the 27 lying
# runs, every one of them in chaos mode `wrong`.
_VERIFY_NEGATIVE_ANSWER = (
    "Сделал, но проверка показала, что нужного изменения на результате нет — "
    "он остался прежним. Выдавать это за готовое не буду: скажи, "
    "попробовать ещё раз?"
)

_VERIFY_PARTIAL_ANSWER = (
    "Сделал, но проверка говорит, что изменение получилось не полностью. "
    "Посмотри — если не то, скажи, что поправить."
)


_VERIFY_DRAW_ANSWER = (
    "Нарисовал, но проверка видит недочёты — посмотри. Если что-то не так, "
    "скажи, что поправить, или попроси перерисовать."
)


_NO_CALC_ANSWER = (
    "Не смог посчитать: калькулятор не вернул результат. "
    "Повтори запрос, пожалуйста — называть число наугад я не буду."
)


# Measured on the chaos bench: `img_inpaint [wrong]`, calls=['inpaint_image'].
# The tool returned a plausible "Edit applied", the model never called
# inspect_image at all, and answered "Готово! Я добавил ей очки." The guard that
# believes a negative verdict could not fire -- there was no verdict. The file
# exists and is delivered, so this is not a failure and must not be reported as
# one; what is missing is the RIGHT to assert that the specific change landed.
def _gp_edit_tools():
    """The one definition lives in graph_personality, with the other tool facts.

    Duplicating it here is how a new edit tool ends up guarded in one place and
    not the other.
    """
    import graph_personality as _gp
    return _gp._EDIT_TOOLS


_UNVERIFIED_EDIT_ANSWER = (
    "Картинку обновил и прикладываю. Проверить результат я не смог, так что "
    "не берусь утверждать, что вышло именно то, что нужно — посмотри сам, и "
    "если нет, скажи, поправлю."
)


def _closing_prefill(ctx, default):
    """The prefill for the forced closing reply.

    On Gemma 4 this is its own empty thought block (llm.GEMMA_NO_THINK_PREFILL):
    the closing reply only reports work already done, and left to think the
    model spent 3 x 3000 tokens deliberating and never answered (live
    2026-09-22, the mods.rar turn -- the user got a sentence cut off mid-word).
    Shrinking max_tokens could never help: llm floors it to GEMMA_MIN_TOKENS.
    """
    import llm
    model = getattr(ctx, "model_name", "") or llm.MODEL_NAME
    return llm.GEMMA_NO_THINK_PREFILL if llm._is_gemma4(model) else default


def _finalize_answer(ctx: Context, state: AgentState, messages: list,
                     gen: "_GenSettings", last_message, guard_popped: bool,
                     tools_called_this_turn: set, user_input: str,
                     original_input: str, failed_promises=None,
                     verification_negative: bool = False) -> str:
    """Turn whatever the loop ended on into the one answer the user gets.

    Three sources, in order: the model's own last message; a forced closing
    call with NO tools offered (retried smaller and blunter, because more
    budget makes a deliberating model worse, not better); and finally an honest
    last-resort sentence. Whichever wins goes through the SAME finalisation --
    de-duplicate a stalled model's repeated sentence, then match the reply to
    the user's language.

    Appends the forced closing message to `messages` when it makes one. Returns
    "" when nothing usable came back; the tts node skips empty text rather than
    voicing an English error into a Russian conversation.
    """
    import graph as _g
    final_answer = ""
    tools_ran = any(m.get("role") == "tool" for m in messages)
    if last_message:
        final_answer = strip_textual_tool_calls(
            strip_reasoning_leak(last_message.get("content", "") or ""))

    # The round budget can run out while the model is still mid-chain (its last
    # message carries tool_calls, not text), and a model occasionally returns an
    # empty final turn. In both cases force one closing call WITHOUT tools so the
    # user always gets an answer grounded in the tool results gathered so far.
    if not final_answer and (last_message is not None or guard_popped) \
            and not ctx.is_cancelled():
        logger.info("No final answer after the tool loop — forcing a closing reply")
        ctx.set_stage("Writing a response")
        reply_lang_hint = (f"the same language as the user's original message "
                           f"(«{original_input[:120]}»)" if original_input
                           else "the user's language")
        nudge = {"role": "system", "content": (
            "The tool phase is over — tool calls are IMPOSSIBLE now, and writing "
            "one out as text (<tool_call>, <function=...) is forbidden and will be "
            f"discarded. Reply with plain text only, in {reply_lang_hint}: tell the "
            "user what was done, based on the tool results above. If something "
            "failed or is unfinished, say so honestly instead of pretending it "
            "succeeded."
        )}
        # The nudge is sent to the API but NOT appended to `messages`, so the
        # persisted history stays clean.
        # ONE forced call is not enough on a model with no reasoning-effort
        # knob. Measured on google/gemma-4-26b-a4b-qat: 4 of 10 identical calls
        # came back as pure reasoning with no answer after it, so stripping the
        # thought left nothing — and the user saw the literal string
        # "(no response)". More budget makes it WORSE, not better: a bigger
        # allowance is a bigger invitation to keep deliberating. So retry
        # smaller and blunter.
        forced = None
        for _try, _shrink in enumerate((1.0, 0.5, 0.3)):
            _budget = max(384, int(gen.max_tokens * _shrink))
            _msgs = messages + [nudge]
            if _try:
                _msgs = _msgs + [{"role": "system", "content": (
                    "Answer NOW, in plain text. Do not deliberate any further — "
                    "write the reply itself, however short.")}]
                logger.warning("closing reply was empty — retry %d at %d tokens",
                               _try, _budget)
            forced = _g.send_to_lm_studio(ctx, _msgs,
                                       temperature=gen.temperature,
                                       max_tokens=_budget,
                                       prefill=_closing_prefill(ctx, gen.prefill))
            if strip_textual_tool_calls(strip_reasoning_leak(
                    (forced or {}).get("content", "") or "")).strip():
                break
            if ctx.is_cancelled():
                break
        if forced:
            # This closing call was made with NO tools offered, so a compliant
            # backend returns text. A misbehaving model may still emit tool_calls
            # here — they can never be executed or answered, and appending them
            # would leave a dangling unanswered-tool_calls message in the persisted
            # history (invalid if ever re-sent). Drop them: only the text matters now.
            forced.pop("tool_calls", None)
            messages.append(forced)
            final_answer = strip_textual_tool_calls(
                strip_reasoning_leak(forced.get("content", "") or ""))

    # Last resort: tools did real work this turn but the model never produced
    # speakable text (e.g. it kept emitting tool syntax). Give the user an honest
    # generic close instead of silence — the work itself (image, report) is done
    # and visible in the UI.
    # This used to require `tools_ran`, so a plain conversational turn whose
    # model output was all reasoning fell through to an EMPTY answer, and the
    # delivery layer printed the literal placeholder "(no response)". Silence
    # is never an acceptable reply — say something true instead.
    if not final_answer and not ctx.is_cancelled():
        # Non-ASCII original => the user writes in Russian (this project's user
        # base); pure-ASCII turns get the English variant.
        _ru = bool(re.search(r"[А-Яа-яЁё]", original_input or user_input or ""))
        if tools_ran:
            # Say what ACTUALLY happened. The old text was a generic apology
            # about running out of steps — sent even when the edit had just
            # succeeded and the picture was already on its way to the user,
            # which reads as a failure report attached to a working result.
            final_answer = _outcome_sentence(state, tools_called_this_turn, _ru)
        else:
            final_answer = _STUCK_RU if _ru else _STUCK_EN

    # ONE place where the answer is finalised, so every branch above (normal
    # close, forced close, last-resort text) gets the same treatment.
    if final_answer:
        # A stalled model can emit the same sentence twice; never ship that.
        final_answer = collapse_verbatim_repetition(final_answer)
        # Drop any picture the model "attached" as a URL — it did not.
        final_answer = _strip_fabricated_images(final_answer)
        # A stated number, with the calculator broken, is an invented number.
        # There is no salvageable remainder in such an answer: the value IS the
        # answer, so it is replaced rather than edited.
        # A degenerate answer -- "4.", "0.", "." -- with no letters in it at
        # all. Observed after the corrective round pops a draft that stated a
        # fabricated number: told not to give a number, a small model sometimes
        # emits the fragment instead of a sentence. That is worse for the user
        # than an honest failure, so it gets the same replacement.
        _degenerate = not any(ch.isalpha() for ch in final_answer)
        if ("calculate" in (failed_promises or ())
                and (_STATED_RESULT_RE.search(final_answer) or _degenerate)):
            logger.warning("Answer states a computed value (or collapsed to a "
                           "fragment) but calculate failed this turn — replacing")
            final_answer = _NO_CALC_ANSWER
        # Same treatment for the tools that were supposed to hand over a FILE.
        # The corrective round asks the model not to claim a result that does
        # not exist; this is the backstop for when it claims one anyway, which
        # it did on 9 of 27 lying runs ("Вот реальное фото Эйфелевой башни"
        # after find_photo failed). Deliberately narrow: it fires only when the
        # tool errored AND left no artifact AND the answer asserts delivery, so
        # an honest "не удалось, но вот что могу предложить" is untouched.
        elif failed_promises:
            import graph_personality as _gp
            _artifact_fails = {t for t in failed_promises
                               if t in _PROMISE_NOUN
                               and not str(state.get(
                                   _gp._TOOL_ARTIFACT.get(t, "")) or "").strip()}
            _claims = (_gp._DELIVERY_CLAIM_RE.search(final_answer)
                       or _gp._PROMISE_CLAIM_RE.search(final_answer)
                       or _gp._ACTION_CLAIM_RE.search(final_answer))
            _line = (_honest_failure_line(_artifact_fails, _tool_said_unsupported(messages))
                     if _artifact_fails else "")
            if _artifact_fails and _claims and _line:
                logger.warning("Answer claims a deliverable from failed "
                               "tool(s) %s — replacing", sorted(_artifact_fails))
                final_answer = _line
            elif (_artifact_fails and _line
                  and not _ADMISSION_RE.search(final_answer)):
                # Says nothing either way. The draft may still be useful
                # (a suggestion, a question), so the failure is PREPENDED
                # rather than swallowing it.
                logger.warning("Answer is silent about failed tool(s) %s — "
                               "prepending the failure", sorted(_artifact_fails))
                final_answer = _line + "\n\n" + final_answer
        # The verifier contradicting a tool that reported success. No tool
        # errored, so nothing above fires; the artifact exists and is still
        # delivered, only the claim about it is corrected.
        elif verification_negative and not _ADMISSION_RE.search(final_answer):
            import graph_personality as _gp
            if (_gp._DELIVERY_CLAIM_RE.search(final_answer)
                    or _gp._PROMISE_CLAIM_RE.search(final_answer)
                    or _gp._ACTION_CLAIM_RE.search(final_answer)):
                logger.warning("Answer claims a result the verifier said is "
                               "not there — replacing")
                _edits = {"inpaint_image", "redraw_image", "transfer_image", "edit_image"}
                _called = set(tools_called_this_turn or ())
                if "generate_image" in _called and not (_called & _edits):
                    # A fresh picture has no "change" that could be missing:
                    # "нужного изменения нет — остался прежним" under a new logo
                    # read as nonsense (live 2026-09-28).
                    final_answer = _VERIFY_DRAW_ANSWER
                else:
                    final_answer = (_VERIFY_PARTIAL_ANSWER if verification_negative == "partial"
                                    else _VERIFY_NEGATIVE_ANSWER)
        # A file that was promised and does not exist. Checked before the
        # verification guards below because it is the bluntest of the three:
        # there is no artifact of ANY kind, so there is nothing to hedge about.
        _asked = " ".join(str(state.get(k) or "") for k in ("user_input_original", "user_input"))
        from prompt_guard import has_quote, user_words
        _fwd_turn = has_quote(_asked)
        if (_FILE_READY_RE.search(final_answer)
                and (not _fwd_turn or _user_read(user_words(_asked))["asks_for_file"])
                and not any(str(state.get(k) or "").strip()
                            for k in _ARTIFACT_KEYS)
                and not _ADMISSION_RE.search(final_answer)):
            logger.warning("Answer says a file is ready but none exists — replacing")
            final_answer = _NO_FILE_ANSWER

        # An edit that was never verified. A separate check, not another
        # `elif`: an edit turn can ALSO have had a tool fail (inpaint came
        # back empty, redraw delivered), and in the chain above that case is
        # swallowed by the failed_promises branch, which correctly does
        # nothing because a file exists -- and then this never ran. Safe to
        # run unconditionally: every replacement the chain writes begins
        # with an admission, which the guard below already excludes.
        # Only _ACTION_CLAIM_RE and
        # _PROMISE_CLAIM_RE count here, never _DELIVERY_CLAIM_RE -- "вот твоя
        # картинка" hands over a file that genuinely exists and is honest;
        # "я добавил ей очки" asserts a change nothing confirmed.
        if (not verification_negative
              and "inspect_image" not in (tools_called_this_turn or ())
              and (tools_called_this_turn or set()) & _gp_edit_tools()
              and str(state.get("image_path") or "").strip()
              and not _ADMISSION_RE.search(final_answer)):
            import graph_personality as _gp
            if (_gp._ACTION_CLAIM_RE.search(final_answer)
                    or _gp._PROMISE_CLAIM_RE.search(final_answer)):
                logger.warning("Answer asserts an edit landed but nothing "
                               "verified it — replacing with a hedged handover")
                final_answer = _UNVERIFIED_EDIT_ANSWER
        # …and after a long, all-English tool chain the reply drifts back to
        # English even though the user wrote in another language.
        final_answer = _match_reply_language(ctx, final_answer, original_input)
        final_answer = _unparrot(ctx, final_answer, original_input, messages)
        final_answer = _prompt_leak_guard(ctx, _ru_thousands(final_answer))
        final_answer = _fix_totals(final_answer)

    # Leave it empty when the model produced nothing usable. The tts node skips
    # empty text, so the assistant never voices an English error message in the
    # middle of a non-English (e.g. Russian) conversation.
    if not final_answer:
        logger.warning("No usable answer from the model on this turn")
    try:
        # The model's own words, kept when the guards above changed them: a
        # replaced answer used to leave no trace of what had been replaced.
        import turn_trace
        _draft = (last_message or {}).get("content", "") or ""
        if _draft.strip() and _draft.strip() != final_answer.strip():
            turn_trace.answer("draft", _draft)
        turn_trace.answer("final", final_answer)
    except Exception:
        pass
    return final_answer
