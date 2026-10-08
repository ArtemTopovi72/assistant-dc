"""How one turn is composed: sampling settings and the user message.

Lifted out of graph.py. Two decisions that happen before the model is called
and nothing after it:

  _generation_settings  what temperature/top_p/max_tokens this turn runs at,
                        and the system prompt it runs under.
  _compose_user_message what the model actually reads as "the user said" —
                        the translated text, the attached image note, the
                        pinned facts, the retrieved documents.

Neither touches send_to_lm_studio, execute_tool or TOOL_SCHEMAS, which is what
made them movable: those three are patched on the graph module by the suites,
so anything calling them has to stay where the patch lands.
"""
import os
import re as _re
from typing import NamedTuple

from models import AgentState, Context
from prompts import build_system_prompt


class _GenSettings(NamedTuple):
    """Everything about HOW this turn is generated, in one place.

    Two independent axes feed it: the thinking toggle (ctx.no_think) picks the
    behavioural mode, and ctx.response_length picks the token budget. Keeping
    them together is what makes the difference between `max_tokens` (the
    user-facing answer) and `loop_max_tokens` (the tool rounds) legible.
    """
    concise: bool
    length: str
    temperature: float
    max_tokens: int          # ceiling for the ANSWER
    loop_max_tokens: int     # ceiling for the tool-calling rounds
    prefill: object          # closed <think></think>, or None
    system_prompt: str


def _ask(text: str) -> dict:
    """The model's read of what the message asks the program to do exactly
    (agent/ask_read.py). Counting, reversing and sorting stay here."""
    import ask_read
    return ask_read.read(text)


# Live 2026-09-28: 'о' in «обороноспособность» -> 5 (7), «программирование»
# reversed -> «еироваиммаргорп», 9 words -> 10. Letters are not tokens.
def _text_ops_note(text: str) -> str:
    a = _ask(text)
    op = a["op"]
    if op == "letter_count":
        ch, word = a["letter"], a["word"]
        n = word.lower().count(ch.lower())
        return (f"[Counted exactly by the program: «{word}» contains the letter «{ch}» "
                f"{n} times. Use this number.]")
    # «отсортируй по алфавиту: яблоко, груша, ...» put груша after ёжевика (live).
    if op == "sort":
        key = lambda w: w.lower().replace("ё", "е￿")
        return ("[Sorted exactly by the program (Russian alphabet, ё after е): "
                + ", ".join(sorted(a["items"], key=key)) + ". Use this order.]")
    if op == "reverse":
        return (f"[Reversed exactly by the program: «{a['target']}» backwards is "
                f"«{a['target'][::-1]}». Use this exact string.]")
    if op == "word_count":
        n = len(_re.findall(r"[\w-]+", a["target"]))
        return f"[Counted exactly by the program: «{a['target']}» has {n} words. Use this number.]"
    if op == "percent":
        return ("[Computed exactly by the program: "
                + "; ".join(f"{p['pct']:g}% of {p['of']:g} = {p['pct'] * p['of'] / 100:g}"
                            for p in a["percents"])
                + ". Start from these values; answer only what was asked, no extra totals.]")
    return ""


def _dose_note(text: str) -> str:
    """Live 2026-09-28: a 14 kg child's paracetamol dose came back as
    «7-14 mg/kg, 100-200 mg» -- the standard single dose is 10-15 mg/kg."""
    a = _ask(text)
    if not a["dose"]:
        return ""
    kg = a["weight_kg"]
    calc = (f" For {kg:g} kg: paracetamol {kg*10:.0f}-{kg*15:.0f} mg per dose, max "
            f"{kg*60:.0f} mg/day; ibuprofen {kg*5:.0f}-{kg*10:.0f} mg per dose, max "
            f"{kg*30:.0f} mg/day." if 0 < kg < 200 else "")
    return ("[Reference doses (standard guidance): paracetamol 10-15 mg/kg per dose, every "
            "4-6 h, max 60 mg/kg/day (adults max 4 g/day); ibuprofen 5-10 mg/kg per dose, "
            "every 6-8 h, max 30 mg/kg/day, not under 3 months." + calc +
            " Use these numbers, name the syrup volume only from the concentration on the "
            "package, and tell them to see a doctor for infants or fever lasting over 3 days.]")


def _word_count_note(text: str) -> str:
    ops = _text_ops_note(text)
    if ops:
        return ops
    a = _ask(text)
    if a["op"] != "word_occurrences":
        return _dose_note(text)
    word = a["word"].lower()
    # the counted text is what came before the question about it
    cut = (text or "").lower().rfind(word)
    body = (text or "")[:cut] if cut > 0 else (text or "")
    if len(body) < 40:
        return ""
    n = sum(1 for w in _re.findall(r"[\w-]+", body.lower()) if w == word)
    return (f"[Counted exactly by the program: the word «{word}» occurs {n} times in "
            f"the user's message before the question. Use this number.]")


def no_history_note(text: str, history) -> str:
    """After /clear, «о каком городе мы говорили?» got an invented «Санкт-Петербург»."""
    if any(m.get("role") in ("user", "assistant") and m.get("content") for m in history or []):
        return ""
    if not _ask(text)["earlier_talk"]:
        return ""
    return ("[This chat has no earlier messages (it is new or was cleared). Say plainly that "
            "you have nothing earlier to go on; do not invent a past conversation.]")


def facts_parts(facts_text: str) -> list:
    """Saved-facts block for both the fast path and the full loop."""
    parts = ["Saved facts (you remembered these earlier; treat them as known and use "
             "them when relevant; newest are last — if two facts conflict, the later "
             f"one is correct):\n{facts_text}"]
    # «шоколадный медовик -- в нём точно нет арахиса»: a guarantee nobody can give.
    # 10-08 live: after «аллергия на орехи, я вегетарианец» dinner advice was a pasta with
    # «кедровыми орешками» -- the allergy only forbade PROMISING, never SUGGESTING.
    import intent
    if intent.ask_yes("Saved facts about a user: {text} -- do they say the user has a food "
                      "allergy, an intolerance or a diet (vegetarian, no gluten...)?", facts_text):
        parts.append("The user has a food allergy, intolerance or diet. It is a hard limit on "
                     "every food suggestion: never offer a dish or ingredient that contains the "
                     "allergen or breaks the diet (not even 'swap it out' -- offer only what is "
                     "safe from the start), and check each ingredient against the saved facts "
                     "before answering. Never promise a dish, cake or product is free of the "
                     "allergen (no «точно нет», «гарантированно»); tell them to confirm the "
                     "ingredients and cross-contamination with the maker or the label.")
    return parts


# An explicit length in the request ("на 1200 слов", "подробно"). Live
# 2026-09-28: "инструкция на 1200 слов" hit the 700-token concise ceiling and
# arrived as 2.5k chars cut mid-word.
def _asked_tokens(text: str) -> int:
    a = _ask(text)
    if 100 <= a["words_asked"] <= 20000:
        return min(8000, a["words_asked"] * 3 + 300)   # ~2.5 tokens per RU word + markup
    return 2600 if a["detailed"] else 0


def _generation_settings(ctx: Context, user_text: str = "") -> "_GenSettings":
    """Resolve the sampling profile, the token budgets and the system prompt."""
    # Direct mode (thinking toggle OFF) is a real behavioural mode, not just a
    # hidden reasoning channel: a concise system directive + a colder, tighter
    # sampling profile so the model acts instead of deliberating in prose.
    concise = bool(getattr(ctx, "no_think", False))
    gen_temperature = 0.2 if concise else 0.5
    # Token budget is driven by the explicit response-length setting (a separate
    # axis from the thinking toggle). "auto" keeps the historical defaults
    # (concise 700 / verbose 1500); "short"/"long" override both so the user's
    # length choice wins even when thinking is off. The budget covers reasoning +
    # answer, so "long" needs generous headroom to finish a detailed reply.
    length = getattr(ctx, "response_length", "auto") or "auto"
    if length == "ultra":
        gen_max_tokens = 160 if concise else 256
    elif length == "short":
        gen_max_tokens = 400 if concise else 600
    elif length == "long":
        gen_max_tokens = 2600
    else:  # auto
        gen_max_tokens = 700 if concise else 1500
    # `gen_max_tokens` clamps the user-facing ANSWER only. The tool-calling rounds
    # (reasoning + filling tool arguments like an image description) must NOT be
    # starved by a short answer setting — a 160-token cap made the agent pass
    # barebones tool args and route poorly ("dumber"). So the loop always gets at
    # least the AUTO budget; only "long" raises it further. Brevity of the final
    # reply is enforced by the system-prompt length directive, not this ceiling.
    gen_max_tokens = max(gen_max_tokens, _asked_tokens(user_text)) if length != "ultra" else gen_max_tokens
    auto_floor = 700 if concise else 1500
    loop_max_tokens = max(gen_max_tokens, auto_floor)
    # The load-bearing reasoning suppressor. Measured on the live finetune:
    # `/no_think` + enable_thinking=False are only honored for trivial prompts
    # and IGNORED for tool/agent prompts (the model still emits a full <think>
    # block, ~97 completion tokens). Prefilling a CLOSED <think></think> forces
    # generation to start AFTER the reasoning block, so the model genuinely
    # produces fewer tokens (97 -> 40 WITH tools, tool_calls still returned
    # correctly) instead of merely hiding stripped CoT. This is what makes the
    # model reason LESS, not just expose less.
    gen_prefill = "<think></think>" if concise else None
    system_prompt = build_system_prompt(ctx.custom_personality_text, concise=concise,
                                         length=length,
                                         reply_lang=getattr(ctx, "reply_lang", "ru") or "ru",
                                         tz=getattr(ctx, "user_tz", "") or "")
    return _GenSettings(concise=concise, length=length,
                        temperature=gen_temperature,
                        max_tokens=gen_max_tokens,
                        loop_max_tokens=loop_max_tokens,
                        prefill=gen_prefill,
                        system_prompt=system_prompt)


def _wants_fresh_picture(ctx, state) -> bool:
    """'нарисуй кота' with a picture in the chat is a NEW picture; 'нарисуй ей
    шляпу' / 'draw a hat on it' refers to the existing one. The model's read
    of the turn (agent/intent.py): generate_image wanted and no edit tool --
    'нарисуй набережную Сочи в такую погоду' after a lighthouse was redrawn
    ON the lighthouse (live 2026-09-13) when this was a verb list."""
    from graph_personality import _turn_intent
    from graph_fastpath import _IMAGE_ACTION_TOOLS
    text = state.get("user_input_original") or state.get("user_input") or ""
    if not text.strip():
        return False
    wants = set(_turn_intent(ctx, state, text)["wants"])
    return "generate_image" in wants and not (wants & _IMAGE_ACTION_TOOLS - {"generate_image"})


FORGOTTEN_NOTE = (
    "[The user asked you to forget what they told you about themselves, and the "
    "saved facts were deleted. Anything about the user (name, diet, allergies, "
    "pets, preferences) that still appears earlier in this conversation is VOID: "
    "do not use it, and if asked, say you have nothing saved about that.]")


# Ponytail (github.com/DietrichGebert/ponytail, MIT, v4.10.0), condensed for
# the coding sandbox: the "lazy senior dev" ladder. Only on turns that ask for
# code, so a picture or a weather question never pays its ~250 tokens.
# PONYTAIL=off|lite|full|ultra (default full).
_PONYTAIL_LEVEL = {
    "lite": "Build what is asked, but name the lazier alternative in one line.",
    "full": "The ladder is enforced: stdlib and native first, shortest working diff.",
    "ultra": "YAGNI extremist: deletion before addition; ship the one-liner and challenge the rest.",
}
_PONYTAIL_RULES = (
    "[Coding style -- lazy senior dev ({level}). {level_line} Read the task and "
    "the code it touches FIRST; then stop at the first rung that holds: "
    "1) does it need to exist? 2) already in this folder? reuse it; 3) stdlib? "
    "4) native platform feature? 5) an installed dependency? never add one for a "
    "few lines; 6) one line? 7) only then the minimum code that works. "
    "Bug fix = root cause in the shared function, not a guard per caller. No "
    "unrequested abstractions, no scaffolding for later, fewest files. Never "
    "simplify away input validation, error handling that prevents data loss, "
    "security, or anything the user asked for. Non-trivial logic leaves ONE "
    "runnable check (an assert self-test or one small test file). Mark a "
    "deliberate shortcut with a `ponytail:` comment naming its limit. Reply: "
    "code first, then at most three short lines -- what was skipped, when to add it.]")


def _ponytail_block(ctx, text: str) -> str:
    level = (os.getenv("PONYTAIL", "full") or "full").strip().lower()
    if level not in _PONYTAIL_LEVEL or getattr(ctx, "sandbox", None) is None:
        return ""
    if not _ask(text)["code"]:
        return ""
    return _PONYTAIL_RULES.format(level=level, level_line=_PONYTAIL_LEVEL[level])


def _working_folder_record(ctx) -> str:
    box = getattr(ctx, "sandbox", None)
    if box is None:
        return ""
    try:
        from tool_code_handlers import read_record
        return read_record(box)
    except Exception:
        return ""


def _compose_user_message(ctx: Context, state: AgentState) -> tuple:
    """Assemble the user turn the model actually sees.

    Returns (user_message_text, effective_input, original_input).
    `user_message_text` is the bare input wrapped in this turn's scaffolding:
    pinned facts, session memory, any vision summary, the image-tool steering
    block, and the reply-language instruction when the turn was translated at
    entry. `effective_input` is the input with the bare "image" placeholder
    expanded - the fast path sends that instead of the scaffolded text.
    `original_input` is the pre-translation message, which doubles as the
    reply-language sample.

    Pure assembly: reads ctx and state, mutates neither.
    """
    user_input = state.get("user_input", "")
    vision_summary = state.get("vision_summary", "").strip()
    memory_text = (state.get("session_memory_text", "") or ctx.memory_text()).strip()

    # Build user message
    parts = []
    _cnt = (_word_count_note(state.get("user_input_original") or user_input)
            or no_history_note(state.get("user_input_original") or user_input,
                               state.get("messages")))
    facts_text = ctx.facts_text(state.get("user_input_original") or user_input)
    if facts_text:
        parts.extend(facts_parts(facts_text))
    elif getattr(ctx, "facts_forgotten", False):
        parts.append(FORGOTTEN_NOTE)
    if memory_text:
        # Neutral label: "Session memory:" was read back as a feature to announce
        # ("память сессии загружена", live 2026-09-25).
        parts.append(f"Earlier in this chat (background only; never mention or summarise it):\n{memory_text}")
    if vision_summary:
        parts.append(f"Image description:\n{vision_summary}")
    # Steer edit-type requests to the editing tools. Only forbid generate_image
    # when an image was sent THIS turn; for a merely-previous image, leave room
    # for a brand-new generation (otherwise "draw a dog" after an earlier image
    # would wrongly redraw the old one).
    if state.get("picture_question"):
        parts.append(
            "[The user asks a QUESTION about the picture. Answer it in words from the "
            "image description above — describe what is there, small details included. "
            "Do NOT call any image tool (no redraw_image, inpaint_image, generate_image, "
            "enhance, upscale); nothing is to be changed.]"
        )
    elif state.get("image_data"):
        parts.append(
            "[The user just sent an image — work on THAT image, never generate_image. "
            "When they want to change ONE specific thing in it — clothing (dress, shirt, "
            "skirt), hair, face, an accessory, or the background — which is the usual case "
            "(e.g. 'replace her dress with red', 'change the shirt', 'remove the background') "
            "— call inpaint_image with that thing as `region` (region='dress', 'shirt', "
            "'hair', 'face', 'background', …) and `instructions` describing the new look. It "
            "regenerates ONLY that region and keeps the person and everything else identical. "
            "Use redraw_image ONLY when the user explicitly wants the WHOLE picture re-rendered "
            "or restyled — it replaces the entire image, including the person.]"
        )
    elif getattr(ctx, "last_image_path", None) and _wants_fresh_picture(ctx, state):
        parts.append(
            "[The user asks for a NEW picture. Call generate_image with a full prompt "
            "for it — do not edit, redraw or inpaint the previously created image.]"
        )
    elif getattr(ctx, "last_image_path", None):
        parts.append(
            "[A previously created image is available. If the user asks to change or "
            "REMOVE ONE named thing in it — clothing, hair, face, an object, a spill or "
            "puddle, a stain, a logo, the sky, the background, the color of one element "
            "— call inpaint_image with that thing as `region` (e.g. region='sky' for "
            "'make the sky orange'; region='puddle' for 'remove the puddle'); it edits "
            "ONLY that area and keeps everything else identical. Removing something ('убери', "
            "'remove', 'delete') is STILL inpaint_image with the thing to erase as `region` "
            "— never redraw_image, which would rebuild the whole picture and destroy the rest. "
            "Use redraw_image ONLY when the whole picture must be re-rendered (different style, "
            "season, or a transformation touching everything). ALSO use redraw_image with "
            "mode='upscale' when the user asks to upscale/increase resolution (keep person "
            "faithful); mode='restore' to repair/enhance quality; mode='enhance' to improve "
            "detail without changes. For a brand-new, unrelated picture use generate_image. "
            "Any change REQUIRES one of these tool calls — never claim an edit happened "
            "without calling a tool.]"
        )
    # What the file tools already did in this working folder (find_content,
    # dedupe_photos, scripts): kept on disk, so a compacted history cannot
    # turn «сколько мостов нашёл?» into «я не проводил поиск».
    _pony = _ponytail_block(ctx, (state.get("user_input_original", "") or "") + " " + user_input)
    if _pony:
        parts.append(_pony)
    _rec = _working_folder_record(ctx)
    if _rec:
        parts.append("[Working-folder record -- things YOU already did here with "
                     "the file tools; answer questions about them from this, and "
                     "never deny having done them:\n" + _rec + "]")
    # English-first pipeline: when the turn was translated at entry, the model
    # reads and reasons over the English text, but the FINAL reply must come
    # back in the user's own language. The original message doubles as the
    # language sample, so no separate language classifier is needed.
    original_input = state.get("user_input_original", "")
    if original_input and original_input != user_input:
        parts.append(
            "[The user's message was translated into English for processing. "
            "Do all reasoning, planning and tool arguments in English, but write "
            f"your final reply to the user in the same language as their original "
            f"message: «{original_input}»]"
        )
    # "image" is the neutral placeholder set when a photo arrives with no caption.
    # Replace it with a clear instruction so the model acknowledges receipt properly.
    effective_input = user_input
    if user_input.strip() == "image" and state.get("image_data"):
        # vision_agent_node already ran the vision model on this exact photo
        # (see "Image description:" above, from vision_summary) -- the old
        # wording only asked for a bare "photo received, what next?"
        # acknowledgement, so a real description sat unused right above it in
        # the same prompt. Say what is actually in the photo first.
        effective_input = ("[The user sent a photo with no caption. Briefly say what is in "
                           "it, using the image description above (one or two sentences, "
                           "the concrete subject/scene, not a generic \"a photo of a "
                           "person\"), then ask what they would like to do with it.]")
    if _ask(original_input or user_input)["earlier_talk"]:
        _asked = [str(m.get("content") or "").strip() for m in state.get("messages", [])
                  if m.get("role") == "user"]
        if _asked and _asked[-1] == user_input.strip():
            _asked = _asked[:-1]
        _asked = [a[:200] for a in _asked if a][-20:]
        if _asked:
            # Live 2026-09-28: "что я спросил три сообщения назад?" named the
            # previous question and called the current one "the first".
            effective_input = (
                "[The user's earlier messages in this chat, oldest first; each is "
                "labelled with how many messages ago it was (use the label as is):\n"
                + "\n".join(f"{i}. ({len(_asked) - i + 1} ago) {a}"
                            for i, a in enumerate(_asked, 1)) + "]\n\n"
                + effective_input)
    parts.append(effective_input)
    if _cnt:
        parts.append(_cnt)
    user_message_text = "\n\n".join(parts)
    return user_message_text, effective_input, original_input
