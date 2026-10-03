"""Context management v2: budget-driven, addressable, append-only.

The v1 compactor (graph_history.compact_history_if_needed) folded everything but
the last two user turns into a 120-word paragraph every THIRD turn, whatever the
window held, and re-wrote that paragraph from itself each time. Three failures
followed from that design and were seen live: the picture/file of three turns ago
was gone ("which picture?"), each re-summary lost a little more (the compaction
cliff), and the prompt prefix changed every third turn so LM Studio re-evaluated
the whole transcript.

v2, one idea per mechanism (sources in docs/research_log.md, 2026-09-24):

  budget trigger (ReSum / TokenPilot)   nothing happens below MASK_AT of the
                                        per-request window; compaction only past
                                        COMPACT_AT. The prefix stays byte-stable
                                        between those events, so the KV cache
                                        is reused (llama.cpp prefix caching).
  observation masking                   old tool outputs become one-line markers
                                        -- cheaper than a summary and as good.
  addressable recall (ARC / Pull)       every masked or folded span is archived
                                        on disk under a content id; the marker
                                        carries that id plus the paths/urls it
                                        mentioned, and recall_context(id) brings
                                        the original back on demand.
  anchored structured memory            one system message with sections (goal,
                                        decisions, artifacts, open, prefs, folded)
                                        that are MERGED by a JSON delta, never
                                        re-written: nothing old passes through a
                                        summariser twice, so there is no cliff.
  memory cards (decision-aware)         when the sections outgrow CARD_BUDGET the
                                        rendered memory keeps the cards that share
                                        words / the task kind with the request.
  task isolation (HyMem)                every card and archive entry is tagged
                                        image / research / code / chat.
  dashboard (latent context managers)   the memory message ends with the window
                                        use (rounded to 1k, so it rarely changes)
                                        and how to recall/fold.
  memory as action (AgentFold)          fold_context(note) lets the model ask for
                                        a fold of finished work; it runs at the
                                        end of the turn.
  ACON guidelines                       runtime/context_guidelines.txt is appended
                                        to the delta prompt; bench/context_acon.py
                                        refines it from failed journeys.

Fails open everywhere: any error returns the input list unchanged.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import config as _cfg_env   # env_int/env_float: a bad .env value falls back, never crashes the import
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("assistant.graph")

MEMORY_MARKER = "[Working memory]\n"
V1_SUMMARY_MARKER = "[Conversation summary so far]\n"

MASK_AT = _cfg_env.env_float("CONTEXT_MASK_AT", 0.40)
COMPACT_AT = _cfg_env.env_float("CONTEXT_COMPACT_AT", 0.65)
KEEP_TURNS = _cfg_env.env_int("CONTEXT_KEEP_TURNS", 2)
MASK_MIN_CHARS = 400
CARD_BUDGET = 2400            # chars of rendered memory before card selection kicks in
SECTION_CAP = {"decisions": 14, "artifacts": 16, "open": 8, "prefs": 10, "folded": 12}
SECTIONS = ("goal", "decisions", "artifacts", "open", "prefs", "folded")
_TITLES = {"goal": "Goal", "decisions": "Decisions & facts", "artifacts": "Files & pictures",
           "open": "Open requests", "prefs": "User preferences", "folded": "Folded earlier turns"}

_ARTIFACT_RE = re.compile(
    r"(?:[A-Za-z]:[\\/][^\s\"'<>|*?]+?\.(?:png|jpe?g|webp|gif|mp4|webm|mov|wav|mp3|ogg|"
    r"pdf|pptx|docx|xlsx|csv|txt|md|py|js|json|zip|html)\b"
    r"|https?://[^\s\"'<>)\]]+)", re.I)

_KIND_BY_TOOL = {
    "generate_image": "image", "redraw_image": "image", "inpaint_image": "image",
    "inspect_image": "image", "transfer_image": "image", "fix_hands": "image",
    "fix_artifact": "image", "find_photo": "image", "generate_video": "image",
    "search": "research", "deep_research": "research", "create_presentation": "research",
    "list_files": "code", "read_file": "code", "write_file": "code", "edit_file": "code",
    "search_files": "code", "run_python": "code", "unpack_archive": "code",
}


# ---------------------------------------------------------------- budget ----

def budget_tokens() -> int:
    """Per-request window: CONTEXT ÷ PARALLEL (see memory lmstudio-context-divided-by-parallel)."""
    env = os.getenv("CONTEXT_BUDGET_TOKENS")
    if env:
        return max(2048, int(env))
    try:
        import config
        n = int(getattr(config, "MODEL_CONTEXT_TOKENS", 0) or 0)
        par = max(1, int(getattr(config, "MODEL_PARALLEL", 1) or 1))
        if n:
            return max(2048, n // par)
    except Exception:
        pass
    return 20480


def est_tokens(messages: List[dict]) -> int:
    n = 0
    for m in messages:
        c = m.get("content")
        n += len(c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)) if c else 0
        if m.get("tool_calls"):
            n += len(json.dumps(m["tool_calls"], ensure_ascii=False))
        n += 12
    return int(n / 3.2)        # mixed RU/EN + JSON: ~3.2 chars per token on Gemma


# --------------------------------------------------------------- archive ----

def _archive_dir() -> str:
    d = os.getenv("CTX_ARCHIVE_DIR")
    if not d:
        try:
            import config
            d = os.path.join(str(config.OUTPUT_DIR), "ctx_archive")
        except Exception:
            d = os.path.join("runtime", "ctx_archive")
    os.makedirs(d, exist_ok=True)
    return d


def archive_put(text: str, *, kind: str = "chat", label: str = "") -> str:
    """Content-addressed: the same span always gets the same id, across restarts
    and chats, so a marker in a persisted Telegram history still resolves."""
    aid = "a" + hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:7]
    p = os.path.join(_archive_dir(), aid + ".json")
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"id": aid, "kind": kind, "label": label, "text": text}, fh, ensure_ascii=False)
    return aid


def archive_get(aid: str) -> Optional[dict]:
    aid = re.sub(r"[^a-z0-9]", "", (aid or "").lower())
    if not aid:
        return None
    p = os.path.join(_archive_dir(), aid + ".json")
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def artifacts_in(text: str) -> List[str]:
    seen, out = set(), []
    for a in _ARTIFACT_RE.findall(text or ""):
        a = a.rstrip(".,;:")
        if a not in seen:
            seen.add(a); out.append(a)
    return out


def _marker(aid: str, what: str, n: int, arts: List[str]) -> str:
    tail = ("; mentions: " + ", ".join(os.path.basename(a) if not a.startswith("http") else a
                                        for a in arts[:4])) if arts else ""
    return (f"[archived {aid}: {what}, {n} chars{tail} -- "
            f"recall_context(id='{aid}') returns the full text]")


def is_marker(content) -> bool:
    return isinstance(content, str) and content.startswith("[archived a")


# -------------------------------------------------------------- masking ----

def _tool_names(messages: List[dict]) -> Dict[str, str]:
    out = {}
    for m in messages:
        for c in m.get("tool_calls") or []:
            out[c.get("id", "")] = (c.get("function") or {}).get("name", "tool")
    return out


def mask_observations(messages: List[dict], keep_from: int) -> Tuple[List[dict], int, List[str]]:
    """Tool outputs before index `keep_from` become archive markers. The
    tool_call_id stays, so call/result pairs remain valid for the template."""
    names = _tool_names(messages)
    out, n, arts_all = [], 0, []
    for i, m in enumerate(messages):
        c = m.get("content")
        if (i < keep_from and m.get("role") == "tool" and isinstance(c, str)
                and len(c) >= MASK_MIN_CHARS and not is_marker(c)):
            name = names.get(m.get("tool_call_id", ""), "tool")
            arts = artifacts_in(c)
            aid = archive_put(c, kind=_KIND_BY_TOOL.get(name, "chat"), label=name)
            m = dict(m, content=_marker(aid, f"{name} result", len(c), arts))
            arts_all += arts
            n += 1
        out.append(m)
    return out, n, arts_all


# ------------------------------------------------------ structured memory ----

def empty_state() -> dict:
    return {k: ([] if k != "goal" else "") for k in SECTIONS}


def render(state: dict, dashboard: str = "") -> str:
    parts = [MEMORY_MARKER.rstrip("\n")]
    if state.get("goal"):
        parts.append(f"## {_TITLES['goal']}\n{state['goal']}")
    for k in SECTIONS[1:]:
        items = state.get(k) or []
        if items:
            parts.append(f"## {_TITLES[k]}\n" + "\n".join(f"- {x}" for x in items))
    if dashboard:
        parts.append(dashboard)
    return "\n".join(parts) + "\n"


def parse(text: str) -> dict:
    st = empty_state()
    if not text or not text.startswith(MEMORY_MARKER.rstrip("\n")):
        return st
    rev = {v: k for k, v in _TITLES.items()}
    cur = None
    for line in text.splitlines()[1:]:
        if line.startswith("## "):
            cur = rev.get(line[3:].strip())
            continue
        if line.startswith("[Context:"):
            cur = None
            continue
        if cur == "goal" and line.strip():
            st["goal"] = (st["goal"] + " " + line.strip()).strip()
        elif cur and line.startswith("- "):
            st[cur].append(line[2:].strip())
    return st


_LEDGER_RE = re.compile(r"^older (.+?) \((\d+)\) -> recall_context\(id='(a[0-9a-f]{7})'\)$")


def _is_ledger(item: str) -> bool:
    return bool(_LEDGER_RE.match(item or ""))


# A summariser that CONTINUES the transcript writes turns and tool calls into the
# memory, which then read as things that happened (Majordomo's history validation).
_TRANSCRIPT_RE = re.compile(r"^\s*(?:user|assistant|system|tool|пользователь|ассистент|бот)\s*:"
                            r"|<\|?/?(?:tool|channel|start|end)|\bcall:\w+\{|\"tool_calls\"", re.I)


def _add(state: dict, key: str, items) -> None:
    for it in items or []:
        it = " ".join(str(it).split())[:300]
        if it and _TRANSCRIPT_RE.search(it):
            logger.warning("context v2: transcript-shaped memory item dropped: %r", it[:120])
            continue
        if it and it not in state[key]:
            state[key].append(it)
    cap = SECTION_CAP.get(key)
    live = [x for x in state[key] if not _is_ledger(x)]
    if not cap or len(live) <= cap:
        return
    spill, keep = live[:-cap], live[-cap:]
    # Overflow is archived, never dropped -- and into ONE cumulative ledger per
    # section, so any old item is exactly one recall away. (A fresh archive per
    # spill chained spills of spills: after 30 turns the first picture sat four
    # recalls deep, lost in practice. tests/test_context_v2 cliff test.)
    title = _TITLES[key].lower()
    prev_text, prev_n = "", 0
    for x in state["folded"]:
        m = _LEDGER_RE.match(x)
        if m and m.group(1) == title:
            e = archive_get(m.group(3))
            prev_text, prev_n = (e or {}).get("text", ""), int(m.group(2))
            state["folded"].remove(x)
            break
    text = (prev_text + "\n" if prev_text else "") + "\n".join(spill)
    aid = archive_put(text, kind="chat", label=f"older {key}")
    note = f"older {title} ({prev_n + len(spill)}) -> recall_context(id='{aid}')"
    ledgers = [x for x in state["folded"] if _is_ledger(x)]
    if key == "folded":
        state[key] = ledgers + [note] + keep
    else:
        state[key] = keep
        state["folded"] = ledgers + [note] + [x for x in state["folded"] if not _is_ledger(x)]


def merge_delta(state: dict, delta: dict) -> dict:
    """Append-only merge: nothing already in the state is rewritten."""
    if not isinstance(delta, dict):
        return state
    g = delta.get("goal")
    if isinstance(g, str) and g.strip():
        state["goal"] = " ".join(g.split())[:300]
    closed = {str(x).strip().lower() for x in (delta.get("close_open") or [])}
    if closed:
        state["open"] = [o for o in state["open"]
                         if not any(c and c in o.lower() for c in closed)]
    _add(state, "decisions", delta.get("add_decisions"))
    _add(state, "open", delta.get("add_open"))
    _add(state, "prefs", delta.get("add_prefs"))
    return state


def _is_preference(sent: str) -> bool:
    """A user sentence that states a standing preference ("always answer short",
    "не присылай голосом"). Read by the model, not a word list."""
    import intent
    return intent.ask_yes(
        "A user wrote this sentence in a chat with an assistant: {text}\n\nDoes it "
        "state a standing preference or rule for how the assistant should behave from "
        "now on (always/never do something, a liked or disliked way)?", sent)


def extract_now(span: List[dict]) -> dict:
    """Extraction over what is about to leave the
    window (Remember-When-It-Matters, write-before-compaction): every path/url,
    and every user sentence that states a standing preference. Runs BEFORE the
    LLM delta, so a summariser that misses them cannot lose them."""
    arts, prefs = [], []
    for m in span:
        c = m.get("content")
        if not isinstance(c, str):
            continue
        arts += artifacts_in(c)
        for tc in m.get("tool_calls") or []:
            arts += artifacts_in(str((tc.get("function") or {}).get("arguments") or ""))
        if m.get("role") == "user":
            for sent in re.split(r"(?<=[.!?\n])\s+", c):
                if 8 <= len(sent) <= 240 and _is_preference(sent):
                    prefs.append(sent.strip())
    return {"artifacts": arts, "prefs": prefs}


def _turn_chunks(span: List[dict], per: int = 2) -> List[List[dict]]:
    """The span cut at user messages into groups of `per` turns: each delta call
    sees a turn or two, never a 25-turn backlog in one go."""
    starts = [k for k, m in enumerate(span) if m.get("role") == "user"] or [0]
    if starts[0] != 0:
        starts = [0] + starts
    cuts = starts[::per] + [len(span)]
    return [span[a:b] for a, b in zip(cuts, cuts[1:]) if b > a]


def _tag(kind: str, items) -> list:
    return [f"[{kind}] {x}" if kind != "chat" and not str(x).startswith("[") else x
            for x in (items or [])]


def _kind_of(text: str) -> str:
    """image | code | research | chat -- what the text is about, the model's read."""
    import intent
    return intent.ask_choice(
        "A piece of a chat with an assistant: {text}. What is it about? image = a "
        "picture or a photo; code = program code or files; research = a search, a "
        "study or a presentation; chat = anything else.",
        (text or "")[:1500], ("image", "code", "research", "chat"), "chat")


def select_cards(state: dict, query: str, budget: int = CARD_BUDGET) -> dict:
    """Decision-aware selection: when the memory is too long to send whole, keep
    the goal, every open request, the newest artifacts, and the cards that share
    words or task kind with the request. The rest stays in the stored state."""
    if len(render(state)) <= budget:
        return state
    words = {w for w in re.findall(r"\w{4,}", (query or "").lower())}
    qk = _kind_of(query)

    def score(card: str) -> float:
        # HyMem isolation: a card tagged with ANOTHER task kind competes only on
        # shared words; the request's own kind (and untagged chat) gets a lift.
        cw = set(re.findall(r"\w{4,}", card.lower()))
        tag = re.match(r"\[(image|research|code)\]", card)
        kind_bonus = 1.0 if (tag and tag.group(1) == qk) else (0.3 if not tag else -0.5)
        return len(words & cw) + kind_bonus

    out = empty_state()
    out["goal"], out["open"] = state["goal"], list(state["open"])
    out["artifacts"] = state["artifacts"][-6:]
    out["folded"] = state["folded"][-4:]
    for k in ("decisions", "prefs"):
        ranked = sorted(state[k], key=score, reverse=True)
        out[k] = [c for c in state[k] if c in ranked[:6]]      # keep original order
    return out


# ----------------------------------------------------------- compaction ----

DELTA_PROMPT = """You maintain the working memory of an assistant conversation.
You get the CURRENT memory and a span of turns that is about to be folded away.
Return ONLY a JSON object with what the span ADDS or CLOSES -- never restate what
the memory already holds:
{"goal": "<the user's current overall goal, or null if unchanged>",
 "add_decisions": ["facts, choices, names, numbers established in the span"],
 "add_open": ["requests the user made that are still NOT done"],
 "close_open": ["short text of memory open-requests the span completed"],
 "add_prefs": ["stable user preferences stated in the span"]}
Each item one short line, in the conversation's language. Keep file names and
ids verbatim. Omit pleasantries, tool mechanics and superseded attempts."""


def _guidelines() -> str:
    for p in (os.getenv("CONTEXT_GUIDELINES", ""),
              os.path.join(_archive_dir(), "..", "context_guidelines.txt")):
        if p and os.path.exists(p):
            try:
                return open(p, encoding="utf-8").read().strip()
            except OSError:
                pass
    return ""


def _render_span(msgs: List[dict]) -> str:
    lines = []
    for m in msgs:
        role, c = m.get("role"), (m.get("content") or "")
        c = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
        if role == "user":
            lines.append(f"User: {c.strip()[:1200]}")
        elif role == "assistant":
            if c.strip():
                lines.append(f"Assistant: {c.strip()[:800]}")
            for tc in m.get("tool_calls") or []:
                f = tc.get("function") or {}
                lines.append(f"(tool {f.get('name')}: {str(f.get('arguments'))[:200]})")
        elif role == "tool":
            lines.append(f"(result: {c.strip()[:300]})")
    return "\n".join(lines)


def _llm_delta(ctx, state: dict, span_text: str) -> Optional[dict]:
    if ctx is None:
        return None
    try:
        import llm
        from utils import safe_json_from_llm
        sys_p = DELTA_PROMPT + (("\n\nExtra rules learned from past failures:\n" + g)
                                if (g := _guidelines()) else "")
        resp = llm.send_to_lm_studio(
            ctx, [{"role": "system", "content": sys_p},
                  {"role": "user", "content": f"CURRENT MEMORY:\n{render(state)}\n\nSPAN:\n{span_text}"}],
            tools=[], tool_choice="none", temperature=0.2, max_tokens=700)
        return safe_json_from_llm((resp or {}).get("content") or "")
    except Exception:
        logger.warning("context v2: delta call failed", exc_info=True)
        return None


def _scrub(state: dict) -> dict:
    """An order smuggled into the chat must not live on in the memory (the
    remember_fact jailbreak, see prompt_guard)."""
    try:
        from prompt_guard import injection_reason
    except Exception:
        return state
    for k in SECTIONS[1:]:
        state[k] = [x for x in state[k] if not injection_reason(x)]
    if state["goal"] and injection_reason(state["goal"]):
        state["goal"] = ""
    return state


def _dashboard(used: int, budget: int, n_arch: int) -> str:
    return (f"[Context: ~{round(used / 1000)}k of {round(budget / 1000)}k tokens; "
            f"{n_arch} archived item(s). recall_context(id) reopens an [archived ...] "
            f"item; fold_context(note) folds finished work.]")


def manage(ctx, messages: List[dict], *, query: str = "", force: bool = False) -> List[dict]:
    """End-of-turn context maintenance. Returns the INPUT list when nothing
    changed (callers rely on identity), else a new list."""
    try:
        if not messages:
            return messages
        budget = budget_tokens()
        used = est_tokens(messages)
        fold_req = getattr(ctx, "ctx_fold_request", "") if ctx is not None else ""
        if ctx is not None and fold_req:
            ctx.ctx_fold_request = ""
        if not (force or fold_req) and used < MASK_AT * budget:
            return messages

        head = messages[0] if messages[0].get("role") == "system" else None
        i = 1 if head else 0
        state = empty_state()
        if len(messages) > i and messages[i].get("role") == "system":
            c = str(messages[i].get("content") or "")
            if c.startswith(MEMORY_MARKER.rstrip("\n")):
                state = parse(c); i += 1
            elif c.startswith(V1_SUMMARY_MARKER):
                # migrate a v1 paragraph once: it becomes one decision card
                _add(state, "decisions", ["Earlier: " + c[len(V1_SUMMARY_MARKER):].strip()[:600]])
                i += 1
        body = messages[i:]
        user_pos = [k for k, m in enumerate(body) if m.get("role") == "user"]
        keep_from = user_pos[-KEEP_TURNS] if len(user_pos) >= KEEP_TURNS else 0

        body, n_masked, arts = mask_observations(body, keep_from if not force else len(body))
        _add(state, "artifacts", arts)

        compacted = False
        if (force or fold_req or est_tokens(([head] if head else []) + body) >= COMPACT_AT * budget) \
                and keep_from > 0:
            span, tail = body[:keep_from], body[keep_from:]
            span_text = _render_span(span)
            _add(state, "artifacts", artifacts_in(span_text))
            kinds = {_KIND_BY_TOOL.get(n) for n in _tool_names(span).values()} - {None}
            kind = kinds.pop() if len(kinds) == 1 else _kind_of(span_text)
            aid = archive_put(json.dumps(span, ensure_ascii=False), kind=kind, label="turns")
            now = extract_now(span)
            _add(state, "artifacts", now["artifacts"])
            _add(state, "prefs", now["prefs"])
            for chunk in _turn_chunks(span):
                ck_text = _render_span(chunk)
                ck_kinds = {_KIND_BY_TOOL.get(n) for n in _tool_names(chunk).values()} - {None}
                ck_kind = ck_kinds.pop() if len(ck_kinds) == 1 else _kind_of(ck_text)
                delta = _llm_delta(ctx, state, ck_text)
                if delta:
                    for key in ("add_decisions", "add_open"):
                        delta[key] = _tag(ck_kind, delta.get(key))
                    merge_delta(state, delta)
                else:
                    # no summariser: keep the user's own lines (their intent)
                    _add(state, "decisions", _tag(ck_kind, [
                        f"User asked: {m['content'][:200]}" for m in chunk
                        if m.get("role") == "user" and isinstance(m.get("content"), str)]))
            first = next((m["content"] for m in span if m.get("role") == "user"
                          and isinstance(m.get("content"), str)), "")
            topic = (fold_req or first or "earlier turns").strip()[:80]
            _add(state, "folded", [f"[{kind}] {topic} -> recall_context(id='{aid}')"])
            body = tail
            compacted = True

        if not n_masked and not compacted:
            return messages
        state = _scrub(state)
        shown = select_cards(state, query or next(
            (m.get("content") for m in reversed(body) if m.get("role") == "user"), "") or "")
        n_arch = sum(1 for m in body if is_marker(m.get("content"))) + len(state["folded"])
        rebuilt = ([head] if head else [])
        mem_used = est_tokens(rebuilt + body)
        rebuilt.append({"role": "system", "content": render(shown, _dashboard(mem_used, budget, n_arch))})
        rebuilt += body
        logger.info("context v2: %d -> %d msgs, ~%d -> ~%d tok (budget %d), masked=%d compacted=%s",
                    len(messages), len(rebuilt), used, est_tokens(rebuilt), budget,
                    n_masked, compacted)
        return rebuilt
    except Exception:
        logger.warning("context v2: skipped", exc_info=True)
        return messages


# ---------------------------------------------------------------- tools ----

def recall(aid: str, limit: int = 6000) -> str:
    e = archive_get(aid)
    if not e:
        return f"[TOOL ERROR] no archived item {aid!r}; ids look like 'a1b2c3d'."
    t = e.get("text") or ""
    if e.get("label") == "turns":
        try:
            t = _render_span(json.loads(t))
        except ValueError:
            pass
    more = f"\n[... {len(t) - limit} more chars]" if len(t) > limit else ""
    return f"[recalled {e['id']} ({e.get('label') or e.get('kind')})]\n{t[:limit]}{more}"
