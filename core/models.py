import threading
import re
import time
import json
import os
import logging
from collections import deque
from itertools import islice
from dataclasses import dataclass, field
from pathlib import Path

from prompt_guard import fact_rejection
from typing import Any, Callable, Deque, List, Optional, TypedDict

from config import (
    MEMORY_LIMIT as _MEMORY_LIMIT,
    MEMORY_DIR as _MEMORY_DIR,
    PINNED_FACTS_LIMIT as _FACTS_LIMIT,
)

logger = logging.getLogger("assistant.models")


def _atomic_write_json(path: Path, data) -> None:
    """Write JSON via a PRIVATE temp file + os.replace. The temp name includes
    pid+tid: with a fixed name, two app instances sharing one profile truncate
    each other's temp mid-write and a half-written file can get renamed into
    place (Windows allows replacing a file that another process keeps writing
    to through an open handle). os.replace itself is retried briefly because a
    concurrent reader's open handle makes it fail with WinError 5/32."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}_{threading.get_ident()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    try:
        for attempt in range(3):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.01 * (attempt + 1))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


@dataclass
class Models:
    whisper: Any
    tts_model: Any
    vocoder: Any
    accentor: Any
    accentor_loaded: bool

    @classmethod
    def load(cls, whisper: bool = True) -> "Models":
        from f5_tts.infer.utils_infer import load_model, load_vocoder
        from f5_tts.model import DiT
        from config import (
            WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, DEVICE,
            WEIGHTS_PATH, VOCAB_PATH,
        )

        # whisper=False: the Skyrim voice server transcribes with GigaAM only.
        if whisper:
            from faster_whisper import WhisperModel
            logger.info("Loading Whisper model...")
            # turbo in int8_float16: 1.0 GB of VRAM vs 3.8 GB for large-v3, 2x faster,
            # same text on RU voice notes and better English language detection.
            # The freed card is what lets a 16-18 GB chat model stop spilling.
            import config as _cfg
            # Speech input is optional like voice output: the first start
            # downloads the model from Hugging Face, and an offline machine
            # (or a blocked hub) must still open the app for text chat.
            try:
                whisper = WhisperModel(
                    getattr(_cfg, "WHISPER_MODEL", "large-v3-turbo"),
                    device=WHISPER_DEVICE,
                    compute_type=WHISPER_COMPUTE_TYPE,
                    cpu_threads=16,
                    num_workers=4,
                )
            except Exception as exc:
                logger.warning("Whisper not loaded (%s) -- speech recognition is off; "
                               "it loads on the next start once the model can be downloaded",
                               str(exc).splitlines()[0][:200])
                whisper = None
        else:
            whisper = None

        # Voice output is optional: the checkpoint and vocoder are private
        # downloads, and without them the rest of the app still works (text
        # replies, images, research). synth_single_segment returns None when
        # tts_model is None, which every caller already treats as "no audio".
        from config import VOCOS_DIR
        vocoder = tts_model = None
        if not (Path(WEIGHTS_PATH).exists() and (Path(VOCOS_DIR) / "config.yaml").exists()):
            logger.warning("F5-TTS weights or vocoder missing (%s, %s) -- voice output is off",
                           WEIGHTS_PATH, VOCOS_DIR)
        else:
            # A checkpoint that does not match the vocab (a new download next to
            # an old vocab.txt) raises a size mismatch; voice goes off, the app
            # still opens.
            try:
                logger.info("Loading vocoder...")
                vocoder = load_vocoder(
                    vocoder_name="vocos",
                    is_local=True,
                    local_path=str(VOCOS_DIR),
                    device=DEVICE,
                )

                logger.info("Loading TTS model %s on %s...", Path(WEIGHTS_PATH).name, DEVICE)
                model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
                tts_model = load_model(DiT, model_cfg, str(WEIGHTS_PATH), vocab_file=str(VOCAB_PATH),
                                       device=DEVICE)
            except Exception as exc:
                logger.error("F5-TTS did not load (%s: %s) -- voice output is off",
                             type(exc).__name__, str(exc).splitlines()[0][:300] if str(exc) else "")
                vocoder = tts_model = None

        accentor = None
        accentor_loaded = False
        try:
            # Bilingual heavy-Transformer accentor: RUAccent (turbo3.1, context-aware
            # homograph model) for Russian + CharsiuG2P (byT5) for English, both
            # emitting '+' before the stressed vowel (F5's char-level format). Models
            # load lazily on first use, so model startup stays fast. The callable
            # carries no apostrophes, so the downstream _apos_to_plus() is a no-op.
            from stress import BilingualAccentor
            from config import (RUACCENT_MODEL, RUACCENT_DEVICE, RUACCENT_WORKDIR,
                                STRESS_ENGLISH, CHARSIU_G2P_MODEL)
            accentor = BilingualAccentor(
                ru_model=RUACCENT_MODEL, ru_workdir=RUACCENT_WORKDIR,
                device=RUACCENT_DEVICE, en_model=CHARSIU_G2P_MODEL,
                enable_english=STRESS_ENGLISH)
            accentor_loaded = True
            if not os.getenv("F5_TEST_RUN"):
                accentor.warm()
            logger.info("Accentor ready (RUAccent %s + CharsiuG2P%s, device=%s; lazy-load)",
                        RUACCENT_MODEL, "" if STRESS_ENGLISH else " disabled", RUACCENT_DEVICE)
        except Exception as exc:
            logger.warning("Accentor not loaded: %s", exc)

        return cls(
            whisper=whisper,
            tts_model=tts_model,
            vocoder=vocoder,
            accentor=accentor,
            accentor_loaded=accentor_loaded,
        )


@dataclass
class Context:
    models: Models
    transcription_cache: dict
    cache_file: Path
    asr_lock: threading.Lock
    tts_lock: threading.Lock
    api_lock: threading.Lock = field(default_factory=threading.Lock)
    memory_lock: threading.Lock = field(default_factory=threading.Lock)
    last_api_call_time: float = 0.0
    api_min_interval: float = 1.0
    session_memory: Deque[dict] = field(default_factory=lambda: deque(maxlen=_MEMORY_LIMIT))
    # Durable facts saved via the remember_fact tool. Unlike session_memory
    # (a rolling window — old entries fall out of the injected snapshot within
    # a few turns), pinned facts are ALWAYS injected into the model's context
    # and persist per memory profile (facts.json).
    pinned_facts: List[dict] = field(default_factory=list)
    # Set by forget_facts("all") and cleared by the next remember_fact: the
    # dropped facts are still IN the chat history ("remember: ... allergic to
    # nuts" three turns up), and the model answered "you are allergic to nuts"
    # from there right after confirming it had forgotten (live 2026-09-13,
    # mega run 3, step 61). While this holds, every turn carries a note that
    # what the user said about themselves earlier is void.
    facts_forgotten: bool = False

    # LLM selection (chosen at startup via model_selector)
    model_name: str = ""
    no_think: bool = True
    # gpt-oss reasoning depth — "low" | "medium" | "high". Sent as `reasoning_effort`
    # only for gpt-oss models (they ignore the Qwen `no_think`/enable_thinking
    # switch); harmless/ignored for every other model.
    reasoning_effort: str = "high"

    # Desired length of the spoken/text reply, set via the GUI "Response length"
    # control. "auto" lets the model size the answer to the question; "short" forces
    # terse 1–3 sentence replies; "long" allows thorough, multi-paragraph answers.
    # Orthogonal to `no_think` (which governs reasoning depth, not answer length):
    # together they pick the system-prompt length directive and the token budget.
    response_length: str = "auto"

    # Optional custom reference voice (any audio format) set via the 'v' command.
    custom_ref_wav: Optional[str] = None

    # Optional personality file path; content is injected into the system prompt each turn.
    custom_personality_path: Optional[str] = None
    custom_personality_text: str = ""

    # Microphone/voice input disabled flag (set via GUI settings).
    mic_disabled: bool = False

    # Internet search master switch (set via GUI). When False, the `search` and
    # `deep_research` tools are withheld from the model each turn AND their
    # handlers refuse, so the assistant answers from its own knowledge instead of
    # going online. Image/other tools are unaffected.
    web_search_enabled: bool = True

    # Reference-person image mode (opt-in, default OFF; mirrors config.REFERENCE_PERSON_MODE).
    # When True, drawing a real named person fetches a web photo to use as an identity
    # reference (+ reads their appearance) instead of a from-scratch text2image guess.
    reference_person_mode: bool = False

    # Image-edit routing mode. "auto" keeps the per-category routing
    # (image.route_edit_request, always rendered by FireRed); "firered_whole" re-renders the
    # ENTIRE frame through the instruction edit with no mask/composite and NO
    # identity guards — complete freedom over the whole frame, face included
    # (user-selected via the sidebar 🖼 Whole frame toggle).
    image_edit_engine: str = "auto"

    # Voice output (TTS) disabled flag — agent still generates text, just never speaks.
    tts_disabled: bool = False

    # GUI mode: the graph stores the TTS wav in state instead of opening an
    # external player, so the GUI can play it inline.
    gui_mode: bool = False

    # Optional UI hook: nodes/tools report the current pipeline stage (e.g.
    # "Searching the web", "Drawing a picture") so the GUI can animate it.
    stage_callback: Optional[Callable[[str], None]] = None
    # A short note sent to the user BEFORE a long tool runs (interim.py); the
    # bot sets it per task, the desktop leaves it None.
    interim_callback: Optional[Callable[[str], None]] = None
    # What the user says WHILE this task runs (steer.Inbox); the agent loop
    # folds it into its next round.
    steer_inbox: Optional[Any] = None
    voice_ref: str = ""
    anim_voices: list = field(default_factory=list)   # 🎙 samples for 🎬 Animate

    # Active memory profile directory — all save/load/compact ops target this path.
    active_memory_dir: Path = field(default_factory=lambda: _MEMORY_DIR / "default")

    # Most recent generated/redrawn image — lets the redraw_image tool and the GUI
    # "Redraw" button operate on the last picture without re-specifying it.
    last_image_path: Optional[str] = None
    last_image_prompt: str = ""
    # This turn's picture was POINTED AT by the user (a reply to it, a button
    # under it); graph.needs_relook then answers any question by looking.
    image_pointed_at: bool = False
    # The plan of the most recent deck, so "добавь слайд" edits it instead of
    # planning a new one from scratch (live: the theme and every slide changed).
    last_deck: Optional[dict] = None

    # A picture this turn actually RENDERED, as opposed to one it merely loaded.
    # last_image_path cannot answer that: it holds the user's own upload just as
    # readily as a fresh render, so a caller that recovers from a half-finished
    # turn would post their photo back at them. This is set only where a render
    # succeeded, and is cleared at the start of every turn -- which is what lets
    # a turn killed by the watchdog still deliver a picture that was already
    # finished. Observed live: a chicken-in-a-jersey render completed and scored
    # 9/10, then two inspect_image calls ran the turn past its deadline and the
    # finished picture was thrown away.
    last_render_path: Optional[str] = None
    last_render_status: str = ""

    # Recently loaded/pasted images, oldest→newest, for multi-image reference
    # editing (transfer_image). The newest is normally the target; the earlier
    # ones are references. GUI appends here on every paste/load. Capped to keep
    # memory bounded.
    reference_images: list = field(default_factory=list)

    # Cooperative cancellation: the GUI Stop button sets this event; long-running
    # loops (ComfyUI poll, deep-research crawl, the tool round loop) check it at
    # iteration boundaries and bail out early. Cleared before each new operation.
    cancel_event: threading.Event = field(default_factory=threading.Event)

    # Last deep-research (Ultra Search) result — full markdown report on disk plus
    # the in-memory text, so the GUI Research tab and the tool can both surface it.
    last_research_report: str = ""
    last_research_path: Optional[str] = None

    # Monotonic count of completed user turns this conversation. Drives automatic
    # rolling compaction of the chat history (graph.compact_history_if_needed):
    # every HISTORY_COMPACT_EVERY turns the older messages are folded into one
    # summary so the context sent to the model stays small. Reset by clear-context.
    total_user_turns: int = 0

    def __setattr__(self, name, value):
        # «нет, верни как было» redrew a new ginger cat: every picture this one
        # replaced is kept, newest last, so an undo hands back the real file.
        if name == "last_image_path":
            old = self.__dict__.get("last_image_path")
            if old and value and old != value:
                # a new list: a copy.copy'd per-chat ctx must not append to the shared one
                self.__dict__["image_undo"] = (self.__dict__.get("image_undo", []) + [old])[-10:]
        object.__setattr__(self, name, value)

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def load_memory(self, memory_dir: Path) -> None:
        """Load summary + previous session from disk into the deque."""
        summary_file = memory_dir / "summary.json"
        session_file = memory_dir / "session_memory.json"
        to_add: list = []
        if summary_file.exists():
            try:
                with open(summary_file, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("text"):
                    to_add.append({"ts": data.get("ts", time.time()), "kind": "summary",
                                   "text": data["text"], "meta": {}})
            except Exception as exc:
                logger.warning("Could not load memory summary: %s", exc)
        if session_file.exists():
            try:
                with open(session_file, encoding="utf-8") as f:
                    items = json.load(f)
                # A corrupted file can be a JSON object/scalar, or a list holding
                # non-dict entries. Keep ONLY dict items (mirrors the facts filter) —
                # otherwise a bare string enters session_memory and memory_text() then
                # crashes with `'str' object has no attribute 'get'` on EVERY turn.
                # Only restore durable kinds across sessions. "assistant", "generate",
                # "search" etc. are in-session noise — they accumulate fast (especially
                # from Telegram), are stale by the time the next session starts, and
                # pollute the context with old unrelated turns. Discard them on load;
                # only "fact" and "summary" survive across sessions.
                _DURABLE = {"fact", "summary"}
                if isinstance(items, list):
                    to_add.extend(it for it in items
                                  if isinstance(it, dict) and it.get("kind") in _DURABLE)
                else:
                    logger.warning("session_memory.json has unexpected shape %s; ignored",
                                   type(items).__name__)
            except Exception as exc:
                logger.warning("Could not load session memory: %s", exc)
        # Facts use REPLACE semantics (the deque is appended to because callers
        # clear it explicitly first) — profile switching relies on this: loading
        # a profile without facts.json must drop the previous profile's facts.
        facts: list = []
        facts_file = memory_dir / "facts.json"
        if facts_file.exists():
            try:
                with open(facts_file, encoding="utf-8") as f:
                    facts = json.load(f)
            except Exception as exc:
                logger.warning("Could not load pinned facts: %s", exc)
        with self.memory_lock:
            for item in to_add:
                self.session_memory.append(item)
            self.pinned_facts[:] = [
                f for f in facts if isinstance(f, dict) and str(f.get("text", "")).strip()
                and not fact_rejection(str(f.get("text", "")))
            ][-_FACTS_LIMIT:]

    def save_memory(self, memory_dir: Path) -> None:
        """Persist current session memory to disk (excludes summary items — those live in summary.json)."""
        try:
            memory_dir.mkdir(parents=True, exist_ok=True)
            _DURABLE = {"fact"}
            with self.memory_lock:
                items = [i for i in self.session_memory
                         if i.get("kind") in _DURABLE]
                facts = list(self.pinned_facts)
            _atomic_write_json(memory_dir / "session_memory.json", items)
            _atomic_write_json(memory_dir / "facts.json", facts)
        except Exception as exc:
            logger.error("Failed to save session memory: %s", exc)

    def set_stage(self, stage: str) -> None:
        try:
            import turn_trace   # every front end's phases, one place
            turn_trace.stage(stage)
        except Exception:
            pass
        cb = self.stage_callback
        if cb is not None:
            try:
                cb(stage)
            except Exception:
                pass

    def save_cache(self) -> None:
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(self.cache_file, self.transcription_cache)
        except Exception as exc:
            logger.error("Failed to save transcription cache: %s", exc)

    def remember(self, kind: str, text: str, meta: Optional[dict] = None) -> None:
        item = {
            "ts": time.time(),
            "kind": kind,
            "text": text.strip(),
            "meta": meta or {},
        }
        if not item["text"]:
            return
        with self.memory_lock:
            self.session_memory.append(item)  # deque(maxlen) evicts oldest automatically

    def memory_text(self, limit: int = 12) -> str:
        with self.memory_lock:
            # deque doesn't support slicing; islice from the tail
            items = list(islice(self.session_memory, max(0, len(self.session_memory) - limit), None))
        lines = []
        for i, it in enumerate(items, 1):
            if not isinstance(it, dict):
                continue  # defense-in-depth: never let a stray non-dict crash memory_text
            text = str(it.get("text", "")).strip()
            if text:
                lines.append(f"{i}. [{it.get('kind', 'note')}] {text}")
        return "\n".join(lines)

    def pin_fact(self, text: str) -> bool:
        """Store a durable fact (remember_fact tool). Returns False for an
        exact duplicate. Oldest facts are dropped past PINNED_FACTS_LIMIT, so
        a re-stated fact (e.g. a changed preference) appears last = newest."""
        text = (text or "").strip()
        if not text or fact_rejection(text):
            # An order to the assistant is not a fact (prompt_guard); saving
            # it would replay the injection at the top of every future turn.
            return False
        with self.memory_lock:
            if any(str(f.get("text", "")).strip().casefold() == text.casefold()
                   for f in self.pinned_facts):
                return False
            self.pinned_facts.append({"ts": time.time(), "text": text})
            self.facts_forgotten = False
            del self.pinned_facts[:-_FACTS_LIMIT]
        return True

    def forget_facts(self, what: str = "") -> int:
        """Drop saved facts. `what` empty or "all"/"everything" clears them
        all; otherwise every fact containing one of the words of `what`
        (case-insensitive) goes. Returns how many were dropped."""
        key = (what or "").strip().casefold()
        with self.memory_lock:
            before = len(self.pinned_facts)
            if key in ("", "all", "everything", "всё", "все", "*"):
                self.pinned_facts.clear()
                self.facts_forgotten = True
            else:
                words = [w for w in re.findall(r"\w{3,}", key) if w]
                self.pinned_facts[:] = [
                    f for f in self.pinned_facts
                    if not any(w in str(f.get("text", "")).casefold() for w in words)]
            return before - len(self.pinned_facts)

    def facts_text(self, query: str = "") -> str:
        with self.memory_lock:
            facts = [str(f.get("text", "")).strip() for f in self.pinned_facts]
        # Read-side filter too: a store poisoned before the write guard existed
        # must not keep feeding the injection back into the prompt.
        facts = [t for t in facts if t and not fact_rejection(t)]
        if query:
            # Just-in-time: only the facts this message needs (memory_select).
            import memory_select
            if memory_select.enabled():
                facts = memory_select.select(facts, query)
        return "\n".join(f"- {t}" for t in facts if t)


class AgentState(TypedDict, total=False):
    messages: List[dict]
    user_input: str
    # Original (pre-translation) user message when the turn arrived in a
    # non-English language; user_input then holds the English translation the
    # whole internal pipeline runs on (see graph.translate_node).
    user_input_original: str
    image_data: Optional[bytes]
    final_answer: str

    session_memory_text: str
    vision_summary: str

    # set by the generate_image tool handler (tools.py)
    image_path: str
    image_score: int
    image_attempt: int
    image_status: str
    # set by the editing tool handlers (tool_image_handlers._note_source): the
    # picture an edit started from, so delivery can register the result as a
    # version of it. Undeclared, StateGraph would drop it silently.
    image_derived_from: str

    # set by the deep_research tool handler (tools.py)
    research_report: str
    research_path: str

    # set by the create_presentation tool handler (tools.py)
    # These MUST be declared. StateGraph is built on this TypedDict, so a key a
    # node writes but the schema does not name is dropped from the state the
    # graph returns — and tg_bot's document delivery keys off exactly these two.
    # The deck really was built (7 slides, 6 minutes) and the user was told it
    # was ready, while the path never survived the graph and no file was ever
    # sent. Same false-success the image delivery guard exists to prevent.
    document_path: str
    document_status: str

    # set by the generate_video tool handler (tools.py) — same rule as the deck
    # keys above, and the same cost for getting it wrong: a clip is MINUTES of
    # GPU, so a path dropped by the schema means the user is told their video is
    # ready and never receives it. Both tg_bot's and the GUI's video delivery
    # read exactly these.
    video_path: str
    video_status: str
    video_seconds: float
    ask_voices: int          # generate_video stopped to ask «свои голоса / стандартные» (speakers)

    # set by the tts node (graph.py): path to the synthesized reply audio
    tts_path: str
