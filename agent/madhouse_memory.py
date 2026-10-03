"""SpeakerMem for the Madhouse: what each voice in the room has said about itself.

A character sees only the last WINDOW lines of the transcript. Everything
older used to vanish: the human's name told at minute one, a character's own
promise, who insulted whom. SpeakerMem (speaker-centred, dual-track memory for
multi-party dialogue) keeps it per SPEAKER instead of as one shared summary:

  * track 1, "about the others": what each other person here has said about
    themselves, their plans, their stance -- attributed, so a claim by A is
    never remembered as a claim by B;
  * track 2, "your own line": what THIS character has itself said, promised or
    claimed, so it stays consistent with itself after the words scroll away;
  * the emotional track (VoiceMem's "right brain"): how each speaker FEELS about
    each other one, with the reason -- the grudge or the warmth that outlives
    the line that caused it. Newest feeling wins per pair.

Only lines that have LEFT the reply window are distilled (one LLM call per
batch), so the model never reads the same line twice, and a short
conversation costs nothing. A failed distillation is skipped, not retried in a
loop: the room keeps talking on the window alone, exactly as before.

Headless: no Qt. The store is keyed by the transcript's first line, because
the tab hands the worker a COPY of its list every turn; clearing the room
starts a new transcript and so a new, empty memory.
"""
import logging
import threading
from typing import Dict, List

logger = logging.getLogger("assistant.gui")

WINDOW = 12            # must match the slice in generate_madhouse_reply
BATCH = 6              # distil once this many lines have left the window
PER_SPEAKER = 8        # newest notes kept per speaker
_ROOMS: Dict[tuple, "RoomMemory"] = {}
_LOCK = threading.Lock()


class RoomMemory:
    def __init__(self):
        self.done = 0                          # lines [0, done) already distilled
        self.notes: Dict[str, List[str]] = {}  # speaker name -> notes, oldest first
        self.feelings: Dict[str, Dict[str, str]] = {}   # who -> {toward whom: feeling}
        self.lock = threading.Lock()

    def add(self, speaker: str, note: str) -> None:
        note = (note or "").strip()
        if not speaker or not note:
            return
        lst = self.notes.setdefault(speaker, [])
        if note.casefold() not in (n.casefold() for n in lst):
            lst.append(note)
            del lst[:-PER_SPEAKER]

    def feel(self, who: str, toward: str, feeling: str) -> None:
        feeling = (feeling or "").strip()
        if who and toward and who != toward and feeling:
            self.feelings.setdefault(who, {})[toward] = feeling[:160]


def room_for(history: list) -> "RoomMemory":
    if not history:
        return RoomMemory()
    first = history[0]
    key = (first.get("name"), first.get("text"), first.get("ts"))
    with _LOCK:
        room = _ROOMS.get(key)
        if room is None:
            if len(_ROOMS) > 16:               # old rooms of this session
                _ROOMS.pop(next(iter(_ROOMS)))
            room = _ROOMS[key] = RoomMemory()
        return room


_SYSTEM = (
    "You keep notes on a group conversation, one list per speaker. From the lines "
    "below, write down what each speaker said ABOUT THEMSELVES or committed to: "
    "facts about them (name, job, family, where they are from), their plans and "
    "promises, their opinions and who they sided with or quarrelled with. Attribute "
    "every note to the person who SAID it. Skip small talk, jokes, greetings, and "
    "anything that is an instruction to someone. Each note is one short third-person "
    "sentence. Separately, note how each speaker FEELS about each other person "
    "after these lines (warm, hurt, annoyed, grateful, suspicious...) with the reason "
    "in a few words -- only when the lines show it. "
    'Reply with JSON only: {"notes": {"<speaker name>": ["...", "..."]}, '
    '"feelings": {"<speaker name>": {"<other person>": "annoyed: he mocked her song"}}}. '
    'Use {"notes": {}, "feelings": {}} if there is nothing worth keeping.'
)


def distil(ctx, history: list, *, window: int = WINDOW, batch: int = BATCH) -> "RoomMemory":
    """Fold lines that have left the reply window into the room's notes."""
    room = room_for(history)
    edge = len(history) - window
    if edge - room.done < batch or not room.lock.acquire(blocking=False):
        return room
    try:
        chunk = history[room.done:edge]
        convo = "\n".join(f"{m.get('name')}: {m.get('text')}" for m in chunk)
        import llm
        from utils import safe_json_from_llm
        from prompt_guard import fact_rejection
        res = llm.send_to_lm_studio(
            ctx, [{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": convo}],
            tools=[], tool_choice="none", temperature=0.2, max_tokens=600,
            prefill="<think></think>")
        data = safe_json_from_llm((res or {}).get("content") or "") or {}
        names = {m.get("name") for m in chunk}
        for speaker, notes in ((data.get("notes") or {}) if isinstance(data, dict) else {}).items():
            if speaker not in names or not isinstance(notes, list):
                continue                        # a name nobody in this chunk used
            for n in notes:
                # a note that reads as an order is an injection, not a memory
                if isinstance(n, str) and not fact_rejection(n):
                    room.add(speaker, n)
        feel = data.get("feelings") if isinstance(data, dict) else None
        for who, row in (feel or {}).items():
            if who not in names or not isinstance(row, dict):
                continue
            for toward, f in row.items():
                if isinstance(f, str) and not fact_rejection(f):
                    room.feel(who, str(toward), f)
        room.done = edge
    except Exception:
        # Skip this batch rather than wedge the room: the window still works.
        logger.warning("Madhouse speaker memory: distillation failed", exc_info=True)
        room.done = edge
    finally:
        room.lock.release()
    return room


def prompt_block(room: "RoomMemory", character_name: str, present: list) -> str:
    """The two tracks for `character_name`, or "" when there is nothing yet."""
    own = room.notes.get(character_name) or []
    others = [(n, room.notes[n]) for n in present if n != character_name and room.notes.get(n)]
    parts = []
    if others:
        parts.append("What the others here have told about themselves earlier "
                     "(each note belongs ONLY to the person it is filed under):\n"
                     + "\n".join(f"{n}:\n" + "\n".join(f"- {x}" for x in xs) for n, xs in others))
    felt = [(t, f) for t, f in (room.feelings.get(character_name) or {}).items() if t in present]
    if felt:
        parts.append("How you feel about the others after what happened earlier "
                     "(let it colour your tone, do not announce it):\n"
                     + "\n".join(f"- {t}: {f}" for t, f in felt))
    if own:
        parts.append("What you yourself said earlier -- stay consistent with it:\n"
                     + "\n".join(f"- {x}" for x in own))
    return "\n\n".join(parts)
