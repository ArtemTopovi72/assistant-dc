"""User-maintained manual stress overrides — a per-word dictionary that takes
precedence over the automatic accentor (RUAccent / CharsiuG2P).

The user edits this from the "Stress" GUI tab for edge cases the automatic model
gets wrong. Each entry is a fully stressed spelling with a single ``+`` before the
stressed vowel (the F5 vocab format). At synthesis time, after the accentor runs,
every word whose base (letters only, case- and ё-folded) matches an override is
replaced by the override's spelling — re-cased to the occurrence — so the manual
mark always wins.

Storage: a JSON object {base_word: stressed_form} at config.STRESS_OVERRIDES_PATH.
The file is hot-reloaded on mtime change, so edits in the GUI take effect on the
next synthesis without a restart (GUI and synthesis share the process, but the
mtime check also covers external edits)."""
import json
import re
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_VOWELS = "аеёиоуыэюяaeiouy"
# A word token may already carry the accentor's '+'; capture it so we can replace
# the whole marked word in one shot.
_TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё+]+")


def base_form(word: str) -> str:
    """Letters-only, lowercased, ё-folded key used to match a word regardless of
    case or existing stress mark."""
    w = word.replace("+", "").replace("́", "").replace("'", "").replace("’", "")
    return w.lower().replace("ё", "е")


def normalize_stress(stressed: str) -> str:
    """Tidy a user-entered stressed form: collapse doubled '+', drop a '+' that is
    not immediately before a vowel, strip surrounding whitespace. Returns ''
    if nothing usable remains."""
    s = (stressed or "").strip()
    s = re.sub(r"\++", "+", s)
    s = re.sub(r"\+(?![" + _VOWELS + _VOWELS.upper() + r"])", "", s)
    return s


def _recase(token: str, override: str) -> str:
    """Apply the casing of ``token`` to ``override`` (which is stored lowercased).

    ALL-CAPS token -> upper the override; capitalized first letter -> capitalize the
    override's first alphabetic char; otherwise leave the override as-is."""
    letters = [c for c in token if c.isalpha()]
    if not letters:
        return override
    if len(letters) > 1 and all(c.isupper() for c in letters):
        return override.upper()
    if letters[0].isupper():
        out = list(override)
        for i, c in enumerate(out):
            if c.isalpha():
                out[i] = c.upper()
                break
        return "".join(out)
    return override


class StressOverrides:
    def __init__(self, path):
        self.path = Path(path)
        self._map: dict[str, str] = {}
        self._mtime = None
        self._lock = threading.Lock()
        self.reload()

    # ----- persistence -------------------------------------------------------
    def reload(self) -> None:
        """Reload from disk if the file's mtime changed (cheap stat each call)."""
        try:
            mt = self.path.stat().st_mtime
        except OSError:
            with self._lock:
                self._map = {}
                self._mtime = None
            return
        if mt == self._mtime:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("stress overrides: cannot read %s: %s", self.path, exc)
            data = {}
        m = {}
        if isinstance(data, dict):
            for _, stressed in data.items():
                s = normalize_stress(str(stressed))
                if s and "+" in s:
                    m[base_form(s)] = s.lower()
        with self._lock:
            self._map = m
            self._mtime = mt
        logger.info("stress overrides: loaded %d entr%s from %s",
                    len(m), "y" if len(m) == 1 else "ies", self.path)

    def save(self) -> None:
        with self._lock:
            data = {b: s for b, s in sorted(self._map.items())}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)
        try:
            self._mtime = self.path.stat().st_mtime
        except OSError:
            self._mtime = None

    # ----- editing (used by the GUI) -----------------------------------------
    def pairs(self):
        """Sorted list of (base_word, stressed_form) for display."""
        with self._lock:
            return sorted(self._map.items())

    def set_from_stressed(self, stressed_forms) -> int:
        """Replace ALL entries from an iterable of stressed spellings. Returns the
        count kept (invalid / markless entries are dropped). Saves to disk."""
        m = {}
        for s in stressed_forms:
            s = normalize_stress(str(s))
            if s and "+" in s and base_form(s):
                m[base_form(s)] = s.lower()
        with self._lock:
            self._map = m
        self.save()
        return len(m)

    def upsert(self, stressed: str) -> bool:
        s = normalize_stress(stressed)
        if not s or "+" not in s or not base_form(s):
            return False
        with self._lock:
            self._map[base_form(s)] = s.lower()
        self.save()
        return True

    def remove(self, word: str) -> None:
        with self._lock:
            self._map.pop(base_form(word), None)
        self.save()

    # ----- application -------------------------------------------------------
    def apply(self, text: str) -> str:
        """Replace each word whose base matches an override with the override's
        stressed spelling (re-cased to the occurrence). Hot-reloads first."""
        self.reload()
        with self._lock:
            m = self._map
            if not m or not text:
                return text
            local = dict(m)

        def repl(mt):
            tok = mt.group(0)
            ov = local.get(base_form(tok))
            return _recase(tok, ov) if ov else tok

        return _TOKEN.sub(repl, text)


_INSTANCE = None
_INSTANCE_LOCK = threading.Lock()


def get_overrides() -> StressOverrides:
    """Process-wide singleton bound to config.STRESS_OVERRIDES_PATH."""
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                from config import STRESS_OVERRIDES_PATH
                _INSTANCE = StressOverrides(STRESS_OVERRIDES_PATH)
    return _INSTANCE
