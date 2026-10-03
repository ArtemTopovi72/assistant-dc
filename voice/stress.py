"""Bilingual stress accentuation for TTS — places a single ``+`` immediately
before the stressed vowel, the format the F5 character-level vocab understands
(verified: ``+`` is a vocab token; the model is letters+``+``, NOT IPA).

Engines (both heavy Transformers, kept on CPU — the 12 GB GPU is shared):
  * Russian  -> RUAccent ``turbo3.1`` (context-aware homograph "omograph" model;
    natively emits ``+`` before the stressed vowel).
  * English  -> CharsiuG2P ``byT5`` G2P (IPA with ``ˈ``/``ˌ`` stress). The primary
    stress syllable is located in the IPA and mapped back onto the matching
    ORTHOGRAPHIC vowel, where the ``+`` is inserted — so the output stays in the
    letters+``+`` format (CharsiuG2P does NOT emit Russian stress, so it is used
    for English only).

A text is split into maximal same-script spans so a Russian sentence is accented
as a whole (preserving homograph context) while English words are handled per
word. ``BilingualAccentor`` is a drop-in ``accentor(text) -> text`` callable;
because it emits ``+`` (no apostrophes) the downstream ``_apos_to_plus`` passes it
through unchanged.
"""
import re
import logging
import threading
from functools import lru_cache

logger = logging.getLogger(__name__)

_CYR = re.compile(r"[А-Яа-яЁё]")
_LAT = re.compile(r"[A-Za-z]")
_EN_VOWEL_GROUP = re.compile(r"[aeiouyAEIOUY]+")
_EN_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
# IPA vowel nuclei (a maximal run of these = one syllable nucleus / one diphthong).
_IPA_VOWELS = set("iɪyʏeøɛœæaɶɑɒɔoʊuʌəɐɚɝɵʉɨ")
_PRIMARY = "ˈ"


def _count_ipa_nuclei(ipa: str) -> int:
    """Number of vowel nuclei in an IPA string (diphthong run counts once)."""
    n, prev_v = 0, False
    for ch in ipa:
        v = ch in _IPA_VOWELS
        if v and not prev_v:
            n += 1
        prev_v = v
    return n


def english_word_stress(word: str, ipa: str) -> str:
    """Insert ``+`` before the orthographic vowel of the primary-stressed syllable.

    The count of IPA nuclei BEFORE the ``ˈ`` mark is the 0-based index of the
    stressed syllable; the ``+`` goes before the correspondingly-indexed
    orthographic vowel group. Falls back gracefully (returns ``word`` unchanged)
    when there is no stress mark or no vowel letters.
    """
    if _PRIMARY not in ipa:
        return word
    k = _count_ipa_nuclei(ipa.split(_PRIMARY, 1)[0])
    groups = list(_EN_VOWEL_GROUP.finditer(word))
    if not groups:
        return word
    start = groups[min(k, len(groups) - 1)].start()
    return word[:start] + "+" + word[start:]


def _script_spans(text):
    """Split ``text`` into (kind, substring) spans where kind is 'cyr' or 'lat'.

    Neutral characters (spaces, punctuation, digits) stay with the current span,
    so a whole Russian clause reaches RUAccent together (homograph context) and a
    span only flips when a letter of the OTHER script appears.
    """
    spans, kind, start = [], None, 0
    for i, ch in enumerate(text):
        if _CYR.match(ch):
            k = "cyr"
        elif _LAT.match(ch):
            k = "lat"
        else:
            continue
        if kind is None:
            kind = k
        elif k != kind:
            spans.append((kind, text[start:i]))
            start, kind = i, k
    spans.append((kind or "lat", text[start:]))
    return spans


class BilingualAccentor:
    """Callable that accents Russian (RUAccent) and English (CharsiuG2P) text.

    Models load lazily on first use and are cached. English G2P results are
    memoised per lowercased word. Any failure degrades to returning the span
    unchanged — accentuation must never break synthesis.
    """

    def __init__(self, *, ru_model="turbo3.1", ru_workdir=None, device="cpu",
                 en_model="charsiu/g2p_multilingual_byT5_small_100",
                 enable_english=True):
        self.ru_model_size = ru_model
        self.ru_workdir = ru_workdir
        self.device = device
        self.en_model_name = en_model
        self.enable_english = enable_english
        self._ru = None
        self._ru_lock = threading.Lock()   # warm-up thread vs first real call
        self._en_tok = None
        self._en_model = None

    # ----- Russian -----------------------------------------------------------
    def warm(self) -> None:
        """Load RUAccent on a daemon thread: cold it costs ~2 min 10 s on the first
        voice reply after a restart (live 2026-09-25)."""
        threading.Thread(target=self._ensure_ru, daemon=True).start()

    def _ensure_ru(self):
        with self._ru_lock:
            return self._ensure_ru_locked()

    def _ensure_ru_locked(self):
        if self._ru is None:
            from ruaccent.ruaccent import RUAccent
            import pathlib
            ru = RUAccent()
            wd = str(self.ru_workdir) if self.ru_workdir else None
            if wd:
                pathlib.Path(wd).mkdir(parents=True, exist_ok=True)
            ru.load(omograph_model_size=self.ru_model_size, use_dictionary=True,
                    tiny_mode=False, workdir=wd, device=self.device)
            self._ru = ru
            logger.info("BilingualAccentor: RUAccent %s ready (device=%s)",
                        self.ru_model_size, self.device)
        return self._ru

    def _accent_russian(self, span: str) -> str:
        try:
            ru = self._ensure_ru()
            out = ru.process_all(span)
        except Exception as exc:
            logger.debug("RU accent failed on %r: %s", span[:40], exc)
            return span
        return self._silero_homographs(ru, out)

    _silero = None

    def _silero_homographs(self, ru, out: str) -> str:
        """On a homograph (замо́к/за́мок, мука́/му́ка) silero-stress reads the
        context better; on rare words (электроу́дочник) RUAccent's dictionary
        wins. So RUAccent's line stands and only a word RUAccent itself lists as
        a homograph takes silero's stress, when silero picks one of its forms.
        Bench 2026-10-02: 20/25 RUAccent, 21/25 silero, see the test."""
        try:
            if BilingualAccentor._silero is None:
                from silero_stress import load_accentor
                BilingualAccentor._silero = load_accentor()
            sil = BilingualAccentor._silero(out.replace("+", ""))
        except Exception as exc:
            logger.debug("silero-stress unavailable: %s", exc)
            return out
        word = re.compile(r"[а-яё+]+", re.I)
        a, b = word.findall(out), word.findall(sil)
        if [w.replace("+", "") for w in a] != [w.replace("+", "") for w in b]:
            return out                      # tokenised differently: keep RUAccent
        it = iter(b)

        def pick(m):
            mine, theirs = m.group(0), next(it)
            forms = getattr(ru, "omographs", {}).get(mine.replace("+", "").lower()) or ()
            return theirs if theirs != mine and theirs.lower() in forms else mine
        return word.sub(pick, out)

    # ----- English -----------------------------------------------------------
    def _ensure_en(self):
        if self._en_model is None:
            from transformers import T5ForConditionalGeneration, AutoTokenizer
            self._en_tok = AutoTokenizer.from_pretrained("google/byt5-small")
            self._en_model = T5ForConditionalGeneration.from_pretrained(self.en_model_name)
            self._en_model.to(self.device).eval()
            logger.info("BilingualAccentor: CharsiuG2P %s ready (device=%s)",
                        self.en_model_name, self.device)
        return self._en_model

    @lru_cache(maxsize=20000)
    def _g2p(self, word_lower: str) -> str:
        import torch
        model = self._ensure_en()
        inp = self._en_tok(f"<eng-us>: {word_lower}", return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = model.generate(**inp, num_beams=1, max_length=64)
        return self._en_tok.batch_decode(out, skip_special_tokens=True)[0]

    def _accent_english(self, span: str) -> str:
        if not self.enable_english:
            return span

        def repl(m):
            w = m.group(0)
            if not _EN_VOWEL_GROUP.search(w) or len(w) < 3:
                return w                      # no vowel / too short to bear marked stress
            try:
                return english_word_stress(w, self._g2p(w.lower()))
            except Exception as exc:
                logger.debug("EN stress failed on %r: %s", w, exc)
                return w
        try:
            return _EN_WORD.sub(repl, span)
        except Exception as exc:
            logger.debug("EN span failed on %r: %s", span[:40], exc)
            return span

    # ----- dispatch ----------------------------------------------------------
    def __call__(self, text: str) -> str:
        if not text or not text.strip():
            return text
        out = []
        for kind, span in _script_spans(text):
            if kind == "cyr":
                out.append(self._accent_russian(span))
            elif kind == "lat":
                out.append(self._accent_english(span))
            else:
                out.append(span)
        return "".join(out)
