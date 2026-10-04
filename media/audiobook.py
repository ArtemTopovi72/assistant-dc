"""📚 Audiobook: a book file -> chapters -> sentences -> the cloned voice -> one voice message per chapter.

How it is built (50 searches, docs/audiobook_sota_2026-10.md):
  * F5 holds ~30 s per pass, so text goes in as whole sentences packed to ~220 characters, never cut mid-clause;
  * the SAME reference voice and transcript for every piece (re-using the reference is what keeps the
    timbre from drifting; a different one per piece is the drift);
  * a chapter opens with its title spoken («Глава 3. …») and a second of silence; a paragraph gets a longer
    breath than a sentence;
  * a piece that came back empty or absurdly short/long for its text is rendered once more (the ASR-free
    cousin of the usual round-trip check);
  * chapters come from the file's own headings; a book with none is cut into equal «Часть N» at paragraph ends.
Pure text and audio work here; the Telegram side is bot/tg_audiobook.py.
"""
import os
import re
import zipfile

BOOK_EXTS = {".txt", ".text", ".md", ".fb2", ".epub", ".pdf", ".docx"}   # + .fb2.zip via the .zip check
MAX_CHUNK = 220            # characters per synthesis call (F5: ~30 s of audio per pass, reference included)
PART_CHARS = 6000          # a book without headings: about this much text per «Часть»
MIN_PIECE = 40             # characters: shorter than this F5 bleeds the reference's last words into the speech
MIN_CHAPTER = 600          # a bare-number «chapter» shorter than this (a page number) is glued to the previous one; a titled short chapter stays
MAX_CHAPTER_SECONDS = 25 * 60   # a voice message is cut here (the rest follows as the next part)

# A chapter heading is a SHORT line that opens with one of these words or is a bare number / roman numeral.
_HEAD = re.compile(
    r"^\s*(?:(?:глава|часть|раздел|пролог|эпилог|предисловие|chapter|part|prologue|epilogue|book)\b.{0,70}"
    r"|(?:\d{1,3}|[IVXLC]{1,7})[.)]?)\s*$", re.I)
_SENT_END = re.compile(r"(?<=[.!?…])[\"»”')\]]*\s+")


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def _fb2_text(raw: bytes) -> str:
    """Paragraphs of the main <body> only: notes/comments bodies, binaries and the cover are not read aloud."""
    from xml.etree import ElementTree as ET
    root = ET.fromstring(raw)
    strip = lambda t: t.split("}", 1)[-1]
    out = []
    for body in root:
        if strip(body.tag) != "body" or body.get("name") in ("notes", "comments"):
            continue
        def walk(el):
            tag = strip(el.tag)
            if tag == "title":
                out.append("\n" + " ".join("".join(p.itertext()).strip() for p in el if strip(p.tag) == "p") + "\n")
            elif tag in ("p", "v", "subtitle", "text-author"):
                out.append("".join(el.itertext()).strip())
            elif tag == "empty-line":
                out.append("")
            else:
                for ch in el:
                    walk(ch)
        walk(body)
    return "\n".join(out)


def read_book(path: str) -> str:
    """The plain text of a book file. ValueError for a type we do not read."""
    low = path.lower()
    if low.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            inner = next((n for n in zf.namelist() if n.lower().endswith(".fb2")), None)
            if not inner:
                raise ValueError("zip without fb2")
            return _fb2_text(zf.read(inner))
    ext = os.path.splitext(low)[1]
    if ext == ".fb2":
        with open(path, "rb") as fh:
            return _fb2_text(fh.read())
    if ext in (".txt", ".text", ".md"):
        with open(path, "rb") as fh:
            return _decode(fh.read())
    if ext in (".pdf", ".epub", ".docx"):
        from knowledge import library        # the project's own extractors; reading only, nothing is indexed
        return library.extract_text(path)
    raise ValueError("unsupported book type: " + ext)


def clean(text: str) -> str:
    """Page furniture out: hyphenated line breaks joined, bare page numbers dropped, blanks collapsed."""
    text = text.replace("\r", "").replace("­", "")
    text = re.sub(r"\[\d{1,3}\]|[¹²³⁰-⁹]+", "", text)             # [12] and superscript note marks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    lines = [ln.rstrip() for ln in text.split("\n")]
    lines = [ln for ln in lines if not re.fullmatch(r"\s*[-–—]\s*\d{1,4}\s*[-–—]\s*", ln)]    # «- 12 -» page marks
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def split_chapters(text: str) -> list:
    """[(title, body)] -- by the book's own headings, else equal «Часть N» cut at paragraph ends."""
    text = clean(text)
    lines = text.split("\n")
    marks = [i for i, ln in enumerate(lines) if ln.strip() and _HEAD.match(ln)
             and (i == 0 or not lines[i - 1].strip())]
    chapters = []
    if len(marks) >= 2:
        if marks[0] > 0 and "\n".join(lines[:marks[0]]).strip():
            chapters.append(("", "\n".join(lines[:marks[0]]).strip()))
        for a, b in zip(marks, marks[1:] + [len(lines)]):
            title = lines[a].strip()
            body_start = a + 1
            # a one-word heading «Глава 3» is often followed by its own name on the next line
            if body_start < len(lines) and lines[body_start].strip() and len(lines[body_start]) < 70 \
                    and not re.search(r"[.!?…]$", lines[body_start].strip()) and re.fullmatch(r"(?i)\s*(глава|chapter|часть|part)\s+\S+\s*", title):
                title += ". " + lines[body_start].strip()
                body_start += 1
            body = "\n".join(lines[body_start:b]).strip()
            if not body:
                continue
            if chapters and len(body) < MIN_CHAPTER and chapters[-1][0] and re.fullmatch(r"(?:\d{1,3}|[IVXLC]{1,7})[.)]?", title.strip()):   # a stray page number
                chapters[-1] = (chapters[-1][0], chapters[-1][1] + "\n" + title + "\n" + body)
            else:
                chapters.append((title, body))
        return chapters
    paras = [p for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    paras = [q for p in paras for q in (_SENT_END.split(p) if len(p) > PART_CHARS else [p])]   # one endless paragraph
    cur, size, n = [], 0, 0
    for p in paras:
        cur.append(p)
        size += len(p)
        if size >= PART_CHARS:
            n += 1
            chapters.append((f"Часть {n}", "\n".join(cur)))
            cur, size = [], 0
    if cur:
        n += 1
        chapters.append((f"Часть {n}" if n > 1 else "", "\n".join(cur)))
    return chapters


def chunk_text(body: str, limit: int = MAX_CHUNK) -> list:
    """[(sentences_text, paragraph_end)] -- whole sentences packed up to `limit`; a longer sentence is cut at
    its last comma/space under the limit. paragraph_end marks the piece that closes a paragraph."""
    out = []
    for para in [p.strip() for p in re.split(r"\n+", body) if p.strip()]:
        para = re.sub(r"\s+", " ", para)
        pieces, cur = [], ""
        for s in _SENT_END.split(para):
            s = s.strip()
            while len(s) > limit:
                cut = max(s.rfind(",", 0, limit), s.rfind(";", 0, limit), s.rfind(" ", 0, limit))
                cut = cut if cut > limit // 3 else limit
                head, s = s[:cut + 1].strip(), s[cut + 1:].strip()
                if cur:
                    pieces.append(cur)
                    cur = ""
                pieces.append(head)
            if not s:
                continue
            if cur and len(cur) + 1 + len(s) > limit:
                pieces.append(cur)
                cur = s
            else:
                cur = (cur + " " + s).strip()
        if cur:
            pieces.append(cur)
        out += [(p, False) for p in pieces[:-1]] + ([(pieces[-1], True)] if pieces else [])
    merged = []
    for text, para_end in out:           # a very short piece makes F5 speak the reference's tail (live 10-04): join it to the next
        if merged and len(merged[-1][0]) < MIN_PIECE:
            merged[-1] = (merged[-1][0] + " " + text, para_end)
        else:
            merged.append((text, para_end))
    if len(merged) > 1 and len(merged[-1][0]) < MIN_PIECE:      # a short last piece joins the one before it
        merged[-2:] = [(merged[-2][0] + " " + merged[-1][0], merged[-1][1])]
    return merged


def spoken_title(title: str, n: int) -> str:
    """What is said before a chapter: its own heading, or «Глава N» when the book had none."""
    t = (title or "").strip().rstrip(".")
    return t if t else f"Глава {n}"


def caption(title: str, n: int, total: int, part: int = 1, parts: int = 1) -> str:
    """The text under a chapter's voice message: which chapter of how many, its name, and the part when it was cut."""
    name = spoken_title(title, n)[:200]
    head = f"📖 {n}/{total} · {name}" if not name.lower().startswith(("глава", "chapter")) else f"📖 {name} · {n}/{total}"
    return head + (f" · часть {part}/{parts}" if parts > 1 else "")


def heard_matches(text: str, heard: str, floor: float = 0.7) -> bool:
    """The piece, transcribed back, still says the text (word overlap): a cut-off, looped or
    reference-bleeding piece fails. An empty transcript is not evidence either way."""
    import difflib
    norm = lambda t: re.findall(r"\w+", (t or "").lower().replace("ё", "е"))
    a, b = norm(text), norm(heard)
    if not b or len(a) < 4:
        return True
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= floor


def _plausible(text: str, seconds: float) -> bool:
    """Speech runs ~12-20 characters a second; far outside that the piece was cut off, empty or looped."""
    chars = max(1, len(re.sub(r"\W", "", text)))
    return 0.03 <= seconds / chars <= 0.35


def _trim(seg, floor_db: float = -45.0, keep_ms: int = 60):
    """The engine's own leading/trailing silence differs piece to piece (it reads as hesitation); cut it,
    keep a hair and fade the edges, then the gaps are put back on purpose."""
    from pydub.silence import detect_leading_silence
    start = max(0, detect_leading_silence(seg, floor_db) - keep_ms)
    end = len(seg) - max(0, detect_leading_silence(seg.reverse(), floor_db) - keep_ms)
    return seg[start:end].fade_in(8).fade_out(8) if end - start > 300 else seg


def render_chapter(synth, title: str, body: str, n: int, out_wav: str, cancelled=lambda: False,
                   on_piece=lambda i, total: None, verify=None) -> bool:
    """Speak one chapter into out_wav. `synth(text) -> wav path | None` is the cloned voice.
    `verify(text, wav) -> bool` (optional) is the ASR check: a piece that does not say its text is re-spoken once.
    False when cancelled or nothing could be spoken."""
    from pydub import AudioSegment
    pieces = chunk_text(body)
    head = spoken_title(title, n) + "."
    # the title rides WITH the first sentences: spoken alone (2-3 words) F5 bleeds the reference tail into it
    pieces = [(head + " " + pieces[0][0], pieces[0][1])] + pieces[1:] if pieces else [(head, True)]
    audio = AudioSegment.silent(duration=300)
    bad = 0
    for i, (text, para_end) in enumerate(pieces):
        if cancelled():
            return False
        seg = None
        for _try in (1, 2):                      # ponytail: one retry; add an ASR check if bad pieces persist
            wav = synth(text)
            if wav and os.path.exists(wav):
                cand = AudioSegment.from_file(wav)
                said = True
                if verify is not None:
                    try:
                        said = verify(text, wav)
                    except Exception:
                        said = True               # a broken checker never blocks the book
                try:
                    os.unlink(wav)
                except OSError:
                    pass
                cand = _trim(cand)
                if said and _plausible(text, len(cand) / 1000.0):
                    seg = cand
                    break
                seg = seg or cand                # keep the odd one if the retry is no better
        if seg is None:
            bad += 1
            continue
        audio += seg + AudioSegment.silent(duration=800 if para_end else 300)
        on_piece(i + 1, len(pieces))
    if bad > len(pieces) // 2 or len(audio) < 1500:
        return False
    audio.export(out_wav, format="wav")
    return True


def split_long(wav_path: str, seconds: int = MAX_CHAPTER_SECONDS) -> list:
    """The chapter as one or more wav files, each at most `seconds` long (cut at the quietest second near the mark)."""
    from pydub import AudioSegment
    seg = AudioSegment.from_wav(wav_path)
    ms = seconds * 1000
    if len(seg) <= ms * 1.2:
        return [wav_path]
    parts, start, k = [], 0, 0
    while len(seg) - start > ms * 1.2:
        window = seg[start + ms - 20000: start + ms]
        quiet = min(range(0, len(window) - 1000, 500), key=lambda o: window[o:o + 1000].dBFS)
        end = start + ms - 20000 + quiet + 500
        k += 1
        p = f"{wav_path[:-4]}_p{k}.wav"
        seg[start:end].export(p, format="wav")
        parts.append(p)
        start = end
    k += 1
    p = f"{wav_path[:-4]}_p{k}.wav"
    seg[start:].export(p, format="wav")
    parts.append(p)
    return parts
