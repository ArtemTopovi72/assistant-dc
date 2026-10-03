"""Does YuE2's own ABC melody put Russian stress on the right syllable?
(2026-09-28: the acute accent in the lyric is ignored -- гнИлое.)

cot=full makes YuE2 write an ABC score first; the vocal notes of a lyric line
are its syllables in order. A stressed syllable should sit on a long note or
a strong beat; when the melody puts a longer/stronger note on an unstressed
syllable, the singer stresses that one. Measure it per word.
music._generate_yue2 re-rolls the plan (yue-plan) and keeps the best-scoring one.
usage: python song_stress.py song.json   (the request json YuE2 leaves next to the song)"""
import json, re, sys

VOW = "аеёиоуыэюя"
NOTE = re.compile(r"(\^{1,2}|_{1,2}|=)?([A-Ga-gz])([,']*)(\d*)(/?\d*)(-?)")


def vocal_bars(abc: str):
    """(section, [bar strings]) for the Vocal voice, in order."""
    sec, cur, out = "", None, []
    for line in abc.splitlines():
        s = line.strip()
        if s.startswith("%"):
            sec = s.lstrip("% ").strip()
        elif s.startswith("V:"):
            cur = s[2:].strip().split()[0]
        elif cur == "Vocal" and s and not s[0].isalpha() or (cur == "Vocal" and s.startswith('"')):
            out.append((sec, [b for b in s.split("|") if b.strip()]))
    return out


def notes(abc: str, unit=16):
    """[(section, start_in_16ths, dur_in_16ths)] of sung notes; ties merged."""
    res, t, tie = [], 0, False
    for sec, bars in vocal_bars(abc):
        for bar in bars:
            bar = re.sub(r'"[^"]*"', "", bar)
            for acc, p, octv, num, den, tie_mark in NOTE.findall(bar):
                d = int(num or 1)
                if den:
                    d = d / (int(den[1:] or 2))
                if p == "z":
                    tie = False
                elif tie and res:
                    res[-1] = (res[-1][0], res[-1][1], res[-1][2] + d)
                else:
                    res.append((sec, t, d))
                t += d
                if p != "z":
                    tie = bool(tie_mark)
    return res


def syllables(word: str) -> int:
    return sum(c in VOW for c in word.lower())


def stress_index(marked: str):
    """'гнило́е' -> 1 (0-based syllable carrying the acute)."""
    n = 0
    for i, c in enumerate(marked):
        if c.lower() in VOW:
            if i + 1 < len(marked) and marked[i + 1] == "́":
                return n
            n += 1
    return None


def line_words(lyrics: str):
    """Sung lines in order: [[word, ...], ...] (tags and empty lines skipped)."""
    out = []
    for ln in lyrics.splitlines():
        if not ln.strip() or ln.strip().startswith("["):
            continue
        out.append([w for w in re.findall(r"[\ẃ]+", ln) if syllables(w)])
    return out


def plan_score(lyrics: str, abc: str) -> tuple:
    """Lower is better: (fraction of words with wrong melodic stress, -coverage).
    A plan that sings fewer words (dropped a verse) must not win by omission."""
    bad, total, _ = check(lyrics, abc)
    words = sum(1 for l in line_words(lyrics) for w in l if syllables(w) > 1)
    cover = total / words if words else 0
    return (1.0 if cover < 0.9 else bad / max(total, 1), -cover)


def check(lyrics: str, abc: str, beat=4):
    """Map syllables to notes in order; per multi-syllable word: stressed on the
    longest note (or tied longest on the strongest beat)? Returns (bad, total, rows)."""
    ns = notes(abc)
    k, bad, total, rows = 0, 0, 0, []
    for words in line_words(lyrics):
        for w in words:
            m = syllables(w)
            seg = ns[k:k + m]
            k += m
            si = stress_index(w)
            if m < 2 or si is None or len(seg) < m:
                continue
            total += 1

            def weight(n):                      # duration first, beat strength second
                _s, st, d = n
                return (d, 2 if st % 16 == 0 else 1 if st % beat == 0 else 0)
            got = max(range(m), key=lambda i: weight(seg[i]))
            ok = weight(seg[si]) >= weight(seg[got])
            bad += not ok
            rows.append((w, si, [(s[2], s[1] % 16) for s in seg], ok))
    return bad, total, rows


def _note_tokens(abc: str):
    """Sung notes as they sit in the TEXT: [(line_no, bar_no, start, end, dur, simple)].
    simple = one untied token with a whole-number length (safe to rewrite)."""
    res, cur, tie = [], None, False
    lines = abc.splitlines()
    for ln, line in enumerate(lines):
        s = line.strip()
        if s.startswith("V:"):
            cur = s[2:].strip().split()[0]
            continue
        if cur != "Vocal" or not s or s.startswith("%") or (s[0].isalpha() and not s.startswith('"')):
            continue
        masked = re.sub(r'"[^"]*"', lambda m: " " * len(m.group()), line)
        bar = 0
        pos = 0
        for m in re.finditer(r"\||" + NOTE.pattern, masked):
            if m.group() == "|":
                bar += 1
                continue
            acc, p, octv, num, den, tie_mark = m.groups()
            if p == "z":
                tie = False
                continue
            d = int(num or 1) if not den else int(num or 1) / int(den[1:] or 2)
            if tie and res:
                last = res[-1]
                res[-1] = (last[0], last[1], last[2], last[3], last[4] + d, False)
            else:
                res.append((ln, bar, m.start(), m.end(), d, not den and not tie_mark))
            tie = bool(tie_mark)
    return lines, res


def repair(lyrics: str, abc: str) -> tuple:
    """Move the long note onto the stressed syllable: inside one word, swap the
    lengths of the stressed note and the longest one. Pitches, bar lengths and
    everything outside the word stay. Only words whose notes are simple and in
    one bar are touched. Returns (new_abc, words_fixed)."""
    lines, toks = _note_tokens(abc)
    edits, k, fixed = [], 0, 0
    for words in line_words(lyrics):
        for w in words:
            m = syllables(w)
            seg = toks[k:k + m]
            k += m
            si = stress_index(w)
            if m < 2 or si is None or len(seg) < m:
                continue
            if not all(t[5] for t in seg) or len({(t[0], t[1]) for t in seg}) > 1:
                continue
            got = max(range(m), key=lambda i: seg[i][4])
            if seg[si][4] >= seg[got][4]:
                continue
            a, b = seg[si], seg[got]
            edits += [(a, int(b[4])), (b, int(a[4]))]
            fixed += 1
    for (ln, _bar, st, en, _d, _s), dur in sorted(edits, key=lambda e: (-e[0][0], -e[0][2])):
        tok = lines[ln][st:en]
        mm = NOTE.fullmatch(tok)
        new = (mm.group(1) or "") + mm.group(2) + mm.group(3) + (str(dur) if dur != 1 else "")
        lines[ln] = lines[ln][:st] + new + lines[ln][en:]
    return "\n".join(lines), fixed


if __name__ == "__main__":
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    bad, total, rows = check(d["lyrics"], d["abc"])
    for w, si, seg, ok in rows:
        if not ok or "гнил" in w.lower():
            print(("OK " if ok else "BAD"), w, "stress syl", si + 1, "notes(dur,pos):", seg)
    print(f"\nwrong-stress words: {bad}/{total}   vocal notes: {len(notes(d['abc']))}   "
          f"syllables: {sum(syllables(w) for l in line_words(d['lyrics']) for w in l)}")
