"""🎚 Remix: one song in, plus either new words or a second song.

  words:  the song's own melody, voice and backing, but it sings YOUR text
          (the way "Rammstein sings <anything>" memes are made).
  voice:  the voice of song 1 sings the melody and words of song 2 over song 2's backing.

Nothing is generated from scratch, so it sounds like the originals by construction.
Pipeline: Demucs stems -> (words: Whisper times the original words, TTS speaks our lines into those
spans, pyworld lays the original singer's pitch on them) -> SoulX-Singer-SVC puts the singer's
timbre on it (Seed-VC if SoulX is not installed) -> mixed over the original backing.
"""
import json
import logging
import os
import re
import subprocess
import tempfile
import time

import numpy as np

import cover
import music
from config import OUTPUT_DIR, scratch_path, venv_python

logger = logging.getLogger("assistant.remix")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SX = os.path.join(ROOT, "models_ext", "soulx_singer")
# On C:, not Z: -- from the QLC Z: drive torch's CUDA libraries (cuDNN...) load lazily inside
# the first segment: 3+ min cold, and the bot's 900 s timeout hit while another job wrote to Z:
# (10-07); warm, the whole 3-min song converts in 49 s.
SX_PY = venv_python(os.path.join(ROOT, "venv_soulx"))
SX_MODEL = os.path.join(SX, "pretrained_models", "SoulX-Singer", "model-svc.pt")
SEEDVC_PY = venv_python(os.path.join(ROOT, "venv_qwen"))
SR = 44100
_VOW = re.compile(r"[аеёиоуыэюяaeiouy]", re.I)


def available() -> bool:
    return os.path.isfile(SX_MODEL) and os.path.isfile(SX_PY) or os.path.isfile(SEEDVC_PY)


def _syl(s: str) -> int:
    return max(1, len(_VOW.findall(s)))


def _stems(song: str, work: str, name: str) -> tuple:
    """song -> (vocals, backing) as float arrays at SR; also kept as wavs in `work`."""
    import soundfile as sf
    import mashup_stems
    ref = cover.to_wav(song, os.path.join(work, name + "_ref.wav"))
    st = mashup_stems.separate(ref)
    vox, back = st["vocals"], mashup_stems.backing_of(st)
    sf.write(os.path.join(work, name + "_vox.wav"), vox, SR)
    return vox, back


def _singer_ref(vox: np.ndarray, work: str, name: str, sec: int = 15) -> str:
    """The loudest `sec` seconds of a vocal stem: the voice the SVC copies."""
    import soundfile as sf
    hop = SR * sec
    best = max(range(0, max(1, len(vox) - hop), SR * 5), key=lambda a: float(np.abs(vox[a:a + hop]).mean()))
    p = os.path.join(work, name + "_singer.wav")
    sf.write(p, vox[best:best + hop], SR)
    return p


def _f0_npy(wav: str, npy: str) -> None:
    import librosa
    import pyworld as pw
    import soundfile as sf
    y, _ = librosa.load(wav, sr=24000, mono=True)
    sf.write(wav, y, 24000)
    f, _ = pw.harvest(y.astype(np.float64), 24000, f0_floor=70, f0_ceil=800, frame_period=20.0)
    np.save(npy, f.astype(np.float32)[: len(y) // 480 + 1])


def _soulx(ctx, source: str, prompt: str, out: str) -> bool:
    if not (os.path.isfile(SX_MODEL) and os.path.isfile(SX_PY)):
        return False
    import shutil
    work = tempfile.mkdtemp(prefix="remix_svc_")
    shutil.copy(source, os.path.join(work, "src.wav"))
    shutil.copy(prompt, os.path.join(work, "prm.wav"))
    for k in ("src", "prm"):
        _f0_npy(os.path.join(work, k + ".wav"), os.path.join(work, k + "_f0.npy"))
    cmd = [SX_PY, os.path.join(SX, "cli", "inference_svc.py"), "--device", "cuda", "--model_path", SX_MODEL,
           "--config", os.path.join(SX, "soulxsinger", "config", "soulxsinger.yaml"),
           "--prompt_wav_path", os.path.join(work, "prm.wav"), "--target_wav_path", os.path.join(work, "src.wav"),
           "--prompt_f0_path", os.path.join(work, "prm_f0.npy"), "--target_f0_path", os.path.join(work, "src_f0.npy"),
           "--save_dir", os.path.join(work, "o"), "--auto_shift", "--pitch_shift", "0", "--n_steps", "32", "--fp16"]
    music.run_gpu_worker(ctx, SX_PY, "", {}, "SoulX-SVC", 900, cmd=cmd, env={**os.environ, "PYTHONPATH": SX})
    gen = os.path.join(work, "o", "generated.wav")
    if not os.path.isfile(gen):
        return False
    shutil.copy(gen, out)
    return True


def _seedvc(ctx, source: str, prompt: str, out: str) -> bool:
    if not os.path.isfile(SEEDVC_PY):
        return False
    jf = os.path.join(tempfile.mkdtemp(prefix="remix_svc_"), "jobs.json")
    with open(jf, "w", encoding="utf-8") as fh:
        json.dump([{"source": source, "target": prompt, "out": out}], fh)
    music.run_gpu_worker(ctx, SEEDVC_PY, "seedvc_batch.py", {}, "Seed-VC", 900,
                         cmd=[SEEDVC_PY, os.path.join(ROOT, "scripts", "seedvc_batch.py"), jf])
    return os.path.isfile(out)


def sing_as(ctx, source: str, prompt: str, out: str, order=("soulx", "seedvc")) -> str:
    """`source` (singing) in the timbre of `prompt`, by the first converter in `order` that
    gives a file. Raises cover.CoverFailed when none does."""
    for name in order:
        if {"soulx": _soulx, "seedvc": _seedvc}[name](ctx, source, prompt, out):
            return out
        logger.warning("remix: %s gave no file", name)
    raise cover.CoverFailed("render")


def _mix(voice_wav: str, back: np.ndarray, out_mp3: str) -> str:
    import librosa
    import soundfile as sf
    v, sr = sf.read(voice_wav, dtype="float32")
    v = v.mean(1) if v.ndim > 1 else v
    if sr != SR:
        v = librosa.resample(v, orig_sr=sr, target_sr=SR)
    n = min(len(v), len(back))
    mix = back[:n] + v[:n]
    mix /= max(1.0, float(np.abs(mix).max()) / 0.95)
    wav = out_mp3[:-4] + ".wav"
    sf.write(wav, mix, SR)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-b:a", "192k", out_mp3], check=True,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    os.unlink(wav)
    return out_mp3


# Whisper's stock lines on a bare backing track: the credits of the subtitle sets it learned
# from. 10-07 «3 сентября» ended in four «Субтитры создавал DimaTorzok», each given a lyric line.
_HALLUCINATED = re.compile(r"субтитр|dimatorzok|редактор|корректор|продолжение следует|"
                           r"спасибо за просмотр|подписывайтесь|thank you|subtitles|amara", re.I)


def _hear(whisper, path: str, language=None, vad: bool = True) -> list:
    """Whisper segments of a bare sung vocal. VAD on and no conditioning on the previous text:
    with the defaults one credit-line hallucination repeats through the whole stem (10-07: a
    clear take came back as 13 x «Продолжение следует...», 0 words, and the take picker
    scored it 0%); with these the same take read almost word for word. Greedy at temperature 0:
    with the fallback ladder the same converted vocal scored 0.03, 0.41, 0.52 on three runs
    (10-08), so takes and voices were picked on noise; at 0 it is 0.52 every time."""
    segs, _ = whisper.transcribe(path, word_timestamps=True, language=language, vad_filter=vad,
                                 condition_on_previous_text=False, temperature=0.0)
    return list(segs)


def _sung_words(segs) -> list:
    """Word timings of a sung vocal from Whisper segments: credit-line hallucinations dropped,
    and a number spelled out the way it is sung («3 сентября» -> «третье», 2 syllables, not 0)."""
    from num2words import num2words
    out = []
    for sg in segs:
        if _HALLUCINATED.search(sg.text or ""):
            continue
        for w in sg.words or []:
            t = w.word.strip()
            if not t:
                continue
            if re.fullmatch(r"\d+[.,]?", t):
                try:
                    t = num2words(int(re.sub(r"\D", "", t)), lang="ru", to="ordinal", gender="n")
                except Exception:
                    pass
            out.append({"w": t, "s": w.start, "e": w.end})
    return out


def _phrases(words: list, gap: float = 0.45, longest: float = 9.0) -> list:
    """The original vocal's sung phrases: words split at breaths (or every ~9 s)."""
    out, cur = [], []
    for w in words:
        if cur and (w["s"] - cur[-1]["e"] > gap or w["e"] - cur[0]["s"] > longest):
            out.append(cur)
            cur = []
        cur.append(w)
    return out + ([cur] if cur else [])


def _syl_marks(words: list) -> list:
    """Onset of every syllable (vowel) in `words`, split evenly inside each word, plus the end."""
    marks = []
    for w in words:
        n = _syl(w["w"]) if _VOW.search(w["w"]) else 0
        marks += [w["s"] + k * (w["e"] - w["s"]) / max(n, 1) for k in range(n)]
    return marks + [words[-1]["e"]] if words else []


def _syl_ends(words: list) -> list:
    """Where each syllable of `_syl_marks` stops sounding: the next syllable, or its word's
    end when a pause follows -- a held vowel must not run through the band's bar."""
    ends = []
    for w in words:
        n = _syl(w["w"]) if _VOW.search(w["w"]) else 0
        ends += [w["s"] + (k + 1) * (w["e"] - w["s"]) / max(n, 1) for k in range(n)]
    return ends


def _fit_line(ctx, line: str, n: int) -> str:
    """`line` reworded to `n` syllables, give or take one (a parody fits the meter, not the
    other way). Gemma cannot count syllables: told only the target it handed the line back
    unchanged every time (10-07), so it is told the count it has and how many to add or cut."""
    if abs(_syl(line) - n) <= 1:
        return line
    import llm
    best = line
    for _ in range(3):
        have = _syl(best)
        verb = f"add {n - have}" if n > have else f"remove {have - n}"
        try:
            got = llm.call_llm_simple(
                ctx, "You fit song lyrics to a melody. Syllables are counted as vowels "
                "(а е ё и о у ы э ю я). Rewrite the line to the asked syllable count by adding or "
                "dropping small words, repeating a word or cutting a phrase -- same language, same "
                "meaning, keep its key words and its last word. Reply with the line only.",
                f"Line ({have} syllables): {best}\nNeeded: {n} syllables, so {verb}.",
                temperature=0.5, max_tokens=200).strip().strip('"«»')
        except Exception:
            return best
        got = got.splitlines()[0].strip() if got else ""
        if got and abs(_syl(got) - n) < abs(_syl(best) - n):
            best = got
        if abs(_syl(best) - n) <= 1:
            break
    return best


def _sung(phrases: list) -> list:
    """Phrases with a vowel, minus Whisper's stock hallucinations on a bare backing track
    ("Thank you.", "Subtitles by...") -- Latin words in a song that is otherwise Cyrillic."""
    def cyr(ph):
        return any(re.search(r"[а-яё]", w["w"], re.I) for w in ph)
    out = [ph for ph in phrases if len(_syl_marks(ph)) > 1]
    if sum(map(cyr, out)) * 2 > len(out):
        out = [ph for ph in out if cyr(ph)]
    return out


def _plan(phrases: list, lines: list) -> list:
    """(words, line) pairs in song order: each line takes the run of sung phrases whose
    syllables come closest to its own, so rewording only trims a syllable or two. One line
    per Whisper phrase gave 3-syllable phrases 19-syllable lines (10-07). The lyric goes
    round again when the song is longer."""
    sung, out, i, k = _sung(phrases), [], 0, 0
    while i < len(sung):
        line = lines[k % len(lines)]
        n, words, have = _syl(line), [], 0
        while i < len(sung):
            more = len(_syl_marks(sung[i])) - 1
            if words and abs(have + more - n) > abs(have - n):
                break
            words, have, i = words + sung[i], have + more, i + 1
        out.append((words, line))
        k += 1
    return out


def _stretch(y: np.ndarray, n: int, path: str) -> np.ndarray:
    """`y` time-stretched to exactly `n` samples, pitch kept (rubberband R3)."""
    import soundfile as sf
    import mashup_auto
    if n <= 0 or len(y) < 64:
        return np.zeros(max(n, 0), np.float32)
    sf.write(path + ".in.wav", y, SR)
    subprocess.run([mashup_auto.RUBBERBAND, "-q", "--fine", "-D", f"{n / SR:.6f}", path + ".in.wav", path + ".wav"],
                   check=True, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    out, _ = sf.read(path + ".wav", dtype="float32")
    out = out.mean(1) if out.ndim > 1 else out
    return np.pad(out, (0, max(0, n - len(out))))[:n]


def _onto_marks(y: np.ndarray, src: list, dst: list, length: int, work: str, tag: str,
                dst_ends: list = None) -> np.ndarray:
    """`y` warped so its syllable onsets `src` (s) land on `dst` (s, from 0); `length` samples.
    Syllable by syllable: one time map over the whole line smears each onset ~70-90 ms early
    where a short spoken gap is stretched to a long sung note (10-07 test)."""
    k = min(len(src), len(dst))
    if k < 2:
        return y[:length]
    xs = np.linspace(0, 1, k)
    s_ = (np.interp(xs, np.linspace(0, 1, len(src)), src) - src[0]) * SR
    d_ = np.interp(xs, np.linspace(0, 1, len(dst)), dst) * SR
    y = y[int(src[0] * SR):]           # the first syllable opens the phrase
    out = np.zeros(length, np.float32)
    fade = int(0.006 * SR)
    for j in range(k - 1):
        a0, a1, b0, b1 = int(s_[j]), int(s_[j + 1]), int(d_[j]), int(d_[j + 1])
        if dst_ends is not None and len(dst_ends) == k - 1:      # stop where the singer stopped
            b1 = max(b0 + fade * 3, min(b1, int(dst_ends[j] * SR)))
        seg = _stretch(y[a0:a1], min(b1, length) - b0, os.path.join(work, f"{tag}_{j}"))
        if len(seg) > 2 * fade:
            seg[:fade] *= np.linspace(0, 1, fade)
            seg[-fade:] *= np.linspace(1, 0, fade)
        out[b0:b0 + len(seg)] += seg
    return out


def _speak_on_melody(ctx, vox: np.ndarray, lines: list, work: str, plan: list = None) -> str:
    """Our lines sung syllable for syllable where the original's were: each line takes a run
    of the song's phrases (_plan), is reworded to their syllable count, spoken, and its
    syllables are stretched onto the original syllables; then the original pitch goes on
    top. `plan`: (words, line) pairs made earlier (lines fitted while the LLM was loaded)."""
    import librosa
    import soundfile as sf
    import audio
    vp = os.path.join(work, "orig_vox.wav")
    sf.write(vp, vox, SR)
    wh = ctx.models.whisper
    if plan is None:
        segs = _hear(wh, vp)
        words = _sung_words(segs)
        if len(words) < 3:
            raise cover.CoverFailed("no_words")
        plan = _plan(_phrases(words), lines)
    canvas = np.zeros(len(vox), dtype=np.float32)
    for i, (ph, line) in enumerate(plan):
        dst = _syl_marks(ph)
        line = _fit_line(ctx, line, len(dst) - 1)
        w = audio.synth_single_segment(ctx, 5000 + i, "DC", line, out_stem=os.path.join(work, f"l{i}"))
        if not w:
            continue
        y, _ = librosa.load(w, sr=SR)
        y, _ = librosa.effects.trim(y, top_db=35)
        tp = os.path.join(work, f"l{i}_t.wav")
        sf.write(tp, y, SR)
        ts = _hear(wh, tp)
        tw = [{"w": x.word.strip(), "s": x.start, "e": x.end} for sg in ts for x in (sg.words or []) if x.word.strip()]
        src = _syl_marks(tw)[:-1] + [len(y) / SR] if tw else [0.0, len(y) / SR]
        t0 = dst[0]
        seg = _onto_marks(y, src, [d - t0 for d in dst], int((dst[-1] - t0) * SR) + SR // 10, work, f"l{i}",
                          dst_ends=[e - t0 for e in _syl_ends(ph)])
        s0 = int(t0 * SR)
        canvas[s0:s0 + len(seg)] += seg[:max(0, len(canvas) - s0)]
    sf.write(os.path.join(work, "canvas.wav"), canvas, SR)    # the words alone, for an F0-driven voice
    return _lay_pitch(canvas, vox, work)


def _lay_pitch(canvas: np.ndarray, vox: np.ndarray, work: str) -> str:
    """`canvas` (speech or another singing on the song's timeline) takes the original vocal's pitch contour."""
    import librosa
    import pyworld as pw
    import soundfile as sf
    R = 22050                                       # the speech takes the original pitch contour on the same timeline
    a = librosa.resample(canvas.astype(np.float64), orig_sr=SR, target_sr=R)
    o = librosa.resample(vox.astype(np.float64), orig_sr=SR, target_sr=R)
    fa, ta = pw.harvest(a, R, f0_floor=70, f0_ceil=500, frame_period=10.0)
    fo, _ = pw.harvest(o, R, f0_floor=70, f0_ceil=700, frame_period=10.0)
    n = min(len(fa), len(fo))
    fa, fo = fa[:n], fo[:n]
    ok = np.flatnonzero(fo > 0)
    if not len(ok):
        raise cover.CoverFailed("no_words")
    f_new = np.where(fa > 0, np.interp(np.arange(n), ok, fo[ok]), 0.0)
    sp_ = pw.cheaptrick(a, fa, ta[:n], R)[:n]
    ap = pw.d4c(a, fa, ta[:n], R)[:n]
    y = librosa.resample(pw.synthesize(f_new, sp_, ap, R, 10.0).astype(np.float32), orig_sr=R, target_sr=SR)
    p = os.path.join(work, "speech_on_melody.wav")
    sf.write(p, np.pad(y, (0, max(0, len(vox) - len(y))))[:len(vox)], SR)
    return p


def _out() -> str:
    return str(OUTPUT_DIR / f"remix_{int(time.time() * 1000)}.mp3")


# ── Re-sing: the way an AI cover gets new words right ──────────────────────────
# 10-07: a day of laying new words into the original's syllables ended in «ты пытаешься в
# слово из 4 букв затолкать целое слово». The user's reference (Udio «Modern Talking —
# Говновоз») RE-SINGS the song: its score, new notes for the new words, its own backing.
# YuE2 does that from the original's full score (melody + chords) -> «МОЛОДЕЦ!!!».
SHEETSAGE = os.path.join(music._CPP_DIR, "gguf", "SheetSage2-Q8_0.gguf")
YUE_TRANSCRIBE = os.path.join(os.path.dirname(music.YUE2_CPP_EXE), "yue-transcribe.exe")
_SECTION = {"intro": "Intro", "verse": "Verse", "pre-chorus": "Pre-Chorus", "chorus": "Chorus",
            "bridge": "Bridge", "interlude": "Interlude", "outro": "Outro", "inst": "Interlude"}


def resing_available() -> bool:
    return music.yue2_cpp_available() and os.path.isfile(YUE_TRANSCRIBE) and os.path.isfile(SHEETSAGE)


def _score(ctx, wav: str, work: str) -> str:
    """The recording's full score (melody + chord symbols + section marks) as ABC."""
    abc = os.path.join(work, "score.abc")
    music.run_gpu_worker(ctx, "", "", {}, "SheetSage", 600,
                         cmd=[YUE_TRANSCRIBE, "--model", SHEETSAGE, "--audio", wav, "--out", abc])
    if not os.path.isfile(abc):
        raise cover.CoverFailed("render")
    with open(abc, encoding="utf-8") as fh:
        return fh.read()


def _score_parts(abc: str) -> list:
    """The score's sections in order ("% verse" marks) as (YuE2 lyric tag, sung notes).
    A section the melody leaves silent is instrumental whatever its name."""
    out, voice = [], ""
    for line in (abc or "").splitlines():
        m = re.match(r"^%\s*([a-z-]+)", line)
        if m:
            tag = _SECTION.get(m.group(1).lower())
            out.append([tag, 0] if tag else None)
            continue
        if line.startswith("V:"):
            voice = line[2:].strip()
        elif voice.startswith("Voc") and out and out[-1] and not re.match(r"^[A-Z]:", line):
            out[-1][1] += len(re.findall(r"[A-Ga-g]", re.sub(r'"[^"]*"', "", line)))
    parts = [(t, n) for t, n in (x for x in out if x)]
    return parts or [("Verse", 60), ("Chorus", 60), ("Verse", 60), ("Chorus", 60)]


def _score_sections(abc: str) -> list:
    return [t for t, _ in _score_parts(abc)]


def _score_tempo_key(abc: str) -> str:
    q = re.search(r"^Q:\s*1/4=(\d+)", abc or "", re.M)
    k = re.search(r"^K:\s*([A-G][#b]?m?)", abc or "", re.M)
    parts = []
    if q:
        parts.append(f"{q.group(1)} bpm")
    if k:
        key = k.group(1)
        name = key[0] + {"#": " sharp", "b": " flat"}.get(key[1:2], "")
        parts.append(name + (" minor" if key.endswith("m") else " major"))
    return ", ".join(parts)


_INSTRUMENTAL = ("Intro", "Interlude", "Outro")
_MIN_SUNG = 8          # a section with fewer melody notes is a fill, not a sung part


_WORD = re.compile(r"[а-яёa-z]+(?:-[а-яёa-z]+)*", re.I)


def _words(s: str) -> list:
    return [w.lower().replace("ё", "е") for w in _WORD.findall(s or "")]


def _repetition_profile(phrases: list) -> list:
    """How the ORIGINAL repeats itself, as short notes for the model -- the shape, not its words.
    A singable adaptation keeps the original's map of repeats (Kim & Goto, ISMIR 2023: the
    self-similarity of sections stays the same across languages when a song is sung well);
    10-08 the user asked for repeats placed with meaning, as in «3 сентября» / «Я русский»."""
    lines = [" ".join(_words(p)) for p in phrases if _words(p)]
    notes = []
    counts = {}
    for l in lines:
        if len(l) >= 8:
            counts[l] = counts.get(l, 0) + 1
    refrains = sorted((n for n in counts.values() if n >= 2), reverse=True)
    if not refrains:
        # Whisper cuts the same sung line at different breaths, so whole phrases rarely match
        # (SHAMAN «Я русский»: no refrain found, 10-08); a 3-word run sung 3+ times is one.
        grams = {}
        for l in lines:
            ws = l.split()
            for g in {" ".join(ws[i:i + 3]) for i in range(len(ws) - 2)}:
                grams[g] = grams.get(g, 0) + 1
        refrains = sorted((n for n in grams.values() if n >= 3), reverse=True)[:5]
    if refrains:
        notes.append(f"{len(refrains)} of its lines come back as a refrain (the most repeated is sung "
                     f"{refrains[0]} times) -- choruses repeat the same refrain word for word")
    run = 0
    for l in lines:
        ws = l.split()
        best = max((sum(1 for _ in g) for w, g in __import__("itertools").groupby(ws) if len(w) >= 4), default=1)
        run = max(run, best)
    if run >= 2:
        notes.append(f"it sings one key word {run} times in a row (like «word, word, word») -- "
                     f"do the same with YOUR key word, at the same kind of place (the hook)")
    starts = [l.split()[0] for l in lines if len(l.split()) >= 3]
    ana = sum(1 for a, b in zip(starts, starts[1:]) if a == b)      # «я ..., я ...» counts
    if ana:
        notes.append(f"{ana} pairs of neighbouring lines start with the same word (anaphora)")
    voc = [l for l in lines if all(len(w) <= 3 for w in l.split()) and len(set(l.split())) == 1 and len(l.split()) >= 2]
    if voc:
        notes.append("it fills short gaps with a sung interjection (like «ой-ой-ой»); "
                     "keep such interjections where your lyric already has them")
    return notes


_ARRANGE = (
    "You fit a song lyric onto a melody's sections the way a good songwriter adapts words to an "
    "existing tune. Each section can carry about the given number of syllables (one per note). "
    "Rules:\n"
    "1. Use ONLY words that are already in that section's lines (you may also use the lyric's "
    "hook/title words in any section). Never invent new words, never change the meaning.\n"
    "2. Repeat with meaning: when a section has notes to spare, fill them by repeating the KEY "
    "word or the hook phrase (the title, the name the song is about), never a filler word like "
    "'and', 'the', 'и', 'а', 'же'. A key word may be sung 2-3 times in a row («word, word, word»).\n"
    "3. Put the hook at the start of a chorus, and again at its end when there is room.\n"
    "4. Every chorus is the same refrain, word for word.\n"
    "5. When a section has too few notes, drop the least important line or words, never the hook.\n"
    "6. Keep each line a natural sung phrase; the stressed syllables of the words stay natural.\n"
    "Reply with JSON only: {\"sections\": [[\"line\", ...], ...]} -- one list per given section, "
    "in that order.")


def _arrange_score(cand: list, sung: list, plain: list, hook: set, spare: list = ()) -> float:
    """Deterministic check of one arrangement (rule-based feedback beats a model judge for form,
    ICCC 2025): syllables on budget, no foreign words, choruses alike, hook repeated. Lower is
    better; None when the arrangement broke a hard rule."""
    if not isinstance(cand, list) or len(cand) != len(sung):
        return None
    extra = set(_words(" ".join(spare)))
    allowed_all = set(_words(" ".join(" ".join(p) for p in plain))) | hook | extra
    cost, choruses = 0.0, []
    for (tag, notes), lines, base in zip(sung, cand, plain):
        if not (isinstance(lines, list) and lines and all(isinstance(l, str) and l.strip() for l in lines)):
            return None
        allowed = set(_words(" ".join(base))) | hook | (extra if tag != "Chorus" else set())
        if not set(_words(" ".join(lines))) <= (allowed if base else allowed_all):
            return None
        syl = sum(_syl(l) for l in lines)
        if syl > notes * 1.25:
            return None
        cost += (2 if syl > notes else 1) * abs(notes - syl) / max(1, notes)   # crowding hurts more
        # Repeats must not eat the meaning: «я работал говновозом» -> «говновоз, говновоз» lost
        # who did what (10-08 draft); every content word the plain lines had and this one lost costs.
        kept = {w for w in _words(" ".join(base)) if len(w) >= 4}
        cost += 0.5 * len(kept - set(_words(" ".join(lines)))) / max(1, len(kept))
        if tag == "Chorus":
            choruses.append([l.strip().lower() for l in lines])
    if len({tuple(c) for c in choruses}) > 1:
        cost += 0.5
    if hook and choruses and not any(set(_words(c[0])) & hook for c in choruses):
        cost += 0.3
    return cost


ARRANGE_TRIES = int(os.getenv("ARRANGE_TRIES", "3"))


_SENT = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯЁA-Z])")


def _need(notes: int, lines: list) -> str:
    have = sum(_syl(l) for l in lines)
    if have < notes * 0.85:
        return f"{notes - have} notes to spare: FILL them (repeat the hook, or add an unused line in a verse)"
    if have > notes * 1.1:
        return f"{have - notes} syllables too many: cut the least important words"
    return "fits; repeat the hook only where the original repeats"


def _arrange(ctx, sung: list, plain: list, profile: list, hook: set, spare: list = ()) -> list:
    """The plain line placement (`plain`, one list of lines per sung section) re-arranged with
    meaningful repeats. Several drafts, the best by `_arrange_score`; none that keeps the rules
    -> the plain placement."""
    import llm
    from utils import safe_json_from_llm
    user = ("Sections (syllables the melody holds):\n"
            + "\n".join(f"{i}. {t}: ~{n} notes, lines now {sum(_syl(l) for l in p)} syllables -- {_need(n, p)}:\n   "
                        + "\n   ".join(p) for i, ((t, n), p) in enumerate(zip(sung, plain), 1))
            + ("\nUnused lines of the lyric (a verse with room may take them, in order):\n   "
               + "\n   ".join(spare) if spare else "")
            + "\nThe hook / title words: " + (", ".join(sorted(hook)) or "pick the chorus's key word")
            + ("\nHow the original song repeats itself (copy this shape with these words):\n- "
               + "\n- ".join(profile) if profile else ""))
    best, best_cost = plain, _arrange_score(plain, sung, plain, hook, spare)
    best_cost = 9.0 if best_cost is None else best_cost
    for k in range(max(1, ARRANGE_TRIES)):
        try:
            raw = llm.call_llm_simple(ctx, _ARRANGE, user, temperature=0.4 + 0.2 * k, max_tokens=1200)
            got = safe_json_from_llm(raw or "", ["sections"]) or {}
        except Exception as exc:
            logger.warning("resing: arrange failed: %s", exc)
            continue
        cand = got.get("sections") if isinstance(got, dict) else None
        cost = _arrange_score(cand, sung, plain, hook, spare)
        logger.info("resing: arrangement %d cost %s", k + 1, None if cost is None else round(cost, 2))
        if cost is not None and cost < best_cost:
            # one sung phrase per line: the model glued two lines into one string with «. »
            best = [[p.strip() for l in sec for p in _SENT.split(l.strip()) if p.strip()] for sec in cand]
            best_cost = cost
    return best


def _hook_words(lines: list) -> set:
    """The lyric's hook: its one or two most sung content words (the song's subject -- «говновоз»,
    «говночист»), counted over every line. The most repeated whole LINE was tried first and
    named «должен быть закалённый плечист» the hook (10-08)."""
    wc = {}
    for l in lines:
        for w in _words(l):
            if len(w) >= 4:
                wc[w] = wc.get(w, 0) + 1
    top = sorted(wc.items(), key=lambda kv: -kv[1])[:2]
    return {w for w, n in top if n >= 2 and n >= top[0][1] * 0.5}


def _layout(ctx, lyrics: str, parts: list, profile: list = None) -> str:
    """The new lyric laid out on the original's sections, each sung section holding about as
    many syllables as its melody has notes (10-07: Cheri Cheri Lady got 144 syllables on a
    44-note chorus and YuE2 mumbled them). The model places lines; the budget is enforced
    here by dropping a section's last lines; a broken plan falls back to an even split.
    With the original's `profile` (how it repeats itself) the placed lines are then arranged
    with meaningful repeats of the hook (`_arrange`)."""
    lines = [l.strip() for l in (lyrics or "").splitlines() if l.strip() and not re.fullmatch(r"\[.*\]", l.strip())]
    sung = [(t, n) for t, n in parts if t not in _INSTRUMENTAL and n >= _MIN_SUNG]
    import llm
    from utils import safe_json_from_llm
    got = {}
    try:
        raw = llm.call_llm_simple(
            ctx, "You lay a song's lyric out on a melody's sections. Each section can carry about the "
            "given number of syllables (one per note) -- never more. Return JSON {\"sections\": "
            "[[line numbers], ...]} with one list per given section in that order. A chorus repeats the "
            "same refrain lines each time; verses take the other lines in order; lines that do not fit "
            "anywhere are left out. Reply with JSON only.",
            "Sections:\n" + "\n".join(f"{i}. {t}: ~{n} syllables" for i, (t, n) in enumerate(sung, 1))
            + "\nLines (syllables):\n" + "\n".join(f"{i}. {l} ({_syl(l)})" for i, l in enumerate(lines, 1)),
            temperature=0.2, max_tokens=800)
        got = safe_json_from_llm(raw or "", ["sections"]) or {}
    except Exception as exc:
        logger.warning("resing: layout failed: %s", exc)
    plan = got.get("sections") if isinstance(got, dict) else None
    if not (isinstance(plan, list) and len(plan) == len(sung)
            and all(isinstance(x, list) and x for x in plan)
            and all(isinstance(n, int) and 1 <= n <= len(lines) for x in plan for n in x)):
        per = max(1, len(lines) // max(1, len(sung)))
        plan = [list(range(1 + i * per, 1 + min(len(lines), (i + 1) * per))) or [len(lines)]
                for i in range(len(sung))]
    # Only a chorus repeats. A verse that re-sings lines another verse had (10-07: the model
    # gave verse 2 the lines of verse 1 and the tourists' verse was never sung) takes the
    # next lines nobody has sung yet instead.
    refrain = {n for (tag, _), nums in zip(sung, plan) if tag == "Chorus" for n in nums}
    fitted, used = [], set()
    for (tag, notes), nums in zip(sung, plan):
        nums = list(nums)
        if tag != "Chorus":
            kept = [n for n in nums if n not in used]
            fresh = [n for n in range(1, len(lines) + 1) if n not in used and n not in refrain and n not in kept]
            for _ in range(len(nums) - len(kept)):
                if fresh and sum(_syl(lines[n - 1]) for n in kept) + _syl(lines[fresh[0] - 1]) <= notes * 1.1:
                    kept.append(fresh.pop(0))
            nums = kept
            nums = sorted(nums) or [n for n in range(1, len(lines) + 1) if n not in refrain][:1] or [1]
        while len(nums) > 1 and sum(_syl(lines[n - 1]) for n in nums) > notes * 1.1:
            nums.pop()
        if tag != "Chorus":
            used.update(nums)
        fitted.append(nums)
    placed = [[lines[n - 1] for n in nums] for nums in fitted]
    if profile is not None:
        sung_now = {l for p in placed for l in p}
        placed = _arrange(ctx, sung, placed, profile, _hook_words(lines),
                          [l for n, l in enumerate(lines, 1) if l not in sung_now and n not in refrain])
    blocks, k = [], 0
    for tag, notes in parts:
        if tag in _INSTRUMENTAL or notes < _MIN_SUNG:
            blocks.append(f"[{'Interlude' if tag not in _INSTRUMENTAL else tag}]")
            continue
        blocks.append(f"[{tag}]\n" + "\n".join(placed[k]))
        k += 1
    return "\n\n".join(blocks)


def _song_title(path: str) -> str:
    """«Artist - Title» from the file's own tags (a Telegram audio, a downloaded mp3), or ''."""
    try:
        import mutagen
        f = mutagen.File(path, easy=True)
        tags = getattr(f, "tags", None) or {}
        artist = " ".join(tags.get("artist") or [])
        title = " ".join(tags.get("title") or [])
        return " - ".join(x for x in (artist, title) if x)[:120]
    except Exception:
        return ""


def _resing_style(ctx, heard: str, tempo_key: str, title: str = "") -> tuple:
    """(YuE2 tags for the original's sound, its lead singer). The model names the song from its
    first sung lines (it knows «3 сентября» is 90s Russian chanson by Shufutinsky); tempo and
    key come from the score. The singer keys the RVC voice kept for that artist."""
    from utils import safe_json_from_llm
    got = {}
    try:
        import llm
        raw = llm.call_llm_simple(
            ctx, "These are sung lines of a song. Return JSON {\"artist\": the lead singer's name if you "
            "recognize the song, else \"\", \"tags\": 8-12 comma-separated music tags: genre, era, lead "
            "vocal (gender, timbre), main instruments, mood}. JSON only.",
            # The title names the singer; the sung lines alone made SHAMAN's «Я русский»
            # «Alexander Marshal», and the cover went out in Marshal's voice (10-08).
            (f"File title: {title}\n" if title else "") + heard[:600], temperature=0.2, max_tokens=200)
        got = safe_json_from_llm(raw or "", ["tags"]) or {}
    except Exception as exc:
        logger.warning("resing: style failed: %s", exc)
    tags = got.get("tags") if isinstance(got, dict) else ""
    tags = ", ".join(tags) if isinstance(tags, list) else str(tags or "")
    tags = re.sub(r"[^\w ,'&/-]", "", tags)[:240] or "pop ballad, male lead vocal, synthesizer, drums, bass"
    artist = str(got.get("artist") or "").strip()[:60] if isinstance(got, dict) else ""
    return ", ".join(t for t in (tags, tempo_key) if t), artist


def _mix_vocal(conv: str, yvox, yback) -> str:
    import soundfile as sf
    import mashup_auto
    v, sr = sf.read(conv, dtype="float32")
    v = v.mean(1) if v.ndim > 1 else v
    if sr != SR:
        import librosa
        v = librosa.resample(v, orig_sr=sr, target_sr=SR)
    ratio = mashup_auto._rms(yvox, gate=0.01) / max(1e-6, mashup_auto._rms(yback))
    return mashup_auto.mix(v, yback, ratio, _out())


# A take of YuE2 is a draw: on «3 сентября» the same score, words and settings gave 27..36 of
# 42 lines heard right depending on the seed (10-07, outputs/yue_cover/sweep). The user asked to
# keep re-rolling until the words are clear, so up to RESING_TAKES takes are sung, each scored
# by Whisper against the laid-out lyric, and the first clear one (or the clearest) is kept.
RESING_TAKES = int(os.getenv("RESING_TAKES", "3"))
RESING_CLEAR = float(os.getenv("RESING_CLEAR", "0.9"))


def _letters(s: str) -> str:
    return re.sub(r"[^а-яa-z]+", "", (s or "").lower().replace("ё", "е"))


def _line_heard(heard: str, line: str) -> float:
    """How well `line` appears anywhere in `heard`: the best fuzzy window, 0..1."""
    import difflib
    n = len(line)
    best = 0.0
    for i in range(0, max(1, len(heard) - n + 1), 3):
        r = difflib.SequenceMatcher(None, heard[i:i + n + 4], line, autojunk=False).ratio()
        if r > best:
            best = r
            if best >= 0.95:
                break
    return best


def _clarity(heard: str, lyric: str) -> float:
    """Share of the lyric's distinct lines Whisper heard in the take (each >= 0.8 similar)."""
    h = _letters(heard)
    lines = {_letters(l) for l in (lyric or "").splitlines() if l.strip() and not l.strip().startswith("[")}
    lines = [l for l in lines if len(l) >= 8]
    if not lines:
        return 1.0
    return sum(1 for l in lines if _line_heard(h, l) >= 0.8) / len(lines)


def _likeness(heard: str, lyric: str) -> float:
    """How close Whisper's hearing is to the lyric: the mean best-window similarity of its lines.
    A voice is judged by this, not _clarity: a sung line through RVC reads «сарок лет как пад
    наркозам» -- 0.79, under _clarity's 0.8 cliff -- so the same conversion swung 0.28..0.59 on
    _clarity while its likeness stayed 0.62..0.77 (10-08 sweep, 6 models x 2 takes)."""
    h = _letters(heard)
    lines = {_letters(l) for l in (lyric or "").splitlines() if l.strip() and not l.strip().startswith("[")}
    lines = [l for l in lines if len(l) >= 8]
    if not lines:
        return 1.0
    return sum(_line_heard(h, l) for l in lines) / len(lines)


def resing(ctx, song: str, lyrics: str, info: dict = None) -> str:
    """`song` re-sung with `lyrics`: YuE2 renders the original's full score with the new words
    (new notes where the words need them, its own backing in the original's style), then the
    original singer's voice goes on the vocal -- the artist's own RVC model when one was
    trained, else a zero-shot timbre. `info` (if given) gets what resing_rvc needs to redo the
    voice once the artist's model is trained. Returns an mp3 path; raises cover.CoverFailed."""
    import rvc_voice
    if not [l for l in (lyrics or "").splitlines() if l.strip() and not re.fullmatch(r"\[.*\]", l.strip())]:
        raise cover.CoverFailed("no_words")
    work = tempfile.mkdtemp(prefix="resing_")
    logger.info("resing: work %s", work)
    vox, _ = _stems(song, work, "a")
    abc = _score(ctx, os.path.join(work, "a_ref.wav"), work)
    segs = _hear(ctx.models.whisper, os.path.join(work, "a_vox.wav"))
    heard = " ".join(w["w"] for w in _sung_words(segs))
    title = str((info or {}).get("title") or "") or _song_title(song)
    style, artist = _resing_style(ctx, heard, _score_tempo_key(abc), title)
    profile = _repetition_profile([" ".join(w["w"] for w in p) for p in _phrases(_sung_words(segs))])
    logger.info("resing: the original repeats itself: %s", profile)
    layout = _layout(ctx, lyrics, _score_parts(abc), profile=profile)
    job = {"lyrics": music.yue2_lyrics(layout), "style": music.yue2_language(style, lyrics), "abc": abc}
    logger.info("resing: artist %r, style %s", artist, job["style"])
    if ctx is not None and hasattr(ctx, "set_stage"):
        ctx.set_stage("Composing a song")
    seed, best = int(time.time()) % 100000, None
    for take in range(max(1, RESING_TAKES)):
        out = os.path.join(work, f"yue{take}.mp3")
        music._render_yue2_once(ctx, True, dict(job, out=out), seed + take)
        if not music._valid_audio_file(out):
            continue
        music._master(out)
        yvox, yback = _stems(out, work, f"y{take}")
        tsegs = _hear(ctx.models.whisper, os.path.join(work, f"y{take}_vox.wav"),
                      "ru" if re.search("[а-яё]", lyrics, re.I) else None)
        clear = _clarity(" ".join(w["w"] for w in _sung_words(tsegs)), layout)
        logger.info("resing: take %d heard %.0f%% of the lines", take + 1, clear * 100)
        if best is None or clear > best[0]:
            best = (clear, take, yvox, yback)
        if clear >= RESING_CLEAR:
            break
    if best is None:
        raise cover.CoverFailed("render")
    _, take, yvox, yback = best
    yvox_wav = os.path.join(work, f"y{take}_vox.wav")
    name = rvc_voice.slug(artist) if rvc_voice.available() else ""
    if info is not None:
        info.update({"work": work, "artist": artist, "rvc": name, "yvox": yvox, "yback": yback,
                     "yvox_wav": yvox_wav, "clarity": best[0], "layout": layout,
                     "trained": bool(name and rvc_voice.model_of(name))})
    lang = "ru" if re.search("[а-яё]", lyrics, re.I) else None
    if info is not None:
        info["lang"] = lang
    chosen = str((info or {}).get("voice") or "")
    if chosen == "none":
        # the user's setting: YuE2's own voice, no conversion -- the clearest words (GigaAM on
        # the mix, 10-09: Лесник 71% bare, 30% through the artist's RVC)
        conv = yvox_wav
    elif chosen and rvc_voice.model_of(chosen):
        conv = _voice_keeping_words(ctx, _rvc_voices(ctx, chosen, yvox_wav), yvox_wav, layout, work, lang, best=True)
    elif name and rvc_voice.model_of(name):
        conv = _voice_keeping_words(ctx, _rvc_voices(ctx, name, yvox_wav), yvox_wav, layout, work, lang, best=True)
    else:
        ref = _singer_ref(vox, work, "a")
        voices = [(v, lambda o, v=v: sing_as(ctx, yvox_wav, ref, o, order=(v,))) for v in ("seedvc", "soulx")]
        conv = _voice_keeping_words(ctx, voices, yvox_wav, layout, work, lang)
    return _mix_vocal(conv, yvox, yback)


# An RVC conversion is a draw too: one model, settings and take came out 0.52..0.78 like the
# lyric over four runs (10-08, runtime/govnovoz/rep_sweep.py), so the voice is sung a few times
# and the closest kept.
RVC_TRIES = int(os.getenv("RVC_TRIES", "3"))


def _rvc_voices(ctx, name: str, src: str) -> list:
    import rvc_voice
    return [(f"rvc{k}", lambda o: rvc_voice.convert(ctx, name, src, o)) for k in range(RVC_TRIES)]


def _voice_keeping_words(ctx, voices: list, yvox_wav: str, layout: str, work: str, lang, best=False) -> str:
    """The first singer's-voice conversion that keeps the words (`best`: the closest of them all,
    stopping early at one as clear as the take); else the take's own vocal.
    10-07 live: a take heard at 97% came out of SoulX as «Субтитры... ЧИИИИИИ...» and the user
    got that mush. A conversion heard less than 0.8 as close to the lyric as the take is dropped
    (by _likeness: on _clarity's cliff every RVC voice fell under it and the singer never sang)."""
    def heard(path):
        return _likeness(" ".join(w["w"] for w in _sung_words(_hear(ctx.models.whisper, path, lang))), layout)
    take = top = None
    for vname, convert in voices:
        out = os.path.join(work, vname + ".wav")
        try:
            convert(out)
        except Exception as exc:
            logger.warning("resing: %s failed: %s", vname, exc)
            continue
        if take is None:
            take = heard(yvox_wav)
        kept = heard(out)
        logger.info("resing: %s voice heard %.2f like the lyric (take %.2f)", vname, kept, take)
        if kept >= take * 0.8 and (not best or kept >= take * 0.95):
            return out
        if kept >= take * 0.8 and (top is None or kept > top[0]):
            top = (kept, out)
    if top:
        return top[1]
    logger.warning("resing: no voice kept the words, sending the take's own vocal")
    return yvox_wav


def resing_rvc(ctx, info: dict) -> str:
    """The same re-sung song in the artist's own RVC voice, trained now on the original's vocal
    stem (once per artist; later covers reuse it). Returns an mp3 path."""
    import rvc_voice
    work, name = info["work"], info["rvc"]
    if ctx is not None and hasattr(ctx, "set_stage"):
        ctx.set_stage("Learning the singer's voice")
    if not rvc_voice.model_of(name) and not rvc_voice.train(ctx, name, os.path.join(work, "a_vox.wav")):
        raise cover.CoverFailed("render")
    conv = _voice_keeping_words(ctx, _rvc_voices(ctx, name, info["yvox_wav"]), info["yvox_wav"], info["layout"],
                                work, info.get("lang"), best=True)
    return _mix_vocal(conv, info["yvox"], info["yback"])


def lyrics_of(ctx, song: str) -> str:
    """The words of `song`, heard off its vocal stem, as a lyric with [Verse]/[Chorus] tags.
    10-07: «говновоз делать по 2 ссылкам» -- the second song gives the words, the first the
    melody and the singer. Whisper's phrases are the lines; the model only groups them into
    sections and mends a mis-heard word -- a tidy-up that drifts from what was heard is
    dropped for the plain lines."""
    import difflib
    work = tempfile.mkdtemp(prefix="lyrics_")
    _stems(song, work, "w")
    # An original's stem under a dense mix: with VAD on Whisper kept ~50 of SHAMAN's words and
    # skipped the verses, off it heard 161 (10-08). Both are heard; the fuller one is kept
    # (credit-line hallucinations are filtered either way).
    heard = []
    for vad in (True, False):
        heard.append([sg for sg in _hear(ctx.models.whisper, os.path.join(work, "w_vox.wav"), vad=vad)
                      if not _HALLUCINATED.search(sg.text or "")])
    segs = max(heard, key=lambda ss: sum(len((sg.text or "").split()) for sg in ss))
    lines = [re.sub(r"\s+", " ", sg.text).strip() for sg in segs if len(_letters(sg.text)) >= 3]
    if sum(len(l.split()) for l in lines) < 12:
        raise cover.CoverFailed("no_words")
    raw = "\n".join(lines)
    text = ""
    try:
        import llm
        from utils import safe_json_from_llm
        got = safe_json_from_llm(llm.call_llm_simple(
            ctx, "These are a song's lines as speech recognition heard them. Return JSON {\"lyrics\": the "
            "song as lyric lines with section tags [Verse] / [Pre-Chorus] / [Chorus] / [Bridge] / [Outro] "
            "on their own lines}. Keep the words and their order; split run-on lines at the sung phrase; "
            "fix only an obviously mis-heard word; drop credits and noise. JSON only.",
            raw[:6000], temperature=0.1, max_tokens=3000) or "", ["lyrics"]) or {}
        text = str(got.get("lyrics") or "") if isinstance(got, dict) else ""
    except Exception as exc:
        logger.warning("lyrics_of: tidy failed: %s", exc)
    body = "\n".join(l for l in text.splitlines() if not l.strip().startswith("["))
    if not text or difflib.SequenceMatcher(None, _letters(raw), _letters(body), autojunk=False).ratio() < 0.8:
        text = raw
    logger.info("lyrics_of: %d lines from %s", len(lines), os.path.basename(song))
    return text


def _sections_of(lyric: str) -> list:
    """[(tag, [lines])] of a tagged lyric; untagged text is one [Verse]."""
    out = []
    for raw in (lyric or "").splitlines():
        l = raw.strip()
        m = re.fullmatch(r"\[([^\]]+)\]", l)
        if m:
            out.append((m.group(1).strip(), []))
        elif l:
            if not out:
                out.append(("Verse", []))
            out[-1][1].append(l)
    return [(t, ls) for t, ls in out if ls]


def _shape(sections: list) -> list:
    """Per line of a lyric: (section index, syllables, index of the earlier identical line or
    None) -- the map of repeats a new lyric has to keep."""
    seen, out = {}, []
    for si, (_, lines) in enumerate(sections):
        for l in lines:
            key = " ".join(_words(l))
            out.append((si, _syl(l), seen.get(key)))
            seen.setdefault(key, len(out) - 1)
    return out


def _shape_rows(orig: list) -> list:
    """The original as the model sees it: per line its syllables, repeats and hook place -- not
    its words, so the new lyric cannot lean on them."""
    import itertools
    shape = _shape(orig)
    hook = _hook_words([l for _, ls in orig for l in ls])
    rows, k = [], 0
    for tag, lines in orig:
        rows.append(f"[{tag}]")
        for l in lines:
            _, n, rep = shape[k]
            k += 1
            ws = _words(l)
            notes = [f"{n} syllables"]
            if rep is not None:
                notes.append(f"= line {rep + 1}")
            at = [i for i, w in enumerate(ws) if w in hook]
            if at:
                notes.append("HOOK at the " + ("start" if at[0] <= 1 else "end" if at[-1] >= len(ws) - 2 else "middle"))
            run = max((sum(1 for _ in g) for w, g in itertools.groupby(ws) if len(w) >= 2), default=1)
            if run >= 2:
                notes.append(f"word x{run}")
            rows.append(f"{k}. " + "; ".join(notes))
    return rows


_WRITE_OVER = (
    "You write NEW words for an existing song -- a cover on a new theme, the way Weird Al writes "
    "parodies: keep the original's shape exactly, say something new. You are given only the "
    "original's SHAPE, line by line: syllables, which lines repeat an earlier line, and where its "
    "hook sits. Rules:\n"
    "1. Same sections, same tags, same number of lines in each section.\n"
    "2. Each new line has the SAME number of syllables as the original line (one less or more at "
    "most), so it sits on the same notes; the stressed syllables fall naturally.\n"
    "3. Make ONE hook for the new theme: a short, punchy phrase (the song's new title). Every line "
    "marked HOOK carries it -- at the start or the end of the line, as marked; the rest of that line "
    "is new and may differ from one HOOK line to the next.\n"
    "4. A line marked (= line N) repeats line N: write the SAME new line there, word for word.\n"
    "5. A line marked 'word xN' sings one word N times in a row: do that with YOUR key word.\n"
    "6. Everything is your own text about the theme.\n"
    "7. Verses tell the theme's story with concrete details; the chorus sums it up around the hook. "
    "Make it witty and singable, in {lang}.\n"
    "Reply with JSON only: {{\"lyrics\": the new lyric with the section tags on their own lines -- "
    "no line numbers, no syllable counts}}.")


def _cover_cost(new: str, orig_sections: list) -> float:
    """How far a new lyric is from the original's shape (lower is better; None if broken):
    section and line counts, syllables per line, the repeat map, words copied."""
    secs = _sections_of(new)
    if not secs:
        return None
    # A long song's shape is rarely hit line for line (10-08: «Крошка моя», three drafts, all
    # off by a line or a section, and the cover failed): a miss costs, it does not refuse.
    a, b = _shape(orig_sections), _shape(secs)
    cost = 0.5 * abs(len(secs) - len(orig_sections)) / len(orig_sections) + abs(len(a) - len(b)) / len(a)
    cost += sum(abs(x[1] - y[1]) / max(1, x[1]) for x, y in zip(a, b)) / len(a)
    rep = [(i, x[2]) for i, x in enumerate(a) if x[2] is not None]
    if rep:
        cost += sum(1 for i, j in rep if i >= len(b) or b[i][2] != j) / len(rep)
    # A new text, never the old one with the title word swapped (10-08: shown the original's
    # lines, the model kept SHAMAN's lyric and changed «русский» to «дачник»).
    ow = {w for _, ls in orig_sections for l in ls for w in _words(l) if len(w) >= 4}
    nw = {w for _, ls in secs for l in ls for w in _words(l) if len(w) >= 4}
    if nw and len(ow & nw) / len(nw) > 0.25:
        return None
    return cost


COVER_WORD_TRIES = int(os.getenv("COVER_WORD_TRIES", "3"))


def cover_words(ctx, song: str, theme: str, lang: str = "Russian") -> str:
    """New words for `song` on `theme`, written line for line over its own lyric: the same
    sections, syllables and map of repeats, a new hook where its hook was (10-08: a lyric
    written blind from the theme alone came out a quiet ballad for SHAMAN's «Я русский» anthem).
    Several drafts, the closest to the original's shape kept. Raises cover.CoverFailed."""
    import llm
    from utils import safe_json_from_llm
    orig = _sections_of(lyrics_of(ctx, song))
    if not orig:
        raise cover.CoverFailed("no_words")
    user = f"Theme of the new words: {theme}\n\nThe original's shape:\n" + "\n".join(_shape_rows(orig))
    best, best_cost = "", None
    for t in range(max(1, COVER_WORD_TRIES)):
        try:
            raw = llm.call_llm_simple(ctx, _WRITE_OVER.format(lang=lang), user,
                                      temperature=0.7 + 0.1 * t, max_tokens=3000)
            got = safe_json_from_llm(raw or "", ["lyrics"]) or {}
        except Exception as exc:
            logger.warning("cover_words: draft failed: %s", exc)
            continue
        text = str(got.get("lyrics") or "") if isinstance(got, dict) else ""
        # the model echoed the prompt's «3. » numbers and «(9 syllables)» notes -- YuE would sing them
        text = "\n".join(re.sub(r"\s*\((?:\d+[^)]*|= ?line[^)]*)\)\s*$", "", re.sub(r"^\s*\d+[.)]\s*", "", l))
                         for l in text.splitlines())
        cost = _cover_cost(text, orig)
        logger.info("cover_words: draft %d cost %s", t + 1, None if cost is None else round(cost, 2))
        if cost is not None and (best_cost is None or cost < best_cost):
            best, best_cost = text, cost
    if not best:
        raise cover.CoverFailed("no_words")
    return best


def remix_words(ctx, song: str, lyrics: str) -> str:
    """`song` sings `lyrics` (its own melody, voice, backing). Returns an mp3 path; raises cover.CoverFailed."""
    lines = [l.strip() for l in (lyrics or "").splitlines() if l.strip() and not re.fullmatch(r"\[.*\]", l.strip())]
    if not lines:
        raise cover.CoverFailed("no_words")
    work = tempfile.mkdtemp(prefix="remix_")
    vox, back = _stems(song, work, "a")
    speech = _speak_on_melody(ctx, vox, lines, work)
    conv = sing_as(ctx, speech, _singer_ref(vox, work, "a"), os.path.join(work, "conv.wav"))
    return _mix(conv, back, _out())


def remix_voice(ctx, voice_song: str, melody_song: str) -> str:
    """The voice of `voice_song` sings `melody_song` (its words and melody) over its backing."""
    work = tempfile.mkdtemp(prefix="remix_")
    vox1, _ = _stems(voice_song, work, "a")
    vox2, back2 = _stems(melody_song, work, "b")
    conv = sing_as(ctx, os.path.join(work, "b_vox.wav"), _singer_ref(vox1, work, "a"), os.path.join(work, "conv.wav"))
    return _mix(conv, back2, _out())
