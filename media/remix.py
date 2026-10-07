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
SX_PY = venv_python("Z:/venvs/venv_soulx")   # on the SSD: C: is nearly full, E: is an HDD
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


def sing_as(ctx, source: str, prompt: str, out: str) -> str:
    """`source` (singing) in the timbre of `prompt`. SoulX-Singer-SVC, else Seed-VC."""
    work = tempfile.mkdtemp(prefix="remix_svc_")
    if os.path.isfile(SX_MODEL) and os.path.isfile(SX_PY):
        import shutil
        shutil.copy(source, os.path.join(work, "src.wav"))
        shutil.copy(prompt, os.path.join(work, "prm.wav"))
        for k in ("src", "prm"):
            _f0_npy(os.path.join(work, k + ".wav"), os.path.join(work, k + "_f0.npy"))
        cmd = [SX_PY, os.path.join(SX, "cli", "inference_svc.py"), "--device", "cuda", "--model_path", SX_MODEL,
               "--config", os.path.join(SX, "soulxsinger", "config", "soulxsinger.yaml"),
               "--prompt_wav_path", os.path.join(work, "prm.wav"), "--target_wav_path", os.path.join(work, "src.wav"),
               "--prompt_f0_path", os.path.join(work, "prm_f0.npy"), "--target_f0_path", os.path.join(work, "src_f0.npy"),
               "--save_dir", os.path.join(work, "o"), "--auto_shift", "--pitch_shift", "0", "--n_steps", "32", "--fp16"]
        env = {**os.environ, "PYTHONPATH": SX}
        music.run_gpu_worker(ctx, SX_PY, "", {}, "SoulX-SVC", 900, cmd=cmd, env=env)
        gen = os.path.join(work, "o", "generated.wav")
        if os.path.isfile(gen):
            shutil.copy(gen, out)
            return out
        logger.warning("remix: SoulX gave no file, falling back to Seed-VC")
    job = [{"source": source, "target": prompt, "out": out}]
    jf = os.path.join(work, "jobs.json")
    with open(jf, "w", encoding="utf-8") as fh:
        json.dump(job, fh)
    music.run_gpu_worker(ctx, SEEDVC_PY, "seedvc_batch.py", {}, "Seed-VC", 900,
                         cmd=[SEEDVC_PY, os.path.join(ROOT, "scripts", "seedvc_batch.py"), jf])
    if not os.path.isfile(out):
        raise cover.CoverFailed("render")
    return out


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
        segs, _ = wh.transcribe(vp, word_timestamps=True)
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
        ts, _ = wh.transcribe(tp, word_timestamps=True)
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


def _layout(ctx, lyrics: str, parts: list) -> str:
    """The new lyric laid out on the original's sections, each sung section holding about as
    many syllables as its melody has notes (10-07: Cheri Cheri Lady got 144 syllables on a
    44-note chorus and YuE2 mumbled them). The model places lines; the budget is enforced
    here by dropping a section's last lines; a broken plan falls back to an even split."""
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
    blocks, k = [], 0
    for tag, notes in parts:
        if tag in _INSTRUMENTAL or notes < _MIN_SUNG:
            blocks.append(f"[{'Interlude' if tag not in _INSTRUMENTAL else tag}]")
            continue
        blocks.append(f"[{tag}]\n" + "\n".join(lines[n - 1] for n in fitted[k]))
        k += 1
    return "\n\n".join(blocks)


def _resing_style(ctx, heard: str, tempo_key: str) -> str:
    """YuE2 tags for the original's sound: the model names the song from its first sung
    lines (it knows «3 сентября» is 90s Russian chanson); tempo and key come from the score."""
    tags = ""
    try:
        import llm
        tags = llm.call_llm_simple(
            ctx, "Name the musical style of the song these sung lines come from, as 8-12 comma-separated "
            "music tags: genre, era, lead vocal (gender, timbre), main instruments, mood. Tags only.",
            heard[:600], temperature=0.2, max_tokens=120).strip().splitlines()[0]
    except Exception as exc:
        logger.warning("resing: style failed: %s", exc)
    tags = re.sub(r"[^\w ,'&/-]", "", tags or "")[:240] or "pop ballad, male lead vocal, synthesizer, drums, bass"
    return ", ".join(t for t in (tags, tempo_key) if t)


def resing(ctx, song: str, lyrics: str) -> str:
    """`song` re-sung with `lyrics`: YuE2 renders the original's full score with the new words
    (new notes where the words need them, its own backing in the original's style), then the
    original singer's timbre goes on the vocal. Returns an mp3 path; raises cover.CoverFailed."""
    import soundfile as sf
    import mashup_auto
    if not [l for l in (lyrics or "").splitlines() if l.strip() and not re.fullmatch(r"\[.*\]", l.strip())]:
        raise cover.CoverFailed("no_words")
    work = tempfile.mkdtemp(prefix="resing_")
    logger.info("resing: work %s", work)
    vox, _ = _stems(song, work, "a")
    abc = _score(ctx, os.path.join(work, "a_ref.wav"), work)
    segs, _ = ctx.models.whisper.transcribe(os.path.join(work, "a_vox.wav"), word_timestamps=True)
    heard = " ".join(w["w"] for w in _sung_words(segs))
    out = os.path.join(work, "yue.mp3")
    job = {"lyrics": music.yue2_lyrics(_layout(ctx, lyrics, _score_parts(abc))),
           "style": music.yue2_language(_resing_style(ctx, heard, _score_tempo_key(abc)), lyrics),
           "abc": abc, "out": out}
    logger.info("resing: style %s", job["style"])
    if ctx is not None and hasattr(ctx, "set_stage"):
        ctx.set_stage("Composing a song")
    music._render_yue2_once(ctx, True, job, int(time.time()) % 100000)
    if not music._valid_audio_file(out):
        raise cover.CoverFailed("render")
    music._master(out)
    yvox, yback = _stems(out, work, "y")
    conv = sing_as(ctx, os.path.join(work, "y_vox.wav"), _singer_ref(vox, work, "a"), os.path.join(work, "conv.wav"))
    v, sr = sf.read(conv, dtype="float32")
    v = v.mean(1) if v.ndim > 1 else v
    if sr != SR:
        import librosa
        v = librosa.resample(v, orig_sr=sr, target_sr=SR)
    ratio = mashup_auto._rms(yvox, gate=0.01) / max(1e-6, mashup_auto._rms(yback))
    return mashup_auto.mix(v, yback, ratio, _out())


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
