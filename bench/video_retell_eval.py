"""Long-video retelling: quality and speed on videos whose content is KNOWN.

The truth comes from the generator, not from a judge model: every scene shows a
planted code word and number on a coloured card (and the "transcript" of its
portion says a planted budget), so a storyboard or a retelling is scored by
counting what it kept, where it put it in time, and what it made up.

    python bench/video_retell_eval.py gen                  # build the test videos once
    python bench/video_retell_eval.py run <config> [case]  # one config over the cases
    python bench/video_retell_eval.py grid                 # the whole grid, one JSON per run

A "request" here is one portion through the real path of tg_video._long_video_job
(cut -> look_at_video -> retelling), plus one overview per video. ASR is left out
on purpose (its own bench: asr_ru_shootout); the transcript is the planted one.

Valid only with the GPU to itself: `lms ps` must show one model and nothing else
may render or chat meanwhile (benchmark-contamination). The first portion of a run
pays any model reload, so a config's first number is thrown away when timing.
"""
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for d in ("", "agent", "imaging", "media", "bot", "core"):
    sys.path.insert(0, str(ROOT / d))
os.chdir(ROOT)
os.environ.setdefault("F5_TEST_RUN", "0")

OUT = ROOT / "tmp" / "vr_eval"
FONT = "C\\:/Windows/Fonts/arialbd.ttf"
WORDS = ["Маяк", "Орбита", "Янтарь", "Бастион", "Родник", "Квазар", "Пилигрим", "Тайга", "Мираж",
         "Кузнец", "Фрегат", "Эхо", "Сапфир", "Ураган", "Витязь", "Лагуна", "Гранит", "Комета",
         "Ладья", "Зенит", "Чердак", "Прибой", "Радуга", "Скипетр", "Туман", "Ястреб", "Бирюза",
         "Горизонт", "Крепость", "Ласточка"]
COLORS = ["0x1b4f9c", "0x8c1f1f", "0x1f6b3a", "0x6a3d9a", "0x9a6a12", "0x1f6b6b", "0x555555", "0x9a2f6b"]

# kind -> (scene seconds, total seconds, font px)
KINDS = {"slow": (14, 300, 110), "mid": (8, 240, 110), "fast": (3, 120, 110), "small": (12, 180, 38)}


def ffmpeg(*a):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *a], check=True, timeout=600)


def make_case(kind: str, seed: int = 7) -> dict:
    """-> {"path", "scenes": [{"a","b","word","num","budget"}], "kind"}; cached on disk."""
    OUT.mkdir(parents=True, exist_ok=True)
    meta = OUT / f"{kind}_{seed}.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))
    sc, total, px = KINDS[kind]
    rnd = random.Random(seed)
    words = rnd.sample(WORDS, len(WORDS))
    scenes, parts = [], []
    n = total // sc
    for i in range(n):
        w, num = words[i % len(words)], rnd.randint(11, 98)
        scenes.append({"a": i * sc, "b": (i + 1) * sc, "word": w, "num": num, "budget": rnd.randint(2, 9) * 10 + num})
        seg = OUT / f"{kind}_{seed}_{i:02d}.mp4"
        txt = f"{w} {num}".replace(":", "\\:")
        ffmpeg("-f", "lavfi", "-i", f"color=c={COLORS[i % len(COLORS)]}:s=960x540:r=10:d={sc}",
               "-vf", f"drawtext=fontfile='{FONT}':text='{txt}':fontcolor=white:fontsize={px}:x=(w-text_w)/2:y=(h-text_h)/2,"
                      f"drawbox=x=0:y=0:w=iw*t/{sc}:h=12:color=white@0.8:t=fill",
               "-pix_fmt", "yuv420p", str(seg))
        parts.append(seg)
    lst = OUT / f"{kind}_{seed}.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    path = OUT / f"{kind}_{seed}.mp4"
    ffmpeg("-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(path))
    for p in parts:
        p.unlink()
    case = {"path": str(path), "scenes": scenes, "kind": kind}
    meta.write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
    return case


def said_for(scenes, a, b) -> str:
    """The planted transcript of [a, b): one spoken sentence per scene in it."""
    return " ".join(f"Теперь о проекте {s['word']}. Бюджет этого проекта {s['budget']} тысяч рублей."
                    for s in scenes if s["a"] >= a and s["b"] <= b + 0.5)


# ── scoring ──────────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    return (s or "").lower().replace("ё", "е")


def _stamps(line: str):
    return [int(m) * 60 + int(s) for m, s in re.findall(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)", line)]


def score_board(board: str, scenes, a: float, b: float, tol: float = 4.0) -> dict:
    """Per scene fully inside [a,b): is its word+number on a storyboard line, and does that line's
    stamp fall inside the scene (±tol)?"""
    inside = [s for s in scenes if s["a"] >= a and s["b"] <= b + 0.5]
    seen = timed = 0
    for s in inside:
        for line in (board or "").splitlines():
            if _norm(s["word"]) in _norm(line) and str(s["num"]) in line:
                seen += 1
                st = _stamps(line)
                if any(s["a"] - tol <= t <= s["b"] + tol for t in st):
                    timed += 1
                break
    return {"scenes": len(inside), "seen": seen, "timed": timed}


def invented(text: str, scenes) -> int:
    """Code-word/number pairs and bare two-digit numbers that are NOT in the truth. Timecodes are
    stripped first; the budget numbers said in the transcript count as true."""
    true_nums = {str(s["num"]) for s in scenes} | {str(s["budget"]) for s in scenes}
    true_words = {_norm(s["word"]) for s in scenes}
    body = re.sub(r"(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?!\d)", " ", text or "")
    bad = 0
    for num in re.findall(r"(?<![\d.,])(\d{2,3})(?![\d:])", body):
        if num not in true_nums:
            bad += 1
    for w in re.findall(r"[А-ЯЁ][а-яё]{3,}", body):
        if _norm(w) in {_norm(x) for x in WORDS} and _norm(w) not in true_words:
            bad += 1
    return bad


def score_retell(text: str, scenes) -> dict:
    ws = [s for s in scenes if _norm(s["word"]) in _norm(text)]
    bud = [s for s in scenes if str(s["budget"]) in (text or "")]
    return {"words": len(ws), "budgets": len(bud), "scenes": len(scenes), "invented": invented(text, scenes)}


# ── the pipeline under test ──────────────────────────────────────────────────
def pipeline(case: dict, *, chunk_min: int, frames: int, vision_tokens: int = 900,
             retell_tokens: int = 700, max_portions: int = 99, tag: str = "x", width: int = 0) -> dict:
    import models
    import tg_video as V
    import video_look as VL
    from seams import text_model, vision_model
    ctx = models.Context(models=None, transcription_cache={}, cache_file=None,
                         asr_lock=threading.Lock(), tts_lock=threading.Lock())
    scenes, src = case["scenes"], case["path"]
    total = VL.duration_seconds(src)
    chunk = chunk_min * 60
    n = min(max_portions, max(1, int((total + chunk - 1) // chunk)))
    work = OUT / f"work_{tag}"
    work.mkdir(parents=True, exist_ok=True)

    def shrink(fr):
        if not width:
            return fr
        from PIL import Image
        out = []
        for label, path in fr:
            im = Image.open(path)
            if im.width > width:
                im = im.resize((width, round(im.height * width / im.width)))
            dst = str(path) + f".w{width}.jpg"
            im.convert("RGB").save(dst, quality=85)
            out.append((label, dst))
        return out

    def vision(fr, q, system):
        fr = shrink(fr)
        return vision_model(ctx)(ctx, image_paths=fr, user_text=q, system_prompt=system,
                                 temperature=0.1, max_tokens=vision_tokens)

    rows, material = [], []
    for k in range(n):
        t0 = k * chunk
        length = min(chunk, total - t0)
        part = str(work / f"part_{k}.mp4")
        c0 = time.time()
        assert V.cut_portion(src, t0, length, part), "cut failed"
        t_cut = time.time() - c0
        said = said_for(scenes, t0, t0 + length)
        c0 = time.time()
        seen = VL.look_at_video(ctx, part, max_frames=frames, work_dir=work / f"look_{k}",
                                lang="ru", transcript=said, offset=t0, vision=vision)
        t_look = time.time() - c0
        board = (seen.get("description") or "").strip()
        payload = ("TRANSCRIPT:\n" + said + "\n\n" if said else "") + "STORYBOARD:\n" + board
        c0 = time.time()
        tale = (text_model(ctx)(ctx, V._RETELL_SYSTEM + V._lang_line("ru") + "\n"
                                + V._structure_rules("ru", timecoded=True, ask=False), payload,
                                temperature=0.3, max_tokens=retell_tokens, prefill="<think></think>") or "").strip()
        t_tell = time.time() - c0
        b = score_board(board, scenes, t0, t0 + length)
        r = score_retell(tale, [s for s in scenes if s["a"] >= t0 and s["b"] <= t0 + length + 0.5])
        rows.append({"part": k, "a": t0, "b": t0 + length, "frames": len(seen.get("times") or []),
                     "t_cut": round(t_cut, 1), "t_look": round(t_look, 1), "t_tell": round(t_tell, 1),
                     "board": b, "retell": r, "board_invented": invented(board, scenes),
                     "board_text": board, "retell_text": tale})
        material.append({"a": V._stamp(t0), "b": V._stamp(t0 + length), "board": board, "note": tale})
    c0 = time.time()
    notes = "\n\n".join(f"=== {p['a']}-{p['b']} ===\n" + (p.get("note") or p.get("board") or "") for p in material)
    if len(material) == 1 and material[0]["note"]:       # as in tg_video: one portion is the whole video
        over = material[0]["note"]
    else:
        over = (text_model(ctx)(ctx, V._OVERVIEW_SYSTEM + V._lang_line("ru") + "\n"
                                + V._structure_rules("ru", timecoded=True, ask=False), notes[:24000],
                                temperature=0.3, max_tokens=1100, prefill="<think></think>") or "").strip()
    t_over = time.time() - c0
    return {"case": case["kind"], "chunk_min": chunk_min, "frames": frames, "vision_tokens": vision_tokens, "width": width,
            "portions": rows, "overview_text": over, "t_overview": round(t_over, 1),
            "overview": score_retell(over, [s for s in scenes if s["b"] <= rows[-1]["b"] + 0.5]),
            "t_total": round(sum(r["t_look"] + r["t_tell"] + r["t_cut"] for r in rows) + t_over, 1)}


def summarize(res: dict) -> dict:
    rows = res["portions"]
    sc = sum(r["board"]["scenes"] for r in rows) or 1
    return {"case": res["case"], "chunk": res["chunk_min"], "frames": res["frames"],
            "board_seen": round(sum(r["board"]["seen"] for r in rows) / sc, 2),
            "board_timed": round(sum(r["board"]["timed"] for r in rows) / sc, 2),
            "retell_words": round(sum(r["retell"]["words"] for r in rows) / sc, 2),
            "retell_budgets": round(sum(r["retell"]["budgets"] for r in rows) / sc, 2),
            "overview_words": round(res["overview"]["words"] / max(1, res["overview"]["scenes"]), 2),
            "invented": sum(r["board_invented"] + r["retell"]["invented"] for r in rows) + res["overview"]["invented"],
            "t_total": res["t_total"], "t_per_portion": round(sum(r["t_look"] + r["t_tell"] for r in rows) / len(rows), 1),
            "t_per_video_min": round(res["t_total"] / (rows[-1]["b"] / 60), 1)}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "gen"
    if cmd == "gen":
        for k in KINDS:
            c = make_case(k)
            print(k, c["path"], len(c["scenes"]), "scenes")
    elif cmd == "grid":
        configs = os.environ.get("CONFIGS", "2x8 2x12 2x16 3x12 3x16 3x24 5x12 5x20 5x30 5x40").split()
        kinds = os.environ.get("KINDS", "slow mid fast small").split()
        mp = int(os.environ.get("PORTIONS", "2"))
        out_rows = []
        for cfg in configs:
            chunk, frames = (int(x) for x in cfg.split("x"))
            for kind in kinds:
                f = OUT / f"res_{kind}_{chunk}x{frames}_p{mp}.json"
                if f.exists():
                    res = json.loads(f.read_text(encoding="utf-8"))
                else:
                    res = pipeline(make_case(kind), chunk_min=chunk, frames=frames, tag=f"{kind}_{chunk}x{frames}",
                                   max_portions=mp)
                    f.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
                sm = summarize(res)
                out_rows.append(sm)
                print(json.dumps(sm, ensure_ascii=False), flush=True)
        (OUT / "grid_summary.json").write_text(json.dumps(out_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    elif cmd == "width":
        for w in (0, 640, 448):
            for kind in ("slow", "small", "fast"):
                res = pipeline(make_case(kind), chunk_min=5, frames=40, tag=f"w{w}_{kind}", max_portions=1, width=w)
                sm = summarize(res); sm["width"] = w
                print(json.dumps(sm, ensure_ascii=False), flush=True)
    elif cmd == "run":
        kind = sys.argv[3] if len(sys.argv) > 3 else "slow"
        chunk, frames = (int(x) for x in sys.argv[2].split("x"))
        res = pipeline(make_case(kind), chunk_min=chunk, frames=frames, tag=f"{kind}_{chunk}x{frames}",
                       max_portions=int(os.environ.get("PORTIONS", "99")))
        (OUT / f"res_{kind}_{chunk}x{frames}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(summarize(res), ensure_ascii=False))
