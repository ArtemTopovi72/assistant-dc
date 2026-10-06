"""Vision shootout: which loaded model SEES better, on pictures with known answers.

The published scores put Qwen3.6-35B-A3B and Gemma 4 26B-A4B within 1.5 points
of each other, so they decide nothing for us. This draws pictures whose answer
is known (Russian and English text at three sizes, a count of shapes, a colour,
which object is where, a cell of a table, tiny text in a corner of a big
picture), asks the model loaded in LM Studio through the app's own vision call
(`llm.analyze_image_with_llm`: same resize, same encoding), and scores it.

    # load a model the way the app does, then:
    venv/Scripts/python.exe bench/vision_shootout.py --label gemma
    venv/Scripts/python.exe bench/vision_shootout.py --label qwen
    venv/Scripts/python.exe bench/vision_shootout.py --compare gemma qwen

Results: outputs/vision_shootout/<label>.json. The pictures are the same on
every run (fixed seed), so two labels are compared on identical inputs.
"""
import argparse
import json
import os
import random
import re
import sys
import time
import types
from difflib import SequenceMatcher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT = os.path.join(ROOT, "outputs", "vision_shootout")
IMG = os.path.join(OUT, "images")

RU_WORDS = ("дом река окно ветер город лампа книга стол поезд мост сад снег утро "
            "рынок лодка море звезда поле дорога чайник письмо гора кошка").split()
EN_WORDS = ("house river window wind city lamp book table train bridge garden snow "
            "morning market boat sea star field road kettle letter mountain cat").split()
COLORS = {"красный": (220, 30, 30), "синий": (30, 60, 220), "зелёный": (30, 160, 60),
          "жёлтый": (240, 210, 20), "фиолетовый": (140, 50, 180), "оранжевый": (245, 130, 20)}
COLOR_ALIASES = {"красный": ("красн", "red"), "синий": ("син", "голуб", "blue"),
                 "зелёный": ("зел", "green"), "жёлтый": ("жёлт", "желт", "yellow"),
                 "фиолетовый": ("фиолет", "пурпур", "лилов", "purple", "violet"),
                 "оранжевый": ("оранж", "orange")}


def _font(size):
    from PIL import ImageFont
    for name in ("arial.ttf", "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _canvas(w=1024, h=768):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (w, h), (250, 250, 247))
    return im, ImageDraw.Draw(im)


def build_tasks(seed=7):
    """[(id, kind, image path, question, answer)] -- the same list for every run."""
    rnd = random.Random(seed)
    os.makedirs(IMG, exist_ok=True)
    tasks = []

    def save(im, tid):
        p = os.path.join(IMG, tid + ".png")
        im.save(p)
        return p

    # text: three sizes, Russian and English
    for lang, words in (("ru", RU_WORDS), ("en", EN_WORDS)):
        for size in (64, 32, 18):
            for k in range(2):
                text = " ".join(rnd.sample(words, 4))
                im, d = _canvas()
                d.text((60, 300), text, fill=(20, 20, 20), font=_font(size))
                tid = f"ocr_{lang}_{size}_{k}"
                q = ("Перепиши текст на картинке точно, буква в букву. Ответь только этим текстом."
                     if lang == "ru" else "Copy the text in the picture exactly. Answer with that text only.")
                tasks.append((tid, f"ocr_{lang}", save(im, tid), q, text))
    # count
    for k in range(6):
        n = rnd.randint(3, 13)
        im, d = _canvas()
        boxes = []
        while len(boxes) < n:
            x, y, r = rnd.randint(60, 960), rnd.randint(60, 700), rnd.randint(18, 34)
            if all((x - a) ** 2 + (y - b) ** 2 > (r + c + 12) ** 2 for a, b, c in boxes):
                boxes.append((x, y, r))
        for x, y, r in boxes:
            d.ellipse((x - r, y - r, x + r, y + r), fill=(40, 90, 200))
        tid = f"count_{k}"
        tasks.append((tid, "count", save(im, tid), "Сколько синих кругов на картинке? Ответь только числом.", str(n)))
    # colour of the one square among grey circles
    for k, (name, rgb) in enumerate(list(COLORS.items())):
        im, d = _canvas()
        for _ in range(6):
            x, y = rnd.randint(80, 940), rnd.randint(80, 680)
            d.ellipse((x - 30, y - 30, x + 30, y + 30), fill=(150, 150, 150))
        x, y = rnd.randint(100, 900), rnd.randint(100, 650)
        d.rectangle((x - 45, y - 45, x + 45, y + 45), fill=rgb)
        tid = f"color_{k}"
        tasks.append((tid, "color", save(im, tid), "Какого цвета квадрат? Ответь одним словом.", name))
    # where: left / right, top / bottom
    for k in range(6):
        im, d = _canvas()
        left = rnd.random() < 0.5
        tx, cx = (200, 800) if left else (800, 200)
        y1, y2 = rnd.randint(200, 560), rnd.randint(200, 560)
        d.polygon(((tx, y1 - 50), (tx - 50, y1 + 40), (tx + 50, y1 + 40)), fill=(200, 40, 40))
        d.ellipse((cx - 45, y2 - 45, cx + 45, y2 + 45), fill=(40, 140, 60))
        tid = f"where_{k}"
        tasks.append((tid, "where", save(im, tid),
                      "Где красный треугольник: слева или справа от зелёного круга? Ответь одним словом: слева или справа.",
                      "слева" if left else "справа"))
    # table cell
    for k in range(4):
        im, d = _canvas()
        f = _font(28)
        vals = [[rnd.randint(10, 99) for _ in range(3)] for _ in range(3)]
        cols = ("Январь", "Февраль", "Март")
        rows = ("Москва", "Казань", "Тула")
        for j, c in enumerate(cols):
            d.text((300 + j * 220, 180), c, fill=(0, 0, 0), font=f)
        for i, r in enumerate(rows):
            d.text((80, 260 + i * 90), r, fill=(0, 0, 0), font=f)
            for j in range(3):
                d.rectangle((290 + j * 220, 250 + i * 90, 480 + j * 220, 320 + i * 90), outline=(120, 120, 120))
                d.text((350 + j * 220, 262 + i * 90), str(vals[i][j]), fill=(0, 0, 0), font=f)
        i, j = rnd.randrange(3), rnd.randrange(3)
        tid = f"table_{k}"
        tasks.append((tid, "table", save(im, tid),
                      f"Какое число в таблице в строке «{rows[i]}», в столбце «{cols[j]}»? Ответь только числом.",
                      str(vals[i][j])))
    # tiny text in a corner of a big picture: is the detail kept after resizing?
    for k in range(4):
        im, d = _canvas(2400, 1600)
        for _ in range(30):
            x, y = rnd.randint(0, 2300), rnd.randint(0, 1500)
            d.rectangle((x, y, x + 90, y + 60), fill=(rnd.randint(150, 230),) * 3)
        code = f"{rnd.randint(100, 999)}-{rnd.choice('АБВГДЕЖКЛМН')}{rnd.randint(10, 99)}"
        d.rectangle((2050, 1450, 2380, 1560), fill=(255, 255, 255))
        d.text((2070, 1470), code, fill=(0, 0, 0), font=_font(44))
        tid = f"tiny_{k}"
        tasks.append((tid, "tiny", save(im, tid),
                      "В правом нижнем углу написан код. Перепиши его точно. Ответь только кодом.", code))
    return tasks


def _norm(s):
    s = (s or "").strip().lower().replace("ё", "е")
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    return re.sub(r"[\s\"'«»`.,:;!?]+", " ", s).strip()


def score(kind, answer, got):
    g = _norm(got)
    if kind in ("count", "table"):
        nums = re.findall(r"\d+", g)
        return 1.0 if nums and nums[0] == answer else 0.0
    if kind == "color":
        return 1.0 if any(a in g for a in COLOR_ALIASES[answer]) else 0.0
    if kind == "where":
        other = "справа" if answer == "слева" else "слева"
        return 1.0 if answer in g and other not in g else 0.0
    # text: similarity, so a near miss on 18 px text shows as a near miss
    return round(SequenceMatcher(None, _norm(answer), g).ratio(), 3)


def run(label, model, n_max=None):
    from llm import analyze_image_with_llm
    ctx = types.SimpleNamespace(model_name=model, no_think=True, reasoning_effort="low",
                                is_cancelled=lambda: False, set_stage=lambda *_: None)
    rows = []
    tasks = build_tasks()[:n_max] if n_max else build_tasks()
    for tid, kind, path, q, ans in tasks:
        t0 = time.time()
        try:
            got = analyze_image_with_llm(ctx=ctx, image_path=path, user_text=q,
                                         system_prompt="You read pictures accurately and answer briefly.",
                                         max_tokens=300) or ""
        except Exception as exc:
            got = f"[ERR {type(exc).__name__}: {exc}]"
        dt = time.time() - t0
        s = score(kind, ans, got)
        rows.append({"id": tid, "kind": kind, "answer": ans, "got": got.strip()[:300], "score": s, "sec": round(dt, 2)})
        print(f"{tid:14s} {s:5.2f} {dt:5.1f}s  want={ans!r}  got={got.strip()[:80]!r}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, label + ".json"), "w", encoding="utf-8") as f:
        json.dump({"label": label, "model": model, "rows": rows}, f, ensure_ascii=False, indent=1)
    summary(label, rows)


def _by_kind(rows):
    out = {}
    for r in rows:
        out.setdefault(r["kind"], []).append(r["score"])
    return {k: sum(v) / len(v) for k, v in out.items()}


def summary(label, rows):
    kinds = _by_kind(rows)
    total = sum(r["score"] for r in rows) / max(1, len(rows))
    sec = sorted(r["sec"] for r in rows)
    print(f"\n{label}: TOTAL {total:.3f}  median {sec[len(sec) // 2]:.1f}s  "
          + "  ".join(f"{k} {v:.2f}" for k, v in sorted(kinds.items())))


def compare(labels):
    data = {}
    for lb in labels:
        with open(os.path.join(OUT, lb + ".json"), encoding="utf-8") as f:
            data[lb] = json.load(f)["rows"]
    kinds = sorted({r["kind"] for rows in data.values() for r in rows})
    print("kind".ljust(10) + "".join(lb.rjust(12) for lb in labels))
    for k in kinds + ["TOTAL", "median s"]:
        line = k.ljust(10)
        for lb in labels:
            rows = data[lb]
            if k == "TOTAL":
                v = sum(r["score"] for r in rows) / len(rows)
            elif k == "median s":
                s = sorted(r["sec"] for r in rows)
                v = s[len(s) // 2]
            else:
                xs = [r["score"] for r in rows if r["kind"] == k]
                v = sum(xs) / len(xs) if xs else float("nan")
            line += f"{v:12.3f}"
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--label", help="name for this run's results (e.g. gemma, qwen)")
    ap.add_argument("--model", default="", help="served model id (default: MODEL_NAME from .env)")
    ap.add_argument("--max", type=int, default=0, help="only the first N tasks (a quick check)")
    ap.add_argument("--compare", nargs="+", help="labels to put side by side")
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare)
    if not a.label:
        ap.error("--label or --compare")
    from config import MODEL_NAME
    run(a.label, a.model or MODEL_NAME, a.max or None)


if __name__ == "__main__":
    main()
