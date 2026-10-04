"""Measure how long YuE2 sings per sung line (the line_budget table was measured on Music3).

    venv/Scripts/python.exe bench/yue2_seconds_per_line.py [8 16 24 36]

One render per size, same style and seed, a fresh Russian lyric of N sung lines in 4-line sections.
Prints lines, seconds of audio, seconds per line, wall time. Needs the card: stop the desktop app first.
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "media"))
sys.path.insert(0, os.path.join(ROOT, "voice"))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import music

STANZAS = [
    ["Над городом гаснет закат", "Неон загорается вновь", "И окна, как свечи, горят", "Сжигая чужую любовь"],
    ["Я иду по пустой мостовой", "Тень моя убегает вперёд", "Ветер шепчет мне что-то с тобой", "Только сердце опять не уснёт"],
    ["Расскажи мне про лето в окне", "Про цветы на балконной стене", "Про дорогу, что вела домой", "Где ты ждал меня тихой весной"],
    ["Пусть весь мир замирает на миг", "Пусть звенит тишина в проводах", "Я услышу твой голос родной", "Даже если ты где-то вдали"],
    ["Облака уплывают на юг", "Журавли замыкают круги", "Я тебя никогда не забуду", "Хоть и вряд ли вернусь я туда"],
    ["Свет в окне, и дрожит занавеска", "Кто-то ждёт у порога меня", "Тихий дождь, словно старая песня", "Снова шепчет про завтрашний день"],
    ["Звёзды падают в синюю тьму", "Я загадаю одно лишь желанье", "Чтоб не страшно мне было идти", "Чтобы ты был со мною в дороге"],
    ["Дальний поезд уходит во мглу", "Огоньки исчезают за лесом", "Я остался один на перроне", "Но в кармане тепло от письма"],
    ["Тихо падает первый снег", "Город спит под белым одеялом", "Я стою у твоего окна", "И не знаю, что сказать тебе"],
]


def lyric(n_lines: int) -> str:
    out, i, k = [], 0, 0
    tags = ["verse", "chorus"]
    while k < n_lines:
        st = STANZAS[i % len(STANZAS)][: min(4, n_lines - k)]
        out.append(f"[{tags[i % 2]}]\n" + "\n".join(st))
        k += len(st)
        i += 1
    out.append("[outro]\nИ тихо гаснет свет")
    return "\n\n".join(out)


def seconds(path: str) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


def main(sizes):
    rows = []
    for n in sizes:
        t0 = time.time()
        path = music._generate_yue2(None, lyric(n), "Russian pop, 100 BPM, female vocal, piano, warm", 42)
        dt = time.time() - t0
        sec = seconds(path)
        sung = music.count_sung_lines(lyric(n))
        rows.append({"lines": sung, "seconds": round(sec, 1), "sec_per_line": round(sec / sung, 2), "wall": round(dt)})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    print("TABLE", json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [8, 16, 24, 36])
