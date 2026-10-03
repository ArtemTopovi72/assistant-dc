"""F5 flow steps A/B: speed and intelligibility (GigaAM WER) per nfe_step.

    venv/Scripts/python.exe bench/tts_nfe_ab.py [--steps 16,12,10,7] [--reps 2]

Same phrases, same voice, per step count: synth seconds per phrase and the word
error rate of GigaAM listening to the result. WAVs are kept in
outputs/nfe_ab/<nfe>/ for a listen.
"""
import argparse, json, re, shutil, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
import config as C
from tts_whisper_roundtrip import _ctx

PHRASES = [
    "Привет! Сегодня в Москве облачно, около двенадцати градусов, к вечеру возможен дождь.",
    "Я нашёл три подходящих варианта, самый дешёвый стоит полторы тысячи рублей.",
    "Картинка готова: рыжий кот сидит на подоконнике и смотрит на снег.",
    "Напомню завтра в девять утра про встречу с врачом.",
    "Если коротко, то ошибка была в том, что файл открывался дважды.",
    "Хорошо, переведу этот текст на английский и пришлю тебе через минуту.",
    "В этой книге шесть глав, а главная героиня живёт в маленьком северном городе.",
    "Поезд отправляется с Ленинградского вокзала в двадцать три тридцать.",
    "Чтобы установить программу, скачай архив, распакуй его и запусти установщик.",
    "Спасибо, что подождал! Вот что удалось выяснить по твоему вопросу.",
    "Температура воды в озере сейчас семнадцать градусов, купаться прохладно.",
    "Музыка получилась спокойной, с гитарой и лёгким шумом дождя на фоне.",
]


def words(s):
    return re.findall(r"[а-яёa-z0-9]+", s.lower().replace("ё", "е"))


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="16,12,10,7")
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    import audio as A
    ctx = _ctx()
    A.synth_single_segment(ctx, 0, "DC", "Прогрев.", apply_stress=True)   # warm-up
    out = {}
    for nfe in [int(x) for x in a.steps.split(",")]:
        C.TTS_NFE_STEP = nfe
        d = ROOT / "outputs" / "nfe_ab" / str(nfe)
        d.mkdir(parents=True, exist_ok=True)
        errs = n = 0
        secs = []
        for rep in range(a.reps):
            for i, ph in enumerate(PHRASES):
                t = time.time()
                wav = A.synth_single_segment(ctx, i, "DC", ph, apply_stress=True)
                secs.append(time.time() - t)
                if not wav:
                    errs += len(words(ph)); n += len(words(ph)); continue
                heard = A.transcribe_audio_file(ctx, wav, engine="gigaam") or ""
                e, m = wer(ph, heard); errs += e; n += m
                if rep == 0:
                    shutil.copy(wav, d / f"{i:02d}.wav")
                if e:
                    print(f"  nfe {nfe} [{i}] {e} err: {heard}", flush=True)
        secs.sort()
        out[nfe] = {"wer_pct": round(100 * errs / n, 2), "median_s": round(secs[len(secs) // 2], 3)}
        print(f"nfe {nfe}: {out[nfe]}", flush=True)
    json.dump(out, open(ROOT / "outputs" / "nfe_ab" / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
