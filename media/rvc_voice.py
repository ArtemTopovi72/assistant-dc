"""A singer's own RVC voice (Applio), trained once on the original's vocal stem and kept per artist.

10-07: «3 сентября» re-sung by YuE2 got Shufutinsky's voice from an RVC model trained on his
vocal stem («МОЛОДЕЦ!!!»); a zero-shot timbre (SoulX / Seed-VC) is only close. Training takes
~4 s an epoch on 6 min of vocal, so the first cover of an artist pays it once and the model is
reused for every later cover of that artist."""
import glob
import logging
import os
import re
import shutil

import music
from config import venv_python

logger = logging.getLogger("assistant.rvc")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPLIO = os.path.join(ROOT, "models_ext", "applio")
PY = venv_python(os.path.join(ROOT, "venv_applio"))
# 10-08 sweep on Marshal (4.9 min of vocal, YuE take heard at 0.97): the stock pretrain at 200
# epochs kept 0.52 of the lines, TITAN at 60 -- 0.34, the Russian Snowie V3.1 pretrain at 60 --
# 0.59 in 5 min instead of 10. «Epochs x minutes of vocal ~ 300» is the usual budget.
EPOCHS = int(os.getenv("RVC_EPOCHS", "60"))
PRETRAIN_URL = "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v2/40k/Snowie/"
PRETRAIN = ("G_SnowieV3.1_40k.pth", "D_SnowieV3.1_40k.pth")
TRAIN_TIMEOUT = int(os.getenv("RVC_TRAIN_TIMEOUT", "3600"))
_TRANSLIT = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                     ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s",
                      "t", "u", "f", "h", "ts", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya"]))


def available() -> bool:
    return os.path.isfile(PY) and os.path.isfile(os.path.join(APPLIO, "core.py"))


def slug(artist: str) -> str:
    """Folder name for an artist: «Михаил Шуфутинский» -> mihail_shufutinskiy. The same singer
    spelled in Latin («Mikhail Shufutinsky») lands on the voice already trained for them."""
    s = "".join(_TRANSLIT.get(c, c) for c in (artist or "").strip().lower())
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")[:40]
    if not s:
        return ""
    import difflib
    key = re.sub(r"[^a-z]", "", s).replace("kh", "h").replace("sky", "skiy")
    for d in os.listdir(os.path.join(APPLIO, "logs")) if os.path.isdir(os.path.join(APPLIO, "logs")) else ():
        other = re.sub(r"[^a-z]", "", d).replace("kh", "h").replace("sky", "skiy")
        same = difflib.SequenceMatcher(None, key, other).ratio() >= 0.85 or (
            min(len(key), len(other)) >= 6 and (key in other or other in key))
        if d != s and other and same and model_of(d):
            return d
    return s


# Star voices offered in the cover's voice picker: trained on ~15 songs of the singer's lead
# vocal each (runtime/covers_batch/tsoi_night/night.py, 10-09), then the one-song models kept
# from earlier covers. Only the trained ones are shown; one per label, the first wins.
STARS = (("tsoi_hq", "Виктор Цой"), ("gorshok_hq", "Горшок (КиШ)"), ("anders_hq", "Томас Андерс"),
         ("lindemann_hq", "Тиль Линдеманн"), ("viktor_tsoy", "Виктор Цой"), ("korol_i_shut", "Горшок (КиШ)"),
         ("shaman", "SHAMAN"), ("mihail_shufutinskiy", "Михаил Шуфутинский"), ("lyube", "Любэ"),
         ("ruki_vverh", "Руки Вверх"))


# The song's artist as the style step names it -> the star voice trained on many songs, which
# beats the one-song model made from that artist's first cover.
ALIASES = {"viktor_tsoy": "tsoi_hq", "viktor_tsoi": "tsoi_hq", "kino": "tsoi_hq",
           "korol_i_shut": "gorshok_hq", "mihail_gorshenev": "gorshok_hq", "gorshok": "gorshok_hq",
           "modern_talking": "anders_hq", "tomas_anders": "anders_hq", "thomas_anders": "anders_hq",
           "rammstein": "lindemann_hq", "till_lindemann": "lindemann_hq", "til_lindemann": "lindemann_hq",
           "lindemann": "lindemann_hq"}


def best_voice(name: str) -> str:
    """`name`'s star model when one is trained, else `name` itself."""
    star = ALIASES.get(name or "")
    return star if star and model_of(star) else name


def stars() -> list:
    """[(model name, label)] of the star voices trained here."""
    seen, out = set(), []
    for name, label in STARS:
        if label not in seen and model_of(name):
            seen.add(label)
            out.append((name, label))
    return out


def model_of(name: str):
    """(pth, index) of a trained voice, or None. The latest checkpoint, unless best.txt in the
    model's folder names another: the last is not the best (10-09 night run, words heard in the
    mix: Tsoi singing «Du hast» 71 % at 25 epochs, 32 % at 75)."""
    d = os.path.join(APPLIO, "logs", name)
    pths = sorted(glob.glob(os.path.join(d, f"{name}_*e_*s.pth")), key=os.path.getmtime)
    idx = [p for p in glob.glob(os.path.join(d, "*.index")) if "trained" not in os.path.basename(p)]
    try:
        pinned = os.path.join(d, open(os.path.join(d, "best.txt"), encoding="utf-8").read().strip())
        if pinned in pths:
            pths.append(pinned)
    except OSError:
        pass
    return (pths[-1], idx[0]) if pths and idx else None


def _run(ctx, args: list, label: str, timeout: int) -> str:
    env = {**os.environ, "PYTHONUTF8": "1"}
    _, tail = music.run_gpu_worker(ctx, PY, "", {}, label, timeout,
                                   cmd=[PY, os.path.join(APPLIO, "core.py")] + args, env=env, cwd=APPLIO)
    return tail


SEPARATOR = os.path.join(ROOT, "venv_applio", "Scripts", "audio-separator.exe")
LEAD_MODEL = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
DRY_MODEL = "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"


def split_lead(ctx, vocal_wav: str, work: str) -> tuple:
    """A vocal stem -> (the lead voice dry, the backing vocals or None): the karaoke model takes
    the lead off the choir, the dereverb model the hall off the lead. Raises when it can't."""
    if not os.path.isfile(SEPARATOR):
        raise RuntimeError("no audio-separator")
    models = os.path.join(APPLIO, "sep_models")
    src, backing = vocal_wav, None
    for i, (model, keep) in enumerate(((LEAD_MODEL, "(Vocals)"), (DRY_MODEL, "(noreverb)"))):
        out = os.path.join(work, f"clean{i}")
        os.makedirs(out, exist_ok=True)
        music.run_gpu_worker(ctx, PY, "", {}, "RVC clean", 900, env={**os.environ, "PYTHONUTF8": "1"},
                             cmd=[SEPARATOR, src, "-m", model, "--output_dir", out, "--model_file_dir", models,
                                  "--output_format", "WAV"], cwd=APPLIO)
        wavs = glob.glob(os.path.join(out, "*.wav"))
        got = [f for f in wavs if keep in os.path.basename(f)]
        if not got:
            raise RuntimeError(f"{model}: no {keep} stem")
        if i == 0:
            backing = next((f for f in wavs if keep not in os.path.basename(f)), None)
        src = got[0]
    return src, backing


def _clean_vocal(ctx, vocal_wav: str, work: str) -> str:
    """The singer alone and dry: the lead voice off the backing vocals, then the reverb off it.
    10-08 sweep (Whisper likeness of a YuE take sung through, 2 takes x 3 settings): Snowie on
    the bare Demucs stem 0.62, on the cleaned one 0.68 -- the model no longer learns the choir
    and the hall as part of the voice. The bare stem when the separator is missing or fails."""
    try:
        return split_lead(ctx, vocal_wav, work)[0]
    except Exception as exc:
        logger.warning("rvc: vocal cleanup failed (%s), training on the bare stem", exc)
        return vocal_wav


def _pretrain_args() -> list:
    """The Russian Snowie pretrain (fetched once into Applio's custom pretraineds); the stock
    one when it can't be had."""
    d = os.path.join(APPLIO, "rvc", "models", "pretraineds", "custom")
    paths = [os.path.join(d, f) for f in PRETRAIN]
    for f, path in zip(PRETRAIN, paths):
        if os.path.isfile(path) and os.path.getsize(path) > 1e8:
            continue
        try:
            import urllib.request
            os.makedirs(d, exist_ok=True)
            urllib.request.urlretrieve(PRETRAIN_URL + f, path + ".part")
            os.replace(path + ".part", path)
        except Exception as exc:
            logger.warning("rvc: no Snowie pretrain (%s), stock one", exc)
            return []
    return ["--custom-pretrained", "--g-pretrained-path", paths[0], "--d-pretrained-path", paths[1]]


def _save_every(epochs: int) -> int:
    """Applio takes --save-every-epoch 1..100 only (RVC_EPOCHS=120 failed: «120 is not in the
    range»); the largest step that lands on the last epoch."""
    return next(k for k in range(min(epochs, 100), 0, -1) if epochs % k == 0)


def train(ctx, name: str, vocal_wav: str):
    """Train `name` on one vocal stem. Returns (pth, index) or None."""
    data = os.path.join(APPLIO, "datasets", name)
    os.makedirs(data, exist_ok=True)
    shutil.copy(_clean_vocal(ctx, vocal_wav, data + "_clean"), os.path.join(data, "vocal.wav"))
    sr = "40000"
    _run(ctx, ["preprocess", "--model-name", name, "--dataset-path", data, "--sample-rate", sr,
               "--cut-preprocess", "Automatic", "--process-effects"], "RVC prep", 900)
    _run(ctx, ["extract", "--model-name", name, "--f0-method", "rmvpe", "--sample-rate", sr, "--gpu", "0"],
         "RVC extract", 900)
    # Applio binds a random port in 20000..55555 for torch.distributed; Windows reserves
    # 53091..53790 and 50000..50059 here, and on those the run dies in TCPStore (10-08: the
    # Russian bind error crashed its own decoding) yet still prints «trained successfully».
    # A run that left no weights is run again.
    pths = lambda: glob.glob(os.path.join(APPLIO, "logs", name, f"{name}_*e_*s.pth"))
    for _ in range(3):
        _run(ctx, ["train", "--model-name", name, "--sample-rate", sr, "--total-epoch", str(EPOCHS),
                   "--save-every-epoch", str(_save_every(EPOCHS)), "--save-only-latest", "--batch-size", "8",
                   "--gpu", "0", "--pretrained", *_pretrain_args(), "--index-algorithm", "Auto"],
             "RVC train", TRAIN_TIMEOUT)
        if pths():
            break
        logger.warning("rvc: training %s left no weights, again", name)
    _run(ctx, ["index", "--model-name", name], "RVC index", 900)
    got = model_of(name)
    logger.info("rvc: trained %s -> %s", name, got)
    return got


def convert(ctx, name: str, vocal_wav: str, out_wav: str, index_rate: float = 0.0) -> str:
    """`vocal_wav` sung in `name`'s voice. 10-08 Whisper sweep (share of the lyric's lines heard):
    index 0 / protect 0.33 kept 0.52-0.59, index 0.4 / protect 0.5 -- 0.48, 0.75 -- 0.45, -12 -- 0.31.
    The index pulls the stem's mumble in; the timbre is in the model."""
    pth, idx = model_of(name)
    _run(ctx, ["infer", "--input-path", os.path.abspath(vocal_wav), "--output-path", os.path.abspath(out_wav),
               "--pth-path", pth, "--index-path", idx, "--pitch", "0", "--index-rate", f"{index_rate:g}", "--protect", "0.33",
               "--f0-method", "rmvpe", "--volume-envelope", "1", "--export-format", "WAV"], "RVC", 900)
    if not os.path.isfile(out_wav):
        raise RuntimeError("rvc convert gave no file")
    return out_wav
