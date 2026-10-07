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


def model_of(name: str):
    """(pth, index) of a trained voice, or None."""
    d = os.path.join(APPLIO, "logs", name)
    pths = sorted(glob.glob(os.path.join(d, f"{name}_*e_*s.pth")), key=os.path.getmtime)
    idx = [p for p in glob.glob(os.path.join(d, "*.index")) if "trained" not in os.path.basename(p)]
    return (pths[-1], idx[0]) if pths and idx else None


def _run(ctx, args: list, label: str, timeout: int) -> str:
    env = {**os.environ, "PYTHONUTF8": "1"}
    _, tail = music.run_gpu_worker(ctx, PY, "", {}, label, timeout,
                                   cmd=[PY, os.path.join(APPLIO, "core.py")] + args, env=env, cwd=APPLIO)
    return tail


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


def train(ctx, name: str, vocal_wav: str):
    """Train `name` on one vocal stem. Returns (pth, index) or None."""
    data = os.path.join(APPLIO, "datasets", name)
    os.makedirs(data, exist_ok=True)
    shutil.copy(vocal_wav, os.path.join(data, "vocal.wav"))
    sr = "40000"
    _run(ctx, ["preprocess", "--model-name", name, "--dataset-path", data, "--sample-rate", sr,
               "--cut-preprocess", "Automatic", "--process-effects"], "RVC prep", 900)
    _run(ctx, ["extract", "--model-name", name, "--f0-method", "rmvpe", "--sample-rate", sr, "--gpu", "0"],
         "RVC extract", 900)
    _run(ctx, ["train", "--model-name", name, "--sample-rate", sr, "--total-epoch", str(EPOCHS),
               "--save-every-epoch", str(EPOCHS), "--save-only-latest", "--batch-size", "8", "--gpu", "0",
               "--pretrained", *_pretrain_args(), "--index-algorithm", "Auto"], "RVC train", TRAIN_TIMEOUT)
    _run(ctx, ["index", "--model-name", name], "RVC index", 900)
    got = model_of(name)
    logger.info("rvc: trained %s -> %s", name, got)
    return got


def convert(ctx, name: str, vocal_wav: str, out_wav: str) -> str:
    """`vocal_wav` sung in `name`'s voice. 10-08 Whisper sweep (share of the lyric's lines heard):
    index 0 / protect 0.33 kept 0.52-0.59, index 0.4 / protect 0.5 -- 0.48, 0.75 -- 0.45, -12 -- 0.31.
    The index pulls the stem's mumble in; the timbre is in the model."""
    pth, idx = model_of(name)
    _run(ctx, ["infer", "--input-path", os.path.abspath(vocal_wav), "--output-path", os.path.abspath(out_wav),
               "--pth-path", pth, "--index-path", idx, "--pitch", "0", "--index-rate", "0", "--protect", "0.33",
               "--f0-method", "rmvpe", "--volume-envelope", "1", "--export-format", "WAV"], "RVC", 900)
    if not os.path.isfile(out_wav):
        raise RuntimeError("rvc convert gave no file")
    return out_wav
