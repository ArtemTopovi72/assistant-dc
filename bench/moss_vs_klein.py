"""MOSS-Image-SFT against Klein on the layout scenes Klein was chosen on (klein_hard + klein_vs_ideogram):
the same planner caption, Klein as the app renders it (caption_to_prompt, KLEIN_STEPS) and MOSS as its
layout JSON (the ComfyUI MOSSImageLayoutPrompt conversion). 1024x1024, two seeds each.

    venv/Scripts/python.exe bench/moss_vs_klein.py [--hard] [scene ...]
--hard runs bench/moss_hard_scenes.py (three seeds) into tmp/moss_vs_klein_hard/.
Scores each lettering element by OCR (exact after case/space folding) and times each render.
Writes tmp/moss_vs_klein/<scene>_<engine>_<seed>.png, <scene>.jpg and results.json.
"""
import difflib, json, math, os, re, sys, time
import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, d) for d in ("bench", "imaging", "core", "agent")] + [ROOT]
os.chdir(ROOT); sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import klein  # noqa: E402
import klein_hard as KH  # noqa: E402
import klein_vs_ideogram as KV  # noqa: E402

API = "http://127.0.0.1:8000"
OUT = os.path.join("tmp", "moss_vs_klein")
SIZE = 1024
SEEDS = (101, 202)
MOSS_UNET = os.getenv("MOSS_UNET", "moss_image_sft_int8_convrot.safetensors")
MOSS_CLIP = os.getenv("MOSS_CLIP", "qwen3.5_9b_int8_convrot.safetensors")
MOSS_STEPS, MOSS_CFG = int(os.getenv("MOSS_STEPS", "25")), float(os.getenv("MOSS_CFG", "2.5"))

SCENES = dict(KH.SCENES)
SCENES.update({s["name"]: KV.ig_caption(s) for s in KV.SCENES})


def texts(caption):
    return [e["text"] for e in caption["compositional_deconstruction"]["elements"] if e.get("text")]


def moss_layout(caption, w=SIZE, h=SIZE) -> str:
    """The MOSSImageLayoutPrompt conversion: 0-1000 [y0,x0,y1,x1] -> pixels around the canvas centre."""
    summary = re.sub(r"\s*<[^>]+>", "", klein.caption_to_prompt(caption).split(' [{"id"')[0])   # Klein's ids mean nothing here
    items = []
    for el in caption["compositional_deconstruction"]["elements"]:
        item = {"label": "text" if el.get("text") else "object", "caption": el.get("desc", "")}
        if el.get("text"):
            item["text"] = el["text"]
        y0, x0, y1, x1 = el["bbox"]
        item["bbox"] = [math.floor(y0 * h / 1000) - h // 2, math.floor(x0 * w / 1000) - w // 2,
                        math.ceil(y1 * h / 1000) - h // 2, math.ceil(x1 * w / 1000) - w // 2]
        items.append(item)
    return json.dumps({"summary": {"caption": summary}, "items": items, "bbox_coord_space": "abs_centralized",
                       "bbox_format": "yxyx"}, ensure_ascii=False, separators=(",", ":"))


def moss_wf(prompt, seed):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MOSS_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": MOSS_CLIP, "type": "moss_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae_small.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"}},
        "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": SIZE, "height": SIZE, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
                                                   "latent_image": ["6", 0], "seed": seed, "steps": MOSS_STEPS, "cfg": MOSS_CFG,
                                                   "sampler_name": "er_sde", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "moss_bench", "images": ["8", 0]}},
    }


def render(wf, save_id):
    t = time.time()
    r = requests.post(f"{API}/prompt", json={"prompt": wf}, timeout=60).json()
    if "prompt_id" not in r:
        raise RuntimeError(json.dumps(r)[:600])
    pid = r["prompt_id"]
    while True:
        h = requests.get(f"{API}/history/{pid}", timeout=30).json().get(pid)
        if h and h["status"].get("completed"):
            break
        if h and h["status"].get("status_str") == "error":
            raise RuntimeError(json.dumps(h["status"])[-800:])
        time.sleep(0.5)
    im = h["outputs"][save_id]["images"][0]
    data = requests.get(f"{API}/view", params={"filename": im["filename"], "subfolder": im["subfolder"], "type": "output"},
                        timeout=60).content
    return data, time.time() - t


def _fold(s):
    return re.sub(r"[\W_]+", "", (s or "").upper().replace("Ё", "Е"))


def text_score(path, wanted):
    """Per lettering element: the best OCR similarity (1.0 = exact after folding)."""
    import ocr_reader
    got = ocr_reader.read_pieces(path) or []
    got = [_fold(g if isinstance(g, str) else g[0] if isinstance(g, (list, tuple)) else str(g)) for g in got]
    joined = "".join(got)
    out = {}
    for t in wanted:
        f = _fold(t)
        best = max([difflib.SequenceMatcher(None, f, g).ratio() for g in got] + [1.0 if f and f in joined else 0.0] + [0.0])
        out[t] = round(best, 2)
    return out


def sheet(name, paths):
    f = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 20)
    H = 420
    tiles = [(lab, Image.open(p).convert("RGB").resize((H, H))) for lab, p in paths]
    s = Image.new("RGB", (len(tiles) * (H + 8), H + 30), "white")
    for i, (lab, t) in enumerate(tiles):
        s.paste(t, (i * (H + 8), 30)); ImageDraw.Draw(s).text((i * (H + 8) + 4, 5), lab, fill="black", font=f)
    s.save(os.path.join(OUT, f"{name}.jpg"), quality=88)


if __name__ == "__main__":
    if "--hard" in sys.argv:
        import moss_hard_scenes
        sys.argv.remove("--hard")
        SCENES, OUT, SEEDS = moss_hard_scenes.SCENES, OUT + "_hard", (101, 202, 303)
    os.makedirs(OUT, exist_ok=True)
    rpath = os.path.join(OUT, "results.json")
    res = json.load(open(rpath, encoding="utf-8")) if os.path.exists(rpath) else {}
    for name in sys.argv[1:] or SCENES:
        cap = SCENES[name]
        tiles = []
        for eng in ("klein", "moss"):
            for seed in SEEDS:
                p = os.path.join(OUT, f"{name}_{eng}_{seed}.png")
                if eng == "klein":
                    data, dt = render(klein.t2i_workflow(klein.caption_to_prompt(cap), SIZE, SIZE, seed), klein.N_SAVE)
                else:
                    data, dt = render(moss_wf(moss_layout(cap), seed), "9")
                open(p, "wb").write(data)
                sc = text_score(p, texts(cap))
                res[f"{name}/{eng}/{seed}"] = {"s": round(dt, 1), "text": sc}
                json.dump(res, open(rpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                tiles.append((f"{eng} #{seed} {dt:.0f}s", p))
                print(name, eng, seed, f"{dt:.1f}s", sc, flush=True)
        sheet(name, tiles)
