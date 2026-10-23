"""Fibo 2.0 Edit Turbo against our Klein on the same edits, by eye.

    venv/Scripts/python.exe bench/fibo_vs_klein.py [case ...]
Fibo runs on a separate ComfyUI (upstream master, port 8010; the live 0.41.0 has no Fibo nodes).
Klein's side is what is already on disk: the raw edits of tmp/lockbench (annotator prompt, no box
lock) and the shipped whole-frame edits of tmp/edit_path_ab (with the lock). Fibo gets the plain
instruction -- Bria's VGL rewrite needs Gemini -- at its card settings: 4 steps, cfg 1.
Writes tmp/fibo_vs_klein/<case>.jpg (source | Klein | Fibo) and times.json.
"""
import json, os, sys, time, uuid
import requests
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.stdout.reconfigure(encoding="utf-8", errors="replace")
API = os.environ.get("FIBO_COMFY", "http://127.0.0.1:8010")
UNET = os.environ.get("FIBO_UNET", "fibo2_edit_turbo_bf16.safetensors")
OUT = os.path.join("tmp", "fibo_vs_klein")

CASES = {   # name: (source, instruction, klein result)
    "aviator": ("tmp/judge_photos/owner_profile.jpg", "wearing aviator sunglasses with the Russian text 'Идите нахуй, я деловой' written at the bottom", "tmp/lockbench/aviator_raw.png"),
    "sunglasses_woman": ("tmp/judge_photos/woman1.jpg", "wearing stylish black sunglasses covering her eyes", "tmp/lockbench/sunglasses_woman_raw.png"),
    "hat_squirrel": ("tmp/judge_photos/hat1.jpg", "add a small hat on the squirrel's head", "tmp/lockbench/hat_squirrel_raw.png"),
    "tie": ("tmp/judge_photos/glasses0.jpg", "change the tie to a solid red tie", "tmp/lockbench/tie_raw.png"),
    "watch_rm": ("tmp/judge_photos/watch1.jpg", "remove the watch from the wrist, bare skin", "tmp/lockbench/watch_rm_raw.png"),
    "shirt": ("tmp/judge_photos/glasses1.jpg", "change the shirt to a solid blue shirt without patterns", "tmp/lockbench/shirt_raw.png"),
    "hair": ("tmp/judge_photos/woman0.jpg", "change her hair colour to ginger", "tmp/lockbench/hair_raw.png"),
    "nose": ("tmp/judge_photos/beard0.jpg", "make his nose much larger", "tmp/lockbench/nose_raw.png"),
    "helmet": ("runtime/comfy/ideogram_00184_.png", "change the space helmet to a shiny, polished gold space helmet", "tmp/edit_path_ab/helmet_whole.png"),
    "armor_girl": ("runtime/_working_input_1792194794750.png", "change the armor of the girl on the right to red armor", "tmp/edit_path_ab/armor_girl_whole.png"),
    "armor_man": ("runtime/upscaled_faceguard_1792207590192.png", "change the man with a shovel to the man wearing red and black plate armor", "tmp/edit_path_ab/armor_man_whole.png"),
    "hand_lenin": ("runtime/comfy/klein_edit_00100_.png", "change the left hand of Lenin to a clean five-fingered hand punching a demon in the face", "tmp/edit_path_ab/hand_lenin_whole.png"),
    "hand_baker": ("runtime/comfy/transfer_00003_.png", "change the right hand of the baker to a clean five-fingered human hand holding bread correctly without any fusion", "tmp/edit_path_ab/hand_baker_whole.png"),
}


def graph(image_name: str, instruction: str, seed: int) -> dict:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b_int8_convrot.safetensors", "type": "fibo2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "5": {"class_type": "TextEncodeFibo2", "inputs": {"clip": ["2", 0], "prompt": instruction, "negative_prompt": "",
                                                          "vae": ["3", 0], "images.image_1": ["4", 0],
                                                          "width": 0, "height": 0, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["5", 0], "negative": ["5", 1],
                                                   "latent_image": ["5", 2], "seed": seed, "steps": 4, "cfg": 1.0,
                                                   "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"filename_prefix": "fibo_bench", "images": ["7", 0]}},
    }


def render(src: str, instruction: str, seed: int = 1234567) -> tuple:
    with open(src, "rb") as f:
        name = requests.post(f"{API}/upload/image", files={"image": (f"fibo_{uuid.uuid4().hex[:8]}{os.path.splitext(src)[1]}", f)},
                             timeout=60).json()["name"]
    t0 = time.time()
    r = requests.post(f"{API}/prompt", json={"prompt": graph(name, instruction, seed)}, timeout=60).json()
    if "prompt_id" not in r:
        raise RuntimeError(json.dumps(r)[:800])
    pid = r["prompt_id"]
    while True:
        h = requests.get(f"{API}/history/{pid}", timeout=30).json().get(pid)
        if h and h.get("status", {}).get("completed"):
            break
        if h and h.get("status", {}).get("status_str") == "error":
            raise RuntimeError(json.dumps(h["status"])[:1500])
        time.sleep(1)
    img = h["outputs"]["8"]["images"][0]
    data = requests.get(f"{API}/view", params={"filename": img["filename"], "subfolder": img["subfolder"], "type": "output"}, timeout=60).content
    return data, time.time() - t0


def sheet(name: str, paths: list):
    H = 480
    tiles = [(lab, Image.open(p).convert("RGB")) for lab, p in paths]
    tiles = [(lab, im.resize((max(1, int(im.width * H / im.height)), H))) for lab, im in tiles]
    s = Image.new("RGB", (sum(t.width for _, t in tiles) + 10 * len(tiles), H + 26), "white")
    x = 0
    for lab, t in tiles:
        s.paste(t, (x, 26)); ImageDraw.Draw(s).text((x + 4, 6), f"{name}: {lab}", fill="black")
        x += t.width + 10
    s.save(os.path.join(OUT, f"{name}.jpg"), quality=88)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    tpath = os.path.join(OUT, "times.json")
    times = json.load(open(tpath)) if os.path.exists(tpath) else {}
    for n in sys.argv[1:] or CASES:
        src, instr, klein = CASES[n]
        data, dt = render(src, instr)
        fibo = os.path.join(OUT, f"{n}_fibo.png")
        open(fibo, "wb").write(data)
        times[n] = round(dt, 1)
        json.dump(times, open(tpath, "w"), indent=1)
        sheet(n, [("source", src), ("klein", klein), ("fibo", fibo)])
        print(n, f"{dt:.1f}s", flush=True)
