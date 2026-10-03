"""Where a Music3 render spends its time: per-node wall clock from ComfyUI's
websocket, for cold and warm runs and several seeds.

    venv/Scripts/python bench/night_music.py [--runs 4] [--seconds 60] [--preset X]

Writes runtime/night_music/timings.json and the rendered files.
"""
import argparse, json, os, sys, time, uuid
import requests, websocket

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import music as M  # noqa: E402

COMFY = "127.0.0.1:8000"
OUT = os.path.join(ROOT, "runtime", "night_music")

LYRICS = """[verse]
Утро тихо входит в город
Солнце греет старый двор
Я иду навстречу ветру
Продолжая разговор
[chorus]
Лети, лети над крышами
Пусть песню все услышат
Лети, лети над крышами
Сегодня мир наш выше"""
STYLE = "upbeat russian pop, female vocal, acoustic guitar, bright drums, 120 bpm"


def free_mb():
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def run_one(wf):
    cid = uuid.uuid4().hex
    ws = websocket.create_connection(f"ws://{COMFY}/ws?clientId={cid}", timeout=3600)
    pid = requests.post(f"http://{COMFY}/prompt", json={"prompt": wf, "client_id": cid}, timeout=60).json()["prompt_id"]
    t0 = time.time()
    marks, cur, steps = [], None, []
    while True:
        m = ws.recv()
        if not isinstance(m, str):
            continue
        d = json.loads(m)
        if d.get("type") == "executing" and d["data"].get("prompt_id") == pid:
            node = d["data"].get("node")
            now = time.time() - t0
            if cur is not None:
                marks.append((cur, round(now - cur_t, 2)))
            if node is None:
                break
            cur, cur_t = node, now
        elif d.get("type") == "progress" and d["data"].get("prompt_id") == pid:
            steps.append((d["data"]["value"], round(time.time() - t0, 2)))
        elif d.get("type") == "execution_error":
            raise RuntimeError(json.dumps(d)[:800])
    ws.close()
    return round(time.time() - t0, 1), marks, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--preset", default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = []
    for i in range(a.runs):
        wf = M.build_workflow(LYRICS, STYLE, duration_s=a.seconds, seed=1000 + i, preset=a.preset)
        titles = {k: v["class_type"] for k, v in wf.items()}
        vram = free_mb()
        wall, marks, steps = run_one(wf)
        row = {"run": i, "seed": 1000 + i, "wall": wall, "vram_before_mb": vram,
               "nodes": [(titles.get(n, n), s) for n, s in marks],
               "dit_steps": len(steps), "first_step_at": steps[0][1] if steps else None}
        res.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        json.dump(res, open(os.path.join(OUT, "timings.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
