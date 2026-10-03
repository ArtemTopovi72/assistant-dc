"""MTP speculative decoding A/B on our own llama.cpp build (LM Studio refuses a
separate MTP head). Same prompt, temperature 0, three runs each; tok/s from the
server's own timings.  venv/Scripts/python bench/mtp_ab.py MODEL.gguf MTP.gguf"""
import json, os, subprocess, sys, time, urllib.request

SERVER = r"C:\llamacpp\cuda-b11177\llama-server.exe"
PORT = 8091
MSG = [{"role": "system", "content": "Ты — Лидия, хускарл из Вайтрана. Отвечай по-русски, живо, 4-6 предложений."},
       {"role": "user", "content": "Расскажи, как ты попала на службу к ярлу и что думаешь о драконах."}]


def post(body):
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json"}), timeout=300)
    return json.loads(r.read())


def run(model, mtp):
    cmd = [SERVER, "-m", model, "-ngl", "99", "-fa", "on", "-c", "8192", "--port", str(PORT),
           "-ctk", "q8_0", "-ctv", "q8_0", "--reasoning-budget", "0"]
    if mtp:
        cmd += ["-md", mtp, "--spec-type", "draft-mtp", "-ngld", "99"] + os.environ.get("MTP_EXTRA", "").split()
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(180):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
            except Exception:
                time.sleep(1)
        post({"messages": MSG, "max_tokens": 16})   # warm-up
        rates, texts = [], []
        for _ in range(3):
            t = time.time()
            r = post({"messages": MSG, "max_tokens": 300, "temperature": 0})
            tm = r.get("timings", {})
            rates.append(tm.get("predicted_per_second") or r["usage"]["completion_tokens"] / (time.time() - t))
            texts.append(r["choices"][0]["message"]["content"])
        return rates, texts, tm
    finally:
        p.kill(); p.wait()


if __name__ == "__main__":
    model, mtp = sys.argv[1], sys.argv[2]
    for name, m in ((("no MTP", None),) if not os.environ.get("MTP_ONLY") else ()) + (("MTP " + os.environ.get("MTP_EXTRA", ""), mtp),):
        rates, texts, tm = run(model, m)
        print(f"{name}: {', '.join('%.1f' % r for r in rates)} tok/s  draft={tm.get('draft_n')} acc={tm.get('draft_n_accepted')}")
        print("   ", texts[0][:160].replace("\n", " "))
