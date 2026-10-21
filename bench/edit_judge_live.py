"""Stock photos + ordinary edit requests through the debug chat: did the edit land, and what did
the bot SAY about it? The owner's complaint (2026-10-21): the judge keeps saying an edit failed
(«увеличь нос», «убери часы») -- an edit that is on the picture must not be reported as failed.

    venv/Scripts/python.exe bench/edit_judge_live.py [case ...]
Photos: tmp/judge_photos (bench fetch from Pexels/Unsplash). Results: tmp/judge_photos/results.jsonl
"""
import json, os, re, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import debug_user as D

P = os.path.join(ROOT, "tmp", "judge_photos")
# name: (photo, button, words) -- the 🧽 Убрать / 🎨 Правка buttons under the photo, as a user does it
CASES = {
    "watch_off":  ("watch1.jpg", "remove_object", "часы"),
    "watch_off2": ("watch0.jpg", "remove_object", "часы с руки"),
    "glasses":    ("glasses0.jpg", "remove_object", "очки"),
    "beard_rm":   ("beard1.jpg", "remove_object", "бороду"),
    "nose":       ("beard0.jpg", "edit", "сделай ему нос побольше"),
    "tie":        ("glasses0.jpg", "edit", "поменяй галстук на красный"),
    "shirt":      ("glasses1.jpg", "edit", "сделай рубашку однотонной синей"),
    "hair":       ("woman0.jpg", "edit", "перекрась волосы в рыжий"),
    "sunglasses": ("woman1.jpg", "edit", "надень ей солнечные очки"),
    "squirrel":   ("hat1.jpg", "edit", "надень белке маленькую шляпу"),
}
LOG = os.path.join(ROOT, "assistant_app.log")


def _log_size() -> int:
    return os.path.getsize(LOG)


def _tail(since_size: int) -> str:
    with open(LOG, "rb") as fh:
        fh.seek(since_size)
        return fh.read().decode("utf-8", "replace")


def _idle() -> bool:
    """No debug-chat task started without its END."""
    with open(LOG, "rb") as fh:
        fh.seek(max(0, os.path.getsize(LOG) - 2_000_000))
        t = fh.read().decode("utf-8", "replace")
    started = re.findall(r"Task (\w+) START chat=777000001", t)
    return all(f"Task {tid} END" in t for tid in started[-5:])


def _picture_id(since: float) -> str:
    for e in reversed(D.outbox(since, 0)):
        rm = e["payload"].get("reply_markup") or ""
        if '"edit:' in rm:
            return rm.split('"edit:', 1)[1].split('"', 1)[0]
    return ""


def _wait_end(size: int, timeout: float) -> None:
    """The task this message started has ended and the debug chat is idle."""
    t = time.time()
    while time.time() - t < timeout:
        m = re.search(r"Task (\w+) START chat=777000001", _tail(size))
        if m and f"Task {m.group(1)} END" in _tail(size) and _idle():
            break
        time.sleep(3)
    time.sleep(4)


def run(name: str, timeout: float = 900) -> dict:
    img, button, text = CASES[name]
    while not _idle():
        time.sleep(5)
    t_photo, size = time.time(), _log_size()
    D.photo(os.path.join(P, img))
    # a photo seen before gets its buttons with no task at all: wait for the buttons
    t = time.time()
    while time.time() - t < timeout and not _picture_id(t_photo):
        time.sleep(2)
    while not _idle():
        time.sleep(3)
    pid = _picture_id(t_photo)
    if not pid:
        print(name, "no keyboard under the photo", flush=True)
        return {}
    D.press(f"{button}:{pid}")
    time.sleep(4)
    t0, size = time.time(), _log_size()
    D.say(text)
    _wait_end(size, timeout)
    evs = D.outbox(t0, 0)
    said = [e["payload"].get("text", "") for e in evs if e["method"] == "sendMessage"
            and not e["payload"].get("text", "").startswith(("⚙", "👁", "⏳", "🎨", "✏"))]
    photos = [f for e in evs if e["method"] == "sendPhoto" for f in e["files"]]
    row = {"case": name, "img": img, "button": button, "ask": text, "said": said[-2:], "photos": photos,
           "sec": round(time.time() - t0)}
    with open(os.path.join(P, "results.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False)[:600], flush=True)
    return row


if __name__ == "__main__":
    for c in sys.argv[1:] or list(CASES):
        run(c)
