"""🧪 Use the RUNNING bot as an ordinary user (bot/debug_user.py): no restart, the same
queue as everyone. Messages go to runtime/debug_user/inbox, answers come from outbox.jsonl.

  python bench/debug_user.py say "привет"
  python bench/debug_user.py album a.jpg b.jpg --caption "замени четвёртого на губку"
  python bench/debug_user.py photo a.jpg --caption "..."
  python bench/debug_user.py press swap_person:ab12cd34
  python bench/debug_user.py wait --until sendPhoto --timeout 1500
  python bench/debug_user.py shot          # the live window with the Debug chat tab
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot"))
import debug_user as D  # noqa: E402

from debug_user import OUTBOX, say, photo, album, press, outbox  # noqa: E402

USER = 0


def wait(since: float, until: str = "", timeout: float = 600, quiet: float = 60) -> list:
    """Answers since `since`, until a `until` method arrives (then `quiet` s more of silence)."""
    end, seen, last = time.time() + timeout, 0, time.time()
    hit = False
    while time.time() < end:
        evs = outbox(since, USER)
        if len(evs) != seen:
            for e in evs[seen:]:
                p = e["payload"]
                print(f"  <- {e['method']:<18} {(p.get('text') or p.get('caption') or '')[:300]!r} "
                      f"{' '.join(e['files'])}", flush=True)
                hit = hit or (until and e["method"] == until)
            seen, last = len(evs), time.time()
        if hit and time.time() - last > min(quiet, 15):
            break
        if not until and seen and time.time() - last > quiet:
            break
        time.sleep(1)
    return outbox(since, USER)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["say", "photo", "album", "press", "wait", "shot"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--caption", default="")
    ap.add_argument("--until", default="")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--user", type=int, default=0, help="debug user 0..4 (several = parallel users)")
    a = ap.parse_args()
    USER = a.user
    if not D.enabled():
        sys.exit("debug user is off: create runtime/debug_user/enabled and keep the app running")
    t0 = time.time()
    if a.cmd == "shot":                  # the live window, Debug chat tab in front
        open(D.SHOT + ".req", "w").write("1")
        while os.path.exists(D.SHOT + ".req") or not os.path.exists(D.SHOT) or os.path.getmtime(D.SHOT) < t0:
            if time.time() - t0 > 30:
                sys.exit("no shot: is the GUI running?")
            time.sleep(0.5)
        sys.exit(print(D.SHOT))
    if a.cmd == "say":
        say(" ".join(a.args), a.user)
    elif a.cmd == "photo":
        photo(a.args[0], a.caption, "", a.user)
    elif a.cmd == "album":
        album(a.args, a.caption, a.user)
    elif a.cmd == "press":
        press(a.args[0], a.user)
    wait(t0 if a.cmd != "wait" else time.time(), a.until, a.timeout)
