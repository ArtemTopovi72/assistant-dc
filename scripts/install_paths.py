"""Put the source folders on this environment's import path (a .pth file in site-packages).

Modules keep flat names (`import tg_bot`, `import config`) while living in folders;
the .pth file is the same mechanism `pip install -e` uses, except its one line PREPENDS
the folders: a plain path line lands after site-packages, where any installed package with
the same name (pip's `lmstudio` vs agent/lmstudio.py) silently wins. Idempotent: run it once
after creating the venv, and again only if the checkout moves.
"""
import os
import site
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDERS = ("core", "agent", "bot", "gui", "imaging", "media", "voice", "research",
           "knowledge", "services", "sandbox", "mantella")


def source_dirs():
    return [os.path.join(ROOT, d) for d in FOLDERS if os.path.isdir(os.path.join(ROOT, d))]


def ensure(write_venv: bool = True):
    """Current process gets the folders now; the venv gets them for every later process."""
    for d in reversed(source_dirs()):
        if d not in sys.path:
            sys.path.insert(0, d)
    target = os.path.join(site.getsitepackages()[-1], "assistant_dc.pth")
    # Importing launch_all (tests do, from any checkout) must not repoint the
    # shared venv at itself -- 2026-10-01 a clone's suite run did, and the main
    # checkout's tests then imported the clone's stale code.
    if not write_venv:
        return target
    want = f"import sys; sys.path[0:0] = [d for d in {source_dirs()!r} if d not in sys.path]\n"
    try:
        if not os.path.exists(target) or open(target, encoding="utf-8").read() != want:
            with open(target, "w", encoding="utf-8") as f:
                f.write(want)
    except OSError:
        pass
    return target


if __name__ == "__main__":
    print("wrote", ensure())
