"""List file paths and function names in memory notes that no longer exist in the repo.

    venv/Scripts/python scripts/check_memory_refs.py [memory_dir]

A note that names voice/audio.py:synth_x() after a rename sends the next session
looking for code that is gone. Paths are checked on disk, `name()` against
any identifier in the repo's .py/.json (agent tool names live in strings).
"""
import os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEM = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / ".claude/projects/C--Users-Artem-f5-tts-project/memory"
SKIP = {"venv", ".git", "runtime", "outputs", "node_modules", ".claude", "__pycache__"}
PATH_RE = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w-]+\.(?:py|json|ps1|bat|txt|yaml|toml))(?::\d+)?\b")
FUNC_RE = re.compile(r"`(?:[\w.]+\.)?([A-Za-z_]\w{2,})\(\)?`?")


def repo_index():
    files, names = set(), set()
    for d, dirs, fs in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x not in SKIP]
        for f in fs:
            rel = os.path.relpath(os.path.join(d, f), ROOT).replace("\\", "/")
            files.add(rel)
            if f.endswith((".py", ".json")):
                try:
                    src = open(os.path.join(d, f), encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                # any identifier, not just def/class: agent tool names live in strings
                names.update(re.findall(r"\w+", src))
    return files, names


def check(text, files, names):
    """Return the referenced paths and function names that do not exist."""
    missing = []
    # only paths rooted in one of our top-level dirs: bare names and other
    # projects' paths (Mantella, scratchpad, ComfyUI) are not ours to check
    tops = {f.split("/", 1)[0] for f in files if "/" in f}
    for p in set(PATH_RE.findall(text)):
        if "/" in p and p.split("/", 1)[0] in tops and p not in files:
            missing.append(p)
    # snake_case only: camelCase is Qt/JS API, not our code
    for n in set(FUNC_RE.findall(text)):
        if "_" in n.strip("_") and n.islower() and n not in names:
            missing.append(n + "()")
    return sorted(missing)


def main():
    files, names = repo_index()
    bad = 0
    for md in sorted(MEM.glob("*.md")):
        if md.name == "MEMORY.md":
            continue
        miss = check(md.read_text(encoding="utf-8", errors="ignore"), files, names)
        if miss:
            bad += 1
            print(f"{md.name}: {', '.join(miss)}")
    print(f"{bad} notes with stale refs")


if __name__ == "__main__":
    if os.environ.get("SELFTEST"):
        assert check("see voice/audio.py and `gone_func()`", {"voice/audio.py"}, {"x"}) == ["gone_func()"]
        assert check("voice/nope.py", {"voice/audio.py"}, set()) == ["voice/nope.py"]
        assert check("`audio.py` `synth_x()` `setParent()` other/x.py", {"voice/audio.py"}, {"synth_x"}) == []
        print("ok")
    else:
        main()
