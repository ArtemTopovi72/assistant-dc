"""Every value .env.example sets must equal the code's own default.

The setup script copies .env.example to .env, so a stale example value is not
documentation drift: it silently overrides the code. Three had already undone
fixes (TG_ETA_RESEARCH_SEC=900 pinned the research ETA the code now measures,
TG_REPORT_FILE_CHARS=6000 hid short reports in the chat again, and
DR_BRIEF_TOKEN_CEILING=2048 cut the brief budget to a third). A knob whose
example value differs on purpose belongs commented out, with a note.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"runtime", "tests", "bench", "venv", ".venv", ".git", "node_modules"}
# Defaults given as a module constant rather than a literal.
CONSTANT_DEFAULTS = {"SANDBOX_IMAGE"}

_READ = re.compile(
    r"""(?:env_int|env_float|env_bool|_env_int|_env_float|_env_bool|_cfg_int|_cfg_float|"""
    r"""_cfg_str|_cfg_bool|getenv|environ\.get)\(\s*["']([A-Z0-9_]+)["']\s*,\s*([^)\n]+?)\s*\)""")


def _norm(v):
    v = v.strip().strip("\"'")
    try:
        return float(v)
    except ValueError:
        return v.lower()


def _example_values():
    out = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Z0-9_]+)=(.*)$", line.strip())
        if m:
            out[m.group(1)] = re.sub(r"\s+#.*$", "", m.group(2)).strip()   # dotenv inline comment
    return out


def _code_defaults():
    found = {}
    for f in ROOT.rglob("*.py"):
        if SKIP_DIRS & set(f.relative_to(ROOT).parts):
            continue
        for m in _READ.finditer(f.read_text(encoding="utf-8", errors="replace")):
            found.setdefault(m.group(1), set()).add(m.group(2))
    return found


def test_example_values_match_code_defaults():
    defaults = _code_defaults()
    bad = []
    for key, value in _example_values().items():
        if key in CONSTANT_DEFAULTS or key not in defaults or value == "":
            continue
        if _norm(value) not in {_norm(d) for d in defaults[key]}:
            bad.append(f"{key}={value} but the code defaults to {sorted(defaults[key])}")
    assert not bad, "\n".join(bad)
