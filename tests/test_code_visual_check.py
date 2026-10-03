"""Eyes for run_code: a picture a script drew is looked at before it is sent.

Live 2026-09-18: a flowchart came as one column of boxes with an arrow drawn
through every box and its text; the script ran without error, so the loop
had no way to know. Now the newest picture goes to the vision model with the
request, plainly visible defects come back as a fix request (once per turn),
flowcharts are steered to graphviz (dot is in the image), and an image built
from an older Dockerfile is rebuilt instead of trusted.
"""
import os, sys, io, types, tempfile
os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import code_visual_check as C
import llm

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

# --- the critic's verdict ---------------------------------------------------------
_orig = llm.analyze_image_with_llm
seen = {}
def _fake(ctx, image_path=None, image_bytes=None, user_text="", system_prompt="", **k):
    seen["user"] = user_text; seen["system"] = system_prompt
    return _fake.answer
llm.analyze_image_with_llm = _fake
try:
    _fake.answer = "OK"
    check("OK -> no defects", C.critique(None, "x.png", "flowchart") == "")
    _fake.answer = "OK."
    check("'OK.' -> no defects", C.critique(None, "x.png", "flowchart") == "")
    _fake.answer = "- An arrow runs straight through every box and its text.\n- The NO branches are floating lines."
    d = C.critique(None, "x.png", "нарисуй блок-схему")
    check("defects come back as short lines", d.startswith("An arrow runs") and "NO branches" in d, d)
    check("the request is shown to the judge", "нарисуй блок-схему" in seen["user"])
    check("the judge is told to look for geometric defects only, not taste",
          "THROUGH a box" in seen["system"] and "Do NOT judge style" in seen["system"])
    _fake.answer = ""
    check("an empty answer passes (no vote)", C.critique(None, "x.png", "q") == "")
    def _boom(*a, **k): raise RuntimeError("vision down")
    llm.analyze_image_with_llm = _boom
    check("a failing vision call passes silently", C.critique(None, "x.png", "q") == "")
finally:
    llm.analyze_image_with_llm = _orig
n = C.note_for("An arrow through the boxes.")
check("the note asks for a fix via edit_file and graphviz", "VISUAL CHECK" in n and "edit_file" in n and "graphviz" in n)

# --- wired into run_code's delivery, once per turn -----------------------------------------
import tool_code_handlers as H
from PIL import Image
root = Path(tempfile.mkdtemp())
box = types.SimpleNamespace(root=root)
def _write(name):
    Image.new("RGB", (64, 64), (255, 255, 255)).save(root / name)
_orig_rec = H._record; H._record = lambda *a, **k: None
_orig_crit = C.critique
C.critique = lambda ctx, path, req: "An arrow through the boxes." if "flow" in req else ""
try:
    state = {"user_input": "draw a flowchart"}
    before = H._pictures_in(box); _write("chart.png")
    out = H._deliver_new_pictures(box, state, before, types.SimpleNamespace())
    check("the delivered picture carries the visual-check defects", "VISUAL CHECK" in out and "chart.png" in out, out)
    check("the picture is still the current one (the user sees it if the model stops)", state.get("image_path", "").endswith("chart.png"))
    before = H._pictures_in(box); _write("chart2.png")
    out2 = H._deliver_new_pictures(box, state, before, types.SimpleNamespace())
    check("the second picture in the same turn is not judged again", "VISUAL CHECK" not in out2 and "chart2.png" in out2, out2)
    state = {"user_input": "make a collage"}
    before = H._pictures_in(box); _write("c.png")
    check("a clean picture gets no note", "VISUAL CHECK" not in H._deliver_new_pictures(box, state, before, types.SimpleNamespace()))
finally:
    H._record = _orig_rec; C.critique = _orig_crit

# --- graphviz is offered and the image keeps up with the Dockerfile ----------------------
import tool_descriptions as TD
check("run_code says: graphs with graphviz, never hand-placed boxes",
      "graphviz.Digraph" in TD._RUN_CODE_DESC and "NEVER by hand-placing boxes" in TD._RUN_CODE_DESC)
df = open("docker/sandbox/Dockerfile", encoding="utf-8").read()
check("the house image installs the dot binary and the python bindings",
      "graphviz" in df.split("apt-get install")[1].split("&&")[0] and "pip install" in df and " graphviz " in df.replace("\\\n", " ") + " ")
import code_runner as R
check("the Dockerfile digest is stable and short", R.dockerfile_digest() == R.dockerfile_digest() and len(R.dockerfile_digest()) == 12)
src = open(R.__file__, encoding="utf-8").read()
check("ensure_image rebuilds a stale image and stamps the digest as a label",
      "if image_present() and not stale:" in src and '"--label", f"{_DOCKERFILE_LABEL}={dockerfile_digest()}"' in src)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
