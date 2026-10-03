"""find_content: the universal by-content search over the working folder.

Live 2026-09-14: «найди мост» over a 173-photo DCIM.zip. open_image +
inspect_image look at ONE picture per two calls; the task did not fit in a
turn and the model narrated a plan instead of acting, twice. Contact sheets
+ verification make the whole folder one call; text files are grepped.
"""
import os, sys, tempfile, inspect
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from pathlib import Path
from PIL import Image
import photo_search as P

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

root = Path(tempfile.mkdtemp(prefix="findc_"))
folder = root / "DCIM"; folder.mkdir()
for i in range(30):
    Image.new("RGB", (400, 300), (i * 8, 40, 200 - i * 5)).save(folder / f"P{i:03d}.JPG")
(folder / "notes.txt").write_text("we crossed the old bridge in Mostar", encoding="utf-8")

# a vision stub: on a sheet, cells 2 and 5 "show a bridge"; alone, only P001 and P016 do
calls = []
def vision(path, question, system):
    calls.append(Path(path).name)
    if "sheet" in Path(path).name:
        return "[2, 5]"
    return "YES" if Path(path).name in ("P001.JPG", "P016.JPG", "P004.JPG") else "NO"

res = P.find_in_photos(None, folder, "a bridge", vision=vision, work_dir=root / "sheets")
check("all 30 photos are scanned in 3 sheets", res["scanned"] == 30 and res["sheets"] == 3, res)
check("grid hits become candidates (2 per sheet)", res["candidates"] == 6, res)
check("only verified candidates are matches", [m.name for m in res["matches"]] == ["P001.JPG", "P004.JPG", "P016.JPG"], [m.name for m in res["matches"]])
check("3 sheet calls + 6 verifications, not 30 looks", len(calls) == 9, len(calls))
check("the sheet is a numbered 4x3 grid", Image.open(root / "sheets" / "sheet_000.jpg").size == (P.GRID_COLS * P.CELL, P.GRID_ROWS * P.CELL))
check("odd model output still parses", P._parse_cells("Cells 3 and 7 show it: [3, 7].", 12) == [3, 7])
check("out-of-range cells are dropped", P._parse_cells("[0, 5, 99]", 12) == [5])

# the tool is registered, in the file kit, and points the current picture at the first hit
import tools, tool_retrieval as R, tool_code_handlers as H
check("find_content is a registered tool", any(getattr(t, "name", "") == "find_content" for t in tools.TOOLS))
check("...and rides with the file kit", "find_content" in R._CODE_KIT)
src = inspect.getsource(H._handle_find_content.__wrapped__ if hasattr(H._handle_find_content, "__wrapped__") else H._handle_find_content)
check("the first match becomes the current picture (delivered)", 'state["image_path"] = str(first)' in src and '"image_status"] = "ok"' in src)
check("text files are grepped too", "box.search(" in src)

# the promise guard: a plan with no tool call is not an answer
import graph_finalize as F, graph_personality as G
check("«Начинаю поиск» with no tool call is a promise", F._FILE_PROMISE_RE.search("Сначала я распакую архив. Начинаю поиск."))
check("an actual answer is not", not F._FILE_PROMISE_RE.search("В архиве 173 фото; мост есть на P1012796.JPG."))
check("the guard is wired into the loop", "_FILE_PROMISE_RE.search(draft)" in inspect.getsource(G.personality_node))

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
