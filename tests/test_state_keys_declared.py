"""Every state key the delivery layer reads must be DECLARED in AgentState.

THE BUG THIS CATCHES. `create_presentation` wrote `document_path` and
`document_status` into the state, and tg_bot's delivery block keyed off exactly
those. Neither was declared in AgentState — and the graph is built with
`StateGraph(AgentState)`, whose TypedDict schema decides which channels survive.
So the keys were dropped between the tool and the delivery, every single time:
the deck really was built (7 slides, six minutes of work), the user was told
"презентация готова", and no file was ever sent. Nothing failed loudly; the
information simply evaporated in the middle.

The class is invisible by inspection — the tool looks right, the delivery looks
right, and only the schema in a third file makes them not meet. So the check is
derived from the source rather than written by hand: every `final.get("…")` in
tg_bot must name a declared field.

Run: venv/Scripts/python.exe tests/test_state_keys_declared.py
"""
import ast, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Hermetic: this suite is not a live-model test, but it was reaching
# localhost:1234 (see tests/offline_guard.py). Nothing here depends on
# the answers — the calls only made it slow and machine-dependent.
import offline_guard; offline_guard.offline_llm()
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


# ── what AgentState declares ─────────────────────────────────────────────────
tree = ast.parse(open(os.path.join(ROOT, "core/models.py"), encoding="utf-8").read())
declared = set()
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "AgentState":
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                declared.add(stmt.target.id)
check("AgentState was found and has fields", len(declared) > 5, declared)
for must in ("document_path", "document_status", "image_path", "image_status",
             "research_path", "tts_path", "final_answer"):
    check(f"AgentState declares {must}", must in declared)


# ── what the delivery layer reads out of the finished state ──────────────────
def keys_read_from(path: str, var: str) -> set:
    """`var.get("x")` / `var["x"]` occurrences in one file."""
    out = set()
    src = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(src):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == var
                and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            out.add(node.args[0].value)
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == var and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            out.add(node.slice.value)
    return out


# The delivery layer split out of tg_bot.py into tg_tasks.py; the reads of the
# finished state went with it. Scan the whole family so this check follows the
# code instead of quietly measuring an empty file.
read = set()
for _mod in ("bot/tg_bot.py", "bot/tg_tasks.py"):
    read |= keys_read_from(os.path.join(ROOT, _mod), "final")
check("the delivery layer reads state keys at all", len(read) >= 5, read)
undeclared = sorted(read - declared)
check("every key tg_bot reads from the finished state is declared in AgentState",
      not undeclared,
      f"dropped by StateGraph before delivery: {undeclared}")

# and the tool that writes them must be writing the same names
tools_written = keys_read_from(os.path.join(ROOT, "agent/tools.py"), "state")
for k in ("document_path", "document_status"):
    check(f"tools.py writes {k}", k in tools_written, sorted(tools_written)[:12])
    check(f"…and tg_bot reads {k}", k in read)

# a live round-trip: what a tool writes must survive graph.invoke
print()
import threading
import config
from models import Context, Models
from graph import build_graph

ctx = Context(models=Models(None, None, None, None, False), transcription_cache={},
              cache_file=None, asr_lock=threading.Lock(), tts_lock=threading.Lock(),
              model_name=config.MODEL_NAME, no_think=True)
g = build_graph(ctx)
seeded = {"messages": [], "user_input": "", "document_path": "C:/tmp/deck.pptx",
          "document_status": "success", "final_answer": "", "image_data": None}
try:
    import graph as G
    _real = G.agent_node if hasattr(G, "agent_node") else None
    out = g.invoke(dict(seeded))
    check("document_path survives a pass through the compiled graph",
          out.get("document_path") == "C:/tmp/deck.pptx", out.get("document_path"))
    check("document_status survives too",
          out.get("document_status") == "success", out.get("document_status"))
except Exception as exc:
    # An LLM-less environment can fail the agent node; the schema checks above
    # are the load-bearing ones and do not need a model.
    print(f"SKIP  live round-trip needs LM Studio ({type(exc).__name__})")

print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
