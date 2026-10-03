"""A word to the user BEFORE a long tool runs (interim.py).

The user asked for a reaction in real time, not only when the work is done
(2026-09-18). The model's own pre-tool sentence goes out when it is clean,
otherwise a one-line note naming the work; one per turn; long tools only.
"""
import os, sys, json, threading
os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interim as I

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

tc = lambda n: {"id": "x", "type": "function", "function": {"name": n, "arguments": "{}"}}

# --- which rounds deserve a note --------------------------------------------
check("a drawing round is long", I.long_tool_in([tc("generate_image")]) == "generate_image")
check("a research round is long", I.long_tool_in([tc("deep_research")]) == "deep_research")
check("a calculator round is not", I.long_tool_in([tc("calculate"), tc("get_weather")]) is None)
check("a malformed call does not crash it", I.long_tool_in([{"id": 1}, tc("run_code")]) == "run_code")

# --- the model's own words ----------------------------------------------------
check("a clean sentence is kept", I.clean_note("Сейчас нарисую кота в шляпе.") == "Сейчас нарисую кота в шляпе.")
check("only the first sentence", I.clean_note("Понял. Сначала найду фото, потом соберу коллаж, потом…") == "Понял.")
check("a tool-call leak is not sent", I.clean_note('<tool_call>{"name": "generate_image"}') == "")
check("a bracketed control frame is not sent", I.clean_note("[generate] a cat") == "")
check("a colon-ending plan header is not sent", I.clean_note("План действий:") == "")
check("a wall of text is not sent", I.clean_note("слово " * 60) == "")

# --- composition ---------------------------------------------------------------
check("own words win, with the icon", I.compose("generate_image", "Рисую кота!", "ru") == "🎨 Рисую кота!")
check("no words -> the RU note", I.compose("generate_image", "", "ru") == "🎨 Понял, рисую — это займёт пару минут.")
check("no words -> the EN note", I.compose("deep_research", "", "en") == "🔬 Got it, researching — a few minutes.")
check("an unknown language falls back to English", I.compose("run_code", "", "de").startswith("💻 Got it,"))

# --- offer: the callback, the cap ---------------------------------------------
class _Ctx:
    reply_lang = "ru"
    def __init__(self): self.sent = []; self.interim_callback = self.sent.append
c = _Ctx()
check("no long tool -> nothing sent", not I.offer(c, "hi", [tc("calculate")], 0) and not c.sent)
check("a long tool -> one note", I.offer(c, "", [tc("generate_music")], 0) and c.sent == ["🎵 Понял, сочиняю и пою — несколько минут."], c.sent)
check("the cap: a second note in the same turn is not sent", not I.offer(c, "", [tc("generate_music")], 1))
c2 = _Ctx(); c2.interim_callback = None
check("no callback (desktop) -> nothing, no error", not I.offer(c2, "", [tc("generate_image")], 0))
c3 = _Ctx()
def _boom(t): raise RuntimeError("telegram down")
c3.interim_callback = _boom
check("a failing callback is swallowed", not I.offer(c3, "", [tc("generate_image")], 0))

# --- through the real loop -------------------------------------------------------
import test_agent_loop_hardening as H
import tools as tools_mod
spec = tools_mod._BY_NAME["generate_image"]
orig = spec.handler
order = []
object.__setattr__(spec, "handler", lambda c, s, a: (order.append("tool"), "[done] drawn")[1])
try:
    def fn(i):
        if i == 1:
            return H._msg("Сейчас нарисую лису.", tcs=[H._tc("generate_image", {"prompt": "a fox"}, "g1")])
        return H._msg("Готово, лиса нарисована.")
    H._script["fn"] = fn; H._script["n"] = 0
    ctx = H._ctx()
    ctx.reply_lang = "ru"
    ctx.interim_callback = lambda t: order.append(("note", t))
    import graph as graph_mod
    g = graph_mod.build_graph(ctx)
    st = H._base_state(); st["user_input"] = "нарисуй лису"
    final = g.invoke(st)
finally:
    object.__setattr__(spec, "handler", orig)
check("the note goes out BEFORE the tool runs, with the model's own words",
      order[:2] == [("note", "🎨 Сейчас нарисую лису."), "tool"], order)
check("the turn still ends with a final answer", bool((final.get("final_answer") or "").strip()))

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
