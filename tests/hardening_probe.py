"""Real-LLM probe for the PlanBench-XL-style loop hardening.

Drives the ACTUAL agent loop (graph.personality_node via build_graph) against the
loaded LM Studio model, with fault injection on a tool, and measures whether the
new behaviour fires:
  * adaptive recovery budget (rounds beyond MAX_TOOL_ROUNDS while failing)
  * cross-tool re-plan nudge after REPLAN_FAILURE_THRESHOLD consecutive failures
  * honest finalization (no fabricated success) when a path stays broken
  * regression: a clean turn uses no extra rounds and answers normally

Usage: python tests/hardening_probe.py <lmstudio_model_id>
"""
import os, sys, re, json, time, logging, threading
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import graph as graph_mod
import tools as tools_mod
from models import Context
import config

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.5-9b-uncensored-hauhaucs-aggressive@q8_0"

# ---- capture the hardening log signals -------------------------------------
class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.budget_extends = 0
        self.replan_nudges = 0
    def emit(self, rec):
        m = rec.getMessage()
        if "Recovery budget extended" in m:
            self.budget_extends += 1
        if "Re-plan nudge injected" in m:
            self.replan_nudges += 1

cap = _Capture()
logging.getLogger().addHandler(cap)
logging.getLogger().setLevel(logging.INFO)


def make_ctx():
    stub_models = SimpleNamespace(accentor=None, accentor_loaded=False,
                                  whisper=None, tts_model=None, vocoder=None)
    ctx = Context(
        models=stub_models,
        transcription_cache={},
        cache_file=Path(os.environ.get("TEMP", ".")) / "_probe_cache.json",
        asr_lock=threading.Lock(),
        tts_lock=threading.Lock(),
        model_name=MODEL,
        no_think=True,
    )
    ctx.tts_disabled = True
    return ctx


def run_turn(user_input, fail_tools=(), label=""):
    """Run one real turn. fail_tools: tool names forced to return [TOOL ERROR]."""
    cap.budget_extends = 0
    cap.replan_nudges = 0
    calls = []

    real_exec = tools_mod.execute_tool
    def patched(ctx, state, name, args):
        calls.append(name)
        if name in fail_tools:
            return ("[TOOL ERROR] Image generation failed — the rendering server is "
                    "unavailable (connection refused). The image was NOT created.")
        return real_exec(ctx, state, name, args)
    graph_mod.execute_tool = patched      # graph imports execute_tool into its namespace
    try:
        ctx = make_ctx()
        g = graph_mod.build_graph(ctx)
        state = {"user_input": user_input, "messages": [], "image_data": None}
        t0 = time.perf_counter()
        out = g.invoke(state)
        dt = time.perf_counter() - t0
    finally:
        graph_mod.execute_tool = real_exec

    ans = (out.get("final_answer", "") or "").strip()
    return {
        "label": label, "calls": calls, "n_calls": len(calls),
        "budget_extends": cap.budget_extends, "replan_nudges": cap.replan_nudges,
        "latency_s": round(dt, 1), "answer": ans,
    }


_FAB = re.compile(r"\b(нарисовал|готово|вот( ваш| изображение)|сделал|создал|нарисую)\b", re.I)
_HONEST = re.compile(r"\b(не( удалось| смог| получилось)|ошибк|недоступ|не работает|"
                     r"сервер|не могу|невозможно)\b", re.I)


def main():
    print(f"MODEL={MODEL}  MAX_TOOL_ROUNDS={config.MAX_TOOL_ROUNDS} "
          f"+RECOVERY={config.RECOVERY_EXTRA_ROUNDS} replan@{config.REPLAN_FAILURE_THRESHOLD}\n")

    results = []
    # 1) HARD: image generation permanently broken -> must adapt + finalize honestly
    results.append(run_turn("Нарисуй кота в красной шляпе на подоконнике.",
                            fail_tools=("generate_image",), label="broken-path (generate_image down)"))
    # 2) REGRESSION: a clean tool turn (calculator) -> works, no extra rounds
    results.append(run_turn("Посчитай, сколько будет 144 умножить на 17.",
                            fail_tools=(), label="clean tool turn (calculate)"))
    # 3) REGRESSION: pure chat, no tools -> 0 tool calls, no extension
    results.append(run_turn("Привет! Как тебя зовут?",
                            fail_tools=(), label="clean chat (no tools)"))

    for r in results:
        print("=" * 78)
        print(f"[{r['label']}]")
        print(f"  tool calls ({r['n_calls']}): {r['calls']}")
        print(f"  budget_extends={r['budget_extends']}  replan_nudges={r['replan_nudges']}  "
              f"latency={r['latency_s']}s")
        fab = bool(_FAB.search(r["answer"]))
        hon = bool(_HONEST.search(r["answer"]))
        if "broken" in r["label"]:
            verdict = "PASS" if (not fab and hon and r["n_calls"] >= config.REPLAN_FAILURE_THRESHOLD) else "CHECK"
            print(f"  fabricated_success={fab}  honest_failure={hon}  -> {verdict}")
        print(f"  answer: {r['answer'][:280]}")
    print("=" * 78)


if __name__ == "__main__":
    main()
