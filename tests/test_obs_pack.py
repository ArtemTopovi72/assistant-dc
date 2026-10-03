"""ObservationPack (SoL-Pi, 2026-09), driven through the REAL loop with a scripted model.

A big tool result is sent in full for two rounds, then shrunk to a 1 KB excerpt.
Off by default (OBS_PACK=0): nothing may change for the live app until the A/B.
"""
import os, sys, json, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging; logging.basicConfig(level=logging.CRITICAL)

import graph as graph_mod
import llm as llm_mod
from models import Context
from prompts import SYSTEM_PROMPT_PERSONALITY
from config import MODEL_NAME

BIG = "строка результата поиска " * 400          # ~10 KB
_sent = []


def _tc(n):
    return {"id": f"s{n}", "type": "function",
            "function": {"name": "search", "arguments": json.dumps({"query": f"q{n}"})}}


def _run(tmp_path, rounds=4):
    def send(ctx, messages, tools=None, **k):
        _sent.append([str(m.get("content") or "") for m in messages if m.get("role") == "tool"])
        n = len(_sent)
        if n <= rounds:
            return {"role": "assistant", "content": "", "tool_calls": [_tc(n)]}
        return {"role": "assistant", "content": "Готово."}
    graph_mod.send_to_lm_studio = send
    llm_mod.send_to_lm_studio = send
    saved_exec = graph_mod.execute_tool
    graph_mod.execute_tool = lambda ctx, name, args, *a, **k: BIG
    pass
    _sent.clear()
    c = Context(models=None, transcription_cache={}, cache_file=tmp_path / "c.json",
                asr_lock=threading.Lock(), tts_lock=threading.Lock(),
                model_name=MODEL_NAME, no_think=True)
    c.tts_disabled = True; c.gui_mode = True; c.web_search_enabled = True
    c.active_memory_dir = tmp_path / "mem"; c.active_memory_dir.mkdir()
    st = {"messages": [{"role": "system", "content": SYSTEM_PROMPT_PERSONALITY}],
          "user_input": "найди что-нибудь", "image_data": None, "final_answer": "",
          "session_memory_text": "", "vision_summary": "", "image_path": "",
          "image_score": 0, "image_attempt": 0, "image_status": ""}
    try:
        graph_mod.build_graph(c).invoke(st)
    finally:
        graph_mod.execute_tool = saved_exec
    return list(_sent)


def _first_result_len(sent):
    return [len(tools[0]) if tools else 0 for tools in sent]


def test_off_nothing_shrinks(tmp_path, monkeypatch):
    # On by default since the 2026-09-24 A/B; OBS_PACK=0 is the way back.
    monkeypatch.setenv("OBS_PACK", "0")
    sent = _run(tmp_path)
    lens = [x for x in _first_result_len(sent) if x]
    assert lens and min(lens) >= len(BIG), lens


def test_on_full_for_two_rounds_then_excerpt(tmp_path, monkeypatch):
    monkeypatch.delenv("OBS_PACK", raising=False)      # the default
    sent = _run(tmp_path)
    lens = _first_result_len(sent)
    seen = [x for x in lens if x]
    assert seen[0] >= len(BIG) and seen[1] >= len(BIG), lens     # full, two rounds
    assert seen[-1] < 1500, lens                                  # then the excerpt
    assert any("shortened" in t[0] for t in sent if t), "no marker for the model"


def test_small_results_are_never_packed(tmp_path, monkeypatch):
    global BIG
    monkeypatch.setenv("OBS_PACK", "1")
    old, BIG = BIG, "короткий ответ"
    try:
        sent = _run(tmp_path)
    finally:
        BIG = old
    assert all("shortened" not in x for t in sent for x in t)
