"""The delivery round, driven through the REAL loop with a scripted model.

Bench deliver_fix 2026-09-23: the model made the right edit on the very last
tool round and answered "Готово" with nothing packed. Now a spent budget with
an unpacked-and-changed folder buys exactly one more round, for pack_archive.
"""
import os, sys, json, zipfile, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path

import graph as graph_mod
import llm as llm_mod
import sandbox_access as A
from code_sandbox import Sandbox
from models import Context
from prompts import SYSTEM_PROMPT_PERSONALITY
from config import MODEL_NAME

_seen = []


def _tc(name, args, cid):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


class _User:
    prefs = {"sandbox": A.CODE}


def _run(tmp_path, script):
    def send(ctx, messages, tools=None, **k):
        _seen.append([m.get("content") for m in messages if m.get("role") == "user"])
        return script(len(_seen), messages)
    graph_mod.send_to_lm_studio = send
    llm_mod.send_to_lm_studio = send
    _seen.clear()
    box = Sandbox(tmp_path / "sbx")
    with zipfile.ZipFile(box.root / "mod.jar", "w") as zf:
        zf.writestr("data/mod/tags/block/a.json", '{"values": []}')
    c = Context(models=None, transcription_cache={}, cache_file=tmp_path / "c.json",
                asr_lock=threading.Lock(), tts_lock=threading.Lock(),
                model_name=MODEL_NAME, no_think=True)
    c.tts_disabled = True; c.gui_mode = True; c.web_search_enabled = False
    c.active_memory_dir = tmp_path / "mem"; c.active_memory_dir.mkdir()
    c.sandbox, c.sandbox_user = box, _User()
    st = {"messages": [{"role": "system", "content": SYSTEM_PROMPT_PERSONALITY}],
          "user_input": "добавь блок в тег мода", "image_data": None, "final_answer": "",
          "session_memory_text": "", "vision_summary": "", "image_path": "",
          "image_score": 0, "image_attempt": 0, "image_status": ""}
    pass
    pass
    try:
        return graph_mod.build_graph(c).invoke(st)
    finally:
        pass


def _asked_for_delivery(messages):
    return any("[DELIVERY ROUND]" in str(m.get("content") or "") for m in messages)


def test_a_spent_budget_buys_one_round_to_pack(tmp_path):
    def script(n, messages):
        if _asked_for_delivery(messages):
            if any(m.get("role") == "tool" and "Packed" in str(m.get("content")) for m in messages):
                return {"role": "assistant", "content": "Готово, архив отправлен."}
            return {"role": "assistant", "content": "",
                    "tool_calls": [_tc("pack_archive", {"path": "mod_unpacked",
                                                        "output": "mod_fixed.jar"}, f"p{n}")]}
        if not any("Unpacked into" in str(m.get("content")) for m in messages):
            return {"role": "assistant", "content": "",
                    "tool_calls": [_tc("unpack_archive", {"path": "mod.jar"}, "u1")]}
        # Busy until the budget runs out: a fresh write every round.
        return {"role": "assistant", "content": "",
                "tool_calls": [_tc("write_file", {"path": "mod_unpacked/data/mod/tags/block/a.json",
                                                  "content": '{"values": ["x:%d"]}' % n}, f"w{n}")]}
    final = _run(tmp_path, script)
    assert _asked_for_delivery(final["messages"]), "no delivery round was granted"
    assert str(final.get("document_path") or "").endswith("mod_fixed.jar"), final.get("document_path")
    assert final.get("document_status") == "success"


def test_no_delivery_round_when_nothing_is_pending(tmp_path):
    def script(n, messages):
        if n <= 40:
            return {"role": "assistant", "content": "",
                    "tool_calls": [_tc("list_files", {"path": "."}, f"l{n}")]}
        return {"role": "assistant", "content": "ok"}
    final = _run(tmp_path, script)
    assert not _asked_for_delivery(final["messages"])
