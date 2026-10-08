"""The fast path (short, tool-free turns) must still see Saved facts.

Live 2026-09-18, journey 9: "запомни: меня зовут Марат..." saved the fact
correctly (remember_fact ran, ctx.pinned_facts held it). Three turns later
"как меня зовут и что ты обо мне знаешь?" is short and has no tool-trigger
keyword, so it took graph_fastpath's tool-less path -- which built its prompt
from a bare system message + the user's line, with a comment saying facts were
deliberately left out to save tokens. The model, given nothing about Marat,
answered "you haven't told me your name yet" -- a confident denial of a fact
it genuinely had saved, just never shown.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")

import graph_fastpath as F
from graph_compose import _GenSettings, FORGOTTEN_NOTE


class _Ctx:
    def __init__(self, facts=(), forgotten=False):
        self._facts = list(facts)
        self.facts_forgotten = forgotten
        self.custom_personality_text = ""
        self.reply_lang = "ru"
    def facts_text(self, query=""):
        return "\n".join(f"- {t}" for t in self._facts)
    def is_cancelled(self):
        return False
    def set_stage(self, *a, **k):
        pass


_GEN = _GenSettings(concise=False, length="auto", temperature=0.3,
                    max_tokens=600, loop_max_tokens=600, prefill=None,
                    system_prompt="")


def _fake_llm(captured):
    def fn(ctx, messages, **kw):
        captured.append(messages)
        return {"role": "assistant", "content": "Тебя зовут Марат."}
    return fn


def test_fast_path_includes_saved_facts(monkeypatch):
    import graph as _g
    captured = []
    monkeypatch.setattr(_g, "send_to_lm_studio", _fake_llm(captured))
    ctx = _Ctx(facts=["Пользователя зовут Марат, он вегетарианец и у него аллергия на орехи."])
    F._fast_path_reply(ctx, [{"role": "system", "content": "s"}], _GEN,
                       "what is my name?", "what is my name?",
                       "как меня зовут и что ты обо мне знаешь?")
    assert captured, "the fast path never called the model"
    sent = " ".join(m.get("content", "") for m in captured[0])
    assert "Марат" in sent, f"the saved fact never reached the fast-path prompt: {sent!r}"
    assert "Saved facts" in sent


def test_fast_path_with_no_facts_sends_no_facts_block(monkeypatch):
    """No false positives: an empty fact store must not inject an empty block."""
    import graph as _g
    captured = []
    monkeypatch.setattr(_g, "send_to_lm_studio", _fake_llm(captured))
    ctx = _Ctx(facts=[])
    F._fast_path_reply(ctx, [{"role": "system", "content": "s"}], _GEN,
                       "hi", "hi", "привет")
    sent = " ".join(m.get("content", "") for m in captured[0])
    assert "Saved facts" not in sent


def test_fast_path_after_forget_shows_the_forgotten_note(monkeypatch):
    import graph as _g
    captured = []
    monkeypatch.setattr(_g, "send_to_lm_studio", _fake_llm(captured))
    ctx = _Ctx(facts=[], forgotten=True)
    F._fast_path_reply(ctx, [{"role": "system", "content": "s"}], _GEN,
                       "what's my name?", "what's my name?", "как меня зовут?")
    sent = " ".join(m.get("content", "") for m in captured[0])
    assert FORGOTTEN_NOTE in sent


def test_fast_path_cut_by_token_cap_falls_through(monkeypatch):
    """Live: a 60-row table stopped at row 25 and was delivered as the answer."""
    import graph as _g
    monkeypatch.setattr(_g, "send_to_lm_studio", lambda ctx, m, **kw:
                        {"role": "assistant", "content": "| 25 | Mn", "finish_reason": "length"})
    assert F._fast_path_reply(_Ctx(), [{"role": "system", "content": "s"}], _GEN,
                              "table", "table", "таблица") is None


if __name__ == "__main__":
    import subprocess
    r = subprocess.run([sys.executable, "-m", "pytest", __file__, "-q"])
    sys.exit(r.returncode)


# 10-08 live: «аллергия на орехи» was in the facts and dinner advice still offered
# «кедровые орешки» -- the allergy note only forbade PROMISING a dish free of the allergen.
import intent as _intent_mod
from graph_compose import facts_parts as _facts_parts
_intent_mod.YES_STUB = lambda q, t: "аллерги" in t
try:
    _txt = " ".join(_facts_parts("Пользователя зовут Марат, он вегетарианец и у него аллергия на орехи."))
    assert "never offer a dish or ingredient that contains the allergen" in _txt, _txt
    assert "diet" in _txt
    assert len(_facts_parts("Пользователя зовут Марат.")) == 1
finally:
    _intent_mod.YES_STUB = None
print("ok an allergy or diet is a hard limit on suggestions, not only on promises")
