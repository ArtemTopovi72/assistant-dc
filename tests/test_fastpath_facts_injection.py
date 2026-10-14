"""The fast path (short, tool-free turns) must still see Saved facts.

A fact saved with remember_fact ("запомни: меня зовут Марат...") must reach a
short, keyword-free follow-up ("как меня зовут и что ты обо мне знаешь?") that
takes graph_fastpath's tool-less path; otherwise the model answers "you haven't
told me your name yet" -- a confident denial of a fact it genuinely had saved.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")

import graph_fastpath as F
from graph_compose import _GenSettings, FORGOTTEN_NOTE


class _Ctx:
    def __init__(self, facts=(), forgotten=False, chat_model=None):
        self.chat_model = chat_model
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


def test_fast_path_includes_saved_facts():
    captured = []
    ctx = _Ctx(facts=["Пользователя зовут Марат, он вегетарианец и у него аллергия на орехи."],
               chat_model=_fake_llm(captured))
    F._fast_path_reply(ctx, [{"role": "system", "content": "s"}], _GEN,
                       "what is my name?", "what is my name?",
                       "как меня зовут и что ты обо мне знаешь?")
    assert captured, "the fast path never called the model"
    sent = " ".join(m.get("content", "") for m in captured[0])
    assert "Марат" in sent, f"the saved fact never reached the fast-path prompt: {sent!r}"
    assert "Saved facts" in sent


def test_fast_path_with_no_facts_sends_no_facts_block():
    """No false positives: an empty fact store must not inject an empty block."""
    captured = []
    ctx = _Ctx(facts=[], chat_model=_fake_llm(captured))
    F._fast_path_reply(ctx, [{"role": "system", "content": "s"}], _GEN,
                       "hi", "hi", "привет")
    sent = " ".join(m.get("content", "") for m in captured[0])
    assert "Saved facts" not in sent


def test_fast_path_after_forget_shows_the_forgotten_note():
    captured = []
    ctx = _Ctx(facts=[], forgotten=True, chat_model=_fake_llm(captured))
    F._fast_path_reply(ctx, [{"role": "system", "content": "s"}], _GEN,
                       "what's my name?", "what's my name?", "как меня зовут?")
    sent = " ".join(m.get("content", "") for m in captured[0])
    assert FORGOTTEN_NOTE in sent


def test_fast_path_cut_by_token_cap_falls_through():
    """A 60-row table cut at row 25 by the token cap is not delivered as the answer."""
    cut = lambda ctx, m, **kw: {"role": "assistant", "content": "| 25 | Mn", "finish_reason": "length"}
    assert F._fast_path_reply(_Ctx(chat_model=cut), [{"role": "system", "content": "s"}], _GEN,
                              "table", "table", "таблица") is None


if __name__ == "__main__":
    import subprocess
    r = subprocess.run([sys.executable, "-m", "pytest", __file__, "-q"])
    sys.exit(r.returncode)


# An allergy or diet in the facts is a hard limit on suggestions, not only on promises.
from graph_compose import facts_parts as _facts_parts
import types as _types
_ALLERGY = _types.SimpleNamespace(yes_no=lambda q, t: "аллерги" in t)
_txt = " ".join(_facts_parts("Пользователя зовут Марат, он вегетарианец и у него аллергия на орехи.", _ALLERGY))
assert "never offer a dish or ingredient that contains the allergen" in _txt, _txt
assert "diet" in _txt
assert len(_facts_parts("Пользователя зовут Марат.", _ALLERGY)) == 1
print("ok an allergy or diet is a hard limit on suggestions, not only on promises")
