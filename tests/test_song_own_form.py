import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The model's reads are stubbed here; the phrases run live in bench/intent_rest_live.py.
import intent
intent.YES_STUB = lambda q, t: (("structure" in q and any(w in t.lower() for w in ("ровно", "без припева", "exactly")))
                                or ("NO singing" in q and "без вокала" in t))
import music

seen = []
class _Stop(Exception): pass
def _fake(ctx, msgs, **kw):
    seen.append(msgs[0]["content"]); raise _Stop
import llm
llm.send_to_lm_studio = _fake
for topic, capped in (("блюз: ровно три куплета по четыре строки, без припева", False),
                      ("песня про кота", True)):
    seen.clear()
    try: music.build_structured_caption(None, topic, "ru", duration_s=120)
    except Exception: pass
    assert seen and (("LENGTH:" in seen[0]) == capped), topic
print("PASS the user's own form overrides the line budget")

seen.clear()
try: music.build_structured_caption(None, "эмбиент для сна, без слов и без вокала", "ru", duration_s=60)
except Exception: pass
assert "INSTRUMENTAL" in seen[0] and "LENGTH:" not in seen[0]
print("PASS 'без вокала' in the wish is an instrumental")
import utils as _u
assert _u.safe_json_from_llm('{"lyrics": "[verse]\nМягко светят\nАнгел мой", "style": "x"}') == \
    {"lyrics": "[verse]\nМягко светят\nАнгел мой", "style": "x"}
print("PASS raw newlines inside a JSON string are accepted")
