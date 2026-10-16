"""A lone surrogate in crawled text must not cost the page.

A Basler page carried an unpaired U+DC00: LM Studio's JSON parser answered 400
three times and the page's brief was lost, and the cross-encoder's tokenizer
raised "TextInputSequence must be str", dropping the whole batch to lexical.
"""
import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm
import rerank
from tokenizers import Tokenizer, models, pre_tokenizers

ok = 0
checks = []


def check(name, cond):
    global ok
    checks.append(name)
    ok += bool(cond)
    print(("PASS  " if cond else "FAIL  ") + name)


BAD = "acA4096-11gc \udc00 4096 x 2160"

posted = []


class _Resp:
    status_code = 200
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def iter_lines(self, decode_unicode=False):
        yield b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}'
        yield b"data: [DONE]"
    def close(self): pass


def _post(url, json=None, **kw):
    posted.append(json)
    return _Resp()


ctx = types.SimpleNamespace(deps=types.SimpleNamespace(requests=types.SimpleNamespace(post=_post)))
llm._stream_chat_raw(ctx, {"messages": [{"role": "user", "content": BAD}]})
body = json.dumps(posted[0], ensure_ascii=False)
try:
    body.encode("utf-8")
    encodable = True
except UnicodeEncodeError:
    encodable = False
check("the request LM Studio gets is valid UTF-8", encodable)
check("the rest of the page text is kept", "4096 x 2160" in posted[0]["messages"][0]["content"])


class _TokenizingCE:
    """Tokenizes like the real cross-encoder does, so a lone surrogate raises."""
    def __init__(self, *a, **k):
        self.tk = Tokenizer(models.WordLevel({"x": 0}, unk_token="x"))
        self.tk.pre_tokenizer = pre_tokenizers.Whitespace()

    def predict(self, pairs, **k):
        self.tk.encode_batch(list(pairs))
        return [float("4096" in t) for _, t in pairs]


r = rerank.Reranker("cross", cross_encoder=_TokenizingCE, allow_download=True,
                    is_cached=lambda m: True)
check("cross backend loaded", r.backend == "cross")
check("a lone surrogate is scored by the cross-encoder, not lexical fallback",
      r.relevance("4096 camera", [BAD, "unrelated"]) == [1.0, 0.0])

print(f"\n{ok}/{len(checks)} checks passed")
sys.exit(0 if ok == len(checks) else 1)
