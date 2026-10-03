"""A JSON tool call written into the prose is a call, not a leak.

Live 2026-09-13 (mega run 3, steps 72/75/77): after a model reload three
drawing turns in a row came back with the call as text —
`<tool_call>{"name": "generate_image", "arguments": {...}}</tool_call>` —
utils.strip_textual_tool_calls deleted the block as a leak and the user got
«Вот ваш логотип» with no picture. llm.extract_gemma4_tool_calls now parses
that shape (wrapped or bare) and the stream path recognises it as markup.
"""
import os, sys, json, inspect
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import llm as L
import utils as U

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

def calls_of(text):
    tc, rem = L.extract_gemma4_tool_calls(U.strip_think_tags(text))
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in tc], rem

WRAPPED = ('Рисую логотип.\n<tool_call>\n{"name": "generate_image", "arguments": '
           '{"prompt": "flat bakery logo: an ear of grain, text \\"ХЛЕБ У ДОМА\\"", '
           '"aspect": "square"}}\n</tool_call>')
names, rem = calls_of(WRAPPED)
check("the wrapped JSON call parses", names and names[0][0] == "generate_image", names)
check("...with its arguments intact",
      names and names[0][1].get("aspect") == "square" and "ХЛЕБ У ДОМА" in names[0][1].get("prompt", ""), names)
check("...and the prose around it is what remains", rem == "Рисую логотип.", repr(rem))

BARE = '{"name":"calculate","arguments":{"expression":"(2340*0.15)/3"}}'
names, rem = calls_of(BARE)
check("a bare JSON call parses", names == [("calculate", {"expression": "(2340*0.15)/3"})] and rem == "", (names, rem))

FENCED = 'Sure:\n```json\n{"name": "search", "parameters": {"query": "highest road in Europe"}}\n```'
names, rem = calls_of(FENCED)
check("a fenced call with 'parameters' parses", names and names[0] == ("search", {"query": "highest road in Europe"}), names)
check("...and the fence goes with it", "```" not in rem, repr(rem))

STRARGS = '{"name": "remember_fact", "arguments": "{\\"text\\": \\"cat is Barsik\\"}"}'
names, rem = calls_of(STRARGS)
check("string-encoded arguments are decoded", names and names[0][1] == {"text": "cat is Barsik"}, names)

for prose in ('The JSON {"name": "Marat", "age": 30} describes a user.',
              'Use {"name": "x"} as a template', 'plain text with {braces}', ""):
    names, rem = calls_of(prose)
    check(f"prose untouched: {prose[:40]!r}", names == [] and rem == prose.strip(), (names, rem))

src = inspect.getsource(L)
check("the stream path treats a JSON call as markup", "_JSON_CALL_START_RE.search(content)" in src)

# Format F, live 2026-09-29: `world[TOOL_CALL]generate_image(description='…')` was the reply.
_calls, _rest = L.extract_gemma4_tool_calls(
    "world[TOOL_CALL]generate_image(description='A ginger cat (fluffy) on a sofa', width=944)[/TOOL_CALL]")
check("a python-style call is parsed", [c["function"]["name"] for c in _calls] == ["generate_image"]
      and json.loads(_calls[0]["function"]["arguments"]) == {"description": "A ginger cat (fluffy) on a sofa", "width": 944},
      _calls)
check("and leaves only the prose", _rest == "world", _rest)
check("prose with brackets is not a call", L.extract_gemma4_tool_calls("текст (в скобках)")[0] == [])
check("the stream path treats it as markup", "_PY_CALL_START_RE.search(content)" in src)
import utils as _U
check("a leftover is stripped", _U.strip_textual_tool_calls("ok [TOOL_CALL]search(query='x')").strip() == "ok")

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
