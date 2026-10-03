"""A tool call must never be turned into a chat message by our own stripper.

From the live bot log (2026-07-28 23:29): the user asked a question, the bot
answered with

    call:search{query:<|"|>who is Evgeniy Prigozhin<|"|>}

and no search ever ran. Five times in twelve minutes, across five questions.

Cause: `utils._CONTROL_TOKEN_RE` (added by the control-token-leak fix) treats
`<|tool_call>` and `<tool_call|>` as decorative control tokens and deletes them.
`llm.send_to_lm_studio` runs `strip_think_tags()` on the content BEFORE
`extract_gemma4_tool_calls()`, so by parse time the delimiters were gone, the
Format-A regex could not match, and the naked body became the final answer.
`strip_textual_tool_calls` could not catch it either — it also keys off the
delimiters — so the residue sailed through the last guard to the user.

Note the delimiters must survive stripping but not APPEAR: three different
mechanisms (parse, strip-as-block, reasoning-block boundary) all key off them.

Run: venv/Scripts/python.exe tests/test_tool_call_delimiter_leak.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import utils as U
import llm as L

OK = BAD = 0
def check(name, cond, detail=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {detail}")


Q = '<|"|>'
WRAPPED = '<|tool_call>call:search{query:' + Q + 'who is Evgeniy Prigozhin' + Q + '}<tool_call|>'
BARE    = 'call:search{query:' + Q + 'Who is the Wagner Group' + Q + '}'


def calls_of(text):
    """The full production path: strip first (as send_to_lm_studio does), parse after."""
    tc, remaining = L.extract_gemma4_tool_calls(U.strip_think_tags(text))
    return [(c["function"]["name"], c["function"]["arguments"]) for c in tc], remaining


print("=" * 70)
print("THE REPORTED LINE PARSES AS A CALL, NOT AS A REPLY")
print("=" * 70)

names, remaining = calls_of(WRAPPED)
check("the wrapped call survives stripping and parses",
      names and names[0][0] == "search", str(names))
check("the query text is intact",
      names and "Evgeniy Prigozhin" in names[0][1], str(names))
check("nothing is left over to send to the user", remaining == "", repr(remaining))

# The exact string the user received. Even if some future stripper eats the
# wrapper again, the call must RUN rather than be shown.
names, remaining = calls_of(BARE)
check("the bare body parses as a call too",
      names and names[0][0] == "search", str(names))
check("the bare body leaves no user-visible residue", remaining == "", repr(remaining))

# Each of the five queries from the log, including Cyrillic.
for q in ["who is Evgeniy Prigozhin", "Who is the Wagner Group",
          "Who is Joseph Kabzon?", "Волос СПБГЭТУ ЛЭТИ",
          "Волос преподаватель ЛЭТИ"]:
    n, _ = calls_of('<|tool_call>call:search{query:' + Q + q + Q + '}<tool_call|>')
    check(f"log line parses: {q[:32]}", n and q in n[0][1], str(n))


print()
print("=" * 70)
print("THE DELIMITERS SURVIVE THE STRIPPER")
print("=" * 70)

check("strip_think_tags keeps <|tool_call>", "<|tool_call>" in U.strip_think_tags(WRAPPED))
check("strip_think_tags keeps <tool_call|>", "<tool_call|>" in U.strip_think_tags(WRAPPED))
check("strip_control_tokens keeps them too",
      "<|tool_call>" in U.strip_control_tokens(WRAPPED))

# ...but every OTHER control token must still go, or this fix would have
# reopened the leak it was built to close.
check("channel reasoning is still stripped",
      U.strip_think_tags('<|channel>analysis my secret plan<|channel>final<|message|>Hello')
      == "Hello")
check("bare <|channel>> residue is still stripped",
      "channel" not in U.strip_think_tags('<|channel>>Here is a list'))
check("<|message|> is still stripped",
      "<|message|>" not in U.strip_think_tags('a<|message|>b'))
check("assistantfinal is still stripped",
      U.strip_think_tags('assistantfinal Hello').strip() == "Hello")

# A reasoning block with no closing marker used to run to end-of-text and
# swallow the call that followed it.
n, _ = calls_of('<|channel>thought I should look this up.'
                '<|tool_call>call:search{query:' + Q + 'x' + Q + '}<tool_call|>')
check("an unterminated reasoning block does not swallow the call", bool(n), str(n))
check("the reasoning text itself is not delivered",
      "look this up" not in (calls_of('<|channel>thought I should look this up.'
                                      '<|tool_call>call:search{query:' + Q + 'x' + Q + '}<tool_call|>')[1]))


print()
print("=" * 70)
print("THE LAST-RESORT STRIPPER CATCHES WHAT PARSING MISSES")
print("=" * 70)

check("a bare body is stripped from a final answer",
      U.strip_textual_tool_calls("Sure. " + BARE).strip() == "Sure.")
check("a wrapped call is stripped from a final answer",
      U.strip_textual_tool_calls("Sure. " + WRAPPED).strip() == "Sure.")
check("an orphan opener is stripped",
      "tool_call" not in U.strip_textual_tool_calls("text <|tool_call>"))
check("an orphan closer is stripped",
      "tool_call" not in U.strip_textual_tool_calls("text <tool_call|>"))

# The Format-C pattern is gated on <|"|> precisely so prose cannot match it.
for prose in ["I will call: the doctor about a{b} appointment",
              "Use the call:site feature",
              "def f(): call:x{y}",
              "Здесь нет вызова: call: поиск"]:
    check(f"prose untouched: {prose[:34]}",
          U.strip_textual_tool_calls(prose) == prose and
          U.strip_think_tags(prose) == prose)
check("a call without the delimiter is NOT silently eaten",
      U.strip_textual_tool_calls("call:search{query:hello}") == "call:search{query:hello}")


print()
print("=" * 70)
print("MUTATION: PUT THE OLD GRAMMAR BACK, THE SUITE MUST NOTICE")
print("=" * 70)

import re as _re
_OLD_GRAMMAR = _re.compile(r"<\|[a-z_]{1,24}\|?>+|<[a-z_]{1,24}\|>", _re.IGNORECASE)
_NEVER = _re.compile(r"(?!x)x")

_saved_re, _saved_bare = U._CONTROL_TOKEN_RE, L._G4_BARE_RE
try:
    # Mutant 1 — grammar only. Format C is a SECOND, independent layer, so the
    # call must still go through: losing the wrapper is survivable now.
    U._CONTROL_TOKEN_RE = _OLD_GRAMMAR
    rescued, _ = calls_of(WRAPPED)
    check("old grammar alone: the bare-body parser rescues the call",
          rescued and rescued[0][0] == "search", str(rescued))
    check("and the stripped text is exactly what the user saw",
          U.strip_think_tags(WRAPPED)
          == 'call:search{query:' + Q + 'who is Evgeniy Prigozhin' + Q + '}',
          repr(U.strip_think_tags(WRAPPED)))

    # Mutant 2 — BOTH layers removed, which is the shipped-before state. Two
    # redundant guards hide each other: mutating either one alone proves nothing.
    L._G4_BARE_RE = _NEVER
    broke, leftover = calls_of(WRAPPED)
    check("both layers off: the call is destroyed (the suite sees the real bug)",
          not broke, str(broke))
    check("both layers off: the raw body would be sent to the user",
          leftover.startswith("call:search{"), repr(leftover))
finally:
    U._CONTROL_TOKEN_RE, L._G4_BARE_RE = _saved_re, _saved_bare

check("the grammar is restored after the mutation",
      "<|tool_call>" in U.strip_think_tags(WRAPPED))


print()
print("=" * 70)
print("A REACT-STYLE JSON CALL IN THE CLOSING MESSAGE IS SWEPT")
print("=" * 70)
# Live, 2026-09-12: the picture was drawn, then the closing message began with
# { "action": "generate_image", "action_input": "{...}" } and the voice note
# read the JSON aloud.
_LEAK = ('{ "action": "generate_image", "action_input": "{' + chr(39) + 'description' + chr(39) + ': '
         + chr(39) + 'A large, bright blue elephant on a beach. It is peaceful.' + chr(39) + '}" }' + chr(10) * 2
         + 'Вот твой синий слон на пляже!')
check("the JSON call is gone, the sentence stays",
      U.strip_textual_tool_calls(_LEAK) == "Вот твой синий слон на пляже!",
      U.strip_textual_tool_calls(_LEAK))
check("name/arguments shape too",
      U.strip_textual_tool_calls('{"name": "search", "arguments": {"query": "x {y}"}} Нашёл.') == "Нашёл.")
check("a JSON object the user asked for is untouched",
      U.strip_textual_tool_calls('Вот JSON: {"name": "Иван", "age": 30}') == 'Вот JSON: {"name": "Иван", "age": 30}')
check("prose with braces is untouched",
      U.strip_textual_tool_calls("формула {x} и action: go") == "формула {x} и action: go")
check("an unterminated call is cut to the end, prose before it survives",
      U.strip_textual_tool_calls('Готово. {"action": "x", "action_input": "{oops') == "Готово.")

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
