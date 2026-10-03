"""One JSON extractor, hardened against the reply shapes these models actually emit.

Four call sites parsed LLM JSON four different ways, and every one of them lost a
perfectly good answer on some real reply:

  utils.safe_json_from_llm      first '{' .. last '}'   -> spans two objects
  deep_research outline parser  first '{' .. last '}'   -> same, falls back to a
                                                           generic skeleton
  image._item_attributes        NON-greedy first {...}  -> a leading "{}" or a
                                                           nested object wins
  image EDIT_PROMPT_REFINER     first '{' .. last '}'   -> same as the first

None of them raised. They fell back, so the failure looked like the model being
bad at the task. Same class as the Ideogram styleless-caption bug.

Run: venv/Scripts/python.exe tests/test_llm_json_extraction.py
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

from utils import safe_json_from_llm, iter_json_objects

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


print("=" * 66)
print("REPLY SHAPES THAT USED TO LOSE THE ANSWER")
print("=" * 66)

GOOD = '{"instruction": "make the sky blue", "fill_prompt": "a clear blue sky"}'
KEYS = ("instruction", "fill_prompt")

SHAPES = {
    "clean object":                 GOOD,
    "fenced":                       f"```json\n{GOOD}\n```",
    "trailing prose":               GOOD + "\nHope that helps!",
    "trailing prose with a brace":  GOOD + "\nUse {} if you are unsure.",
    "duplicate fenced copy":        GOOD + '\n```json\n{"instruction": "dup"}\n```',
    "leading chatter":              "Sure, here you go:\n" + GOOD,
    "leading empty object":         "{}\n" + GOOD,
    "think block":                  "<think>hmm</think>\n" + GOOD,
    "windows newlines":             GOOD.replace("\n", "\r\n") + "\r\nDone.",
}
for name, raw in SHAPES.items():
    got = safe_json_from_llm(raw, required_keys=KEYS)
    check(f"survives: {name}",
          isinstance(got, dict) and got.get("instruction") == "make the sky blue",
          json.dumps(got, ensure_ascii=False)[:80])

# Non-ASCII content must round-trip untouched.
RU = '{"instruction": "сделай небо синим", "fill_prompt": "чистое синее небо"}'
got = safe_json_from_llm(RU + "\nГотово!", required_keys=KEYS)
check("non-ascii values survive",
      (got or {}).get("instruction") == "сделай небо синим",
      json.dumps(got, ensure_ascii=False)[:90])


print()
print("=" * 66)
print("THE RIGHT OBJECT WINS")
print("=" * 66)

# A nested object must not be mistaken for the answer.
NESTED = '{"attrs": {"worn": true}, "worn": true, "small": false}'
got = safe_json_from_llm(NESTED, required_keys=("worn", "small", "large", "multi"))
check("a nested object does not stand in for the outer one",
      (got or {}).get("worn") is True and "attrs" in (got or {}),
      json.dumps(got)[:90])

# Two plausible objects: the one carrying more of the wanted keys wins.
TWO = '{"worn": true}\n{"worn": false, "small": true, "large": false, "multi": true}'
got = safe_json_from_llm(TWO, required_keys=("worn", "small", "large", "multi"))
check("the object with the most wanted keys wins",
      (got or {}).get("small") is True, json.dumps(got)[:90])

# Nothing of the right shape must return None, not a fragment.
FRAGMENT = '{"desc": "a ginger cat", "x": 0.3, "y": 0.3}'
check("a fragment of the wrong shape yields None",
      safe_json_from_llm(FRAGMENT, required_keys=KEYS) is None,
      json.dumps(safe_json_from_llm(FRAGMENT, required_keys=KEYS)))

check("prose with no JSON yields None",
      safe_json_from_llm("I couldn't do that.", required_keys=KEYS) is None)
check("an empty reply yields None", safe_json_from_llm("", required_keys=KEYS) is None)

# Without required_keys the helper must still skip a leading empty object.
check("a leading {} does not win when no keys are given",
      safe_json_from_llm("{}\n" + GOOD) == json.loads(GOOD),
      json.dumps(safe_json_from_llm("{}\n" + GOOD))[:80])


print()
print("=" * 66)
print("BRACE SCANNING IS STRING-AWARE")
print("=" * 66)

BRACEY = '{"instruction": "write {OPEN on the sign", "fill_prompt": "a sign"}\nok!'
got = safe_json_from_llm(BRACEY, required_keys=KEYS)
check("an unbalanced brace inside a string does not swallow the object",
      (got or {}).get("instruction") == "write {OPEN on the sign",
      json.dumps(got, ensure_ascii=False)[:90])

ESCAPED = r'{"instruction": "say \"hi\" then {stop", "fill_prompt": "x"}' + "\nend"
got = safe_json_from_llm(ESCAPED, required_keys=KEYS)
check("escaped quotes do not break the scan", isinstance(got, dict),
      json.dumps(got, ensure_ascii=False)[:90])

objs = list(iter_json_objects('{"a": 1} junk {"b": 2}'))
check("every top-level object is yielded in order",
      objs == [{"a": 1}, {"b": 2}], str(objs))
check("an unterminated object yields nothing",
      list(iter_json_objects('{"a": 1')) == [], str(list(iter_json_objects('{"a": 1'))))
check("a stray closing brace does not desync the scan",
      list(iter_json_objects('} {"a": 1}')) == [{"a": 1}],
      str(list(iter_json_objects('} {"a": 1}'))))


print()
print("=" * 66)
print("THE CALL SITES USE IT")
print("=" * 66)

import deep_research as DR
OUTLINE = '{"title": "Survey", "sections": [{"heading": "Intro"}]}'
for name, raw in {
    "trailing prose":        OUTLINE + "\nHope that helps!",
    "duplicate fenced copy": OUTLINE + '\n```json\n{"title": "dup"}\n```',
    "note containing braces": f"```json\n{OUTLINE}\n```\nNote: use {{}} for defaults.",
}.items():
    got = DR._extract_outline_json(raw)
    check(f"outline survives: {name}",
          isinstance(got, dict) and got.get("title") == "Survey"
          and got.get("sections"),
          json.dumps(got, ensure_ascii=False)[:80])

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
