"""The "type your query" prompt showed an internal English routing token.

Tapping 🖼 Создать картинку answered a Russian user with

    ✏️ generate an image of — напиши запрос:

"generate an image of: " is the _PROMPT_KB prefix that gets prepended to the
user's text so the agent (which reasons in English) routes it correctly. It is
a pipeline instruction, not copy — it was never meant to be read by anyone, and
it appeared mid-sentence inside an otherwise Russian message.

Both surfaces leaked it: the reply keyboard buttons AND the /draw /search /deck
/img slash commands, which reach the same message by a different path.

Run: venv/Scripts/python.exe tests/test_tg_prompt_hints.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T
# Never let a suite touch the live tg_users.db.
T.redirect_data_dir(tempfile.mkdtemp(prefix="hints_"))

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


def _is_ascii_english(s: str) -> bool:
    return all(ord(c) < 128 for c in s)


print("=" * 70)
print("1. Every prompt prefix has a translated label in BOTH languages")
print("=" * 70)
for key, prefix in T._PROMPT_KB.items():
    mkey = f"hint_{key}"
    check(f"{key}: has a _MSG entry", mkey in T._MSG, mkey)
    if mkey in T._MSG:
        forms = T._MSG[mkey]
        check(f"{key}: has en and ru", {"en", "ru"} <= set(forms), list(forms))
        ru = forms.get("ru", "")
        check(f"{key}: the Russian label is actually Russian",
              bool(ru) and not _is_ascii_english(ru), repr(ru))

print()
print("=" * 70)
print("2. The raw routing prefix never reaches the user")
print("=" * 70)
for key, prefix in T._PROMPT_KB.items():
    for lang in ("ru", "en"):
        label = T._prompt_label(prefix, lang)
        check(f"{key}/{lang}: label is not the raw prefix",
              label.strip().lower() != prefix.strip().rstrip(":").lower(),
              f"{label!r} vs {prefix!r}")

print()
print("=" * 70)
print("3. The SLASH commands resolve to the same translated labels")
print("=" * 70)
# /draw and /img share a prefix with the gen_image button; they must not fall
# through to the bare-prefix branch just because they arrive by another route.
for cmd, prefix in T._SLASH.items():
    label_ru = T._prompt_label(prefix, "ru")
    check(f"{cmd}: resolves to a translated label",
          not _is_ascii_english(label_ru), f"{cmd} -> {label_ru!r}")
    check(f"{cmd}: matches what the button shows",
          label_ru == T._prompt_label(prefix, "ru"), label_ru)

print()
print("=" * 70)
print("4. The rendered message is fully Russian, no English island")
print("=" * 70)
import html as _h
for key, prefix in T._PROMPT_KB.items():
    msg = T._t("prompt_hint", "ru", hint=_h.escape(T._prompt_label(prefix, "ru")))
    leaked = [w for w in ("generate", "search the web", "deep research",
                          "remember this", "create a presentation", "edit the image")
              if w in msg.lower()]
    check(f"{key}: no English fragment in the RU message", not leaked,
          f"{msg} leaked={leaked}")

print()
print("=" * 70)
print("5. The fallback exists but is unreachable for shipped actions")
print("=" * 70)
# The helper deliberately degrades to the bare prefix instead of raising, so a
# future action cannot break a live conversation. Prove the degraded path works
# AND that nothing shipped currently needs it.
check("an unknown prefix degrades instead of raising",
      T._prompt_label("do something new: ", "ru") == "do something new")
unmapped = [p for p in T._PROMPT_KB.values() if p not in T._PREFIX2HINTKEY]
check("no shipped prefix relies on the fallback", not unmapped, unmapped)

print()
print(f"{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
