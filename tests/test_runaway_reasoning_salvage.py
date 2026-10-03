"""Regression: an UNTERMINATED <|channel>thought block must not delete a real answer.

BUG (observed live on google/gemma-4-26b-a4b-qat): the model opened
`<|channel>thought`, wrote its reasoning, then wrote a finished Markdown report —
and never emitted a closing channel marker. _CHANNEL_BLOCK_RE therefore ran to
end-of-string and strip_control_tokens returned "", so the user got the empty
"(no response)" placeholder while a 13.8k-char document sat in the raw reply.

The salvage must be narrow: it may only fire when stripping left NOTHING, only
for a block that runs to end-of-string, and only when the tail looks like a
document (heading / list), never when it is ordinary reasoning prose.

Run: venv/Scripts/python.exe tests/test_runaway_reasoning_salvage.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

from utils import strip_control_tokens, strip_think_tags

ok = fail = 0

def check(label, cond, extra=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"PASS {label}")
    else:
        fail += 1
        print(f"FAIL {label} {extra}")


REPORT = (
    "# Отчёт по рынку тракторов\n\n"
    "## Введение\n\n"
    "Рынок сельхозтехники в 2026 году показывает устойчивый рост, "
    "обусловленный обновлением парка и программами субсидирования.\n\n"
    "## Основные игроки\n\n"
    "- Ростсельмаш\n- Кировец\n- John Deere\n\n"
    "## Динамика\n\nПоставки тяжёлых моделей растут третий год подряд, тогда как "
    "лёгкий сегмент стагнирует и постепенно вытесняется импортом.\n\n"
    "## Вывод\n\nСпрос смещается в сторону тяжёлых моделей."
)

# ── 1. the live failure: runaway block swallows a finished document ───────────
runaway = ("<|channel>thought\nThe user wants a market report. Let me plan the "
           "sections: introduction, players, conclusion. I should write it in "
           "Russian because the user asked in Russian. Now writing it.\n\n" + REPORT)

out = strip_control_tokens(runaway)
check("runaway block: answer is salvaged, not emptied", out.strip() != "",
      f"got {out!r}")
check("runaway block: salvage starts at the heading",
      out.strip().startswith("# Отчёт"), f"got {out[:60]!r}")
check("runaway block: the whole document survives",
      "Ростсельмаш" in out and "Спрос смещается" in out)
check("runaway block: the reasoning is NOT shown",
      "Let me plan the sections" not in out and "user wants a market report" not in out)
check("strip_think_tags agrees with strip_control_tokens",
      "Ростсельмаш" in strip_think_tags(runaway))

# ── 2. pure reasoning must still yield empty (no resurrection) ────────────────
pure = ("<|channel>thought\nThe user greeted me. I should greet back warmly. "
        "Maybe mention the weather. Actually no, keep it short. Let me think "
        "about the tone once more before answering the greeting properly.")
check("pure reasoning still strips to empty", strip_control_tokens(pure).strip() == "",
      f"got {strip_control_tokens(pure)!r}")

# ── 3. a short list-ish tail is below the salvage floor → still empty ─────────
short = "<|channel>thought\nHmm.\n\n- maybe A\n- maybe B"
check("short tail is not salvaged (below the 200-char floor)",
      strip_control_tokens(short).strip() == "", f"got {strip_control_tokens(short)!r}")

# ── 4. a properly TERMINATED block is untouched by the salvage path ───────────
terminated = ("<|channel>thought\nplanning the answer, considering the options "
              "carefully and at some length so it is not trivially short<channel|>"
              "Столица Франции — Париж.")
out4 = strip_control_tokens(terminated)
check("terminated block: answer kept", "Париж" in out4, f"got {out4!r}")
check("terminated block: reasoning dropped", "planning the answer" not in out4)

# ── 5. harmony spelling with a real final channel is unaffected ───────────────
harmony = ("<|start|>assistant<|channel>analysis<|message|>the user asks for the "
           "capital, I recall it is Paris<|channel>final<|message|>Paris.<|return|>")
out5 = strip_control_tokens(harmony)
check("harmony: final answer kept", out5.strip() == "Paris.", f"got {out5!r}")
check("harmony: analysis dropped", "I recall" not in out5)

# ── 6. clean text is passed through byte-identically ─────────────────────────
clean = "Просто обычный ответ без всякой разметки."
check("clean text untouched", strip_control_tokens(clean) == clean)

# ── 7. a numbered-list document (no heading) is salvaged too ──────────────────
numbered = ("<|channel>thought\nThe user asked for five jokes. Let me recall "
            "them and write them out in full rather than summarising.\n\n"
            "1. Василий Иванович спрашивает Петьку про приборы, и тот отвечает "
            "что приборы показывают тридцать, а сколько тридцать никто не знает.\n"
            "2. Петька докладывает об обстановке на фронте очень подробно.\n"
            "3. Анка садится за пулемёт и открывает огонь по позициям.\n"
            "4. Василий Иванович сдаёт экзамен по математике с трудом.\n"
            "5. Петька летит на аэроплане и теряет управление над рекой.")
# A LIST is NOT enough to salvage on, and this is the important case. Reasoning
# is full of lists ("- first I should check…"), so accepting them shipped 13k
# characters of the model's private deliberation as a live research report —
# English headings, first-person planning, 30% Cyrillic in a Russian document.
# Only a Markdown heading counts as document structure.
out7 = strip_control_tokens(numbered)
check("a bare numbered list is NOT salvaged (it reads like reasoning)",
      out7.strip() == "", f"got {out7[:80]!r}")

reasoning_with_bullets = (
    "<|channel>thought\nThe user asks about tractors. Let me plan the work:\n\n"
    "- first I should check what the K-700 is and when it was made\n"
    "- then compare it against the K-744 on power, cab and transmission\n"
    "- I must remember to answer in Russian rather than in English\n"
    "- finally I will write the sections out and cite each of the sources\n")
check("reasoning that merely contains bullets is never emitted",
      strip_control_tokens(reasoning_with_bullets).strip() == "",
      strip_control_tokens(reasoning_with_bullets)[:80])

headed = ("<|channel>thought\nplanning the document at some considerable length\n\n"
          "# Отчёт\n\n## Раздел\n\n" + "Содержательный текст раздела. " * 20)
check("a real heading IS still salvaged",
      strip_control_tokens(headed).strip().startswith("# Отчёт"),
      strip_control_tokens(headed)[:60])

print(f"\n{ok}/{ok + fail} checks passed")
sys.exit(1 if fail else 0)
