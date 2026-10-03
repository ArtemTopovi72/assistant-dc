"""The live status a user watches during a turn.

`ctx.set_stage(...)` is called from the shared agent core — graph.py, tools.py,
image.py, llm.py — and those strings are English by design: the pipeline reasons
in English and the same text doubles as an operator label in the activity log and
the admin panel. So a Russian user got a fully Russian bot that then narrated
every turn in English: "🔎 Searching the web…", "🎨 Drawing a picture…".

stages.translate() sits at the presentation boundary. Two things make it safe:

  · passthrough for anything unknown — an untranslated stage is the status quo,
    so it can only ever be neutral, never a blank status or a crash;
  · the icon in tg_bot is chosen from the ENGLISH label BEFORE translating, or
    every icon silently disappears in Russian.

The last section is the one that matters over time: it re-derives the stage
vocabulary from the source and fails when somebody adds a set_stage() call the
table has never heard of.

Run: venv/Scripts/python.exe tests/test_stages_i18n.py
"""
import sys, os, re, ast, types, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import stages

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")

CYR = re.compile(r"[А-Яа-яЁё]")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


print("=" * 70)
print("FIXED LABELS")
print("=" * 70)

for en in sorted(stages._STAGES):
    ru = stages.translate(en, "ru")
    check(f"translated: {en}", bool(CYR.search(ru)) and ru != en, repr(ru))

check("English is returned untouched",
      all(stages.translate(en, "en") == en for en in stages._STAGES))
check("the default language is English",
      stages.translate("Searching the web") == "Searching the web")
check("an unsupported language falls back to the English text",
      stages.translate("Searching the web", "de") == "Searching the web")


print()
print("=" * 70)
print("PASSTHROUGH IS ALWAYS SAFE")
print("=" * 70)

for bad, label in ((None, "None"), ("", "an empty string"), (123, "a non-string"),
                   ({}, "a dict")):
    try:
        got = stages.translate(bad, "ru"); ok = True
    except Exception as exc:
        got, ok = repr(exc), False
    check(f"{label} does not raise", ok, str(got))
    if ok:
        check(f"{label} is returned unchanged", got == bad, repr(got))

check("an unknown stage is shown as-is, not blanked",
      stages.translate("Refactoring the flux capacitor", "ru")
      == "Refactoring the flux capacitor")
check("an unknown stage is never returned empty",
      bool(stages.translate("Something New", "ru")))
check("a None language is treated as English",
      stages.translate("Speaking", None) == "Speaking")


print()
print("=" * 70)
print("PARAMETERISED STAGES")
print("=" * 70)

dr = stages.translate("Searching · 12 src / 40 pg / 7 found", "ru")
check("the deep-research progress line is translated", bool(CYR.search(dr)), repr(dr))
check("its numbers survive intact",
      all(n in dr for n in ("12", "40", "7")), repr(dr))
check("its English units are gone",
      not re.search(r"\b(src|pg|found)\b", dr), repr(dr))
check("its phase is translated",
      stages._PHASES["Searching"]["ru"] in dr, repr(dr))
check("the same line in English is untouched",
      stages.translate("Searching · 12 src / 40 pg / 7 found", "en")
      == "Searching · 12 src / 40 pg / 7 found")

for phase in stages._PHASES:
    line = f"{phase} · 1 src / 2 pg / 3 found"
    out = stages.translate(line, "ru")
    check(f"deep-research phase translated: {phase}",
          stages._PHASES[phase]["ru"] in out, repr(out))

# An unknown phase must still get its units translated rather than fall through.
odd = stages.translate("Percolating · 5 src / 6 pg / 7 found", "ru")
check("an unknown phase keeps its name but still translates the units",
      "Percolating" in odd and not re.search(r"\bsrc\b", odd), repr(odd))

retry = stages.translate("Retrying the model call (2/3)", "ru")
check("the retry stage is translated", bool(CYR.search(retry)), repr(retry))
check("the retry counters survive", "2/3" in retry, repr(retry))
check("a malformed retry line falls through",
      stages.translate("Retrying the model call", "ru")
      == "Retrying the model call")

ultra = stages.translate("Ultra: Searching", "ru")
check("a composite Ultra stage translates its inner phase",
      stages._PHASES["Searching"]["ru"] in ultra, repr(ultra))
check("and keeps the Ultra prefix", ultra.startswith("Ultra: "), repr(ultra))

check("bare phase names translate on their own",
      CYR.search(stages.translate("Building report", "ru")),
      repr(stages.translate("Building report", "ru")))


print()
print("=" * 70)
print("THE BOT SHOWS THE TRANSLATION AND LOGS THE ENGLISH")
print("=" * 70)

import tg_bot as T
T.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_stage_"))

CID = 999901

class _Ctx:
    def __init__(self):
        import threading
        self.cancel_event = threading.Event()
        self.stage_callback = None
        self.session_memory = []
        self.memory_lock = threading.Lock()
        self.memory_text = lambda *a, **k: ""
        self.pinned_facts = []
        self.total_user_turns = 0
        self.last_image_path = ""
        self.last_image_prompt = ""
        self.tts_disabled = True
    def set_stage(self, s):
        if self.stage_callback: self.stage_callback(s)


def drive(lang, stage):
    """Run one turn far enough to emit a stage, and capture every sink."""
    ctx = _Ctx()
    seen = {"status": [], "gui": [], "log": [], "panel": []}

    bot = T.TelegramBot("123:TEST", lambda: ctx, lambda: graph,
                        lambda: {"messages": []}, silent_mode=True,
                        on_stage=lambda cid, s, done: seen["gui"].append(s))
    bot._send_text = lambda cid, text, **k: 1
    bot._send_get_id = lambda cid, text, **k: 11
    bot._edit_text = lambda cid, mid, text, **k: seen["status"].append(text)
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._activity.log = lambda cid, ev, text, uname="": (
        seen["log"].append(text) if ev == "stage" else None)
    bot._send_voice_text = lambda *a, **k: True

    class _Graph:
        def invoke(self, state, cfg=None):
            ctx.set_stage(stage)
            return {"messages": [types.SimpleNamespace(content="done")]}
    graph = _Graph()

    T._User(chat_id=CID, name="U", status="approved")
    bot._user_store.put(T._User(chat_id=CID, name="U", status="approved"))
    sess = bot._get_session(CID); sess.lang = lang; sess.voice_on = False
    bot._store.put(sess)

    task = T._Task(task_id="t1", chat_id=CID, user_text="hello")
    bot._run_task_inner(task, bot._get_session(CID), bot._user_store.get(CID),
                        "U", lang, ctx, graph)
    with bot._stages_lock:
        seen["panel"] = list(bot._active_stages.values())
    return seen


ru = drive("ru", "Searching the web")
status_ru = " ".join(ru["status"])
check("the Russian user's live status is Russian",
      bool(CYR.search(status_ru)) and "Searching the web" not in status_ru,
      repr(status_ru[:100]))
check("and it keeps the stage icon",
      "🔎" in status_ru, repr(status_ru[:100]))
check("the activity log stays English",
      any("Searching the web" == t for t in ru["log"]), str(ru["log"]))
check("the desktop feed stays English",
      any("Searching the web" in s for s in ru["gui"]), str(ru["gui"]))

en = drive("en", "Searching the web")
status_en = " ".join(en["status"])
check("the English user's live status is unchanged",
      "Searching the web" in status_en, repr(status_en[:100]))
check("and it has the same icon", "🔎" in status_en, repr(status_en[:100]))

# Every icon must still be selected, in Russian too — this is what breaks if the
# icon lookup is moved after the translation.
ICON_CASES = {
    "Searching the web": "🔎", "Drawing a picture": "🎨",
    "Editing the picture": "✏️", "Looking at the image": "👁",
    "Compacting the chat history": "📦", "Speaking": "🔊",
    "Upscaling (face-safe)": "🔍", "Removing from the picture": "🗑",
    "Restoring the photo": "🪄",
}
for stage, icon in ICON_CASES.items():
    got = " ".join(drive("ru", stage)["status"])
    check(f"icon survives translation: {stage}", icon in got, repr(got[:90]))

# An unknown stage must still reach the user rather than vanish.
unk = " ".join(drive("ru", "Doing something unheard of")["status"])
check("an unknown stage still reaches the Russian user",
      "Doing something unheard of" in unk, repr(unk[:90]))


# Deep research does NOT go through the agent loop — it is dispatched directly,
# with its own status callback. That second path has to translate too, and it is
# the one a user stares at for ten minutes.
def drive_research(lang, phases):
    ctx = _Ctx()
    seen = {"status": [], "gui": []}
    bot = T.TelegramBot("123:TEST", lambda: ctx, lambda: None,
                        lambda: {"messages": []}, silent_mode=True,
                        on_stage=lambda cid, s, done: seen["gui"].append(s))
    bot._send_text = lambda cid, text, **k: 1
    bot._send_get_id = lambda cid, text, **k: 11
    bot._edit_text = lambda cid, mid, text, **k: seen["status"].append(text)
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._send_report_file = lambda *a, **k: True

    fake_dr = types.ModuleType("deep_research")
    def _run(c, topic, progress=None, **kw):
        for phase in phases:
            progress(phase, {"sources": 3, "pages": 9, "findings": 4}, "")
        return {"report": "отчёт", "stats": {"sources": 3, "pages": 9,
                                             "findings": 4, "elapsed": 1.0}}
    fake_dr.run_deep_research = _run
    # tg_bot.py's deep-research bypass calls _dr.lang_of_text(topic) to pick
    # out_lang from the topic itself — this stub replaces the real module in
    # sys.modules, so it needs the attribute too or that call raises before
    # ever reaching the stage-translation behavior this test actually checks.
    fake_dr.lang_of_text = lambda text, default="en": default
    real = sys.modules.get("deep_research")
    sys.modules["deep_research"] = fake_dr

    bot._user_store.put(T._User(chat_id=CID, name="U", status="approved"))
    sess = bot._get_session(CID); sess.lang = lang; sess.voice_on = False
    bot._store.put(sess)
    task = T._Task(task_id="dr", chat_id=CID,
                   user_text="do a deep research on: термоядерный синтез")
    try:
        bot._run_task_inner(task, bot._get_session(CID),
                            bot._user_store.get(CID), "U", lang, ctx, None)
    finally:
        if real is not None: sys.modules["deep_research"] = real
        else: sys.modules.pop("deep_research", None)
    return seen


dr_ru = drive_research("ru", ["Searching", "Crawling", "Building report"])
dr_status = " ".join(dr_ru["status"])
check("the deep-research status is Russian",
      bool(CYR.search(dr_status)), repr(dr_status[:110]))
for phase in ("Searching", "Crawling", "Building report"):
    check(f"deep-research phase reaches the user translated: {phase}",
          stages._PHASES[phase]["ru"] in dr_status, repr(dr_status[:110]))
check("no English phase name leaks into the Russian status",
      not any(p in dr_status for p in ("Searching", "Crawling", "Building report")),
      repr(dr_status[:110]))
check("the deep-research feed stays English for the operator",
      any("Searching" in s for s in dr_ru["gui"]), str(dr_ru["gui"])[:110])

dr_en = drive_research("en", ["Searching"])
check("and an English user still sees English",
      "Searching" in " ".join(dr_en["status"]), str(dr_en["status"])[:110])


print()
print("=" * 70)
print("THE TABLE KEEPS UP WITH THE CODE")
print("=" * 70)

# Re-derive the vocabulary from the source. A new ctx.set_stage("…") anywhere in
# the agent core must either be translated or be listed as deliberately skipped —
# otherwise it silently reintroduces an English phrase into a Russian turn.
SKIP = {
    # Dynamic or already handled by a pattern above.
    "",
}

# Pinned on the STATE of the tree, not on a list of file names. A curated list
# went stale the moment the monoliths were split: graph.py's emitters moved to
# graph_fastpath/graph_finalize/graph_personality and image.py's to
# tool_image_handlers/image_transforms, so the scan found 7 stages instead of
# 25 and the reverse check then declared every real translation dead weight.
emitted = set()
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from install_paths import source_dirs
for path in sorted(os.path.join(d, f) for d in source_dirs() for f in os.listdir(d) if f.endswith(".py")):
    fname = os.path.basename(path)
    tree = ast.parse(open(path, encoding="utf-8").read())
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        name = f.attr if isinstance(f, ast.Attribute) else \
               (f.id if isinstance(f, ast.Name) else "")
        if name not in ("set_stage", "_set_stage") or not n.args:
            continue
        # A plain literal, or either branch of "a if cond else b".
        for arg in ([n.args[0]] if not isinstance(n.args[0], ast.IfExp)
                    else [n.args[0].body, n.args[0].orelse]):
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                emitted.add(arg.value)

check("stage emitters were actually found in the source", len(emitted) >= 15,
      f"{len(emitted)} found")
missing = sorted(s for s in emitted if s not in SKIP
                 and s not in stages.known_stages())
check("every literal stage in the agent core has a translation",
      not missing, "; ".join(missing))

# …and the reverse: a table entry for a stage nobody emits is dead weight that
# will quietly rot out of date.
BOT_OWNED = {"Deep Research", "Reading your documents", "Starting",
             "Transcribing", "Searching documents", "Ready", "Translating"}
stale = sorted(s for s in stages._STAGES
               if s not in emitted and s not in BOT_OWNED)
check("no translation exists for a stage nothing emits", not stale,
      "; ".join(stale))

# Both tables must be complete for every language they claim to support.
LANGS = {"ru"}
gaps = [f"{k}/{lang}" for table in (stages._STAGES, stages._PHASES, stages._UNITS)
        for k, forms in table.items() for lang in LANGS if not forms.get(lang)]
check("every entry has every supported language", not gaps, "; ".join(gaps[:6]))
untranslated = [k for table in (stages._STAGES, stages._PHASES)
                for k, forms in table.items()
                if forms.get("ru") and not CYR.search(forms["ru"])]
check("no entry was left as English in the Russian slot",
      not untranslated, "; ".join(untranslated))

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
