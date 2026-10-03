"""Mutation harness for tests/test_tg_pipeline.py.

Each mutant breaks one real behaviour in the intake pipeline. A survivor means the
suite is decorative for that behaviour.

Run: PYTHONIOENCODING=utf-8 venv/Scripts/python.exe tests/_mut_pipeline.py
"""
import os, sys, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

SUITE = "tests/test_tg_pipeline.py"

MUTANTS = [
    # ── quota ────────────────────────────────────────────────────────────────
    ("an image request no longer charges the task bucket",
     "        self._user_store.bump_usage(chat_id, KIND_TASK)\n"
     "        if kind != KIND_TASK:",
     "        if kind != KIND_TASK:"),

    ("the quota check is skipped entirely",
     "        kind = _classify_task(merged)\n        if not is_admin:",
     "        kind = _classify_task(merged)\n        if False:"),

    ("a spent quota refuses but still charges the user",
     '                                       user.name if user else str(chat_id))\n'
     '                    return',
     '                                       user.name if user else str(chat_id))\n'
     '                    pass'),

    ("admins lose their quota exemption",
     "        is_admin = bool(user and user.is_admin)",
     "        is_admin = False"),

    ("deep research is classified as an ordinary task",
     '    if low.startswith("do a deep research on:"):\n        return KIND_RESEARCH',
     '    if False:\n        return KIND_RESEARCH'),

    ("image intents are classified as ordinary tasks",
     "    if _IMAGE_INTENT_RE.match(text or ''):".replace("'", '"'),
     "    if False:"),

    # ── wait estimate ────────────────────────────────────────────────────────
    ("a deep research is estimated like any other task",
     "    secs = sum(per_res if _classify_task(getattr(t, \"user_text\", \"\")) == KIND_RESEARCH\n"
     "               else per_task for t in tasks)",
     "    secs = sum(per_task for t in tasks)"),

    # ── debounce / albums ────────────────────────────────────────────────────
    ("the debounce window collapses to nothing (one task per message)",
     "            batch = [first]",
     "            batch = [first]\n            self._resolve_and_push(chat_id, batch); continue"),

    ("an album keeps only its first photo's caption",
     '        cap = next((p["caption"] for p in entry["photos"] if p["caption"]), "")',
     '        cap = entry["photos"][0]["caption"] if entry["photos"] else ""'),

    ("flushing an album twice enqueues it twice",
     "        with self._album_lock: entry = self._albums.pop(group_id, None)",
     "        with self._album_lock: entry = self._albums.get(group_id, None)"),

    # ── documents ────────────────────────────────────────────────────────────
    # The early `return` after the "could not read it" notice and the later
    # "nothing to say and no image" guard both stop a blank turn, so removing
    # either alone is equivalent. Remove both — that is the regression that would
    # actually send the agent an empty request.
    ("an unreadable document is queued anyway (both guards removed)",
     [('_t("doc_unreadable", lang, name=_html_mod.escape(fname)),\n'
       '                        parse_mode="HTML",\n'
       '                        keyboard=self._main_menu_kb(sess, lang)); return',
       '_t("doc_unreadable", lang, name=_html_mod.escape(fname)),\n'
       '                        parse_mode="HTML",\n'
       '                        keyboard=self._main_menu_kb(sess, lang))'),
      ('        if not merged and img_bytes is None:\n            return',
       '        if False:\n            return')],
     None),

    ("document extraction ignores the file type",
     '        if ext in (".txt", ".md", ".csv", ".log", ".yaml", ".yml", ".json"):',
     "        if True:"),

    ("a failed download still produces a turn",
     "        merged = \"\\n\".join(texts).strip()\n"
     "        if not merged and img_bytes is None:\n            return",
     "        merged = \"\\n\".join(texts).strip()\n        if False:\n            return"),

    # ── stop / cancel ────────────────────────────────────────────────────────
    ("Stop no longer marks earlier tasks stale",
     "        return bool(ts) and task.enqueue_ts <= ts",
     "        return False"),

    ("one chat's Stop marks every chat's tasks stale",
     "            ts = self._stop_requests.get(task.chat_id, 0.0)",
     "            ts = max(self._stop_requests.values() or [0.0])"),

    ("the cancelled-id set grows without bound",
     "            if len(self._cancelled) > 512:\n                self._cancelled.clear()",
     "            if False:\n                self._cancelled.clear()"),

    # ── offset ───────────────────────────────────────────────────────────────
    ("the poll offset never advances (every update replays forever)",
     '            self._offset = result[-1]["update_id"] + 1',
     '            self._offset = self._offset'),

    ("the offset is not persisted across a restart",
     "            self._offset = result[-1][\"update_id\"] + 1\n            self._save_offset(self._offset)",
     "            self._offset = result[-1][\"update_id\"] + 1"),

    ("a corrupt offset file propagates its exception",
     "        try:\n            if _OFFSET_FILE.exists():\n"
     "                return int(json.loads(_OFFSET_FILE.read_text()))\n"
     "        except Exception:\n            pass",
     "        if _OFFSET_FILE.exists():\n"
     "            return int(json.loads(_OFFSET_FILE.read_text()))"),

    # ── crash recovery ───────────────────────────────────────────────────────
    ("a crash no longer tells the user their task died",
     '                self._send_text(task.chat_id, _t("interrupted", lang),',
     '                _ = (task.chat_id, _t("interrupted", lang),'),

    ("the interrupted text is not stashed, so Retry has nothing to run",
     "                sess.last_task_text = task.user_text",
     "                pass"),

    ("a corrupt inflight file is left in place to wedge every restart",
     "        try:\n            _INFLIGHT_FILE.unlink()\n        except Exception:\n            pass",
     "        pass"),

    ("one bad inflight entry aborts the whole recovery",
     "            except Exception:\n                logger.exception"
     '("inflight recovery failed for %s", d)',
     "            except Exception:\n                raise"),
]

env = dict(os.environ, PYTHONIOENCODING="utf-8")
path = os.path.join(ROOT, "bot/tg_bot.py")
with open(path, encoding="utf-8") as fh:
    ORIGINAL = fh.read()

caught = survived = skipped = 0
for label, find, repl in MUTANTS:
    edits = find if isinstance(find, list) else [(find, repl)]
    bad = [f for f, _ in edits if ORIGINAL.count(f) != 1]
    if bad:
        print(f"SKIP (anchor missing x{len(bad)})  {label}")
        skipped += 1
        continue
    mutated = ORIGINAL
    for f, rep in edits:
        mutated = mutated.replace(f, rep, 1)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(mutated)
        r = subprocess.run([sys.executable, SUITE], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=600)
    finally:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(ORIGINAL)
    if r.returncode != 0:
        caught += 1
        fails = [l for l in (r.stdout or "").splitlines() if l.startswith("FAIL")]
        print(f"CAUGHT    {label}\n            by: "
              f"{(fails[0][6:].strip() if fails else '(crashed)')[:95]}")
    else:
        survived += 1
        print(f"SURVIVED  {label}   <-- not caught")

print()
print(f"{caught} caught, {survived} survived, {skipped} skipped ({len(MUTANTS)} mutants)")
sys.exit(1 if survived or skipped else 0)
