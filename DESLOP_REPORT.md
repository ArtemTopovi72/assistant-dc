# Deslop integration report — 2026-10-11

## Merge
- Base: `origin/main` 49695cc (tag `pre-deslop`), branch `deslop-integration`.
- All 35 `origin/deslop/*` branches merged `--no-ff`: the 21 top branches in handoff group order, then the 14 lower ones.
- One conflict, resolved as the handoff prescribes: `tests/test_text_beside_voice_for_latin.py` deleted (in `tests-reportonly-4`).
- Branches were merged in one pass, then tested together. Each failure was traced to its file, so bisecting was not needed.

## Unit suite (`tests/run_all.py -j 8`, offscreen)
| Run | Result |
|---|---|
| Baseline, clean main | 535/535 |
| After the 35 merges | 530/534. 4 suites failed |
| After the fixes | 534/534 |

The baseline has one more suite (535 against 534) because `test_text_beside_voice_for_latin.py` was deleted, as instructed.

Failures and fixes (commit `fix(deslop): …`):
- `test_tg_voice_clone`: the test patched `tg_voice_clone.threading`, but deslop removed that unused import. The test now patches `threading.Thread` directly; the code is unchanged.
- `test_deep_research_fetch`: deslop removed the try/except around `list.append` in `snap_formula`. The deleted test block simulated a failing `append`, which a plain list cannot do. The code change stands.
- `test_gui_supplement11` / `test_gui_supplement8`: `_default_window_size` had been narrowed to `except AttributeError`. Restored to `except Exception`: if the screen query fails at startup, the window must still open.

## Bugs from the handoff's "probable bugs" list (commit `fix(research): …`)
- **Confirmed and fixed:** `knob_defaults` used `vars(deep_research)`, which cannot see the knobs behind the dr_settings proxy. Every default came back `None`. It now uses `getattr`, in both `research_client.py` and `research_service.py`. Regression test: `tests/test_research_knob_defaults.py`.
- **Fixed:** `weather.what_to_wear_periods` resolved the city without the chat's `lang`.
- Not checked in this pass: `tg_transport` getattr on a ContextVar, the bench knowledge shadowing, `validate_geometry`, tests after `sys.exit`, the chat id in `shot_gui_tabs.py`.

## Live GPU checks (Gemma 4 26B in LM Studio, ComfyUI)
- **`bench/tg_sim.py`, 10 requests.** Chat, picture (sendPhoto), music (sendAudio), video (sendVideo), deck (sendDocument), search, reminder and weather all answered.
  - "Скажи голосом…" came back as text, not voice.
  - "Напиши документ docx" came back as text: "не могу создать файл".
  - Both prompts behave the same on `pre-deslop`, so these are existing gaps, not regressions.
  - The search answer named the 2022 champion as the latest. That is an LLM freshness issue that predates deslop.
- **`bench/live_tg_scenarios.py` music / picture / chat** (the handoff's `live_tg_drive --scenario` is the library; the scenarios run from `live_tg_scenarios.py`): no findings.
- **`bench/tc_run.py`:** 60/69 (87.0%) on `deslop-integration` and 60/69 on `pre-deslop`, the same in every category.
- **`bench/planbench.py`:** not run. Its default model (qwen3.5-9b) is not the house model, and `tc_run` covers the same planning path.
- **GUI:** `tests/shot_gui_tabs.py` opened all 24 tabs plus settings with no errors. The main and Research tabs were checked by eye.

## Local main
Local `main` had 7 unpushed commits (video continue / plan / close-up inserts, fed9c99..bd36433) on top of 49695cc. They are merged together with `deslop-integration`, and the suite is re-run on the result before the push.
