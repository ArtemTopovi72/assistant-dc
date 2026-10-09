---
type: Steering
description: Authoritative steering for the signals/wiki inferrer when operating under docs/wiki/.
---

<!-- steering note: user hints to correct framework detection / domain grouping / build-test
 commands; the inferrer reads this and treats it as authoritative. -->

## Framework

Python 3 desktop/Telegram AI assistant (PyQt5 GUI + Telegram bot) over local models: LLM via LM Studio, F5-TTS voice, ComfyUI image/video engines, YuE2 music. Not a web framework.

## Privacy (hard rule — this repo is PUBLIC)

- Never write personal data into any wiki page or card: Telegram user IDs, home-directory paths (write repo-relative paths only), real people's names or photos, usernames, emails.
- Never mention the local-only game-integration stack, even though some tracked code references it. Its paths are listed in the local (gitignored) `.claude/atomic.toml` `[scan] ignore`; leave those files and any code branches serving them out entirely.
- Never quote system prompts / persona texts verbatim (agent prompts, `personalities/`); describe their role in one line at most.

## Domains

<!-- none yet; let the inferrer cluster -->

## Build

- Test: `python tests/run_all.py`
- No build step.

## Ignore for domains

- runtime/
- venv/, venv_applio/
- tests/test_ozon_tools.py
- .claude/ (including .claude/worktrees/)
- Everything in `[scan] ignore` of `.claude/atomic.toml`
