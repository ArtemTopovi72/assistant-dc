"""Reset every stored per-user image-quality choice, so all users follow the
config default again.

The default tier moved to the cheapest one (config.DEFAULT_IMAGE_QUALITY), but a
user who had ever opened Settings and picked a tier carries that choice in their
session and would keep rendering at the old size. This clears the stored value;
it does not pin anyone to "draft" — an empty value means "follow the default",
so anyone can raise it again in Settings and it sticks for them.

NB the setting lives in tg_sessions.json, NOT in tg_users.db: the users database
holds accounts (approval, password hash, quotas), while per-chat preferences live
in the session store.

    venv/Scripts/python.exe scripts/reset_image_quality.py --dry-run
    venv/Scripts/python.exe scripts/reset_image_quality.py --apply

A timestamped backup is written beside the file before anything is changed.
"""
import argparse
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as _config

SESSION_FILE = os.path.join(ROOT, "tg_sessions.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="write the change (default is a dry run)")
    ap.add_argument("--dry-run", action="store_true", help="show what would change")
    ap.add_argument("--file", default=SESSION_FILE, help="session store to operate on")
    args = ap.parse_args()

    if not os.path.exists(args.file):
        print(f"no session store at {args.file} — nothing to do")
        return 0

    with open(args.file, encoding="utf-8") as fh:
        data = json.load(fh)

    explicit = {k: v.get("image_quality", "") for k, v in data.items()
                if isinstance(v, dict) and v.get("image_quality")}

    print(f"session store : {args.file}")
    print(f"sessions      : {len(data)}")
    print(f"default tier  : {_config.DEFAULT_IMAGE_QUALITY} "
          f"({_config.IMAGE_QUALITIES.get(_config.DEFAULT_IMAGE_QUALITY)} MP)")
    print(f"explicit picks: {len(explicit)}")
    for chat_id, q in sorted(explicit.items()):
        print(f"    chat {chat_id}: {q!r} -> '' (follows the default)")

    if not explicit:
        print("\nnothing to reset — every session already follows the default")
        return 0

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to make the change.")
        return 0

    backup = f"{args.file}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copyfile(args.file, backup)
    print(f"\nbackup written: {backup}")

    for chat_id in explicit:
        data[chat_id]["image_quality"] = ""

    # Write via a temp file and replace, so an interrupted run cannot leave the
    # live session store truncated.
    tmp = f"{args.file}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, args.file)
    print(f"reset {len(explicit)} session(s) — all users now follow "
          f"{_config.DEFAULT_IMAGE_QUALITY!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
