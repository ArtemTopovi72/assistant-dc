"""Restore verified legacy archives. Stop the bot before applying a reviewed manifest.

Preview: python scripts/migrate_context_archives.py --manifest private-plan.json
Apply:   python scripts/migrate_context_archives.py --manifest private-plan.json --apply
Manifest: {"a1234567": {"chat_id": "101", "generation": 4, "active": true}}.
Ownership evidence and manifests contain private identifiers; keep them outside Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from datetime import datetime
from pathlib import Path

if not __package__:
    import install_paths
    install_paths.ensure(write_venv=False)

import context_v2 as context
from models import _atomic_write_json


def migrate(root: Path, assignments: dict, *, apply=False) -> dict:
    root = Path(root)
    session_path = root / 'tg_sessions.json'
    original = session_path.read_bytes()
    sessions = json.loads(original)
    source = root / 'runtime' / 'ctx_archive'
    memory = root / 'tg_memory'
    copies = []
    active = retired = 0
    for aid, owner in assignments.items():
        cid = str(owner.get('chat_id', ''))
        if (not re.fullmatch(r'a[0-9a-f]{7}', aid) or not re.fullmatch(r'-?\d+', cid)
                or cid not in sessions or type(owner.get('active')) is not bool):
            raise ValueError(f'Invalid archive ownership: {aid}')
        path = source / f'{aid}.json'
        if path.resolve().parent != source.resolve():
            raise ValueError(f'Invalid source path: {aid}')
        entry = json.loads(path.read_bytes())
        text = entry.get('text')
        if (entry.get('id') != aid or not isinstance(text, str)
                or 'a' + hashlib.sha1(text.encode('utf8', 'replace')).hexdigest()[:7] != aid):
            raise ValueError(f'Archive hash mismatch: {aid}')
        sess = sessions[cid]
        generation = sess.get('context_generation', 0)
        if type(generation) is not int or generation < 0:
            raise ValueError(f'Invalid context generation for {cid}')
        if owner['active'] and owner.get('generation') != generation:
            raise ValueError(f'Context generation changed for {cid}; review the manifest again')
        directory = memory / f'chat_{cid}'
        directory /= Path('archive') / str(generation) if owner['active'] else Path('legacy_archive')
        destination = directory / path.name
        if not destination.resolve().is_relative_to(memory.resolve()):
            raise ValueError(f'Invalid destination path: {aid}')
        exists = destination.exists()
        if exists and json.loads(destination.read_bytes()) != entry:
            raise ValueError(f'Archive destination conflict: {aid}')
        copies.append((destination, entry, exists))
        if not owner['active']:
            retired += 1
            continue
        active += 1
        history = sess.setdefault('history', [])
        working = next((m for m in history if m.get('role') == 'system'
                        and str(m.get('content', '')).startswith(context.MEMORY_MARKER)), None)
        state = context.parse(working['content']) if working else context.empty_state()
        if any(re.search(r"recall_context\(id='" + aid + r"'\)", str(m.get('content', '')))
               for m in history if m.get('role') in ('system', 'tool')):
            continue
        date = datetime.fromtimestamp(path.stat().st_mtime).date()
        state['folded'].append(f"[legacy] Historical conversation {date} -> recall_context(id='{aid}')")
        if working:
            dashboard = next((line for line in working['content'].splitlines() if line.startswith('[Context:')), '')
            working['content'] = context.render(state, dashboard)
        else:
            index = 1 if history and history[0].get('role') == 'system' else 0
            history.insert(index, {'role': 'system', 'content': context.render(state)})
    changed = sessions != json.loads(original)
    report = {'active': active, 'retired': retired, 'copy_files': sum(not exists for _, _, exists in copies),
              'history_changed': changed, 'applied': False, 'backup': None}
    if not apply or not (changed or report['copy_files']):
        return report
    if session_path.read_bytes() != original:
        raise RuntimeError('Sessions changed during planning; stop the bot and retry')
    backup = memory / '_legacy_migration_backups' / uuid.uuid4().hex / 'tg_sessions.json'
    backup.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(backup, json.loads(original))
    report['backup'] = str(backup)
    for destination, entry, exists in copies:
        if not exists:
            if destination.exists():
                if json.loads(destination.read_bytes()) != entry:
                    raise ValueError(f'Archive destination conflict: {destination.name}')
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(destination, entry)
    if changed:
        if session_path.read_bytes() != original:
            raise RuntimeError(f'Sessions changed; history left untouched. Backup: {backup}')
        _atomic_write_json(session_path, sessions)
    report['applied'] = True
    _atomic_write_json(memory / 'legacy_migration_report.json', dict(report, assignments=assignments))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    result = migrate(args.root, json.loads(args.manifest.read_text('utf8')), apply=args.apply)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
