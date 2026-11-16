"""Qualify bundled additions into a private bank without losing private experts.

This reads only the installed checkout. It does not fetch network updates.
All candidate admissions happen in a temporary bank; a single compare-and-swap
snapshot commit publishes the result after the whole reconciliation succeeds.
"""
from pathlib import Path
import tempfile

from skill_bank import RegistryError, SkillBank, digest


class ReconciliationConflict(RegistryError):
    """A concurrent private update is authoritative; a later refresh can retry."""


def reconcile(private_bank, bundled_path):
    """Append missing bundled experts through normal gates, preserving old order.

    A rejected expert or concurrent private change raises RegistryError without
    publishing any staged expert. The caller may retain the existing private
    bank and retry on its next refresh. Local checksum trust is unchanged.
    """
    source, _ = SkillBank(bundled_path)._read()
    previous, _ = private_bank._read()
    existing = {entry['id'] for entry in previous['entries']}
    pending = [entry for entry in source['entries'] if entry['id'] not in existing]
    if not pending:
        return {'action': 'unchanged', 'revision': previous['revision'], 'added': 0}
    expected = digest(previous)
    with tempfile.TemporaryDirectory(prefix='airi-managed-skills-') as directory:
        staged = SkillBank(Path(directory) / 'bank.json')
        if previous['revision']:
            staged._write(previous)
        for entry in pending:
            result = staged.learn({
                'trusted': True, 'lesson_id': entry['lesson_id'],
                'examples': entry['train'], 'qualification': entry['qualification'],
            })
            if result['action'] != 'activated':
                raise RegistryError('bundled expert admission failed; private bank retained')
        candidate, _ = staged._read()
    with private_bank._lock():
        current, _ = private_bank._read()
        if digest(current) != expected:
            raise ReconciliationConflict('private bank changed during reconciliation; retry refresh')
        publication = private_bank._write(candidate)
    return {
        'action': 'activated', 'revision': candidate['revision'],
        'added': len(pending), **publication,
    }
