"""A single stable model identity; bounded, independently checked skill updates."""
import copy
import hashlib
from success_bank import canonical, row_key, priority

MODEL_ID = 'airi-native'


def new_state(weights_digest):
    return {'schema': 1, 'model_id': MODEL_ID, 'revision': 0,
            'weights_digest': weights_digest, 'parent_weights_digest': None,
            'round': 0, 'attempted_updates': 0, 'skills': {}, 'history': []}


def inherit_state(previous, weights_digest):
    if previous is None:
        return new_state(weights_digest)
    if (previous['schema']!=1 or previous['model_id']!=MODEL_ID
            or previous['weights_digest']!=weights_digest or not isinstance(previous['revision'],int)
            or isinstance(previous['revision'],bool) or previous['revision']<0):
        raise ValueError('cannot reset or attach a different AIRI model lineage')
    return copy.deepcopy(previous)


def adopt(state, weights_digest, *, step, reason):
    if weights_digest == state['weights_digest']:
        raise ValueError('unchanged weights cannot become a new model revision')
    state['parent_weights_digest'] = state['weights_digest']
    state['weights_digest'] = weights_digest
    state['revision'] += 1
    state['history'].append({'action': 'adopted', 'revision': state['revision'],
                             'step': step, 'reason': reason})
    state['history'] = state['history'][-64:]


def choose_skill(bank, state):
    groups = {}
    for entry in bank['entries'].values():
        row = entry['row']; key = row['language'] + ':' + row['family']
        groups.setdefault(key, []).append(entry)
    ready = []
    for key, entries in sorted(groups.items()):
        entries.sort(key=lambda e: (priority(e), row_key(e['row'])), reverse=True)
        digest = hashlib.sha256(canonical(entries).encode()).hexdigest()
        previous = state['skills'].get(key, {})
        same = previous.get('donor_digest') == digest and previous.get('base_revision') == state['revision']
        if same and state['round'] < previous.get('next_round', 0):
            continue
        ready.append((previous.get('attempts', 0), -sum(e['whole_answer_gain'] for e in entries), key, digest, entries))
    return min(ready)[2:] if ready else None


def consolidate(state, bank, before, *, train, evaluate, restore, can_train,
                can_adopt, step, max_updates=32):
    """One skill per transaction. The caller supplies a champion-based student.

    The train callback sees ONLY an archived training lesson. Evaluation examples
    are private to evaluate(); their answers cannot enter this function's bank.
    restore() must restore BOTH champion weights and Adam, preserving compute.
    """
    if not isinstance(max_updates, int) or isinstance(max_updates, bool) or not 1 <= max_updates <= 64:
        raise ValueError('invalid finite consolidation update budget')
    state['round'] += 1
    selected = choose_skill(bank, state)
    if selected is None:
        return {'action': 'no_ready_skill', 'assessment': before, 'updates': 0}
    key, donor_digest, entries = selected
    updates = 0
    try:
        for index in range(max_updates):
            if not can_train():
                break
            train(entries[index % len(entries)])
            updates += 1; state['attempted_updates'] += 1
        if not updates:
            return {'action': 'budget_exhausted', 'assessment': before, 'updates': 0}
        after = evaluate()
        accepted = bool(can_adopt(before, after))
    except BaseException:
        restore()
        raise
    if not accepted:
        restore()
    previous = state['skills'].get(key, {})
    attempts = previous.get('attempts', 0) + 1
    state['skills'][key] = {'attempts': attempts, 'donor_digest': donor_digest,
                           'base_revision': state['revision'], 'accepted': accepted,
                           'next_round': state['round'] + min(16, 2 ** min(attempts, 4))}
    state['history'].append({'action': 'skill_accepted' if accepted else 'skill_rejected',
                             'skill': key, 'step': step(), 'updates': updates})
    state['history'] = state['history'][-64:]
    return {'action': 'skill_accepted' if accepted else 'skill_rejected', 'skill': key,
            'assessment': after if accepted else before, 'updates': updates,
            'donors': sorted({e['source']['model_sha256'] for e in entries})}


def model_record(state, *, release, fingerprint, step, model_files):
    return {'schema': 1, 'model_id': MODEL_ID, 'revision': state['revision'],
            'weights_digest': state['weights_digest'], 'parent_weights_digest': state['parent_weights_digest'],
            'checkpoint_release': release, 'fingerprint': fingerprint, 'completed_steps': step,
            'model_file_sha256': model_files['model/model.pt'], 'model_files': copy.deepcopy(model_files),
            'scope': 'verified offline champion',
            'production_qualified': False, 'live_promoted': False}
