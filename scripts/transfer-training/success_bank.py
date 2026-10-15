"""Bounded, checkpointed lessons from native candidates, never evaluation labels.

The bank stores distributions on solver-labelled answer tokens. A token gain is
not evidence that a complete answer or a general conversational skill improved.
"""
import copy
import hashlib
import json
import math
import re

SCHEMA = 1
CAPACITY = 64
MAX_POSITIONS = 8


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def prompt_key(row):
    return canonical([{'role': m['role'], 'content': ' '.join(m['content'].casefold().split())}
                      for m in row['messages'][:-1]])


def row_key(row):
    return hashlib.sha256(canonical({k: row[k] for k in
                                    ('messages', 'expected', 'language', 'family')}).encode()).hexdigest()


def catalog(training, forbidden):
    """Only explicit membership in the authored training set grants eligibility."""
    forbidden = {prompt_key(row) for row in forbidden}
    result = {}
    for row in training:
        if (prompt_key(row) in forbidden or row['messages'][-1]['role'] != 'assistant'
                or row.get('expected') != row['messages'][-1]['content']
                or not 1 <= len(row['expected'].encode()) <= 128):
            continue
        # Human references and self-generated labels need a separate verifier.
        provenance = row.get('provenance', '')
        if not isinstance(provenance, str) or not provenance.startswith(('authored', 'deterministic')):
            continue
        result[row_key(row)] = copy.deepcopy(row)
    if not result:
        raise ValueError('no verified training lessons')
    return result


def probe_rows(rows, round_number, limit=24):
    """Rotate deterministic examples over families and languages, without eval input."""
    groups = {}
    for key, row in sorted(rows.items()):
        group = (row['language'], row['family'], row.get('length_bucket', -1))
        groups.setdefault(group, []).append(key)
    ordered = sorted(groups)
    start = (round_number * limit) % len(ordered)
    chosen = []
    for offset in range(min(limit, len(rows))):
        group = ordered[(start + offset) % len(ordered)]
        values = groups[group]
        index = (round_number + offset // len(ordered)) % len(values)
        key = values[index]
        if key not in chosen:
            chosen.append(key)
    return [rows[key] for key in chosen]


def empty_bank(fingerprint):
    return {'schema': SCHEMA, 'fingerprint': fingerprint, 'entries': {},
            'source_fingerprints': [fingerprint],
            'harvests': 0, 'donations': 0, 'transfer_updates': 0}


def validate_bank(bank, rows, fingerprint, vocab_size=264):
    if bank['schema'] != SCHEMA or bank['fingerprint'] != fingerprint:
        raise ValueError('success bank provenance mismatch')
    if len(bank['entries']) > CAPACITY:
        raise ValueError('success bank exceeds finite capacity')
    allowed=bank.get('source_fingerprints',[fingerprint])
    if fingerprint not in allowed or len(allowed)>CAPACITY+1:
        raise ValueError('invalid success bank ancestry')
    for key, entry in bank['entries'].items():
        if key not in rows or row_key(entry['row']) != key or entry['row'] != rows[key]:
            raise ValueError('lesson is not an eligible training row')
        positions = entry['positions']
        if not 1 <= len(positions) <= MAX_POSITIONS:
            raise ValueError('invalid teacher token count')
        if len({p['position'] for p in positions}) != len(positions):
            raise ValueError('duplicate teacher position')
        for token in positions:
            p = token['probabilities']
            if (not isinstance(token['position'], int) or isinstance(token['position'], bool)
                    or token['position'] < 1 or len(p) != vocab_size
                    or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in p)
                    or abs(sum(p) - 1.) > 1e-5
                    or not 0 <= token['target'] < vocab_size
                    or max(range(vocab_size), key=p.__getitem__) != token['target']
                    or not math.isfinite(token['gain']) or token['gain'] <= 0):
                raise ValueError('invalid verified donor distribution')
        source = entry['source']
        if (not isinstance(source['step'], int) or isinstance(source['step'], bool) or source['step'] < 0
                or source['fingerprint'] not in allowed or not re.fullmatch('[a-f0-9]{64}',source['model_sha256'])
                or entry['scope'] != 'authored training only'):
            raise ValueError('invalid donor lineage')
    return bank


def migrate_bank(bank, rows, source_fingerprint, fingerprint, vocab_size=264):
    """Carry authenticated donor provenance into a new authorized experiment.

    Only still-eligible training rows survive. A newly held-out example cannot
    enter training through an inherited bank. Original donor identities stay intact.
    """
    previous_rows={key:e['row'] for key,e in bank['entries'].items()}
    validate_bank(bank,previous_rows,source_fingerprint,vocab_size)
    result=copy.deepcopy(bank);result['fingerprint']=fingerprint
    result['entries']={k:e for k,e in result['entries'].items() if k in rows and e['row']==rows[k]}
    result['source_fingerprints']=sorted({fingerprint}|{e['source']['fingerprint'] for e in result['entries'].values()})
    return validate_bank(result,rows,fingerprint,vocab_size)


def priority(entry):
    return (int(entry['whole_answer_gain']), sum(p['gain'] for p in entry['positions']) / len(entry['positions']))


def remember(bank, entry, rows):
    key = row_key(entry['row'])
    if key not in rows or rows[key] != entry['row']:
        raise ValueError('cannot remember an evaluation or foreign lesson')
    previous = bank['entries'].get(key)
    if previous is None or priority(entry) > priority(previous):
        bank['entries'][key] = copy.deepcopy(entry)
        bank['donations'] += 1
    if len(bank['entries']) > CAPACITY:
        # Keep both languages/families represented before filling remaining slots.
        ranked = sorted(bank['entries'], key=lambda k: (priority(bank['entries'][k]), k), reverse=True)
        kept = []; groups = set()
        for k in ranked:
            r = bank['entries'][k]['row']; group = (r['language'], r['family'])
            if group not in groups and len(kept) < CAPACITY:
                kept.append(k); groups.add(group)
        kept += [k for k in ranked if k not in kept][:CAPACITY-len(kept)]
        bank['entries'] = {k: bank['entries'][k] for k in kept}


def summary(bank):
    entries = list(bank['entries'].values())
    return {'lessons': len(entries), 'verified_whole_answer_gains': sum(e['whole_answer_gain'] for e in entries),
            'verified_token_positions': sum(len(e['positions']) for e in entries),
            'harvests': bank['harvests'], 'donations': bank['donations'],
            'transfer_updates': bank['transfer_updates'],
            'scope': 'training observations; not proof of generalization'}


def retains_correct_answers(before, after):
    """Do not trade a previously correct development answer for a different one."""
    key = lambda s: canonical({k: s.get(k) for k in ('messages', 'expected', 'family', 'language', 'pair_id')})
    old = {key(s) for s in before['samples'] if s['correct']}
    new = {key(s) for s in after['samples'] if s['correct']}
    return old <= new
