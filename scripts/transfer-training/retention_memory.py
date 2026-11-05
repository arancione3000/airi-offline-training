"""Persistent, bounded certificates for explicitly eligible TRAIN answers.

Membership in the caller's audited ``success_bank.catalog`` is mandatory.
Certificates cover every causal answer token, including EOS; they are training
observations and make no claim about held-out wording. Old memories are never
silently evicted, even when a new experiment or champion is adopted. Torch is
imported only by ``make_certifier`` when its callback is actually used.
"""
import copy
import hashlib
import math
import re

from success_bank import canonical, catalog, row_key

SCHEMA = 1
DEFAULT_CAPACITY = 1024
MAX_CAPACITY = 8192
SCOPE = 'certified eligible TRAIN answers only; no evaluation labels'


def _integer(value, name, low=0, high=None):
    if (not isinstance(value, int) or isinstance(value, bool) or value < low
            or (high is not None and value > high)):
        raise ValueError('invalid retention memory ' + name)
    return value


def _digest(value, name, *, optional=False):
    if optional and value is None:
        return value
    if not isinstance(value, str) or re.fullmatch('[a-f0-9]{64}', value) is None:
        raise ValueError('invalid retention memory ' + name)
    return value


def _fingerprint(value):
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError('invalid retention experiment fingerprint')
    return value


def _eligible_row(eligible, key):
    """Check complete row equality/provenance, not merely a matching answer."""
    if not isinstance(eligible, dict) or key not in eligible:
        raise ValueError('memory row is not in the explicit eligible TRAIN catalog')
    row = eligible[key]
    try:
        allowed = catalog([row], [])
        valid = key == row_key(row) and allowed.get(key) == row
    except (KeyError, TypeError, AttributeError):
        valid = False
    if not valid:
        raise ValueError('memory row lacks exact eligible TRAIN provenance')
    return row


def _catalog_digest(eligible):
    """Bind the explicit catalog without copying its potentially large corpus."""
    if not isinstance(eligible, dict) or not eligible:
        raise ValueError('retention requires an explicit nonempty TRAIN catalog')
    result = hashlib.sha256()
    for key in sorted(eligible):
        _digest(key, 'catalog row key')
        _eligible_row(eligible, key)
        # Full metadata/provenance is included; row_key alone omits metadata.
        result.update(canonical([key, eligible[key]]).encode())
        result.update(b'\n')
    return result.hexdigest()


def _certificate(value):
    if not isinstance(value, dict):
        raise ValueError('retention needs a complete-answer certificate, not a boolean')
    if (not isinstance(value.get('correct'), bool)
            or value.get('complete') is not True or value.get('includes_eos') is not True):
        raise ValueError('retention certificate must include a complete causal answer and EOS')
    _integer(value.get('token_count'), 'certificate token count', 1)
    for name in ('min_margin', 'mean_nll'):
        measured = value.get(name)
        if (isinstance(measured, bool) or not isinstance(measured, (int, float))
                or not math.isfinite(measured) or (name == 'mean_nll' and measured < 0)):
            raise ValueError('invalid retention certificate ' + name)
    if value['correct'] and value['min_margin'] < 0:
        raise ValueError('correct retention certificate has a negative target margin')
    # Keep the schema finite and JSON-safe even when callbacks expose tensors.
    return {name: value[name] for name in
            ('correct', 'complete', 'includes_eos', 'token_count', 'min_margin', 'mean_nll')}


def empty_memory(fingerprint, capacity=DEFAULT_CAPACITY):
    _fingerprint(fingerprint)
    _integer(capacity, 'capacity', 1, MAX_CAPACITY)
    return {'schema': SCHEMA, 'fingerprint': fingerprint,
            'source_fingerprints': [fingerprint], 'capacity': capacity, 'scope': SCOPE,
            'base_champion_sha256': None, 'eligible_sha256': None, 'entries': {},
            'seen': 0, 'probe_cursor': 0, 'probe_group_cursors': {},
            'probe_observations': {}, 'sampling_cursors': {},
            'refreshes': 0, 'observations': 0, 'sampling_calls': 0, 'samples': 0,
            'capacity_exhausted': False, 'new_skipped': 0}


def validate_memory(memory, eligible, fingerprint, champion_digest=None):
    """Authenticate every stored row against the current audited TRAIN mapping.

    Adding eligible rows is allowed. Removing/changing a protected row requires
    an explicit experiment decision rather than silent migration or forgetting.
    ``champion_digest`` authenticates the base teacher, when supplied by resume.
    """
    if (not isinstance(memory, dict) or type(memory.get('schema')) is not int
            or memory.get('schema') != SCHEMA
            or memory.get('fingerprint') != _fingerprint(fingerprint)
            or memory.get('scope') != SCOPE):
        raise ValueError('retention memory schema or experiment provenance mismatch')
    capacity = _integer(memory.get('capacity'), 'capacity', 1, MAX_CAPACITY)
    entries = memory.get('entries')
    if not isinstance(entries, dict) or len(entries) > capacity:
        raise ValueError('retention memory exceeds capacity')
    base = _digest(memory.get('base_champion_sha256'), 'base champion digest', optional=True)
    if entries and base is None:
        raise ValueError('certified memories require an authenticated champion')
    if champion_digest is not None and base != _digest(champion_digest, 'expected champion digest'):
        raise ValueError('retention memory belongs to a different champion')
    _digest(memory.get('eligible_sha256'), 'eligible catalog digest', optional=True)
    ancestors = memory.get('source_fingerprints')
    if (not isinstance(ancestors, list) or fingerprint not in ancestors
            or len(ancestors) != len(set(ancestors)) or len(ancestors) > MAX_CAPACITY + 1):
        raise ValueError('invalid retention memory ancestry')
    for ancestor in ancestors:
        _fingerprint(ancestor)
    for counter in ('seen', 'probe_cursor', 'refreshes', 'observations',
                    'sampling_calls', 'samples', 'new_skipped'):
        _integer(memory.get(counter), counter)
    for name in ('probe_group_cursors', 'sampling_cursors'):
        if not isinstance(memory.get(name), dict):
            raise ValueError('invalid persisted retention ' + name)
        for key, cursor in memory[name].items():
            if not isinstance(key, str):
                raise ValueError('invalid retention cursor stratum')
            _integer(cursor, name)
    if (not isinstance(memory.get('capacity_exhausted'), bool)
            or memory['capacity_exhausted'] != (len(entries) >= capacity)):
        raise ValueError('invalid retention capacity accounting')
    for key, entry in entries.items():
        row = _eligible_row(eligible, key)
        if not isinstance(entry, dict) or entry.get('row') != row:
            raise ValueError('protected row differs from exact eligible TRAIN row')
        if entry.get('first_fingerprint') not in ancestors:
            raise ValueError('protected row has unauthenticated experiment ancestry')
        _digest(entry.get('first_champion_sha256'), 'first champion digest')
        if entry.get('certified_champion_sha256') != base:
            raise ValueError('protected row was certified by a different base champion')
        certificate = _certificate(entry.get('certificate'))
        observation = _certificate(entry.get('observation'))
        if not certificate['correct']:
            raise ValueError('uncertified answer in retention memory')
        for name, expected in (
                ('margin_drop', max(0., certificate['min_margin'] - observation['min_margin'])),
                ('nll_increase', max(0., observation['mean_nll'] - certificate['mean_nll']))):
            if entry.get(name) != expected:
                raise ValueError('invalid current retention degradation statistic')
    observations = memory.get('probe_observations')
    if not isinstance(observations, dict) or len(observations) > MAX_CAPACITY:
        raise ValueError('unbounded retention probe observations')
    for key, observation in observations.items():
        _eligible_row(eligible, key)
        _certificate(observation)
    # Detect non-JSON or NaN metadata before the caller writes a checkpoint.
    canonical(memory)
    return memory


def migrate_memory(memory, eligible, source_fingerprint, fingerprint, champion_digest=None):
    """Change experiment identity while retaining every authenticated memory."""
    validate_memory(memory, eligible, source_fingerprint, champion_digest)
    result = copy.deepcopy(memory)
    result['fingerprint'] = _fingerprint(fingerprint)
    result['source_fingerprints'] = sorted(set(result['source_fingerprints']) | {fingerprint})
    result['eligible_sha256'] = _catalog_digest(eligible)
    return validate_memory(result, eligible, fingerprint, champion_digest)


def _stratum(row):
    # No family whitelist: new context/copy/recall skills enter by eligibility.
    history = row.get('history_exchanges')
    if history is None:
        history = max(0, sum(m['role'] == 'user' for m in row['messages'][:-1]) - 1)
    intent = row['expected'] if row['family'] in {'extra-dialogue', 'foundation-dialogue'} else ''
    return canonical([row['language'], row['family'], row.get('length_bucket', -1), history, intent])


def _probe_keys(memory, eligible, limit):
    groups = {}
    for key, row in eligible.items():
        if key not in memory['entries']:
            groups.setdefault(_stratum(row), []).append(key)
    for values in groups.values():
        values.sort()
    ordered = sorted(groups)
    chosen, used = [], set()
    available = sum(map(len, groups.values()))
    if not ordered:
        return chosen
    # Each successful visit advances one persistent group cursor. Saturated
    # small groups are skipped within this batch rather than reprobed twice.
    index = memory['probe_cursor'] % len(ordered)
    while len(chosen) < min(limit, available):
        group = ordered[index % len(ordered)]
        values = groups[group]
        cursor = memory['probe_group_cursors'].get(group, 0)
        for offset in range(len(values)):
            key = values[(cursor + offset) % len(values)]
            if key not in used:
                memory['probe_group_cursors'][group] = cursor + offset + 1
                chosen.append(key); used.add(key)
                break
        index += 1
    memory['probe_cursor'] = index
    return chosen


def _set_observation(entry, observation):
    entry['observation'] = observation
    entry['margin_drop'] = max(0., entry['certificate']['min_margin'] - observation['min_margin'])
    entry['nll_increase'] = max(0., observation['mean_nll'] - entry['certificate']['mean_nll'])


def refresh(champion_digest, eligible, certify, previous, *, probe_limit=128):
    """Recheck ALL old memories, then certify a bounded balanced new sample.

    The returned bank is a fresh value. A lost old certificate or callback
    failure leaves ``previous`` untouched and rejects this proposed champion.
    At capacity new correct rows are counted as skipped; no old row is evicted.
    """
    _digest(champion_digest, 'champion digest')
    _integer(probe_limit, 'probe limit', 0, MAX_CAPACITY)
    validate_memory(previous, eligible, previous['fingerprint'])
    result = copy.deepcopy(previous)
    result['eligible_sha256'] = _catalog_digest(eligible)
    for key, entry in result['entries'].items():
        certificate = _certificate(certify(_eligible_row(eligible, key)))
        if not certificate['correct']:
            raise ValueError('proposed champion lost a previously certified TRAIN answer: ' + key)
        entry['certificate'] = certificate
        entry['certified_champion_sha256'] = champion_digest
        _set_observation(entry, copy.deepcopy(certificate))
    observations = {}
    for key in _probe_keys(result, eligible, probe_limit):
        row = _eligible_row(eligible, key)
        certificate = _certificate(certify(row))
        observations[key] = certificate
        result['seen'] += 1
        if not certificate['correct']:
            continue
        if len(result['entries']) >= result['capacity']:
            result['new_skipped'] += 1
            continue
        entry = {'row': copy.deepcopy(row), 'first_fingerprint': result['fingerprint'],
                 'first_champion_sha256': champion_digest,
                 'certified_champion_sha256': champion_digest, 'certificate': certificate}
        _set_observation(entry, copy.deepcopy(certificate))
        result['entries'][key] = entry
    result['probe_observations'] = observations
    result['base_champion_sha256'] = champion_digest
    result['refreshes'] += 1
    result['capacity_exhausted'] = len(result['entries']) >= result['capacity']
    return validate_memory(result, eligible, result['fingerprint'], champion_digest)


def observe(memory, eligible, certify, keys=None):
    """Measure current candidate drift without changing the certified teacher.

    Incorrect current answers remain in the bank and receive high replay
    priority. The caller's original bank survives any failed observation.
    """
    validate_memory(memory, eligible, memory['fingerprint'])
    selected = list(memory['entries']) if keys is None else list(keys)
    if len(selected) != len(set(selected)) or any(key not in memory['entries'] for key in selected):
        raise ValueError('only protected TRAIN entries can be observed')
    result = copy.deepcopy(memory)
    for key in selected:
        _set_observation(result['entries'][key], _certificate(certify(_eligible_row(eligible, key))))
    result['observations'] += len(selected)
    return validate_memory(result, eligible, result['fingerprint'])


def _priority(entry):
    current = entry['observation']
    return (not current['correct'], entry['margin_drop'], entry['nll_increase'], -current['min_margin'])


def sample_rows(memory, eligible, rng, *, per_language=2, uniform_fraction=.25):
    """Rotate coverage strata, protect current drift, and retain random replay.

    Each language receives the requested number. Nonuniform slots first cover
    its most degraded memory, then rotate family/length/history/intent strata;
    within a stratum the most degraded margin wins. Uniform slots sample the
    rotating stratum using the supplied checkpointed RNG. Cursors are persisted
    in-place; rows are copied so consumers cannot mutate bank provenance.
    A single-slot configuration alternates priority and coverage calls, so a
    persistently degraded answer cannot permanently starve other strata.
    """
    validate_memory(memory, eligible, memory['fingerprint'])
    _integer(per_language, 'samples per language', 1, 64)
    if (isinstance(uniform_fraction, bool) or not isinstance(uniform_fraction, (int, float))
            or not math.isfinite(uniform_fraction) or not 0 <= uniform_fraction <= 1):
        raise ValueError('invalid retention uniform fraction')
    by_language = {}
    for key, entry in memory['entries'].items():
        by_language.setdefault(entry['row']['language'], []).append(key)
    if any(not by_language.get(language) for language in ('it', 'en')):
        raise ValueError('certified retention sampling requires both languages')
    chosen = []
    for language in ('it', 'en'):
        groups = {}
        for key in sorted(by_language[language]):
            groups.setdefault(_stratum(memory['entries'][key]['row']), []).append(key)
        strata = sorted(groups)
        cursor = memory['sampling_cursors'].get(language, 0)
        used = set()
        for slot in range(per_language):
            if len(used) == len(by_language[language]):
                used.clear()
            candidates = [key for key in by_language[language] if key not in used]
            uniform = rng.random() < uniform_fraction
            worst = max(candidates, key=lambda key: (_priority(memory['entries'][key]), key))
            priority_turn = per_language > 1 or memory['sampling_calls'] % 2 == 0
            if (slot == 0 and priority_turn and not uniform
                    and _priority(memory['entries'][worst])[:3] > (False, 0., 0.)):
                key = worst
            else:
                for _ in range(len(strata)):
                    group = strata[cursor % len(strata)]; cursor += 1
                    pool = [key for key in groups[group] if key not in used]
                    if pool:
                        break
                key = (pool[rng.randrange(len(pool))] if uniform else
                       max(pool, key=lambda key: (_priority(memory['entries'][key]), key)))
            chosen.append(copy.deepcopy(_eligible_row(eligible, key)))
            used.add(key)
        memory['sampling_cursors'][language] = cursor
    memory['sampling_calls'] += 1
    memory['samples'] += len(chosen)
    return chosen


def make_certifier(model, encode, eos_token_id):
    """Build a lazy native causal full-answer verifier with EOS validation.

    ``encode`` must return one complete unpadded causal sequence and its SFT
    labels. Correct next-token argmax at every gold position implies the same
    greedy complete answer by induction; a correct prefix alone is insufficient.
    Evaluation mode and every submodule's original train/eval flag are restored.
    """
    _integer(eos_token_id, 'EOS token id')

    def certify(row):
        import torch
        from torch.nn import functional as F
        ids, labels = encode(row)
        if (ids.ndim != 2 or labels.ndim != 2 or ids.shape != labels.shape
                or ids.shape[0] != 1 or ids.shape[1] < 2):
            raise ValueError('retention certificate requires one complete causal sequence')
        targets = labels[:, 1:]
        mask = targets != -100
        positions = mask[0].nonzero().flatten()
        if (not positions.numel() or int(positions[-1]) != targets.shape[1] - 1
                or int(targets[0, -1]) != eos_token_id
                or not bool((ids[:, 1:][mask] == targets[mask]).all())):
            raise ValueError('retention encoding omits or corrupts the complete answer EOS')
        # The final assistant answer must be contiguous; earlier assistant
        # turns can also be supervised and are included in the certificate.
        tail = positions[-1]
        while int(tail) > 0 and bool(mask[0, int(tail) - 1]):
            tail = tail - 1
        if int(targets[0, int(tail)]) == eos_token_id:
            raise ValueError('retention certificate has an empty final answer')
        modes = [(module, module.training) for module in model.modules()]
        try:
            model.eval()
            with torch.no_grad():
                logits = model(ids)['logits']
                if logits.shape[:2] != ids.shape or logits.shape[-1] < 2:
                    raise ValueError('invalid causal retention logits')
                scores = logits[:, :-1, :].float()[mask]
                gold = targets[mask]
                if not bool(torch.isfinite(scores).all()):
                    raise ValueError('nonfinite retention certificate logits')
                selected = scores.gather(1, gold[:, None]).squeeze(1)
                alternatives = scores.clone()
                alternatives.scatter_(1, gold[:, None], -float('inf'))
                margins = selected - alternatives.max(-1).values
                nll = F.cross_entropy(scores, gold, reduction='mean')
                result = {'correct': bool((scores.argmax(-1) == gold).all()),
                          'complete': True, 'includes_eos': True,
                          'token_count': int(mask.sum()), 'min_margin': float(margins.min()),
                          'mean_nll': float(nll)}
                return _certificate(result)
        finally:
            for module, training in modes:
                module.training = training
    return certify


def summary(memory):
    entries = list(memory['entries'].values())
    groups = {_stratum(entry['row']) for entry in entries}
    return {'rows': len(entries), 'capacity': memory['capacity'],
            'languages': sorted({entry['row']['language'] for entry in entries}),
            'strata': len(groups), 'currently_correct': sum(entry['observation']['correct'] for entry in entries),
            'degraded': sum(entry['margin_drop'] > 0 or entry['nll_increase'] > 0 for entry in entries),
            'seen': memory['seen'], 'refreshes': memory['refreshes'],
            'capacity_exhausted': memory['capacity_exhausted'], 'new_skipped': memory['new_skipped'],
            'base_champion_sha256': memory['base_champion_sha256'], 'scope': SCOPE}
