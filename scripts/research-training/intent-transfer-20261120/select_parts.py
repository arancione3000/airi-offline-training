"""Fit a fixed per-operation policy from prior end-to-end development observations.

Only the source-independent question classifier is combined. The caller retains
approved routes and validates the unique source and actual program at execution.
"""
from collections import defaultdict


# Prefer existing cheap gates on tied coverage. Neural tie order is fixed,
# not an empirical claim about their relative CPU latency.
TIE_ORDER = ('approved', 'v6', 'parser', 'minilm_v4', 'minilm_v5', 'learned_expanded')
MINIMUM_WORDINGS = 2


def _measure(records, operations):
    wordings = defaultdict(dict)
    errors = dict.fromkeys(operations.values(), 0)
    for row in records:
        gold = row['lesson_id']
        predictions, outcomes = row['predictions'], row['outcomes']
        if len(predictions) != 2 or len(outcomes) != 2:
            raise ValueError('two held-out payloads required for each wording')
        complete = True
        for predicted, outcome in zip(predictions, outcomes):
            if predicted is None:
                if outcome not in ('positive_abstain', 'negative_abstain'):
                    raise ValueError('answered case lacks a predicted procedure')
                complete = False
                continue
            correct = gold == predicted and outcome == 'positive_correct'
            if not correct:
                errors[operations[predicted]] += 1
                complete = False
        if gold is not None:
            # A duplicated normalized wording has to succeed every time.
            key = row.get('normalized_query', row['id'])
            group = wordings[operations[gold]]
            group[key] = group.get(key, True) and complete
    return {op: {'correct_wordings': sum(wordings[op].values()),
                 'wrong_executions': errors[op]} for op in sorted(errors)}


def _winner(metrics, arms):
    eligible = [arm for arm in arms if metrics[arm]['wrong_executions'] == 0
                and metrics[arm]['correct_wordings'] >= MINIMUM_WORDINGS]
    if not eligible:
        return 'approved'
    return max(eligible, key=lambda arm: metrics[arm]['correct_wordings'])


def select(observations_by_arm, families):
    """Select from pooled historical records only; never accepts query text to fit a rule."""
    arms = [name for name in TIE_ORDER if name in observations_by_arm]
    if 'approved' not in arms or not observations_by_arm['approved']:
        raise ValueError('approved production development control required')
    operations = {family['id']: family['operation'] for family in families}
    counts = {}
    signature = None
    for arm in arms:
        records = observations_by_arm[arm]
        keys = [(r['id'], r['source_lesson_id'], r['lesson_id'],
                 r.get('normalized_query', r['id'])) for r in records]
        if len({k[0] for k in keys}) != len(keys) or (signature is not None and keys != signature):
            raise ValueError('arms must score the same ordered, nonduplicated development IDs')
        signature = keys
        counts[arm] = _measure(records, operations)
    winners = {op: _winner({arm: counts[arm][op] for arm in arms}, arms)
               for op in sorted(set(operations.values()))}
    totals = {arm: {metric: sum(op[metric] for op in counts[arm].values())
                    for metric in ('correct_wordings', 'wrong_executions')} for arm in arms}
    return {'winners': winners, 'global_winner': _winner(totals, arms),
            'development': counts, 'tie_order': arms, 'minimum_correct_wordings': MINIMUM_WORDINGS}


def predict(query, families, selected, callbacks):
    """Keep a proposal only for its own selected operation. Disagreement abstains."""
    operations = {family['id']: family['operation'] for family in families}
    winners = selected['winners']
    accepted = set()
    for arm in dict.fromkeys(winners.values()):
        if arm == 'approved':
            continue
        lesson = callbacks[arm](query, families)
        if lesson is not None and winners.get(operations.get(lesson)) == arm:
            accepted.add(lesson)
    return accepted.pop() if len(accepted) == 1 else None
