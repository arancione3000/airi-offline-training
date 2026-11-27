"""Learn small typed pipelines from trusted examples and optional queried labels."""
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
import json


MAX_DEPTH = 3
MAX_EXAMPLES = 64
MAX_TEXT = 4096
MAX_LIST = 64
MAX_RECORD = 16
MAX_ABS_INTEGER = 64_000_000


def key(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(',', ':'))


def _text(value):
    if type(value) is not str or len(value) > MAX_TEXT or any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise ValueError('expected bounded Unicode text')
    return value


def _integer(value):
    if type(value) is not int or abs(value) > MAX_ABS_INTEGER:
        raise ValueError('expected bounded integer')
    return value


def _sequence(value):
    if type(value) is not list or len(value) > MAX_LIST:
        raise ValueError('expected bounded sequence')
    if not value or type(value[0]) is int:
        for item in value:
            _integer(item)
    else:
        for item in value:
            _text(item)
    return value


def _record_key(value):
    if not _text(value) or len(value) > 64:
        raise ValueError('invalid record key')
    return value


def _data(value, *, record=True):
    if type(value) is str:
        return _text(value)
    if type(value) is int:
        return _integer(value)
    if type(value) is list:
        return _sequence(value)
    if record and type(value) is dict and 1 <= len(value) <= MAX_RECORD:
        for name, item in value.items():
            _record_key(name)
            _data(item, record=False)
        return value
    raise ValueError('unsupported data shape')


def _strings(items):
    return [_text(item) for item in _sequence(items)]


def _integers(items):
    return [_integer(item) for item in _sequence(items)]


def _nonempty(items):
    items = _sequence(items)
    if not items:
        raise ValueError('empty sequence')
    return items


def _reverse(value):
    return (_text(value) if type(value) is str else _sequence(value))[::-1]


OPERATIONS: dict[str, Callable] = {
    'strip': lambda v: _text(v).strip(),
    'lower': lambda v: _text(v).lower(),
    'upper': lambda v: _text(v).upper(),
    'reverse': _reverse,
    'remove_digits': lambda v: ''.join(ch for ch in _text(v) if not ch.isdigit()),
    'split_words': lambda v: _text(v).split(),
    'split_commas': lambda v: _text(v).split(','),
    'map_strip': lambda v: [item.strip() for item in _strings(v)],
    'map_lower': lambda v: [item.lower() for item in _strings(v)],
    'map_length': lambda v: [len(item) for item in _strings(v)],
    'join_space': lambda v: ' '.join(_strings(v)),
    'join_comma': lambda v: ','.join(_strings(v)),
    'stable_unique': lambda v: list(dict.fromkeys(_sequence(v))),
    'sort': lambda v: sorted(_sequence(v)),
    'keep_even': lambda v: [item for item in _integers(v) if item % 2 == 0],
    'keep_positive': lambda v: [item for item in _integers(v) if item > 0],
    'map_abs': lambda v: [abs(item) for item in _integers(v)],
    'count': lambda v: len(_sequence(v)),
    'sum': lambda v: sum(_integers(v)),
    'min': lambda v: min(_nonempty(v)),
    'max': lambda v: max(_nonempty(v)),
    'first': lambda v: _nonempty(v)[0],
    'last': lambda v: _nonempty(v)[-1],
}


def validate_program(program):
    if (type(program) is not dict or set(program) != {'schema', 'steps'}
            or type(program['schema']) is not int or program['schema'] != 1
            or type(program['steps']) is not list or len(program['steps']) > MAX_DEPTH):
        raise ValueError('invalid pipeline')
    for index, step in enumerate(program['steps']):
        if type(step) is not dict or type(step.get('op')) is not str:
            raise ValueError('invalid step')
        if step['op'] == 'field' and index == 0 and set(step) == {'op', 'key'}:
            _record_key(step['key'])
        elif set(step) != {'op'} or step['op'] not in OPERATIONS:
            raise ValueError('unsupported step')
    return True


def _apply(step, value):
    if step['op'] == 'field':
        if type(value) is not dict or step['key'] not in value:
            raise ValueError('missing record field')
        return value[step['key']]
    return _data(OPERATIONS[step['op']](value), record=False)


def execute_program(program, value):
    validate_program(program)
    current = _data(value)
    for step in program['steps']:
        current = _apply(step, current)
    return current


@dataclass(frozen=True)
class Hypothesis:
    steps: tuple
    outputs: tuple

    def as_program(self):
        return {'schema': 1, 'steps': list(self.steps)}


def _observed_value(value):
    try:
        return _data(value)
    except ValueError:
        return None


def _observed_step(step, value):
    if value is None:
        return None
    try:
        return _apply(step, value)
    except ValueError:
        return None


def _steps(values):
    fields = {name for value in values if type(value) is dict for name in value}
    return [{'op': 'field', 'key': name} for name in sorted(fields)] + [
        {'op': name} for name in sorted(OPERATIONS)
    ]


def _enumerate(values, expected, max_steps):
    # Equal outputs on every available input can share a shortest representative.
    # The pool's labels are unavailable at this stage.
    root = Hypothesis((), tuple(_observed_value(value) for value in values))
    initial = key(root.outputs)
    seen = {initial}
    target = key(expected)
    matches = [root] if key(root.outputs[:len(expected)]) == target else []
    frontier = [root]
    first_steps = _steps(root.outputs[:len(expected)])
    for depth in range(max_steps):
        next_frontier = []
        choices = first_steps if depth == 0 else [step for step in first_steps if step['op'] != 'field']
        for hypothesis in frontier:
            for step in choices:
                outputs = tuple(_observed_step(step, item) for item in hypothesis.outputs)
                fingerprint = key(outputs)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                expanded = Hypothesis((*hypothesis.steps, step), outputs)
                if key(outputs[:len(expected)]) == target:
                    matches.append(expanded)
                next_frontier.append(expanded)
        frontier = next_frontier
    return matches, len(seen)


def _rows(rows, minimum):
    if not isinstance(rows, list) or not minimum <= len(rows) <= MAX_EXAMPLES:
        raise ValueError('invalid example count')
    for row in rows:
        if type(row) is not dict or set(row) != {'input', 'expected'}:
            raise ValueError('invalid example')
        # An invalid input or unsupported target may be deliberately held as an
        # abstention outside the learned domain. Canonical JSON still bounds shape.
        if row['expected'] is not None:
            _data(row['expected'])
        if len(key(row['input'])) > 32768:
            raise ValueError('example input exceeds bound')
    return rows


def fit(train, pool, oracle=None, *, budget=3, max_steps=3, policy='disagreement'):
    """Return a proposal plus a trace. The oracle is called only for selected pool indices."""
    _rows(train, 2)
    if (not isinstance(pool, list) or len(pool) > MAX_EXAMPLES
            or type(budget) is not int or not 0 <= budget <= 8
            or type(max_steps) is not int or not 1 <= max_steps <= MAX_DEPTH
            or policy not in {'disagreement', 'ordered', 'shortest'}
            or (budget and not callable(oracle))):
        raise ValueError('invalid learning configuration')
    values = [row['input'] for row in train] + pool
    labels = [row['expected'] for row in train]
    if len(set(map(key, values))) != len(values):
        raise ValueError('training and pool must contain distinct inputs')
    if any(len(key(value)) > 32768 for value in pool):
        raise ValueError('pool input exceeds bound')
    if not any(label is not None for label in labels):
        raise ValueError('positive examples required')

    hypotheses, explored = _enumerate(values, labels, max_steps)
    initial = len(hypotheses)
    observed = []
    remaining = list(range(len(pool)))
    offset = len(train)
    while len(hypotheses) > 1 and len(observed) < budget and remaining and policy != 'shortest':
        if policy == 'ordered':
            index = remaining[0]
        else:
            def score(index):
                counts = Counter(key(hypothesis.outputs[offset + index]) for hypothesis in hypotheses)
                return max(counts.values()), -len(counts), index

            index = min(remaining, key=score)
        value = oracle(index)
        if value is not None:
            _data(value)
        remaining.remove(index)
        hypotheses = [h for h in hypotheses if key(h.outputs[offset + index]) == key(value)]
        observed.append({'index': index, 'hypotheses_after': len(hypotheses)})

    if not hypotheses:
        status = 'no_consistent_program'
    elif len(hypotheses) > 1 and policy != 'shortest':
        status = 'ambiguous'
    else:
        status = 'proposed'
    return {
        'status': status,
        'program': hypotheses[0].as_program() if status == 'proposed' else None,
        'initial_hypotheses': initial,
        'remaining_hypotheses': len(hypotheses),
        'distinct_observed_behaviors': explored,
        'queries': observed,
    }
