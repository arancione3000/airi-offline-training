"""Bounded growth by composing operations and qualifying every new recipe."""
from collections import Counter
from hashlib import sha256
from itertools import combinations_with_replacement, product
from pathlib import Path
import argparse
import json
import random
import sys


RESEARCH = Path(__file__).resolve().parents[1]
INCREMENTAL = RESEARCH / 'incremental-procedure-verification'
sys.path.insert(0, str(INCREMENTAL))
from fixtures import INPUTS, scenarios
from synthesis import RecipeLibrary, next_probe
from verifier import MAX_MAGNITUDE, OPERATIONS, Verifier


UNARY = ('absolute', 'negate')
BINARY = ('add', 'subtract', 'multiply')
ORDER = ('minimum', 'maximum')
SAMPLE_SIZE = 64
MAX_DEEP_RECIPES = 12_000
MAX_COMPLETE_FINALISTS = 64
MAX_LABELS = 16
MAX_QUALIFICATIONS = 8
MAX_SHALLOW_APPROVED_COLUMNS = 48
MAX_CONSTANT_COLUMNS = 3


def definitions(leaves, *, allow_order):
    """Yield one-operation recipes in deterministic order."""
    for op in UNARY:
        for leaf in leaves:
            yield op, leaf
    for op in BINARY + (ORDER if allow_order else ()):
        pairs = product(leaves, repeat=2) if op == 'subtract' else combinations_with_replacement(leaves, 2)
        for left, right in pairs:
            yield op, left, right


def operation(op):
    if op == 'minimum':
        return min
    if op == 'maximum':
        return max
    return OPERATIONS[op][1]


def evaluate(op, columns):
    return tuple(map(operation(op), *columns))


def most_informative(outcome_rows):
    """Choose a separating input with minimum expected remaining recipes."""
    best = None
    for index, values in outcome_rows:
        counts = Counter(values)
        if len(counts) < 2:
            continue
        score = (sum(count * count for count in counts.values()), max(counts.values()), index)
        if best is None or score < best:
            best = score
    return best[-1] if best is not None else None


class GrowingLearner:
    def __init__(self, base_graph, approved_tables, *, allow_order_macros=True):
        self.base_graph = base_graph
        self.tables = {**approved_tables, 'a': tuple(a for a, _ in INPUTS), 'b': tuple(b for _, b in INPUTS)}
        # Keep recent graph nodes eligible as previously verified results
        # accumulate. The original enumerator's operation bound still applies.
        recent = dict(list(approved_tables.items())[-MAX_SHALLOW_APPROVED_COLUMNS:])
        self.shallow, self.shallow_report = RecipeLibrary(cache=False).enumerate(recent)
        self.shallow_report['approved_columns_used'] = len(recent)
        constants = {}
        for name, values in sorted(approved_tables.items()):
            if all(value == values[0] for value in values):
                constants.setdefault(values[0], name)
        self.two = constants.get(2)
        self.allow_order = allow_order_macros and self.two is not None
        # Preserve positive one and two when available and prefer small
        # constants. The fixed feature width bounds all layered search.
        ranked = sorted(constants, key=lambda value: (value not in (1, 2), abs(value), value))
        self.leaves = ('a', 'b', *(constants[value] for value in ranked[:MAX_CONSTANT_COLUMNS]))
        self.sample = tuple(sorted(random.Random(7611).sample(range(len(INPUTS)), SAMPLE_SIZE)))
        self.deep = None

    def deeper_recipes(self):
        if self.deep is not None:
            return self.deep
        # Bound the language growth over raw inputs and existing constant columns.
        # Existing nonconstant features are already eligible at depth one.
        leaves = self.leaves
        features = {}
        columns = {name: self.tables[name] for name in leaves}
        observed = set(columns.values())
        for expr in definitions(leaves, allow_order=False):
            values = evaluate(expr[0], [columns[arg] for arg in expr[1:]])
            if values not in observed and all(abs(value) <= MAX_MAGNITUDE for value in values):
                observed.add(values)
                features[expr] = values
        columns.update(features)
        sampled = {expr: tuple(values[i] for i in self.sample) for expr, values in columns.items()}
        recipes = []
        at_capacity = False
        for expr in definitions(list(columns), allow_order=self.allow_order):
            if expr[0] not in ORDER and all(isinstance(arg, str) for arg in expr[1:]):
                continue  # The existing shallow search already covered these.
            if len(recipes) >= MAX_DEEP_RECIPES:
                at_capacity = True
                break
            recipes.append((expr, evaluate(expr[0], [sampled[arg] for arg in expr[1:]])))
        self.deep = (columns, tuple(recipes), len(features), at_capacity)
        return self.deep

    def compile(self, expr):
        additions = {}
        intermediate = {}
        seen = {}

        def emit(operation, arguments):
            key = operation, arguments
            if key not in seen:
                index = len(seen)
                name = f'growth_recipe_{index}'
                while name in self.base_graph or name in additions:
                    index += 1
                    name = f'growth_recipe_{index}'
                additions[name] = {'op': operation, 'deps': list(arguments)}
                intermediate[name] = evaluate(operation, [intermediate[arg] if arg in intermediate else self.tables[arg]
                                                         for arg in arguments])
                seen[key] = name
            return seen[key]

        def visit(current):
            if isinstance(current, str):
                return current
            op, *args = current
            leaves = tuple(visit(arg) for arg in args)
            if op in ORDER:
                # min/max(x,y) = (x+y ± |x-y|) // 2. This avoids adding a
                # trusted operation to the production verifier.
                total = emit('add', leaves)
                gap = emit('absolute', (emit('subtract', leaves),))
                signed = emit('add' if op == 'maximum' else 'subtract', (total, gap))
                return emit('floor_quotient', (signed, self.two))
            return emit(op, leaves)

        output = visit(expr)
        return additions, output, intermediate

    def learn(self, oracle, qualify, *, max_depth=2):
        if max_depth not in (1, 2):
            raise ValueError('expected max_depth 1 or 2')
        labels = {}
        trace = []
        qualifications = []
        failed_outputs = set()
        deep_report = None

        def ask(index):
            if index not in labels:
                if len(labels) >= MAX_LABELS:
                    raise ValueError('label budget exhausted')
                value = oracle(index)
                if type(value) is not int or abs(value) > MAX_MAGNITUDE:
                    raise ValueError('invalid oracle label')
                labels[index] = value
                trace.append({'index': index, 'input': list(INPUTS[index]), 'label': labels[index]})
            return labels[index]

        def finish(candidates, depth):
            for expr, values in candidates:
                if values in failed_outputs:
                    continue
                if len(qualifications) >= MAX_QUALIFICATIONS:
                    break
                additions, output, intermediate = self.compile(expr)
                if intermediate[output] != values:
                    raise AssertionError('recipe and compiled graph disagree')
                if any(any(abs(value) > MAX_MAGNITUDE for value in column) for column in intermediate.values()):
                    continue
                admitted = qualify(additions, output, intermediate)
                if type(admitted) is not bool:
                    raise ValueError('qualification callback must return a boolean')
                qualifications.append({'depth': depth, 'admitted': admitted, 'expression': expr})
                if admitted:
                    return {'status': 'accepted', 'depth': depth, 'graph_additions': additions, 'output_node': output}
                failed_outputs.add(values)
            return None

        shallow = self.shallow
        while len(shallow) > 1 and len(labels) < MAX_LABELS:
            index = next_probe(shallow)
            answer = ask(index)
            shallow = tuple(row for row in shallow if row.outputs[index] == answer)
        result = (finish([((c.operation, *c.dependencies), c.outputs) for c in shallow], 1)
                  if len(shallow) == 1 else None)
        reason = 'no_candidate' if not shallow else 'qualification_rejected' if len(shallow) == 1 else 'label_budget'
        if result is None and max_depth == 2 and len(labels) < MAX_LABELS:
            columns, pool, feature_count, at_capacity = self.deeper_recipes()
            # Apply all prior labels before choosing any further query.
            remaining = tuple((expr, row) for expr, row in pool if all(
                operation(expr[0])(*(columns[arg][i] for arg in expr[1:])) == v for i, v in labels.items()))
            deep_report = {'recipes': len(pool), 'sample_size': len(self.sample), 'derived_feature_tables': feature_count,
                           'complete_finalists': 0, 'trigger': reason, 'recipe_budget_reached': at_capacity}
            while len(remaining) > MAX_COMPLETE_FINALISTS and len(labels) < MAX_LABELS:
                position = most_informative((pos, (row[pos] for _, row in remaining))
                                            for pos, index in enumerate(self.sample) if index not in labels)
                if position is None:
                    break
                answer = ask(self.sample[position])
                remaining = tuple((expr, row) for expr, row in remaining if row[position] == answer)
            reason = 'finalist_budget'
            if len(remaining) <= MAX_COMPLETE_FINALISTS:
                candidates = {}
                for expr, _ in remaining:
                    values = evaluate(expr[0], [columns[arg] for arg in expr[1:]])
                    if values not in failed_outputs and all(abs(value) <= MAX_MAGNITUDE for value in values):
                        candidates.setdefault(values, expr)
                deep_report['complete_finalists'] = len(remaining)
                while len(candidates) > 1 and len(labels) < MAX_LABELS:
                    index = most_informative((i, outputs) for i, outputs in enumerate(zip(*candidates, strict=True))
                                             if i not in labels)
                    if index is None:
                        break
                    answer = ask(index)
                    candidates = {values: expr for values, expr in candidates.items() if values[index] == answer}
                result = (finish([(expr, values) for values, expr in candidates.items()], 2)
                          if len(candidates) == 1 else None)
                reason = ('no_candidate' if not candidates else
                          'qualification_rejected' if len(candidates) == 1 else 'label_budget')
        if result is None:
            result = {'status': 'abstained', 'depth': None, 'graph_additions': {}, 'output_node': None, 'reason': reason}
        return {**result, 'labels': trace, 'qualifications': qualifications, 'deep_search': deep_report}


def qualify_with_verifier(base_graph, baseline, oracle, *, mutate_unseen_index=None):
    verifier = Verifier(INPUTS)
    if verifier.update(base_graph, baseline)['status'] != 'accepted':
        raise ValueError('invalid approved graph')

    def qualify(additions, output, intermediate):
        expected = tuple(oracle(index) for index in range(len(INPUTS)))
        if mutate_unseen_index is not None:
            i = mutate_unseen_index
            expected = expected[:i] + (expected[i] + 1,) + expected[i+1:]
        # The final table is supplied by the independent reference. Earlier
        # tables are deterministic artifacts of already approved operations.
        result = verifier.update({**base_graph, **additions}, {**baseline, **intermediate, output: expected})
        if result['status'] != 'accepted' and (verifier.version != 1 or verifier.tables != baseline):
            raise AssertionError('failed admission changed the approved graph')
        return result['status'] == 'accepted'

    return qualify


def experiment():
    sys.path.insert(0, str(INCREMENTAL / 'audit'))
    try:
        from scope_stress import TARGETS
    finally:
        sys.path.pop(0)
    _, graph, baseline, accepted = next(scenarios())
    if not accepted:
        raise AssertionError('baseline fixture not qualified')
    learner = GrowingLearner(graph, baseline)
    levels = {}
    arithmetic = GrowingLearner(graph, baseline, allow_order_macros=False)
    for label, limit, engine in (('1', 1, learner), ('2_arithmetic_only', 2, arithmetic), ('2', 2, learner)):
        cases = {}
        for name, reference in TARGETS.items():
            def oracle(index):
                return reference(*INPUTS[index])
            cases[name] = engine.learn(oracle, qualify_with_verifier(graph, baseline, oracle), max_depth=limit)
        levels[label] = {'approved': sum(case['status'] == 'accepted' for case in cases.values()), 'tasks': cases}
    if any(row['status'] == 'accepted' and levels['2']['tasks'][name]['status'] != 'accepted'
           for name, row in levels['1']['tasks'].items()):
        raise AssertionError('automatic expansion regressed earlier admissions')
    if [levels[level]['approved'] for level in ('1', '2_arithmetic_only', '2')] != [5, 7, 9]:
        raise AssertionError('the scope-growth result changed')
    grown_name = next(name for name, row in levels['2']['tasks'].items() if row['depth'] == 2)
    grown = levels['2']['tasks'][grown_name]
    used = {row['index'] for row in grown['labels']}
    missed_index = next(i for i in range(len(INPUTS)) if i not in used)

    def oracle(index):
        return TARGETS[grown_name](*INPUTS[index])

    corrupted = learner.learn(oracle, qualify_with_verifier(graph, baseline, oracle, mutate_unseen_index=missed_index))
    if corrupted['status'] != 'abstained' or not any(row['depth'] == 2 and not row['admitted']
                                                  for row in corrupted['qualifications']):
        raise AssertionError('an unseen conflicting result did not prevent admission')
    paths = (Path(__file__), INCREMENTAL / 'fixtures.py', INCREMENTAL / 'synthesis.py',
             INCREMENTAL / 'verifier.py', INCREMENTAL / 'audit' / 'scope_stress.py')
    return {'domain_pairs': len(INPUTS), 'shallow_enumeration': learner.shallow_report, 'levels': levels,
            'negative_control': {'task': grown_name, 'conflicting_index': missed_index, 'result': corrupted},
            'sha256': {path.name: sha256(path.read_bytes()).hexdigest() for path in paths},
            'bounds': {'sample_size': SAMPLE_SIZE, 'deep_recipes': MAX_DEEP_RECIPES,
                       'complete_finalists': MAX_COMPLETE_FINALISTS, 'labels_per_task': MAX_LABELS,
                       'max_qualifications': MAX_QUALIFICATIONS,
                       'recent_approved_columns': MAX_SHALLOW_APPROVED_COLUMNS,
                       'constant_columns': MAX_CONSTANT_COLUMNS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = experiment()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print({k: v['approved'] for k, v in result['levels'].items()})


if __name__ == '__main__':
    main()
