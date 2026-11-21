"""Compare sample efficiency for coupled and independent procedural synthesis."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import product
import json
import math
from pathlib import Path
import random
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'assistant-runtime'))
from learn import BUNDLED
from skill_bank import SkillBank
import skill_programs

OPTIONS = tuple(product((0, 1), repeat=3))
PROBES = tuple(product((-2, -1, 0, 1, 2), repeat=2))
CHECKS = tuple(product((-97, -11, 3, 19, 73), repeat=2))
AUDIT_PROBES = tuple(product((-41, -17, 7, 17, 41), repeat=2))
BUDGETS = (4, 8, 12, 16)


def inventory():
    entries = SkillBank(BUNDLED)._read()[0]['entries']
    operations = {'add', 'subtract', 'multiply'}
    return tuple((e['lesson_id'], e['program']['operation']) for e in entries
                 if e['program']['operation'] in operations and not e['lesson_id'].startswith('v5_'))


def execute(op, config, values):
    left, right = values
    a, b, absolute = OPTIONS[config]
    operands = [{'value': left + a}, {'value': right + 2 * b}]
    result = skill_programs._transform(op, '', operands)
    return str(abs(int(result))) if absolute else result


def oracle(op, config, values):
    # Independent formula for the synthetic job specification.
    x = values[0] + (1 if config & 4 else 0)
    y = values[1] + (2 if config & 2 else 0)
    if op == 'add':
        answer = x + y
    elif op == 'subtract':
        answer = x - y
    elif op == 'multiply':
        answer = x * y
    else:
        raise ValueError(op)
    return str(-answer if config % 2 and answer < 0 else answer)


def possible_worlds(size):
    worlds = set()
    for common in range(len(OPTIONS)):
        original = (common,) * size
        worlds.add(original)
        for changed in range(size):
            for replacement in range(len(OPTIONS)):
                worlds.add(original[:changed] + (replacement,) + original[changed + 1:])
    return tuple(sorted(worlds))


def resolved(projections):
    return {i: next(iter(p)) for i, p in enumerate(projections) if len(p) == 1}


@lru_cache(maxsize=4096)
def projection(worlds):
    return tuple(frozenset(options) for options in zip(*worlds, strict=True))


def partition(options, outcomes):
    buckets = defaultdict(set)
    for option in options:
        buckets[outcomes[option]].add(option)
    return buckets


def entropy(sizes):
    n = sum(sizes)
    return -sum((s / n) * math.log2(s / n) for s in sizes)


class Search:
    def __init__(self, operations, strategy, seed):
        self.strategy = strategy
        self.rng = random.Random(seed)
        self.worlds = None if strategy == 'independent' else possible_worlds(len(operations))
        self.choices = tuple(frozenset(range(len(OPTIONS))) for _ in operations)
        self.answers = tuple(tuple(tuple(execute(op, i, probe) for i in range(len(OPTIONS)))
                                   for probe in PROBES) for op in operations)
        self.contradiction = False

    def select(self):
        projections = self.choices if self.worlds is None else projection(self.worlds)
        known = len(resolved(projections))
        candidates = []
        for task, options in enumerate(projections):
            # Multiple input pairs can induce the exact same partition.
            explored = set()
            for probe, outcomes in enumerate(self.answers[task]):
                masks = partition(options, outcomes)
                blocks = frozenset(frozenset(indices) for indices in masks.values())
                if len(blocks) <= 1 or blocks in explored:
                    continue
                explored.add(blocks)
                if self.worlds is None:
                    sizes = tuple(map(len, blocks))
                    gain = sum(1 for n in sizes if n == 1) / sum(sizes)
                else:
                    survival = defaultdict(list)
                    for world in self.worlds:
                        survival[outcomes[world[task]]].append(world)
                    sizes = tuple(map(len, survival.values()))
                    gain = 0.
                    if self.strategy == 'impact':
                        gain = sum(len(rows) * (len(resolved(projection(tuple(rows)))) - known)
                                   for rows in survival.values()) / len(self.worlds)
                candidates.append(((gain, entropy(sizes)), task, probe))
        if not candidates:
            return None
        if self.strategy == 'random':
            _, task, probe = self.rng.choice(candidates)
        else:
            # Stable ties by inventory and probe order.
            _, task, probe = max(candidates, key=lambda candidate: candidate[0])
        return task, probe

    def learn(self, task, probe, answer):
        outcomes = self.answers[task][probe]
        local = self.choices[task]
        remaining = frozenset(i for i in local if outcomes[i] == answer)
        if not remaining:
            raise ValueError('oracle contradicted all local hypotheses')
        self.choices = self.choices[:task] + (remaining,) + self.choices[task + 1:]
        if self.worlds is not None:
            survivors = tuple(world for world in self.worlds if outcomes[world[task]] == answer)
            if not survivors:
                # Preserve each observed label and withdraw assumptions about sharing.
                self.contradiction = True
                self.worlds = None
            else:
                self.worlds = survivors

    def solved(self):
        return resolved(self.choices if self.worlds is None else projection(self.worlds))


def cases(size):
    rng = random.Random(472916)
    yield from [('shared', (i,) * size) for i in range(len(OPTIONS))]
    for count in (1, 2):
        for i in range(len(OPTIONS)):
            allowed = [j for j in range(len(OPTIONS)) if j != i]
            positions = [tuple(rng.sample(range(size), count)) for _ in range(4)]
            for changed in positions:
                world = [i] * size
                for index in changed:
                    world[index] = rng.choice(allowed)
                yield f'{count}_exceptions', tuple(world)


def evaluate(operations, hidden, strategy, seed):
    search = Search(operations, strategy, seed)
    record = {'first_all_correct': None, 'ever_wrong': 0, 'first_contradiction': None,
              'local_audit_detected': False, 'budgets': {}, 'queries': []}
    count = 0
    for step in range(max(BUDGETS) + 1):
        solved = search.solved()
        failures = sum(execute(operations[task], config, values) != oracle(operations[task], hidden[task], values)
                       for task, config in solved.items() for values in CHECKS)
        right = sum(hidden[i] == option for i, option in solved.items())
        wrong = len(solved) - right
        record['ever_wrong'] = max(wrong, record['ever_wrong'])
        if record['first_all_correct'] is None and right == len(hidden):
            record['first_all_correct'] = count
        if step in BUDGETS:
            record['budgets'][step] = {'correct_procedures': right, 'wrong_procedures': wrong,
                                       'wrong_unseen_executions': failures, 'queries': count}
        if step == max(BUDGETS):
            break
        selected = search.select()
        if selected is None:
            continue
        task, probe = selected
        answer = oracle(operations[task], hidden[task], PROBES[probe])
        search.learn(task, probe, answer)
        count += 1
        record['queries'].append([task, PROBES[probe], answer])
        if search.contradiction and record['first_contradiction'] is None:
            record['first_contradiction'] = count
    # Audit each published task at one separate, fixed point (cost counted).
    # This is a single regression observation per task, not exhaustive validation.
    audit_input = (-41, 17)
    record['local_audit_queries'] = len(solved)
    record['local_audit_detected'] = any(execute(operations[i], config, audit_input) !=
                                         oracle(operations[i], hidden[i], audit_input)
                                         for i, config in solved.items())
    record['targeted_audit'] = audit_transferred(search, solved, operations, hidden)
    return record


def audit_transferred(search, solved, operations, hidden):
    """Check confirmations against local alternatives without assuming sharing."""
    remaining = list(search.choices)
    checks = 0
    rejected = 0
    for task, proposed in solved.items():
        options = remaining[task]
        while proposed in options and len(options) > 1:
            # Choose the probe that leaves the fewest local alternatives if
            # its answer agrees with the shared proposal.
            op = operations[task]
            probes = [(values, partition(options, [execute(op, i, values) for i in range(len(OPTIONS))]))
                      for values in AUDIT_PROBES]
            values, buckets = min(probes, key=lambda probe: len(probe[1][execute(op, proposed, probe[0])]))
            expected = execute(op, proposed, values)
            if len(buckets[expected]) == len(options):
                raise AssertionError('local alternatives cannot be distinguished by audit')
            checks += 1
            observed = oracle(op, hidden[task], values)
            options = buckets[observed]
            rejected += observed != expected
        remaining[task] = options
    approved = {task: proposed for task, proposed in solved.items()
                if remaining[task] == {proposed}}
    if any(hidden[task] != proposed for task, proposed in approved.items()):
        raise AssertionError('incorrect local proof')
    return {'checks': checks, 'rejected_proposals': rejected, 'approved': len(approved)}


def summarize(runs):
    out = {}
    for scenario, arms in runs.items():
        out[scenario] = {}
        for strategy, records in arms.items():
            times = [r['first_all_correct'] for r in records if r['first_all_correct'] is not None]
            out[scenario][strategy] = {
                'episodes': len(records), 'finished_within_budget': len(times),
                'median_queries_when_finished': statistics.median(times) if times else None,
                'episodes_ever_wrong': sum(r['ever_wrong'] > 0 for r in records),
                'contradictions': sum(r['first_contradiction'] is not None for r in records),
                'local_audit_detections': sum(r['local_audit_detected'] for r in records),
                'total_local_audit_queries': sum(r['local_audit_queries'] for r in records),
                'targeted_audit': {
                    'detected_episodes': sum(r['targeted_audit']['rejected_proposals'] > 0 for r in records),
                    'median_extra_checks': statistics.median(r['targeted_audit']['checks'] for r in records),
                    'median_total_checks': statistics.median(len(r['queries']) + r['targeted_audit']['checks']
                                                             for r in records),
                    'median_approved': statistics.median(r['targeted_audit']['approved'] for r in records),
                },
                'budgets': {n: {key: round(statistics.mean(r['budgets'][n][key] for r in records), 4)
                                for key in records[0]['budgets'][n]} for n in BUDGETS}}
    return out


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    args = cli.parse_args()
    catalog = inventory()
    operations = tuple(op for _, op in catalog)
    if len(catalog) != 7:
        raise ValueError('unexpected numerical expert inventory')
    for op in set(operations):
        for inputs in (PROBES, CHECKS, AUDIT_PROBES):
            signatures = {tuple(execute(op, code, values) for values in inputs) for code in range(len(OPTIONS))}
            if len(signatures) != len(OPTIONS):
                raise ValueError(f'unidentifiable policies for {op}')
            if any(execute(op, i, values) != oracle(op, i, values)
                   for i in range(len(OPTIONS)) for values in inputs):
                raise ValueError('executor and reference disagree')
    runs = defaultdict(lambda: defaultdict(list))
    for seed, (scenario, target) in enumerate(cases(len(catalog))):
        for strategy in ('independent', 'random', 'entropy', 'impact'):
            result = evaluate(operations, target, strategy, seed)
            if scenario != '2_exceptions' and result['ever_wrong']:
                raise AssertionError('wrong in-domain certification')
            runs[scenario][strategy].append({'target': target, **result})
    report = {'families': catalog, 'candidate_configurations': OPTIONS, 'shared_worlds': len(possible_worlds(7)),
              'query_values': PROBES, 'test_values': CHECKS, 'audit_values': AUDIT_PROBES,
              'summary': summarize(runs), 'episodes': runs}
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, separators=(',', ':'))
        stream.write('\n')


if __name__ == '__main__':
    main()
