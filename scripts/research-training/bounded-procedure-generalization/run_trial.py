"""Evaluate frozen procedure-learning arms without exposing qualification or final labels to fit."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from learner import execute_program, fit, key, validate_program
from registry import ProgramRegistry


HERE = Path(__file__).resolve().parent
CASES_SHA256 = 'b1e67abc956eec9161c2d524d92bf25470d6d41bed2d545fd764a7e3027f02e4'
ARMS = {
    'exact_memory': None,
    'one_step_active': (1, 'disagreement'),
    'three_step_ordered': (3, 'ordered'),
    'three_step_active': (3, 'disagreement'),
}
SPLITS = ('train', 'pool', 'qualification', 'final')
OUTCOMES = ('correct_answers', 'wrong_answers', 'positive_abstentions', 'correct_reviews', 'unsafe_answers')


def read_corpus(path):
    source = path.read_bytes()
    if sha256(source).hexdigest() != CASES_SHA256:
        raise ValueError('frozen corpus SHA-256 mismatch')
    corpus = json.loads(source)
    if corpus.get('schema') != 1:
        raise ValueError('unexpected corpus schema')
    tasks = corpus['tasks']
    ids = {task['id'] for task in tasks}
    if len(ids) != len(tasks):
        raise ValueError('duplicate task IDs')
    for task in tasks:
        seen = set()
        for split in SPLITS:
            rows = task[split]
            inputs = [key(row['input']) for row in rows]
            if len(inputs) != len(set(inputs)) or seen.intersection(inputs):
                raise ValueError(f"input overlap: {task['id']} {split}")
            seen.update(inputs)
    return tasks


def learn_on_public_evidence(config, train, pool):
    max_steps, policy = config
    observed = []

    def oracle(index):
        row = pool[index]
        observed.append(row)
        return row['expected']

    # The learner sees pool inputs and may request at most three real labels.
    proposal = fit(train, [row['input'] for row in pool], oracle, budget=3,
                   max_steps=max_steps, policy=policy)
    indices = [query['index'] for query in proposal['queries']]
    if observed != [pool[index] for index in indices] or len(indices) > 3 or len(set(indices)) != len(indices):
        raise AssertionError('oracle query log mismatches released labels')
    return proposal, [*train, *observed]


def same(actual, expected):
    return key(actual) == key(expected)


def score(respond, rows):
    details = []
    for index, row in enumerate(rows):
        result = respond(row['input'])
        value = result.get('answer')
        expected = row['expected']
        if expected is None:
            outcome = 'correct_reviews' if result['status'] != 'answered' else 'unsafe_answers'
        elif result['status'] != 'answered':
            outcome = 'positive_abstentions'
        else:
            outcome = 'correct_answers' if same(value, expected) else 'wrong_answers'
        details.append({'row': index, 'outcome': outcome, 'status': result['status'],
                        'answer': value, 'expected': expected})
    return details


def totals(results):
    metrics = Counter({name: 0 for name in OUTCOMES})
    for result in results:
        metrics.update(row['outcome'] for row in result['final'])
    metrics['activated_tasks'] = sum(item['admission']['status'] == 'activated' for item in results)
    metrics['oracle_queries'] = sum(len(item['fit']['queries']) for item in results if item['fit'])
    metrics['fully_correct_answerable_tasks'] = sum(
        any(row['expected'] is not None for row in item['final'])
        and all(row['outcome'] in ('correct_answers', 'correct_reviews') for row in item['final'])
        for item in results
    )
    return dict(metrics)


def trial(tasks, config, runtime_id, path):
    def reopen():
        return ProgramRegistry(path, validate_program=validate_program, execute_program=execute_program,
                               runtime_id=runtime_id)

    registry = reopen() if config else None
    admitted = []
    results = []
    revision = 0
    for task in tasks:
        task_id = task['id']
        training = task['train']
        qualification = task['qualification']
        proposal = None
        if config is not None:
            proposal, observed = learn_on_public_evidence(config, training, task['pool'])
            if proposal['program'] is None:
                admission = {'status': 'not_proposed', 'reason': proposal['status'], 'revision': revision}
            else:
                # Qualification is evaluated only by the registry after the proposal is fixed.
                admission = registry.acquire(task_id, proposal['program'], observed, qualification,
                                             expected_revision=revision)
                if admission['status'] == 'activated':
                    assert admission['revision'] == revision + 1
                    assert admission['replayed_cases'] == sum(len(rows) for _, rows in admitted)
                    revision += 1
                    admitted.append((task_id, qualification))
        else:
            admission = {'status': 'baseline', 'revision': 0}
        results.append({'id': task_id, 'domain': task['domain'], 'kind': task['kind'],
                        'fit': proposal, 'admission': admission})

    replayed = 0
    if registry is not None:
        # Reopen against the exact runtime hash before reading any final rows.
        registry = reopen()
        for task_id, rows in admitted:
            for row in rows:
                result = registry.assess(task_id, row['input'])
                if not same(result.get('answer'), row['expected']):
                    raise AssertionError(f'historical replay after reopen failed: {task_id}')
                replayed += 1

    for task, result in zip(tasks, results, strict=True):
        if registry is None:
            observed = {key(row['input']): row['expected'] for row in task['train']}

            def respond(value):
                answer = observed.get(key(value))
                return {'status': 'answered', 'answer': answer} if answer is not None else {'status': 'review'}
        else:
            def respond(value):
                return registry.assess(task['id'], value)

        result['final'] = score(respond, task['final'])
        if registry is not None and result['admission']['status'] != 'activated':
            if any(row['status'] != 'unknown_skill' for row in result['final']):
                raise AssertionError('unqualified skill exposed through the registry')
    return {
        'configuration': None if config is None else {
            'max_steps': config[0], 'query_policy': config[1], 'query_budget': 3,
        },
        'all': totals(results),
        'by_domain': {domain: totals([item for item in results if item['domain'] == domain])
                      for domain in sorted({task['domain'] for task in tasks})},
        'by_kind': {kind: totals([item for item in results if item['kind'] == kind])
                    for kind in sorted({task['kind'] for task in tasks})},
        'admissions': dict(Counter(item['admission']['status'] for item in results)),
        'qualification_replayed_after_reopen': replayed,
        'tasks': results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cases', type=Path, default=HERE / 'CASES.json')
    args = parser.parse_args()
    tasks = read_corpus(args.cases)
    sources = {name: sha256((HERE / name).read_bytes()).hexdigest()
               for name in ('learner.py', 'registry.py', 'oracle_cases.py', 'run_trial.py')}
    runtime_id = sha256(key({name: sources[name] for name in ('learner.py', 'registry.py')}).encode()).hexdigest()
    with TemporaryDirectory(prefix='airi-procedures-trial-') as directory:
        arms = {name: trial(tasks, config, runtime_id, Path(directory) / f'{name}.sqlite')
                for name, config in ARMS.items()}
    evidence = {'schema': 1, 'cases_sha256': CASES_SHA256, 'sources_sha256': sources,
                'runtime_id': runtime_id, 'arms': arms}
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
                           encoding='utf-8')
    print(json.dumps({name: arm['all'] for name, arm in arms.items()}, indent=2))


if __name__ == '__main__':
    main()
