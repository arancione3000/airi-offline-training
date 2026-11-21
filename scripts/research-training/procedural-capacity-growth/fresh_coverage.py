"""Evaluate the fixed procedure grower on ten new arithmetic goals."""

# Chosen before reading the candidate or the existing test definitions.
# The functions deliberately mix different composition and coverage demands.
def targets():
    return {
        'product_plus_divider': lambda a, b: a * b + b,
        'absolute_difference': lambda a, b: abs(a - b),
        'sum_times_difference': lambda a, b: (a + b) * (a - b),
        'maximum_negated_operands': lambda a, b: max(-a, -b),
        'absolute_floor_quotient': lambda a, b: abs(a // b),
        'maximum_dividend_divisor_magnitude': lambda a, b: max(a, abs(b)),
        'product_offset_by_dividend': lambda a, b: a * b - a,
        'remainder_parity': lambda a, b: (a % b) % 2,
        'squared_norm': lambda a, b: a * a + b * b,
        'conditional_sign': lambda a, b: a if b > 0 else -a,
    }


def run(repo):
    import hashlib
    import sys
    from pathlib import Path

    source = (repo / 'scripts/research-training/procedural-capacity-growth').resolve()
    sys.path.insert(0, str(source))
    from synthesis_growth import GrowingLearner, INPUTS, qualify_with_verifier, scenarios

    _, graph, baseline, accepted = next(scenarios())
    if not accepted:
        raise RuntimeError('fixture baseline failed')
    learners = {
        'shallow': (1, GrowingLearner(graph, baseline)),
        'deep_arithmetic_only': (2, GrowingLearner(graph, baseline, allow_order_macros=False)),
        'deep_with_macros': (2, GrowingLearner(graph, baseline)),
    }
    results = {}
    for strategy, (depth, learner) in learners.items():
        cases = {}
        for name, function in targets().items():
            def oracle(index):
                return function(*INPUTS[index])
            cases[name] = learner.learn(
                oracle, qualify_with_verifier(graph, baseline, oracle), max_depth=depth)
        results[strategy] = {
            'verified': sum(case['status'] == 'accepted' for case in cases.values()),
            'scalar_labels': sum(len(case['labels']) for case in cases.values()),
            'cases': cases,
        }
    return {
        'domain_size': len(INPUTS),
        'number_of_targets': len(targets()),
        'compiler_sha256': hashlib.sha256((source / 'synthesis_growth.py').read_bytes()).hexdigest(),
        'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'results': results,
    }


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = run(args.repo)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    for strategy, report in result['results'].items():
        print(strategy, report['verified'], report['scalar_labels'])
        print([(name, case['status'], case['depth'], case.get('reason'))
               for name, case in report['cases'].items()])
