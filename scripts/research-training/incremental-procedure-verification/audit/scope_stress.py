"""Probe the published one-operation synthesis envelope on new target functions."""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path

import fixtures
import synthesis
from verifier import Verifier

# Fixed before seeing any candidate predictions.
TARGETS = {
    'sum': lambda a, b: a + b,
    'product': lambda a, b: a * b,
    'difference_reversed': lambda a, b: b - a,
    'absolute_dividend': lambda a, b: abs(a),
    'negated_floor_quotient': lambda a, b: -(a // b),
    'absolute_sum': lambda a, b: abs(a + b),
    'squared_sum': lambda a, b: (a + b) ** 2,
    'maximum': max,
    'minimum_magnitude': lambda a, b: min(abs(a), abs(b)),
    'gcd': math.gcd,
}


def test():
    _, graph, baseline, should_accept = next(fixtures.scenarios())
    engine = Verifier(fixtures.INPUTS)
    assert should_accept and engine.update(graph, baseline)['status'] == 'accepted'
    hypotheses, enumeration = synthesis.RecipeLibrary(cache=False).enumerate(engine.tables)
    report = {}
    for name, oracle in TARGETS.items():
        verdict, queries = synthesis.learn(hypotheses, lambda i: oracle(*fixtures.INPUTS[i]))
        gold = tuple(oracle(*inputs) for inputs in fixtures.INPUTS)
        count_matches = sum(candidate.outputs == gold for candidate in hypotheses)
        result = {
            'outcome': 'abstained_no_hypotheses',
            'in_hypothesis_space': bool(count_matches),
            'queried_examples': len(queries),
            'queries': queries,
        }
        if verdict is not None:
            # Each task independently starts from the same approved bank.
            trial = Verifier(fixtures.INPUTS)
            assert trial.update(graph, baseline)['status'] == 'accepted'
            trial_graph = {**graph, 'target': verdict.definition()}
            receipt = trial.update(trial_graph, {**baseline, 'target': gold})
            result['recipe'] = verdict.definition()
            result['qualification'] = receipt
            result['outcome'] = 'verified' if receipt['status'] == 'accepted' else 'rejected_by_exhaustive_verifier'
            if receipt['status'] != 'accepted':
                assert trial.version == 1 and trial.tables == baseline
        assert (result['outcome'] == 'verified') == bool(count_matches)
        report[name] = result
    return {
        'domain_pairs': len(fixtures.INPUTS),
        'approved_base_nodes': len(graph),
        'candidate_enumeration': enumeration,
        'source_sha256': {m.__name__: hashlib.sha256(Path(inspect.getfile(m)).read_bytes()).hexdigest()
                          for m in (synthesis, fixtures, inspect.getmodule(Verifier))},
        'tasks': report,
        'total_solved': sum(r['outcome'] == 'verified' for r in report.values()),
        'total_abstained': sum(r['outcome'] == 'abstained_no_hypotheses' for r in report.values()),
        'total_rejected': sum(r['outcome'] == 'rejected_by_exhaustive_verifier' for r in report.values()),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = test()
    with args.output.open('x') as fd:
        json.dump(result, fd, indent=2)
        fd.write('\n')
    print({k: v['outcome'] for k, v in result['tasks'].items()})
    print({k: v for k,v in result.items() if k.startswith('total_')})
