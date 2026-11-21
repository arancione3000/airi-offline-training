"""Reproduce equivalence, rollback, work and measured CPU cost on one revision stream."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from time import process_time_ns

import fixtures
import verifier


MODES = ('full', 'structural', 'semantic')
COUNTERS = ('evaluated_nodes', 'evaluated_cells', 'reused_nodes', 'compared_reference_cells',
            'compared_dependency_cells')


def fingerprint(tables):
    payload = json.dumps(dict(tables), sort_keys=True, separators=(',', ':')).encode()
    return sha256(payload).hexdigest()


def measure(mode, versions):
    engine = verifier.Verifier(fixtures.INPUTS, mode)
    results = []
    for name, graph, expected, should_accept in versions:
        before = engine.tables
        previous_version = engine.version
        start = process_time_ns()
        report = engine.update(graph, expected)
        elapsed = process_time_ns() - start
        if (report['status'] == 'accepted') != should_accept:
            raise AssertionError((mode, name, report))
        after = engine.tables
        if should_accept:
            if after != expected or engine.version != previous_version + 1:
                raise AssertionError('a successful update did not commit exactly the reference')
        elif before != after or engine.version != previous_version:
            raise AssertionError('a rejected update changed the committed table')
        results.append({'name': name, 'nodes': len(graph), 'cpu_ns': elapsed,
                        'committed_sha256': fingerprint(after), **report})
    return results


def summarize(traces):
    first = traces[0][0]
    warm = [r for r in traces[0][1:] if r['status'] == 'accepted']
    return {'first': {k: first[k] for k in COUNTERS},
            'accepted_updates': len(warm),
            'accepted_updates_total': {k: sum(row[k] for row in warm) for k in COUNTERS},
            'rejected_updates': sum(row['status'] == 'rejected' for row in traces[0]),
            'cpu_ms': {'first_median': median(trace[0]['cpu_ns'] for trace in traces) / 1e6,
                       'accepted_updates_total_median': median(sum(r['cpu_ns'] for r in trace[1:]
                                                                     if r['status'] == 'accepted')
                                                                 for trace in traces) / 1e6,
                       'complete_sequence_median': median(sum(r['cpu_ns'] for r in trace)
                                                          for trace in traces) / 1e6}}


def run(repeats):
    preparation = []
    for _ in range(repeats):
        start = process_time_ns()
        versions = tuple(fixtures.scenarios())
        preparation.append(process_time_ns() - start)
    runs = {mode: [] for mode in MODES}
    for i in range(repeats):
        # Interleave methods to reduce systematic ordering effects on timings.
        for mode in MODES[i % len(MODES):] + MODES[:i % len(MODES)]:
            runs[mode].append(measure(mode, versions))
    stable = {mode: [{k: v for k, v in row.items() if k != 'cpu_ns'} for row in traces[0]]
              for mode, traces in runs.items()}
    equal_fields = ('name', 'status', 'version', 'committed_sha256', 'mismatch')
    baseline = [tuple(row[k] for k in equal_fields) for row in stable['full']]
    for rows in stable.values():
        if [tuple(row[k] for k in equal_fields) for row in rows] != baseline:
            raise AssertionError('verification modes disagree')
    metrics = {mode: summarize(traces) for mode, traces in runs.items()}
    computations = {mode: row['accepted_updates_total']['evaluated_cells']
                    for mode, row in metrics.items()}
    if not computations['semantic'] < computations['structural'] < computations['full']:
        raise AssertionError('the semantic cutoff failed to reduce execution work')
    return {'domain_pairs': len(fixtures.INPUTS), 'versions': len(versions), 'cpu_repeats': repeats,
            'preparation_cpu_ms_median': median(preparation) / 1e6,
            'sha256': {name: sha256(Path(module.__file__).read_bytes()).hexdigest()
                       for name, module in [('verifier', verifier), ('fixtures', fixtures)]},
            'metrics': metrics, 'versions_by_method': stable}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    cli.add_argument('--repeats', type=int, default=3)
    args = cli.parse_args()
    if not 1 <= args.repeats <= 9:
        cli.error('repeats must be between one and nine')
    result = run(args.repeats)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
