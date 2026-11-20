"""Secondary diagnostic on the observed pool. Rank expected compositional yield."""
import argparse
import json
from pathlib import Path

from trial import POOL, SEEDS, V5, archive, article, key, mask, recombinations, selection, tie
from trial import typed_router
import gzip
import tempfile


def run(output):
    families = archive('precommitted/FAMILIES.json.gz')['families']
    rows = {row['id']: row for row in POOL}
    frames = {i: mask(row['query'], families) for i, row in rows.items()}
    reach = {
        i: sum(f['id'] != source['id'] and f['operation'] == source['operation']
               and article(f) == article(source) for f in families)
        for i, (source, _) in frames.items()
    }
    with tempfile.TemporaryDirectory() as temporary:
        candidate = Path(temporary) / 'candidate.json'
        candidate.write_bytes(gzip.decompress(V5.read_bytes()))
        verifier = typed_router.compile_model(typed_router.read_model(candidate))
        # Fixed rough probabilities; only the V5 prediction is used here.
        likelihood = {i: .85 if verifier.predict(row['query']) else .50 for i, row in rows.items()}
    positives, _ = recombinations(rows, frames, families)
    results = {}
    for name, weights in (('reach', reach), ('v5_times_reach', {i: reach[i] * likelihood[i] for i in rows})):
        counts = []
        for seed in SEEDS:
            chosen = sorted(weights, key=lambda i: (-weights[i], tie(seed, i)))[:16]
            installed, kept = selection(chosen, rows, frames)
            found = sum(key(family, mask(query, families)[1]) in installed
                        for family, query in positives.values())
            counts.append({'accepted': len(kept), 'recombined_pairs': found})
        results[name] = {'repetitions': counts, 'means': {
            k: sum(row[k] for row in counts) / len(counts) for k in counts[0]}}
    report = {'observed_pool_diagnostic': True, 'verification_budget': 16, 'methods': results}
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: value['means'] for name, value in results.items()}))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    run(cli.parse_args().output)
