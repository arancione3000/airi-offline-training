"""Confirm proposal priority on a fresh pool without changing the ranking weights."""
import argparse
import gzip
import json
from pathlib import Path
import tempfile

from pool_confirmation import POOL
from trial import SEEDS, V5, archive, article, fingerprint, key, mask, recombinations, selection, tie, typed_router
from skill_bank import digest

BUDGET = 16


def rankings(requests, families):
    """Use question text and configured family metadata only."""
    with tempfile.TemporaryDirectory() as temporary:
        artifact = Path(temporary) / 'v5.json'
        artifact.write_bytes(gzip.decompress(V5.read_bytes()))
        donor = typed_router.compile_model(typed_router.read_model(artifact))
        reach, weighted = {}, {}
        for request in requests:
            identifier, query = request
            match = mask(query, families)
            if match is None:
                raise ValueError('ambiguous proposal')
            family, _ = match
            reach[identifier] = sum(
                other['id'] != family['id'] and other['operation'] == family['operation']
                and article(other) == article(family) for other in families)
            weighted[identifier] = reach[identifier] * (.85 if donor.predict(query) else .50)
        return {
            name: [sorted(weights, key=lambda i: (-weights[i], tie(seed, i)))[:BUDGET] for seed in SEEDS]
            for name, weights in (('reach', reach), ('v5_times_reach', weighted))
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    families = archive('precommitted/FAMILIES.json.gz')['families']
    order = rankings([(row['id'], row['query']) for row in POOL], families)
    design = dict(algorithm_sha256=fingerprint(Path(__file__)), donor_sha256=fingerprint(V5),
                  input_sha256=digest(POOL), budget=BUDGET, rankings=order)
    (output / 'FREEZE.json').write_text(json.dumps(design, indent=2) + '\n')
    rows = {row['id']: row for row in POOL}
    frames = {identifier: mask(row['query'], families) for identifier, row in rows.items()}
    if any(frames[i][0]['id'] != row['family'] for i, row in rows.items()):
        raise ValueError('proposal matches a different family')
    possible, _ = recombinations(rows, frames, families)
    pair_keys = [key(family, mask(query, families)[1]) for family, query in possible.values()]
    result = {}
    for name, schedules in order.items():
        counts = []
        for identifiers in schedules:
            installed, kept = selection(identifiers, rows, frames)
            counts.append({'useful_labels': len(kept), 'recombined_pairs': sum(k in installed for k in pair_keys)})
        result[name] = {'repetitions': counts,
                        'mean': {k: sum(row[k] for row in counts) / len(counts) for k in counts[0]}}
    summary = dict(freeze_sha256=digest(design), pool_size=len(POOL),
                   possible_pairs=len(pair_keys), results=result)
    (output / 'RESULTS.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({name: value['mean'] for name, value in result.items()}))


if __name__ == '__main__':
    main()
