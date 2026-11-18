"""Freeze a TRAIN-only research candidate, then independently replay qualification."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import typed_router
from learn import selected_bank
from query_learning import bindings, evaluate

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent / 'semantic-routing-20261116' / 'evidence'


def load(relative):
    return json.loads(gzip.decompress((ARCHIVE / relative).read_bytes()))


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write('\n')


def run(phase, output):
    families = load('precommitted/FAMILIES.json.gz')['families']
    if phase == 'freeze':
        output.mkdir(parents=True, exist_ok=False)
        (output / 'bank.json').write_bytes(gzip.decompress((ARCHIVE / 'bank.json.gz').read_bytes()))
        bank = selected_bank(output / 'bank.json')
        rows = load('precommitted/TRAIN.json.gz')['rows']
        model = typed_router.train(rows, bindings(bank, families), families)
        write(output / 'candidate.json', model)
        print(json.dumps(model['calibration']), model['thresholds'], flush=True)
        return
    model = typed_router.read_model(output / 'candidate.json')
    bank = selected_bank(output / 'bank.json')
    rows = load('precommitted/QUALIFICATION.json.gz')['rows']
    result = evaluate(bank, output / 'candidate.json', rows, families,
                      value_offset=10000, backend=typed_router)
    write(output / 'qualification.json', result)
    observations = result.pop('observations')
    by_family = {}
    source = {r['id']: r['source_lesson_id'] for r in rows}
    for r in observations:
        if r['positive'] and r['variant'] == 0:
            counts = by_family.setdefault(source[r['id']], [0, 0])
            counts[0] += r['correct']
            counts[1] += 1
    summary = dict(result, by_family_distinct_phrasings=by_family,
                   model_sha256=model['sha256'], calibration=model['calibration'],
                   thresholds=model['thresholds'], activated=False, final_observed=False,
                   qualification_observed_previously=True)
    write(output / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'historical_gate'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['freeze', 'qualification'])
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.phase, args.output)
