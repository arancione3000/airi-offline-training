"""Freeze question-only donor decisions before joining trusted TRAIN labels."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import runpy


HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
POOLS = tuple(RESEARCH / 'verified-frames-20261120' / name for name in ('pool.py', 'pool_confirmation.py'))
FAMILIES = RESEARCH / 'semantic-routing-20261116/evidence/precommitted/FAMILIES.json.gz'
DONOR = HERE / 'joint.py'
RECEIPT = HERE / 'donor_decisions.json'


def input_hashes():
    return {str(path.relative_to(RESEARCH)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (*POOLS, FAMILIES, DONOR)}


def proposals():
    rows = {}
    for path in POOLS:
        for row in runpy.run_path(str(path))['POOL']:
            if row['id'] in rows:
                raise ValueError('duplicate TRAIN proposal identity')
            rows[row['id']] = row
    return rows


def freeze(families):
    before = input_hashes()
    predict = runpy.run_path(str(DONOR))['predict']
    # The classifier never sees the row id, source lesson or independent intent.
    predictions = {key: predict(row['query'], families) for key, row in sorted(proposals().items())}
    if input_hashes() != before:
        raise ValueError('donor or TRAIN inputs changed during freeze')
    return {'schema': 1, 'inputs': before, 'predictions': predictions}


def reviewed_rows(families, expected_sha256, receipt_path=RECEIPT):
    raw = receipt_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('wrong frozen TRAIN donation receipt')
    receipt = json.loads(raw)
    if receipt['schema'] != 1 or receipt['inputs'] != input_hashes():
        raise ValueError('stale TRAIN donation inputs')
    pool = proposals()
    choices = receipt['predictions']
    definitions = {family['id']: family['operation'] for family in families}
    if choices.keys() != pool.keys():
        raise ValueError('incomplete donor decisions')
    rows = []
    for key, row in pool.items():
        family, intent = row['family'], row['intent']
        if family not in definitions or intent not in (None, definitions[family]):
            raise ValueError('adjudicated TRAIN label disagrees with supported family')
        choice = choices[key]
        if choice is not None and choice not in definitions:
            raise ValueError('unknown donor prediction')
        if intent is None:
            condition = 'unsupported_donation'
        elif choice == family:
            condition = 'supported_paraphrase'
        else:
            condition = 'unconfirmed_donation'
        rows.append({'id': 'joint-donor-' + key, 'query': row['query'], 'source_lesson_id': family,
                     'lesson_id': family if intent is not None else None, 'condition': condition})
    return rows


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    args = cli.parse_args()
    with gzip.open(FAMILIES, 'rt') as stream:
        manifest = freeze(json.load(stream)['families'])
    with args.output.open('x') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')
