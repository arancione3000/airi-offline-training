"""Extract one exact, independently labeled TRAIN request per procedural family.

The donor proposes the binding only. The normal curriculum supplies independent
answers, admission, fresh-value checks and retention before publication.
"""
import argparse
import gzip
import json
import tempfile
from pathlib import Path

from experiment import HERE, load
import typed_router
from auto_curriculum import BASE_CATALOG


def extract():
    rows = load('precommitted/TRAIN.json.gz')['rows']
    artifact = HERE / 'evidence/candidate.json.gz'
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'candidate.json'
        path.write_bytes(gzip.decompress(artifact.read_bytes()))
        model = typed_router.read_model(path)
    if typed_router.common._digest(rows) != model['train_digest']:
        raise ValueError('donor TRAIN identity mismatch')
    donor = typed_router.compile_model(model)
    labels = {}
    for row in rows:
        labels.setdefault(row['query'], set()).add(row['lesson_id'])
    selected = []
    for family in BASE_CATALOG:
        for row in rows:
            query = row['query']
            if (row['lesson_id'] != family['id'] or labels[query] != {family['id']}
                    or query == family['query']):
                continue
            prediction = donor.predict(query)
            if prediction and prediction['lesson_id'] == family['id']:
                selected.append(dict(parent=family['id'], query=query, train_id=row['id']))
                break
    return dict(donor_sha256=model['sha256'], train_sha256=model['train_digest'], fragments=selected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(extract(), indent=2, ensure_ascii=False) + '\n')
