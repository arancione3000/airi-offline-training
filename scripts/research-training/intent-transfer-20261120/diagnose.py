"""Attribute the first abstention on supported new wordings to a frozen stage."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from evaluate import load_split
import learner
import parser


def parser_stage(query, families):
    normalized = unicodedata.normalize('NFC', query).strip().casefold().replace('’', "'")
    normalized = re.sub(r'[?!.]$', '', normalized)
    if not re.fullmatch(r"[a-zàèéìòù0-9' ,:]+", normalized):
        return 'input syntax'
    bound = parser._bind(parser._canonical(normalized), families)
    if bound is None:
        return 'missing or ambiguous named object'
    return 'accepted' if parser.predict(query, families) else 'no full operation rule'


def learned_stage(query, model):
    if learner.query_router._normal(query) is None:
        return 'input guard'
    bound = learner._frame(query, model.families)
    if bound is None:
        return 'input syntax or missing or ambiguous named object'
    family, frame = bound
    keys = learner._keys(frame)
    if any(key not in model.vocab for key in keys):
        return 'unseen word prefix'
    rank = model._rank(family['operation'], keys, model.vectors @ learner._embedding(frame))
    if rank is None:
        return 'insufficient prototypes'
    if rank[0] < model.score:
        return 'insufficient similarity'
    return 'insufficient margin' if rank[1] < model.margin else 'accepted'


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('holdout', type=Path)
    cli.add_argument('freeze', type=Path)
    cli.add_argument('output', type=Path)
    args = cli.parse_args()
    freeze = json.loads(args.freeze.read_text())['sha256']
    for name, path in [('parser', Path(parser.__file__)), ('learner', Path(learner.__file__)),
                       ('prospective', args.holdout)]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != freeze[name]:
            raise ValueError('input differs from freeze: ' + name)
    families = load_split('FAMILIES')['families']
    rows = [row for row in json.loads(args.holdout.read_bytes())['rows'] if row['lesson_id'] is not None]
    report = {'parser': dict(Counter(parser_stage(row['query'], families) for row in rows))}
    for expanded, name in [(False, 'learned_train'), (True, 'learned_expanded')]:
        model = learner.fit(families, expanded)
        report[name] = dict(Counter(learned_stage(row['query'], model) for row in rows))
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
