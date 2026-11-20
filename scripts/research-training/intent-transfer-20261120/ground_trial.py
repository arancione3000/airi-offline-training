"""Isolate object identification and additionally score full skill execution."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import runpy
import tempfile
import unicodedata

import evaluate as ev
import grounding
import parser


def old_bind(query, families):
    text = unicodedata.normalize('NFC', query).strip().casefold().replace('’', "'")
    text = re.sub(r'[?!.]$', '', text)
    if not re.fullmatch(r"[a-zàèéìòù0-9' ,:]+", text):
        return None
    match = parser._bind(parser._canonical(text), families)
    return match[0]['id'] if match else None


def object_scores(method, rows, families):
    counts = Counter()
    records = []
    for row in rows:
        got = method(row['query'], families)
        expected = row['expected_object']
        if got == expected:
            outcome = 'correct_object' if got is not None else 'correct_abstention'
        elif got is None:
            outcome = 'missed_object'
        else:
            outcome = 'incorrect_object' if expected is not None else 'unsupported_activation'
        counts[outcome] += 1
        records.append({'id': row['id'], 'expected': expected, 'actual': got, 'outcome': outcome})
    return {'counts': counts, 'records': records}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('holdout', type=Path)
    cli.add_argument('digest')
    cli.add_argument('candidate_digest')
    cli.add_argument('output', type=Path)
    cli.add_argument('--aliases', type=Path, default=ev.APPROVED_ALIASES)
    cli.add_argument('--bank', type=Path, default=ev.BUNDLED)
    args = cli.parse_args()
    if args.output.exists():
        cli.error('use a fresh output path')
    if (ev.file_hash(args.holdout) != args.digest
            or ev.file_hash(Path(grounding.__file__)) != args.candidate_digest):
        raise ValueError('incorrect holdout or candidate digest')
    data = json.loads(args.holdout.read_bytes())
    families = ev.load_split('FAMILIES')['families']
    known = {f['id']: f for f in families}
    bank = ev.SkillBank(args.bank)
    aliases = ev.AliasRouter(bank, args.aliases)
    entries = bank._read()[0]['entries']
    owners = {}
    for entry in entries:
        p = entry['program']
        owner, = [f['id'] for f in families if p['operation'] == f['operation'] and p['source'] == f['source']]
        owners[entry['id']] = owner
    splits = {'first_prospective': json.loads((ev.HERE / 'prospective.json').read_bytes()),
              'v6_challenge': runpy.run_path(str(ev.V6 / 'blind/cases.py'))['dataset'](),
              'frames_challenge': runpy.run_path(str(ev.FRAMES / 'holdout.py'))['dataset'](),
              'qualification': {'rows': ev.load_split('QUALIFICATION')['rows'], 'value_offset': 10000}}
    exposure = {name: [row['query'] for row in split['rows']] for name, split in splits.items()}
    exposure['train'] = [r['query'] for r in ev.load_split('TRAIN')['rows']]
    exposure['aliases'] = list(ev.read_aliases(args.aliases))
    exposure['bank'] = [e['program']['query'] for e in entries]
    for name in ('pool', 'pool_confirmation'):
        exposure[name] = [r['query'] for r in runpy.run_path(str(ev.FRAMES / (name + '.py')))['POOL']]
    report = {'candidate_sha256': args.candidate_digest, 'holdout_sha256': args.digest,
              'bank_sha256': ev.file_hash(args.bank), 'aliases_sha256': ev.file_hash(args.aliases),
              'overlap': ev.audit_split(data['rows'], exposure, families), 'methods': {}}
    splits['new_names'] = data
    with tempfile.TemporaryDirectory() as work:
        model = Path(work) / 'bindings.json'
        model.write_text(json.dumps({'bindings': ev.bindings(bank, families)}))
        for name, binder, predict in [('parser', old_bind, parser.predict),
                                       ('grounded_parser', grounding.bind, grounding.predict)]:
            arm = ev.composed(bank, aliases, model, lambda query: predict(query, families))
            metrics = {'object': object_scores(binder, data['rows'], families), 'retention': ev.retention(arm, bank)}
            for split, batch in splits.items():
                metrics[split], records = ev.score(arm, batch['rows'], known, batch['value_offset'], owners)
                if split == 'new_names':
                    metrics['new_names_records'] = records
            report['methods'][name] = metrics
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
