"""Confirm one frozen speech gate on independently authored requests through chat."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

import run
import joint
import speech_gate


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--holdout', type=Path, required=True)
    cli.add_argument('--holdout-sha256', required=True)
    cli.add_argument('--gate-sha256', required=True)
    cli.add_argument('--aliases', type=Path, default=run.ev.APPROVED_ALIASES)
    cli.add_argument('--bank', type=Path, default=run.ev.BUNDLED)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    sha = run.ev.file_hash
    if (sha(args.holdout) != args.holdout_sha256 or sha(Path(speech_gate.__file__)) != args.gate_sha256
            or sha(Path(joint.__file__)) != '8e0c35bb9a39d3fc81d3ab42af78528a3e003d215ade158edde85f52870f458d'):
        raise ValueError('incorrect candidate or corpus fingerprint')
    families = run.ev.load_split('FAMILIES')['families']
    known = {f['id']: f for f in families}
    bank = run.ev.SkillBank(args.bank)
    alias = run.ev.AliasRouter(bank, args.aliases)
    owners = {}
    for e in bank._read()[0]['entries']:
        owner, = [f['id'] for f in families if all(f[k] == e['program'][k] for k in ('source', 'operation'))]
        owners[e['id']] = owner
    historical = run.historical()
    historical['first_joint'] = json.loads((run.HERE / 'prospective.json').read_bytes())
    exposure = {name: [r['query'] for r in data['rows']] for name, data in historical.items()}
    exposure['train'] = [r['query'] for r in run.ev.load_split('TRAIN')['rows']]
    exposure['aliases'] = list(run.ev.read_aliases(args.aliases))
    for name in ('pool', 'pool_confirmation'):
        exposure[name] = [row['query'] for row in run.runpy.run_path(str(run.ev.FRAMES / (name + '.py')))['POOL']]
    args.output.mkdir(parents=True, exist_ok=False)
    files = {'gate': Path(speech_gate.__file__), 'joint': Path(joint.__file__), 'bank': args.bank,
             'aliases': args.aliases, 'holdout': args.holdout, 'previous': run.HERE / 'prospective.json',
             'driver': Path(__file__), 'shared_driver': Path(run.__file__)}
    run.write_json(args.output / 'freeze.json', {k: sha(v) for k,v in files.items()})
    raw = args.holdout.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.holdout_sha256:
        raise ValueError('holdout changed')
    data = json.loads(raw)
    if len(data['rows']) != 90 or sum(r['lesson_id'] is not None for r in data['rows']) != 36:
        raise ValueError('expected 36 valid commands and 54 negatives')
    report = {'overlap': run.ev.audit_split(data['rows'], exposure, families), 'methods': {}}
    logs = {}
    with tempfile.TemporaryDirectory() as temp:
        model = Path(temp) / 'bindings.json'
        run.write_json(model, {'bindings': run.ev.bindings(bank, families)})
        for name, candidate in [('joint', joint), ('gated', speech_gate)]:
            selector = run.Selector(lambda q: candidate.predict(q, families))
            arm = run.ev.composed(bank, alias, model, selector.select)
            metrics = {'retention': run.ev.retention(arm, bank)}
            logs[name] = {}
            for split, batch in {**historical, 'confirmation': data}.items():
                metrics[split], logs[name][split] = run.score(arm, selector, None, batch['rows'], known,
                                                              batch['value_offset'], owners)
            report['methods'][name] = metrics
        report['paired_confirmation'] = run.ev.paired(logs['joint']['confirmation'], logs['gated']['confirmation'])
    encoded = json.dumps(logs, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    (args.output / 'observations.json.gz').write_bytes(gzip.compress(encoded, mtime=0))
    run.write_json(args.output / 'results.json', report)
    print(json.dumps({k: v['confirmation']['wordings'] for k, v in report['methods'].items()}))


if __name__ == '__main__':
    main()
