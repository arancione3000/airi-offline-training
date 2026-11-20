"""Replay the predeclared NLI comparison through the frozen common evaluator."""
import argparse
import gzip
import json
from pathlib import Path
import runpy
import tempfile

import evaluate as evaluation
import nli


REFERENCE_SHA256 = 'ddabfc55c5db0c045ba3a4ac7c49a5c877ccc88b836f6163052c43cf87241155'
ORIGINAL_SHA256 = 'e733d7c2e62955aa94852adc03d0349c128cf4c268ae446179b1115b44ecc515'
CANDIDATE_SHA256 = 'ed0a7546ebfe4cfc21c97e6f7abe021ba0f1d09c6c36d739560f2fa5d2988375'


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reference', type=Path, required=True, help='freeze.json from the primary comparison')
    cli.add_argument('--holdout', type=Path, required=True)
    cli.add_argument('--bank', type=Path, default=evaluation.BUNDLED)
    cli.add_argument('--aliases', type=Path, default=evaluation.APPROVED_ALIASES)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if evaluation.file_hash(args.reference) != REFERENCE_SHA256:
        cli.error('reference differs from the primary comparison')
    expected = json.loads(args.reference.read_bytes())['sha256']
    expected['nli'] = CANDIDATE_SHA256
    paths = {
        'bank': args.bank, 'aliases': args.aliases, 'prospective': args.holdout,
        'harness': Path(evaluation.__file__), 'nli': Path(nli.__file__),
        'learner': evaluation.HERE / 'learner.py', 'v6': evaluation.V6 / 'intent_grammar.py',
        'train': nli.TRAIN_FILE, 'families': evaluation.CORPUS / 'FAMILIES.json.gz',
        'qualification': evaluation.CORPUS / 'QUALIFICATION.json.gz',
        'v6_challenge': evaluation.V6 / 'blind/cases.py',
        'frames_challenge': evaluation.FRAMES / 'holdout.py',
    }
    input_hashes = {name: evaluation.file_hash(path) for name, path in paths.items()}
    for name, digest in input_hashes.items():
        if digest != expected[name]:
            cli.error(f'input differs from frozen reference: {name}')
    args.output.mkdir(parents=True, exist_ok=False)
    bank = evaluation.SkillBank(args.bank)
    aliases = evaluation.AliasRouter(bank, args.aliases)
    families = evaluation.load_split('FAMILIES')['families']
    family_by_id = {f['id']: f for f in families}
    valid_bindings = evaluation.bindings(bank, families)
    if {b['lesson_id'] for b in valid_bindings} != family_by_id.keys():
        raise ValueError('family inventory mismatch')
    owners = {}
    for entry in bank._read()[0]['entries']:
        program = entry['program']
        owner, = [f['id'] for f in families if
                  f['operation'] == program['operation'] and f['source'] == program['source']]
        owners[entry['id']] = owner

    select = nli.fit(families)
    freeze = {'reference_sha256': REFERENCE_SHA256, 'inputs': input_hashes,
              'driver_sha256': evaluation.file_hash(Path(__file__)),
              'calibration_repair': {'original_sha256': ORIGINAL_SHA256, 'corrected_sha256': CANDIDATE_SHA256,
                  'reason': 'Original aborted before evaluation: three supported TRAIN frame/operation pairs '
                            'also received synthetic negative labels. Trusted labels now override synthetic alternatives.'},
              'runtime_sha256': {p.name: evaluation.file_hash(p) for p in (
                  evaluation.RUNTIME / 'assistant.py', evaluation.RUNTIME / 'flexible_skills.py',
                  evaluation.RUNTIME / 'skill_programs.py', evaluation.RUNTIME / 'skill_bank.py')},
              'train_calibration': select.calibration}
    write_json(args.output / 'freeze.json', freeze)
    splits = {name: runpy.run_path(str(paths[name]))['dataset']()
              for name in ('v6_challenge', 'frames_challenge')}
    splits['qualification'] = {'rows': evaluation.load_split('QUALIFICATION')['rows'], 'value_offset': 10000}
    if evaluation.file_hash(args.holdout) != input_hashes['prospective']:
        raise ValueError('prospective data changed after calibration')
    splits['prospective'] = json.loads(args.holdout.read_bytes())
    with tempfile.TemporaryDirectory() as temporary:
        model = Path(temporary) / 'bindings.json'
        model.write_text(json.dumps({'bindings': valid_bindings}))
        arm = evaluation.composed(bank, aliases, model, select)
        retained = evaluation.retention(arm, bank)
        report = {'freeze_sha256': evaluation.file_hash(args.output / 'freeze.json'), 'retention': retained}
        observations = {}
        for name, split in splits.items():
            report[name], observations[name] = evaluation.score(arm, split['rows'], family_by_id,
                                                               split['value_offset'], owners)
    raw = json.dumps(observations, ensure_ascii=False, sort_keys=True).encode()
    (args.output / 'observations.json.gz').write_bytes(gzip.compress(raw, mtime=0))
    write_json(args.output / 'results.json', report)
    print(json.dumps({name: report[name]['wordings'] for name in splits}))


if __name__ == '__main__':
    main()
