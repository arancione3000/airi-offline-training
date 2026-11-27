"""Replay the frozen instruction frame candidate without changing the approved router."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import tempfile

from candidate import FrameRouter

HERE = Path(__file__).resolve().parent
JOINT = HERE.parent / 'joint-action-20261121'
sys.path.insert(0, str(JOINT))
import joint
import run
import speech_gate

ev = run.ev

FROZEN = {
    'candidate': (HERE / 'candidate.py', '277f8c31a0dc7583cd3bfba7744d663d82c4b00bc14fed0c2dfe2e536df3a2ff'),
    'reserved': (HERE / 'blind.json', 'e950e7f99d255bec4421f7191f36744df3829cc802fe19c24be1eced34315921'),
    'bank': (ev.BUNDLED, '4460b73bad38102bc9a72c3ff52238b2d1e82a5dad7dd883a0f47e49de4c4614'),
    'aliases': (ev.APPROVED_ALIASES, 'e2efd400b66b2e2d4e13814d7e69253f59bcfe3e7599ed296fa1e93bc81cf57d'),
    'joint': (JOINT / 'joint.py', '8e0c35bb9a39d3fc81d3ab42af78528a3e003d215ade158edde85f52870f458d'),
    'speech_gate': (JOINT / 'speech_gate.py', '9622d0c99fe0f6876daae02362a46373cece1ab30e60cc6f1e5af6d5f071c84c'),
    'prior_first': (JOINT / 'prospective.json', 'aa808942ff23461fa4c58feba2398a117866cfa97d7ba00497dbeda773e0e63a'),
    'prior_confirmation': (JOINT / 'confirmation.json', '3bca9c68baf0347dfa6d5c217c53341013289f95bf4eb2fb42749aa3edbae2b3'),
}


def verify():
    for name, (path, expected) in FROZEN.items():
        if ev.file_hash(path) != expected:
            raise ValueError(f'{name} differs from the frozen readout input')


def leave_family_out(families, aliases, router):
    alternatives = {
        family['id']: FrameRouter(families, [(query, label) for query, label in aliases if label != family['id']])
        for family in families
    }

    def choose(query):
        parsed = router.parse(query)
        if parsed is not None:
            return alternatives[parsed[0]].predict(query)
        return None

    return choose


def score(arm, selector, data, by_id, owners, seen_words):
    summary, records = run.score(arm, selector, None, data['rows'], by_id, data['value_offset'], owners)
    del summary['timing']
    summary['literal_novel'] = ev.counts([row for row in records if row['query_sha256'] not in seen_words])
    return summary, records


def without_conflict_messages(respond, reviews):
    """An explicit request for clarification does not execute a procedure."""
    def respond_to(messages):
        result = respond(messages)
        if result is not None and result.get('source') == 'source_conflict':
            reviews.append(True)
            return None
        return result
    return respond_to


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path, help='new results directory')
    args = cli.parse_args()
    if args.output.exists():
        cli.error('choose a new results directory')
    verify()
    if run.DEFAULT_MODEL.exists():
        raise ValueError('an additional approved semantic model is present')
    families = ev.load_split('FAMILIES')['families']
    by_id = {family['id']: family for family in families}
    bank = ev.SkillBank(ev.BUNDLED)
    aliases = ev.AliasRouter(bank, ev.APPROVED_ALIASES)
    entries = bank._read()[0]['entries']
    bindings = ev.bindings(bank, families)
    installed = {entry['id']: entry for entry in entries}
    if {binding['lesson_id'] for binding in bindings} != by_id.keys():
        raise ValueError('installed program bindings differ from the family inventory')
    names = run.ev.read_aliases(ev.APPROVED_ALIASES)
    for binding in names.values():
        entry = installed.get(binding['program_id'])
        if entry is None or entry['lesson_id'] != binding['lesson_id']:
            raise ValueError('approved alias does not refer to an installed skill')
    examples = [(question, row['lesson_id']) for question, row in names.items() if row['lesson_id'] in by_id]
    owners = {}
    for entry in entries:
        owner, = [
            family['id'] for family in families
            if all(family[key] == entry['program'][key] for key in ('source', 'operation'))
        ]
        owners[entry['id']] = owner

    split_names = ('reserved', 'prior_first', 'prior_confirmation')
    batches = {name: json.loads(FROZEN[name][0].read_bytes()) for name in split_names}
    historical = run.historical()
    exposure = {name: [row['query'] for row in data['rows']]
                for name, data in {**historical, **{k: batches[k] for k in split_names[1:]}}.items()}
    exposure['train'] = [row['query'] for row in ev.load_split('TRAIN')['rows']]
    exposure['approved_aliases'] = list(names)
    exposure['bank_questions'] = [entry['program']['query'] for entry in entries]
    for key in ('pool', 'pool_confirmation'):
        exposure[key] = [row['query'] for row in run.runpy.run_path(str(ev.FRAMES / (key + '.py')))['POOL']]
    reserved = batches['reserved']['rows']
    if len(reserved) != 72 or sum(row['lesson_id'] is not None for row in reserved) != 36:
        raise ValueError('reserved case inventory differs from the protocol')
    batches['approved_alias_replay'] = {'value_offset': 50000, 'rows': [
        {'query': query, 'source_lesson_id': label, 'lesson_id': label,
         'condition': 'approved_alias', 'id': f'approved-alias-{index}'}
        for index, (query, label) in enumerate(examples)
    ]}
    overlaps = ev.audit_split(reserved, exposure, families)
    known_words = {ev.normalized_hash(query) for queries in exposure.values() for query in queries}
    router = FrameRouter(families, examples)
    methods = {
        'approved': None,
        'joint': lambda query: joint.predict(query, families),
        'speech_gate': lambda query: speech_gate.predict(query, families),
        'frame': router.predict,
        'frame_leave_family_out': leave_family_out(families, examples, router),
    }
    report = {
        'schema': 1,
        'sha256': {**{name: sha for name, (_, sha) in FROZEN.items()},
                   **{f'{name.lower()}_corpus': ev.file_hash(ev.CORPUS / (name + '.json.gz'))
                      for name in ('FAMILIES', 'TRAIN', 'QUALIFICATION')},
                   'driver': ev.file_hash(Path(__file__)),
                   'retention_contract': ev.file_hash(ev.RUNTIME / 'skills/stable_contracts.json.gz'),
                   'shared_evaluator': ev.file_hash(Path(ev.__file__)),
                   'shared_readout': ev.file_hash(Path(run.__file__)),
                   'runtime': {name: ev.file_hash(ev.RUNTIME / name) for name in (
                       'assistant.py', 'flexible_skills.py', 'exact_aliases.py', 'skill_programs.py',
                       'skill_bank.py', 'context_memory.py', 'query_learning.py', 'skill_validation.py',
                   )}},
        'verified_aliases': len(names),
        'approved_experts': len(entries),
        'aliases_in_frame_training': len(examples),
        'trained_frames': len(router.labels),
        'overlap_reserved': overlaps,
        'methods': {},
    }
    observations = {}
    with tempfile.TemporaryDirectory(prefix='airi-language-readout-') as folder:
        model = Path(folder) / 'bindings.json'
        run.write_json(model, {'bindings': bindings})
        for name, predict in methods.items():
            selector = run.Selector(predict) if predict else None
            reviews = []
            arm = without_conflict_messages(
                ev.composed(bank, aliases, model, selector.select if selector else None), reviews)
            metrics = report['methods'][name] = {'retention': ev.retention(arm, bank)}
            metrics['retention']['explicit_conflict_reviews'] = len(reviews)
            observations[name] = {}
            for split, data in batches.items():
                metrics[split], observations[name][split] = score(
                    arm, selector, data, by_id, owners, known_words if split == 'reserved' else set())
                if split != 'reserved':
                    del metrics[split]['literal_novel']
            if name != 'approved':
                metrics['paired_reserved_vs_approved'] = ev.paired(
                    observations['approved']['reserved'], observations[name]['reserved'])
    args.output.mkdir()
    encoded = json.dumps(observations, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    (args.output / 'observations.json.gz').write_bytes(gzip.compress(encoded, mtime=0))
    run.write_json(args.output / 'results.json', report)
    print(json.dumps({name: data['reserved']['wordings'] for name, data in report['methods'].items()}, indent=2))


if __name__ == '__main__':
    main()
