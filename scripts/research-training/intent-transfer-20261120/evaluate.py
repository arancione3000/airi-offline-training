"""Compare bounded procedure routing on disjoint Italian instructions and payloads."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import re
import runpy
import sys
import tempfile
import unicodedata

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
RUNTIME = RESEARCH.parent / 'assistant-runtime'
V6 = RESEARCH / 'intent-20261120'
FRAMES = RESEARCH / 'verified-frames-20261120'
CORPUS = RESEARCH / 'semantic-routing-20261116/evidence/precommitted'
sys.path[:0] = [str(HERE), str(RUNTIME), str(V6)]

from assistant import AiriAssistant
from exact_aliases import AliasRouter, DEFAULT as APPROVED_ALIASES, read_aliases
from flexible_skills import LearnedQueryRouter, _QuestionRoutes
from intent_grammar import predict as grammar
from learn import BUNDLED
from query_learning import bindings, case
from skill_bank import SkillBank
from skill_validation import StableGate


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_split(name):
    with gzip.open(CORPUS / (name + '.json.gz'), 'rt') as stream:
        return json.load(stream)


def union(*selectors):
    def select(query):
        choices = {choice for selector in selectors if (choice := selector(query)) is not None}
        return choices.pop() if len(choices) == 1 else None
    return select


class SelectionBackend:
    """Use the existing global query selection and validated typed execution path."""
    def __init__(self, select):
        self.select = select

    @staticmethod
    def read_model(path):
        return json.loads(path.read_bytes())

    def predict(self, model, query):
        lesson = self.select(query)
        return {'lesson_id': lesson} if lesson is not None else None


def build_selectors(families, learned, transformers):
    import parser as candidate
    v6 = lambda q: grammar(q, families)
    new = lambda q: candidate.predict(q, families)
    selectors = {'approved': None, 'v6': v6, 'parser': new, 'primary': union(v6, new)}
    if learned:
        from learner import fit
        for expanded, name in [(False, 'train'), (True, 'expanded')]:
            select = fit(families, expanded)
            selectors['learned_' + name] = select
            selectors['primary_plus_' + name] = union(v6, new, select)
    if transformers:
        from transformer_control import load
        for name in ('minilm_v4', 'minilm_v5'):
            predict = load(name)
            selectors[name] = lambda q, p=predict: p(q, families)
    return selectors


def composed(bank, aliases, model, select):
    router = None if select is None else LearnedQueryRouter(
        bank, model, backend=SelectionBackend(select), frozen=True)
    assistant = AiriAssistant(lambda messages: '', bank, _QuestionRoutes(aliases, router))

    def respond(messages):
        result = assistant.respond(messages)
        if 'component_error' in result:
            raise RuntimeError(result['component_error'])
        return None if result['source'] == 'native_model' else result
    return respond


def outcome(result, gold, family, messages, owners):
    if result is None:
        return ('positive_abstain' if gold is not None else 'negative_abstain'), None
    owner = owners.get(result.get('skill_id'))
    if gold is None:
        return 'negative_activation', owner
    source = messages[0]['content']
    provenance = result.get('provenance') or {}
    if (result['answer'] == gold and result.get('source') == 'learned_skill'
            and owner == family['id'] and result.get('operation') == family['operation']
            and provenance.get('message_index') == 0 and provenance.get('role') == 'user'
            and provenance.get('span') == source and provenance.get('start') == 0
            and provenance.get('end') == len(source)):
        return 'positive_correct', owner
    return 'positive_wrong', owner


def counts(records):
    keys = ('positive_correct', 'positive_abstain', 'positive_wrong', 'negative_abstain', 'negative_activation')
    by_wording = dict.fromkeys((*keys, 'mixed'), 0)
    by_execution = dict.fromkeys(keys, 0)
    for record in records:
        variants = record['outcomes']
        for result in variants:
            by_execution[result] += 1
        by_wording[variants[0] if len(set(variants)) == 1 else 'mixed'] += 1
    return {'wordings': by_wording, 'executions': by_execution}


def paired(before, after):
    if [r['id'] for r in before] != [r['id'] for r in after]:
        raise ValueError('mismatched paired ids')
    good = {'positive_correct', 'negative_abstain'}
    counts = Counter()
    for first, second in zip(before, after):
        pos = first['outcomes'][0].startswith('positive')
        old = all(result in good for result in first['outcomes'])
        new = all(result in good for result in second['outcomes'])
        change = 'gained' if new and not old else 'lost' if old and not new else 'unchanged'
        counts[('positive_' if pos else 'negative_') + change] += 1
    return {prefix + change: counts[prefix + change] for prefix in ('positive_', 'negative_')
            for change in ('gained', 'lost', 'unchanged')}


def score(arm, rows, families, offset, owners):
    records = []
    for i, row in enumerate(rows):
        family = families[row['source_lesson_id']]
        if row['lesson_id'] not in (None, family['id']):
            raise ValueError('unsupported target/source protocol')
        outcomes, predictions = [], []
        for variant in (0, 1):
            messages, answer = case(family, row['query'], offset + i, variant)
            gold = answer if row['lesson_id'] is not None else None
            result, owner = outcome(arm(messages), gold, family, messages, owners)
            outcomes.append(result)
            predictions.append(owner)
        records.append({'id': row['id'], 'operation': family['operation'], 'lesson_id': row['lesson_id'],
                        'source_lesson_id': family['id'], 'normalized_query': normalized_hash(row['query']),
                        'predictions': predictions, 'outcomes': outcomes})
    by_op = defaultdict(list)
    for record in records:
        by_op[record['operation']].append(record)
    return {**counts(records), 'by_operation': {op: counts(group) for op, group in sorted(by_op.items())}}, records


def retention(arm, bank):
    gate = StableGate()
    gate(arm)
    contracts = [r for entry in bank._read()[0]['entries'] for r in entry['train'] + entry['qualification']]
    failures = []
    for i, row in enumerate(contracts):
        result = arm(row['messages'])
        if (result['answer'] if result else None) != row['answer']:
            failures.append(i)
    return {'stored': len(contracts), 'stored_failed_indices': failures, 'historical': gate.last_report}


def normalize(query):
    text = unicodedata.normalize('NFC', query).casefold().replace('’', "'")
    return ' '.join(re.findall(r'[^\W_]+', text))


def normalized_hash(query):
    return hashlib.sha256(normalize(query).encode()).hexdigest()


def masked(query, families):
    text = ' ' + normalize(query) + ' '
    objects = {' ' + normalize(f['object_phrase']) + ' ' for f in families}
    matches = [o for o in objects if o in text]
    if len(matches) == 1 and text.count(matches[0]) == 1:
        return text.replace(matches[0], ' @ ').strip()
    return None


def audit_split(rows, exposure, family_list):
    queries = [row['query'] for row in rows]
    keys = {normalize(q) for q in queries}
    if len(keys) != len(rows) or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate ids or normalized questions within split')
    frames = {value for q in queries if (value := masked(q, family_list)) is not None}
    overlap = {}
    for name, texts in exposure.items():
        known = {normalize(q) for q in texts}
        known_frames = {value for q in texts if (value := masked(q, family_list)) is not None}
        overlap[name] = {'shared_normalized_questions': len(keys & known),
                         'shared_distinct_masked_frames': len(frames & known_frames)}
    return {'normalized_questions': len(keys), 'unique_masked_frames': len(frames), 'with': overlap}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--holdout', type=Path, required=True)
    cli.add_argument('--holdout-sha256', required=True)
    cli.add_argument('--output', type=Path, required=True, help='new directory')
    cli.add_argument('--bank', type=Path, default=BUNDLED)
    cli.add_argument('--aliases', type=Path, default=APPROVED_ALIASES)
    cli.add_argument('--learned', action='store_true')
    cli.add_argument('--transformers', action='store_true')
    cli.add_argument('--select-parts', action='store_true')
    args = cli.parse_args()
    if args.output.exists():
        cli.error('choose a new output directory')
    if file_hash(args.holdout) != args.holdout_sha256:
        cli.error('holdout digest differs from frozen reference')
    bank = SkillBank(args.bank)
    alias = AliasRouter(bank, args.aliases)
    family_list = load_split('FAMILIES')['families']
    families = {f['id']: f for f in family_list}
    train = load_split('TRAIN')['rows']
    valid_bindings = bindings(bank, family_list)
    if {b['lesson_id'] for b in valid_bindings} != families.keys():
        raise ValueError('family inventory and bank mismatch')
    selectors = build_selectors(family_list, args.learned, args.transformers)
    prior_files = {'v6_challenge': V6 / 'blind/cases.py', 'frames_challenge': FRAMES / 'holdout.py'}
    splits = {name: runpy.run_path(str(file))['dataset']() for name, file in prior_files.items()}
    qualification = load_split('QUALIFICATION')
    splits['qualification'] = {'rows': qualification['rows'], 'value_offset': 10000}
    state = bank._read()[0]
    owners = {}
    for entry in state['entries']:
        program = entry['program']
        matches = [f['id'] for f in family_list if
                   program['operation'] == f['operation'] and program['source'] == f['source']]
        if len(matches) != 1:
            raise ValueError('ambiguous program source and operation')
        owners[entry['id']] = matches[0]
    exposure = {'train': [r['query'] for r in train], 'aliases': list(read_aliases(args.aliases)),
                'bank_queries': [e['program']['query'] for e in state['entries']]}
    for filename in ('pool', 'pool_confirmation'):
        exposure[filename] = [r['query'] for r in runpy.run_path(str(FRAMES / (filename + '.py')))['POOL']]

    files = {'bank': args.bank, 'aliases': args.aliases,
             'v6': V6 / 'intent_grammar.py', 'parser': HERE / 'parser.py', 'harness': HERE / 'evaluate.py',
             'families': CORPUS / 'FAMILIES.json.gz', 'train': CORPUS / 'TRAIN.json.gz',
             'qualification': CORPUS / 'QUALIFICATION.json.gz', **prior_files}
    if args.learned:
        files.update(learner=HERE / 'learner.py', pool=FRAMES / 'pool.py', confirmation=FRAMES / 'pool_confirmation.py')
    if args.transformers:
        files['control'] = HERE / 'transformer_control.py'
    if args.select_parts:
        files['selector'] = HERE / 'select_parts.py'
    hashes = {k: file_hash(v) for k, v in files.items()}
    hashes['prospective'] = args.holdout_sha256
    report = {'sha256': hashes, 'aliases': len(read_aliases(args.aliases)), 'experts': len(state['entries']),
              'overlap': {}, 'methods': {}}
    for split, data in splits.items():
        report['overlap'][split] = audit_split(data['rows'], exposure, family_list)
        exposure[split] = [r['query'] for r in data['rows']]
    observations = {}

    with tempfile.TemporaryDirectory() as temporary:
        model = Path(temporary) / 'bindings.json'
        model.write_text(json.dumps({'bindings': valid_bindings}))
        arms = {}
        def historical(name, selector):
            arm = arms[name] = composed(bank, alias, model, selector)
            metrics = report['methods'][name] = {'retention': retention(arm, bank)}
            if hasattr(selector, 'calibration'):
                metrics['train_calibration'] = selector.calibration
            observations[name] = {}
            for split, data in splits.items():
                metrics[split], observations[name][split] = score(
                    arm, data['rows'], families, data['value_offset'], owners)

        for name, selector in selectors.items():
            historical(name, selector)

        plan = None
        if args.select_parts:
            import select_parts
            sources = ('approved', 'v6', 'parser', 'learned_expanded', 'minilm_v4', 'minilm_v5')
            eligible = {}
            for name in sources:
                if name in arms:
                    retained = report['methods'][name]['retention']
                    if not retained['stored_failed_indices'] and retained['historical']['passed']:
                        eligible[name] = [r for split in prior_files for r in observations[name][split]]
            plan = select_parts.select(eligible, family_list)
            callbacks = {name: (lambda q, families, p=selectors[name]: p(q) if p else None) for name in eligible}
            historical('selected_parts', lambda q: select_parts.predict(q, family_list, plan, callbacks))
            winner = plan['global_winner']
            # The single selected teacher is already evaluated on all historical sets.
            arms['selected_global'] = arms[winner]
            report['methods']['selected_global'] = deepcopy(report['methods'][winner])
            observations['selected_global'] = dict(observations[winner])

        args.output.mkdir(parents=True, exist_ok=False)
        freeze = {'sha256': hashes, 'selection': plan, 'arms': list(arms)}
        freeze_file = args.output / 'freeze.json'
        freeze_file.write_text(json.dumps(freeze, sort_keys=True, indent=2) + '\n')
        report['freeze_sha256'] = file_hash(freeze_file)
        # The prospective file has only been hashed up to this point.
        raw_holdout = args.holdout.read_bytes()
        if hashlib.sha256(raw_holdout).hexdigest() != args.holdout_sha256:
            raise ValueError('holdout changed after freeze')
        prospective = json.loads(raw_holdout)
        report['overlap']['prospective'] = audit_split(prospective['rows'], exposure, family_list)
        seen = {normalized_hash(q) for texts in exposure.values() for q in texts}
        for name, arm in arms.items():
            result, rows = score(arm, prospective['rows'], families, prospective['value_offset'], owners)
            result['literal_novel'] = counts([row for row in rows if row['normalized_query'] not in seen])
            report['methods'][name]['prospective'] = result
            observations[name]['prospective'] = rows

    for name, arm_data in observations.items():
        others = ['v6']
        if name.startswith('primary_plus_') or name == 'selected_parts':
            others.append('primary')
        if name == 'selected_parts':
            others.append('selected_global')
        if name == 'learned_expanded':
            others.append('learned_train')
        for split, records in arm_data.items():
            for other in others:
                if name != other:
                    report['methods'][name][split]['vs_' + other] = paired(observations[other][split], records)
    raw = json.dumps(observations, ensure_ascii=False, sort_keys=True).encode()
    (args.output / 'observations.json.gz').write_bytes(gzip.compress(raw, mtime=0))
    (args.output / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({arm: data['prospective']['wordings'] for arm, data in report['methods'].items()}))


if __name__ == '__main__':
    main()
