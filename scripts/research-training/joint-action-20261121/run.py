"""Score global instruction-only routing through the approved, source-bound chat path."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import runpy
import statistics
import sys
import tempfile
from time import perf_counter

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'intent-transfer-20261120'
sys.path.insert(0, str(PRIOR))
import evaluate as ev
import grounding
from flexible_skills import DEFAULT_MODEL


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def historical():
    result = {name: json.loads((PRIOR / file).read_bytes()) for name, file in
              [('previous_requests', 'prospective.json'), ('alternative_names', 'ground_holdout.json')]}
    for name, path in [('v6_challenge', ev.V6 / 'blind/cases.py'), ('frames_challenge', ev.FRAMES / 'holdout.py')]:
        result[name] = runpy.run_path(str(path))['dataset']()
    result['qualification'] = {'rows': ev.load_split('QUALIFICATION')['rows'], 'value_offset': 10000}
    return result


class Selector:
    def __init__(self, method):
        self.method = method
        self.seconds = []
        self.select = lru_cache(maxsize=None)(self._predict)

    def _predict(self, query):
        start = perf_counter()
        result = self.method(query)
        self.seconds.append(perf_counter() - start)
        return result


def make_method(name, families):
    if name == 'approved':
        return None, None
    if name == 'grounded_parser':
        return Selector(lambda q: grounding.predict(q, families)), lambda q: {'object_id': grounding.bind(q, families)}
    if name == 'joint':
        import joint
        return Selector(lambda q: joint.predict(q, families)), lambda q: {
            'object_id': joint.bind(q, families), 'operation': joint.action(q)}
    import teacher
    method = teacher.build(families)
    return Selector(method), getattr(method, 'trace', None)


def score(arm, selector, trace, rows, families, offset, owners):
    records = []
    grouped = defaultdict(list)
    by_operation = defaultdict(list)
    stages = Counter()
    latencies = []
    index = len(selector.seconds) if selector else 0
    started = perf_counter()
    for i, row in enumerate(rows):
        query = row['query']
        family = families[row['source_lesson_id']]
        target = row['lesson_id']
        if target not in (None, family['id']):
            raise ValueError('unsupported target/source contract')
        record = {'id': row['id'], 'query_sha256': ev.normalized_hash(query),
                  'source_lesson_id': family['id'], 'lesson_id': target,
                  'condition': row.get('condition', row.get('category', 'unspecified')),
                  'answers': [], 'outcomes': []}
        for variant in (0, 1):
            messages, answer = ev.case(family, query, offset + i, variant)
            gold = answer if target is not None else None
            start = perf_counter()
            result = arm(messages)
            latencies.append(perf_counter() - start)
            outcome, owner = ev.outcome(result, gold, family, messages, owners)
            record['outcomes'].append(outcome)
            record['answers'].append({'expected': gold, 'answer': result['answer'] if result else None,
                                      'skill_id': result.get('skill_id') if result else None,
                                      'family': owner, 'provenance': result.get('provenance') if result else None})
        # The explicit query-only probe also measures the unguarded selector on requests
        # the approved bank already handled. It never receives the source or gold label.
        if selector:
            record['proposal'] = selector.select(query)
        if trace:
            record['trace'] = details = trace(query)
            if 'expected_object' in row and 'object_id' in details:
                obj = row['expected_object']
                got = details['object_id']
                stages['object_correct' if obj is not None and obj == got else
                       'object_correct_abstain' if obj is None and got is None else
                       'object_missed' if got is None else 'object_wrong'] += 1
            if target is not None and 'operation' in details:
                stages['positive_action_correct' if details['operation'] == family['operation'] else
                       'positive_action_missed' if details['operation'] is None else 'positive_action_wrong'] += 1
        records.append(record)
        grouped[record['condition']].append(record)
        by_operation[family['operation']].append(record)
    summary = ev.counts(records)
    summary['by_condition'] = {key: ev.counts(group)['wordings'] for key, group in grouped.items()}
    summary['by_operation'] = {key: ev.counts(group)['wordings'] for key, group in by_operation.items()}
    summary['stages'] = stages
    intervals = selector.seconds[index:] if selector else []
    summary['timing'] = {'split_wall_seconds_including_diagnostics': perf_counter() - started,
                         'uncached_query_decisions': len(intervals), 'query_decision_seconds': sum(intervals),
                         'median_uncached_query_ms': 1000 * statistics.median(intervals) if intervals else None,
                         'median_first_value_chat_ms': 1000 * statistics.median(latencies[::2]),
                         'median_repeat_value_chat_ms': 1000 * statistics.median(latencies[1::2])}
    return summary, records


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--holdout', type=Path)
    cli.add_argument('--holdout-sha256')
    cli.add_argument('--joint-sha256')
    cli.add_argument('--teacher-sha256')
    cli.add_argument('--teacher', action='store_true', help='run only the pinned instruction-model arm')
    cli.add_argument('--fresh-only', action='store_true', help='score only the new split and retention')
    cli.add_argument('--smoke', action='store_true', help='historical subset only, with published controls')
    cli.add_argument('--bank', type=Path, default=ev.BUNDLED)
    cli.add_argument('--aliases', type=Path, default=ev.APPROVED_ALIASES)
    args = cli.parse_args()
    if args.output.exists() or (args.smoke and (args.holdout or args.fresh_only)):
        cli.error('choose a new output directory; smoke uses historical data only')
    if DEFAULT_MODEL.exists():
        cli.error('approved semantic router present; this experiment assumes the exact bank and approved aliases')
    methods = ['approved', 'grounded_parser']
    files = {'harness': Path(__file__), 'bank': args.bank, 'aliases': args.aliases,
             'grounding': Path(grounding.__file__), 'parser': PRIOR / 'parser.py', 'previous_evaluator': PRIOR / 'evaluate.py',
             'train': ev.CORPUS / 'TRAIN.json.gz', 'families': ev.CORPUS / 'FAMILIES.json.gz',
             'qualification': ev.CORPUS / 'QUALIFICATION.json.gz', 'previous_requests': PRIOR / 'prospective.json',
             'alternative_names': PRIOR / 'ground_holdout.json', 'v6_challenge': ev.V6 / 'blind/cases.py',
             'frames_challenge': ev.FRAMES / 'holdout.py'}
    for filename in ('assistant.py', 'flexible_skills.py', 'exact_aliases.py', 'skill_programs.py',
                     'skill_bank.py', 'query_learning.py', 'skill_validation.py', 'context_memory.py'):
        files[filename] = ev.RUNTIME / filename
    files['stable_contracts'] = ev.RUNTIME / 'skills/stable_contracts.json.gz'
    if not args.smoke and not (args.holdout and args.holdout_sha256):
        cli.error('the prospective run requires the sealed corpus and digest')
    if args.teacher or args.joint_sha256 or not args.smoke:
        name, digest = ('teacher', args.teacher_sha256) if args.teacher else ('joint', args.joint_sha256)
        files[name] = HERE / (name + '.py')
        if not digest or ev.file_hash(files[name]) != digest:
            cli.error(f'provide the frozen {name} digest')
        methods = [name] if args.teacher else [*methods, name]
    families = ev.load_split('FAMILIES')['families']
    by_id = {family['id']: family for family in families}
    bank = ev.SkillBank(args.bank)
    aliases = ev.AliasRouter(bank, args.aliases)
    entries = bank._read()[0]['entries']
    binding = ev.bindings(bank, families)
    if {row['lesson_id'] for row in binding} != by_id.keys():
        raise ValueError('missing family programs in bank')
    owners = {}
    for entry in entries:
        program = entry['program']
        owner, = [f['id'] for f in families if f['operation'] == program['operation'] and f['source'] == program['source']]
        owners[entry['id']] = owner

    splits = historical()
    exposure = {k: [r['query'] for r in v['rows']] for k, v in splits.items()}
    exposure.update(train=[r['query'] for r in ev.load_split('TRAIN')['rows']],
                    aliases=list(ev.read_aliases(args.aliases)), bank=[e['program']['query'] for e in entries])
    for name in ('pool', 'pool_confirmation'):
        exposure[name] = [row['query'] for row in runpy.run_path(str(ev.FRAMES / (name + '.py')))['POOL']]
    args.output.mkdir(parents=True)
    freeze = {'sha256': {k: ev.file_hash(v) for k, v in files.items()}, 'methods': methods,
              'prospective_sha256': args.holdout_sha256, 'smoke': args.smoke,
              'experts': len(entries), 'aliases': len(exposure['aliases'])}
    write_json(args.output / 'freeze.json', freeze)
    report = {'freeze_sha256': ev.file_hash(args.output / 'freeze.json'), 'methods': {}}
    if args.smoke:
        data = splits['alternative_names']
        splits = {'smoke': {'rows': data['rows'][:4], 'value_offset': data['value_offset']}}
    else:
        raw = args.holdout.read_bytes()
        if hashlib.sha256(raw).hexdigest() != args.holdout_sha256:
            raise ValueError('prospective corpus digest changed')
        new = json.loads(raw)
        positives = sum(row['lesson_id'] is not None for row in new['rows'])
        if len(new['rows']) != 108 or positives != 54:
            raise ValueError('prospective corpus requires 54 supported and 54 unsupported wordings')
        report['overlap'] = ev.audit_split(new['rows'], exposure, families)
        splits = {'prospective': new} if args.fresh_only else {**splits, 'prospective': new}
    observations = {}
    known_questions = {ev.normalized_hash(q) for texts in exposure.values() for q in texts}
    with tempfile.TemporaryDirectory() as temp:
        model = Path(temp) / 'bindings.json'
        write_json(model, {'bindings': binding})
        for name in methods:
            loading = perf_counter()
            selector, trace = make_method(name, families)
            metrics = {'initialization_seconds': perf_counter() - loading}
            arm = ev.composed(bank, aliases, model, selector.select if selector else None)
            metrics['retention'] = ev.retention(arm, bank)
            observations[name] = {}
            for key, batch in splits.items():
                summary, rows = score(arm, selector, trace, batch['rows'], by_id, batch['value_offset'], owners)
                if key == 'prospective':
                    summary['literal_novel'] = ev.counts([r for r in rows if r['query_sha256'] not in known_questions])
                if name == 'joint':
                    summary['vs_grounded_parser'] = ev.paired(observations['grounded_parser'][key], rows)
                metrics[key] = summary
                observations[name][key] = rows
            if selector and hasattr(selector.method, 'stats'):
                stats = selector.method.stats
                metrics['selector_stats'] = stats() if callable(stats) else stats
            report['methods'][name] = metrics
    raw = json.dumps(observations, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    (args.output / 'observations.json.gz').write_bytes(gzip.compress(raw, mtime=0))
    write_json(args.output / 'results.json', report)
    target = 'smoke' if args.smoke else 'prospective'
    print(json.dumps({name: result[target]['wordings'] for name, result in report['methods'].items()}))


if __name__ == '__main__':
    main()
