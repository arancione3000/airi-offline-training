"""Exercise newly transferred literal questions through the chat API on independent values."""
import argparse
import gzip
import hashlib
import json
import operator
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parents[1] / 'assistant-runtime'
sys.path.insert(0, str(RUNTIME))
from assistant import AiriAssistant
from exact_aliases import AliasRouter, read_aliases
from flexible_skills import _QuestionRoutes, _semantic_router
from learn import BUNDLED
from skill_bank import SkillBank

ARITHMETIC = {'add': operator.add, 'subtract': operator.sub, 'multiply': operator.mul}
TEXT = {'copy': str, 'uppercase': str.upper, 'lowercase': str.lower, 'reverse': lambda value: value[::-1]}


def examples(family):
    source, op = family['source'], family['operation']
    if op in ARITHMETIC:
        values = [(f"{a}{source['separator']}{b}", str(ARITHMETIC[op](a, b)))
                  for a, b in ((-23, 41), (0, -8), (71, 71), (23, -41), (-7, -51), (1_000_000, -1))]
    else:
        values = [(s, TEXT[op](s)) for s in ('ÉrA-72', 'iĞne', 'Faß-31', 'PèR-4', 'τΣ-83', 'ÄLM-2')]
    return [(source['prefix'] + v + source['suffix'], gold) for v, gold in values]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('before', type=Path)
    cli.add_argument('after', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--bank', type=Path, default=BUNDLED)
    args = cli.parse_args()
    old, new = read_aliases(args.before), read_aliases(args.after)
    if any(new.get(q) != b for q, b in old.items()):
        raise ValueError('previously registered binding lost')
    previous = HERE.parent / 'verified-frames-20261120'
    donor = {r['query']: r for name in ('pool', 'pool_confirmation')
             for r in runpy.run_path(str(previous / (name + '.py')))['POOL'] if r['intent'] is not None}
    with gzip.open(HERE.parent / 'semantic-routing-20261116/evidence/precommitted/FAMILIES.json.gz') as f:
        families = {f['id']: f for f in json.load(f)['families']}
    bank = SkillBank(args.bank)
    assistants = [AiriAssistant(lambda _: '', bank, _QuestionRoutes(AliasRouter(bank, path), _semantic_router(bank)))
                  for path in (args.before, args.after)]
    metrics = {'new_donor_wordings': 0, 'positive_executions': 0, 'before_correct': 0, 'after_correct': 0,
               'positive_incorrect': [0, 0], 'negative_executions': 0, 'false_activations': [0, 0], 'samples': []}
    for query, binding in sorted(new.items()):
        if query in old or query not in donor:
            continue
        f = families[binding['lesson_id']]
        if donor[query]['family'] != f['id'] or donor[query]['intent'] != f['operation']:
            raise ValueError('donor binding disagrees with adjudicated operation')
        metrics['new_donor_wordings'] += 1
        question = {'role': 'user', 'content': query}
        values = examples(f)
        for i, (source, expected) in enumerate(values):
            message = {'role': 'user', 'content': source}
            responses = [assistant.respond([message, question]) for assistant in assistants]
            for j, (name, r) in enumerate(zip(('before_correct', 'after_correct'), responses, strict=True)):
                if 'component_error' in r:
                    raise ValueError(r['component_error'])
                if r['source'] == 'native_model':
                    continue
                if (r['source'] == 'learned_skill' and r['answer'] == expected and r['skill_id'] == binding['program_id']
                        and r['operation'] == f['operation'] and r['provenance']['span'] == source):
                    metrics[name] += 1
                else:
                    metrics['positive_incorrect'][j] += 1
            if i == 0:
                metrics['samples'].append({'query': query, 'source': source, 'expected': expected,
                                           'before': responses[0], 'after': responses[1]})
            metrics['positive_executions'] += 1
        other = next(obj for obj in families.values() if obj['id'] != f['id'])
        first = {'role': 'user', 'content': values[0][0]}
        competing = {'role': 'user', 'content': examples(other)[0][0]}
        for history in ([question], [competing, question], [first, competing, question],
                        [{**first, 'role': 'assistant'}, question], [first, {'role': 'user', 'content': values[1][0]}, question]):
            for j, assistant in enumerate(assistants):
                answer = assistant.respond(history)
                if 'component_error' in answer:
                    raise ValueError(answer['component_error'])
                metrics['false_activations'][j] += answer['source'] != 'native_model'
            metrics['negative_executions'] += 1
    metrics.update(before_aliases=len(old), after_aliases=len(new),
                   snapshots_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.before, args.after)])
    with args.output.open('x') as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    if (not metrics['new_donor_wordings'] or metrics['after_correct'] != metrics['positive_executions']
            or any(metrics['positive_incorrect'] + metrics['false_activations'])):
        raise SystemExit('donated question replay failed')


if __name__ == '__main__':
    main()
