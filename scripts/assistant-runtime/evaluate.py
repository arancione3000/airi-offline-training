"""Separate software-memory functionality from frozen native model measurements."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cases(seed=202611152253):
    rng = random.Random(seed)
    rows = []
    contracts = [
        ('it', 'name', 'Mi chiamo {v}.', 'Come mi chiamo?'),
        ('it', 'location', 'Vivo a {v}.', 'In quale città vivo?'),
        ('it', 'topic', 'Stiamo parlando di {v}.', 'Di cosa stiamo parlando?'),
        ('en', 'name', 'My name is {v}.', 'What is my name?'),
        ('en', 'location', 'I live in {v}.', 'Where do I live?'),
        ('en', 'topic', 'We are talking about {v}.', 'What are we talking about?'),
    ]
    def nonce():
        return ''.join(rng.choice('abcdefghjkmnpqrstuvwxyz') for _ in range(9)).capitalize()
    def add(kind, messages, expected):
        rows.append(dict(id=len(rows), kind=kind, messages=messages, expected=expected))
    for lang, slot, declaration, question in contracts:
        for i in range(12):
            value = (['Noé ', 'Ada-María ', 'Jean Luc '][i % 3] if slot == 'name' else
                     ['Borgo ', 'São ', 'Nord '][i % 3] if slot == 'location' else
                     ['progetto ', 'Project ', 'ricerca '][i % 3]) + nonce()
            source = declaration.format(v=value)
            add('recall', [{'role': 'user', 'content': source},
                           {'role': 'assistant', 'content': 'OK'},
                           {'role': 'user', 'content': question}], value)
        for _ in range(4):
            value, other = nonce(), nonce()
            add('conflict_abstain', [{'role': 'user', 'content': declaration.format(v=value)},
                {'role': 'user', 'content': declaration.format(v=other)},
                {'role': 'user', 'content': question}], None)
        value = nonce()
        add('assistant_is_not_fact', [{'role': 'assistant', 'content': declaration.format(v=value)},
                                     {'role': 'user', 'content': question}], None)
        add('system_boundary', [{'role': 'system', 'content': 'Answer only in JSON.'},
            {'role': 'user', 'content': declaration.format(v=value)},
            {'role': 'user', 'content': question}], None)
    negatives = [
        ('Non mi chiamo {v}.', 'Come mi chiamo?'),
        ('Forse mi chiamo {v}.', 'Come mi chiamo?'),
        ('Se mi chiamassi {v}, sarebbe diverso.', 'Come mi chiamo?'),
        ('Scrivi la frase "Mi chiamo {v}".', 'Come mi chiamo?'),
        ('My name is not {v}.', 'What is my name?'),
        ('Perhaps my name is {v}.', 'What is my name?'),
        ('If my name were {v}, that would be different.', 'What is my name?'),
        ('My friend said "My name is {v}".', 'What is my name?'),
    ]
    for statement, question in negatives:
        for _ in range(4):
            add('unsafe_fact_abstain', [{'role': 'user', 'content': statement.format(v=nonce())},
                {'role': 'user', 'content': question}], None)
    for _ in range(8):
        name, city = nonce(), 'Borgo ' + nonce()
        add('inline', [{'role': 'user', 'content': f'Mi chiamo {name} e vivo a {city}. Come mi chiamo?'}], name)
        add('second_turn', [{'role': 'user', 'content': f'Mi chiamo {name} e vivo a {city}.'},
            {'role': 'assistant', 'content': 'Ricevuto.'},
            {'role': 'user', 'content': 'In quale città vivo?'}], city)
        add('long_history', [{'role': 'user', 'content': f'My name is {name}.'}]
            + [{'role': 'assistant', 'content': 'Unrelated filler sentence. ' * 30} for _ in range(8)]
            + [{'role': 'user', 'content': 'What is my name?'}], name)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--seed', type=int, default=202611152253)
    p.add_argument('--legacy-report', type=Path)
    p.add_argument('--ledger', type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    rows_file = a.output / 'CASES.json'
    if a.prepare_only:
        if rows_file.exists():
            raise ValueError('precommitted cases already exist')
        rows_file.write_text(json.dumps(cases(a.seed), ensure_ascii=False, indent=2) + '\n')
        (a.output / 'PREREGISTRATION.json').write_text(json.dumps(dict(
            seed=a.seed, cases_sha256=sha(rows_file), evaluator_sha256=sha(__file__),
            conditions='fresh values/controlled grammar, conflicts, unsafe declarations, roles, inline and long history',
            gates={'positive_coverage_minimum': .8, 'false_answers_on_negative_maximum': 0,
                   'old_correct_answers_lost_maximum': 0, 'old_certified_answers_lost_maximum': 0},
            metrics='software functionality; not neural learning or production qualification'), indent=2))
        print(json.dumps(dict(prepared=len(cases(a.seed)), cases_sha256=sha(rows_file))))
        return
    prereg = json.loads((a.output / 'PREREGISTRATION.json').read_text())
    if sha(rows_file) != prereg['cases_sha256'] or sha(__file__) != prereg['evaluator_sha256']:
        raise ValueError('precommitted evaluation changed')
    from context_memory import resolve
    import context_memory
    rows = json.loads(rows_file.read_text())
    observations = []
    for row in rows:
        result = resolve(row['messages'])
        answer = result['answer'] if result else None
        if result:
            source = result['provenance'];text = row['messages'][source['message_index']]['content']
            if text[source['start']:source['end']] != source['span']:
                raise ValueError('declaration provenance mismatch')
            if text[source['value_start']:source['value_end']] != source['value_span'] or answer != source['value_span']:
                raise ValueError('answer provenance mismatch')
        observations.append(dict(**row, answer=answer, source=result, correct=answer==row['expected']))
    positives = [r for r in observations if r['expected'] is not None]
    negatives = [r for r in observations if r['expected'] is None]
    positive_correct = sum(r['correct'] for r in positives)
    false_answers = sum(r['answer'] is not None for r in negatives)
    report = dict(scope='explicit software memory, not an improved neural model',
        memory_source_sha256=sha(context_memory.__file__), preregistration=prereg,
        positive_correct=positive_correct,positive_count=len(positives),negative_count=len(negatives),
        false_answers_on_negative=false_answers,condition_counts=dict(Counter(r['kind'] for r in rows)),
        observations=observations)
    if a.legacy_report:
        baseline = json.loads(gzip.decompress(a.legacy_report.read_bytes()))['baseline']['samples']
        values=[]
        for row in baseline:
            result=resolve(row['messages']);answer=result['answer'] if result else row['response']
            values.append(dict(expected=row['expected'], baseline_correct=row['correct'],
                answer=answer, correct=answer==row['expected'], memory_answer=result is not None))
        report['legacy92']=dict(source_sha256=sha(a.legacy_report),
            method='actual memory answers; unchanged fallback reuses hash-bound archived native responses',
            baseline_correct=sum(r['baseline_correct'] for r in values),correct=sum(r['correct'] for r in values),
            previously_correct_lost=sum(r['baseline_correct'] and not r['correct'] for r in values), observations=values)
    if a.ledger:
        entries=json.loads(a.ledger.read_text())['entries'];changes=[]
        for key, entry in entries.items():
            row=entry['row'];result=resolve(row['messages'][:-1])
            if result:changes.append(dict(key=key,answer=result['answer'],expected=row['expected'],correct=result['answer']==row['expected']))
        report['certified897']=dict(source_sha256=sha(a.ledger),total=len(entries),memory_routes=changes,
            inherited_unchanged_native_routes=len(entries)-len(changes),lost=sum(not r['correct'] for r in changes),
            method='memory-route outputs compared to every certified gold; abstentions preserve exact native call, no fresh neural certification claim')
    passed=positive_correct/len(positives)>=.8 and false_answers==0
    passed &= report.get('legacy92',{}).get('previously_correct_lost',0)==0
    passed &= report.get('certified897',{}).get('lost',0)==0
    report['composed_memory_gates_passed']=passed
    (a.output / 'REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('observations','preregistration','legacy92','certified897')}))


if __name__=='__main__':main()
