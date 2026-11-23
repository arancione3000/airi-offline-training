"""Evaluate an append-only Airi bank, a maintained parser, and a local Transformer."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from statistics import median
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

from prepare import FORMATS

HERE = Path(__file__).resolve().parent
ASSISTANT = HERE.parents[1] / 'assistant-runtime'
sys.path[:0] = [str(ASSISTANT), str(HERE.parent / 'joint-action-20261121')]
from assistant import AiriAssistant
from skill_bank import SkillBank
from skill_validation import StableGate

CORPUS_SHA256 = '0a0747e2e818e1cb7d9aff3b53032b33140f66851169deec404299fff170786d'
REVIEW = 'REVISIONE'
UNITS = re.compile(r'\b(?:unità|quantità)\b')
NET = re.compile(r'\b(?:saldo|netto|togliendo)\b')
OTHER = re.compile(r'\b(?:euro|valore|costo|somma|prezzo|totale lordo)\b')


def fixed_parser(messages):
    question = messages[-1]['content'].casefold()
    if not UNITS.search(question) or not NET.search(question) or OTHER.search(question):
        return None
    pairs = set()
    for message in messages[:-1]:
        if message['role'] != 'user':
            continue
        title, sep, body = message['content'].partition(':')
        if not sep:
            continue
        for expected, left, right in FORMATS:
            if title.strip().casefold() != expected.casefold():
                continue
            fields = re.fullmatch(r'\s*([a-z]+)\s+([0-9]{1,6})\s*;\s*([a-z]+)\s+([0-9]{1,6})(?:\s+unità)?\s*\.\s*',
                                  body.casefold())
            if fields:
                name1, value1, name2, value2 = fields.groups()
                quantities = {name1: int(value1), name2: int(value2)}
                if quantities.keys() == {left, right}:
                    pairs.add((quantities[left], quantities[right]))
    if len(pairs) != 1:
        return None
    left, right = pairs.pop()
    return str(left - right)


def airi(bank):
    assistant = AiriAssistant(lambda _: '', skill_bank=bank)

    def respond(messages):
        result = assistant.respond(messages)
        if 'component_error' in result:
            raise RuntimeError(result['component_error'])
        if result['source'] == 'source_conflict':
            return None
        return result['answer'] or None

    return respond


def render(messages):
    return '\n'.join(f'{message["role"]}: {message["content"]}' for message in messages)


def operands(messages):
    # TRAIN only. The synthetic reports have exactly two ASCII integer fields.
    return ','.join(re.findall(r'[0-9]+', messages[0]['content']))


def qwen_reference(corpus, teacher):
    instructions = ("""Per ogni conversazione individua le quantità spedite e stornate se l'utente chiede quante unità
rimangono al netto degli storni. Rispondi soltanto con i due interi nell'ordine spedite,stornate.
Scrivi REVISIONE se manca un dato, se due resoconti utente sono in disaccordo, se l'unica fonte
è un messaggio assistant, o se l'utente chiede un altro calcolo. Le righe user sono dell'utente.
""")
    examples = '\n\n'.join(render(case['messages']) + '\nRisposta: ' + operands(case['messages'])
                           for case in corpus['train'])
    prefix = '<|im_start|>system\n' + instructions + examples + '<|im_end|>\n<|im_start|>user\n'
    grammar = 'root ::= "REVISIONE" | number "," number\nnumber ::= [0-9]+\n'

    def respond(messages):
        prompt = prefix + render(messages) + '\n<|im_end|>\n<|im_start|>assistant\n'
        status, result = teacher._request('/completion', {
            'prompt': prompt, 'grammar': grammar, 'temperature': 0.0, 'repeat_penalty': 1.0,
            'n_predict': 16, 'seed': 7, 'cache_prompt': True,
        })
        if status != 200:
            raise RuntimeError(f'Qwen HTTP {status}: {result}')
        raw = result['content'].strip()
        if raw == REVIEW:
            return None
        if not re.fullmatch(r'[0-9]{1,8},[0-9]{1,8}', raw):
            return 'FORMATO_INCOMPLETO'
        one, two = map(int, raw.split(','))
        return str(one - two)

    return respond


def disposition(expected, actual):
    if expected is None:
        return 'correct_review' if actual is None else 'unsafe_automation'
    if actual is None:
        return 'missed_automation'
    return 'correct_automation' if actual == expected else 'wrong_balance'


def measure(respond, cases):
    predicted, timings = [], []
    totals, by_case = Counter(), {}
    for case in cases:
        start = perf_counter()
        answer = respond(case['messages'])
        timings.append((perf_counter() - start) * 1000)
        outcome = disposition(case['answer'], answer)
        predicted.append(answer)
        totals[outcome] += 1
        by_case.setdefault(case['kind'], Counter())[outcome] += 1
    return {
        'totals': dict(totals), 'by_case': by_case, 'predictions': predicted,
        'median_ms': round(median(timings), 6),
        'total_inference_ms': round(sum(timings), 6),
    }


def trial(corpus, use_qwen):
    scenarios = corpus['final']
    lessons = []
    with TemporaryDirectory(prefix='airi-retail-') as tmp:
        directory = Path(tmp)
        working = directory / 'working'
        working.mkdir()
        path = working / 'bank.json'
        gate = StableGate()
        for form in range(len(FORMATS)):
            lesson = {'lesson_id': f'retail-layout-{form}', 'trusted': True}
            for key, section in [('examples', 'train'), ('qualification', 'qualification')]:
                lesson[key] = [r for r in corpus[section] if r['format'] == form]
            start = perf_counter()
            event = SkillBank(path).learn(lesson, regression_gate=gate, expected_revision=form)
            lessons.append({
                'event': event, 'regression_gate': gate.last_report,
                'wall_ms': round((perf_counter() - start) * 1000, 3),
            })
            if event['action'] != 'activated':
                raise RuntimeError(f'Lesson {form}: {event}')
            if form == 0:
                shutil.copytree(working, directory / 'previous')
        first = measure(airi(SkillBank(directory / 'previous' / path.name)), scenarios)
        final = measure(airi(SkillBank(path)), scenarios)
        methods = {'airi_first': first, 'airi_two': final, 'script': measure(fixed_parser, scenarios)}
        if use_qwen:
            from teacher import Teacher
            model = Teacher([{'object_phrase': 'le quantità', 'id': 'retail', 'operation': 'subtract'}])
            try:
                methods['qwen'] = measure(qwen_reference(corpus, model), scenarios)
            finally:
                model.close()
        retention = Counter()
        for case, old, new in zip(scenarios, first['predictions'], final['predictions'], strict=True):
            if case['format'] != 0:
                continue
            old_ok = old == case['answer']
            new_ok = new == case['answer']
            retention['tested_first_format'] += 1
            retention['old_correct'] += old_ok
            retention['old_correct_lost'] += old_ok and not new_ok
            retention['old_incorrect_resolved'] += not old_ok and new_ok
    return {
        'corpus_sha256': CORPUS_SHA256, 'counts': {k: len(corpus[k]) for k in ('train', 'qualification', 'final')},
        'lessons': lessons, 'methods': methods, 'retention': dict(retention),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cases', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--qwen', action='store_true')
    args = parser.parse_args()
    data = args.cases.read_bytes()
    if sha256(data).hexdigest() != CORPUS_SHA256:
        raise ValueError('The frozen corpus SHA-256 does not match')
    result = trial(json.loads(data), args.qwen)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({method: value['totals'] for method, value in result['methods'].items()}, indent=2))
