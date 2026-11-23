"""Freeze public retail values and synthetic Italian conversations for the pilot."""
import argparse
from collections import defaultdict
from hashlib import file_digest
from io import BytesIO
import json
from pathlib import Path
import random
import re
from zipfile import ZipFile

DATA_SHA256 = 'f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b'
QUESTION = 'Quante unità rimangono spedite al netto degli storni?'
FORMATS = (
    ('Riepilogo spedizioni', 'evase', 'stornate'),
    ('Registro rimborsi', 'spedite', 'rettificate'),
)
PARAPHRASES = (
    'Qual è il saldo delle unità dopo gli storni?',
    'Togliendo gli storni, quante unità restano spedite?',
    'Mi calcoli il netto delle quantità spedite?',
)


def read_values(archive):
    from openpyxl import load_workbook

    with archive.open('rb') as stream:
        if file_digest(stream, 'sha256').hexdigest() != DATA_SHA256:
            raise ValueError('UCI archive SHA-256 mismatch')
    counts = defaultdict(lambda: [0, 0])
    with ZipFile(archive) as package:
        book = load_workbook(BytesIO(package.read('Online Retail.xlsx')), read_only=True, data_only=True)
        try:
            for invoice, stock, _, quantity in book.active.iter_rows(min_row=2, max_col=4, values_only=True):
                if type(quantity) is not int or not re.fullmatch('[0-9]{5}[A-Z]{0,2}', str(stock)):
                    continue
                cancellation = str(invoice).upper().startswith('C')
                if cancellation and quantity < 0:
                    counts[str(stock)][1] -= quantity
                elif not cancellation and quantity > 0:
                    counts[str(stock)][0] += quantity
        finally:
            book.close()
    values = []
    seen = set()
    for _, (shipped, cancelled) in sorted(counts.items()):
        pair = (shipped, cancelled)
        if 5 <= cancelled < shipped <= 10000 and pair not in seen:
            values.append(pair)
            seen.add(pair)
    random.Random(37183).shuffle(values)
    return iter(values)


def source(form, pair, drift=None):
    title, left, right = FORMATS[form]
    shipped, cancelled = pair
    if drift == 'newline':
        return f'{title}: {left} {shipped};\n{right} {cancelled} unità.'
    if drift == 'reordered':
        return f'{title}: {right} {cancelled}; {left} {shipped} unità.'
    return f'{title}: {left} {shipped}; {right} {cancelled} unità.'


def record(form, kind, pair, other=None, variant=0):
    title, left, _ = FORMATS[form]
    question = QUESTION
    role = 'user'
    reports = [source(form, pair)]
    answer = str(pair[0] - pair[1])
    if kind == 'paraphrase':
        question = PARAPHRASES[variant % len(PARAPHRASES)]
    elif kind == 'layout':
        reports = [source(form, pair, ('newline', 'reordered')[variant % 2])]
    elif kind != 'exact':
        answer = None
        if kind == 'missing':
            reports = [f'{title}: {left} {pair[0]} unità.']
        elif kind == 'money':
            question = 'Quanto vale il saldo degli storni in euro?'
        elif kind == 'wrong_operation':
            question = 'Somma le unità spedite e quelle stornate.'
        elif kind == 'conflict':
            reports.append(source(form, other))
        elif kind == 'conflict_between_formats':
            reports.append(source(1 - form, other))
        elif kind == 'assistant_source':
            role = 'assistant'
        else:
            raise ValueError(kind)
    return {
        'format': form, 'kind': kind,
        'messages': [{'role': role, 'content': text} for text in reports] +
                    [{'role': 'user', 'content': question}],
        'answer': answer,
    }


def build(values):
    result = {'schema': 1, 'dataset_sha256': DATA_SHA256, 'train': [], 'qualification': [], 'final': []}
    for form in range(len(FORMATS)):
        result['train'] += [record(form, 'exact', next(values)) for _ in range(4)]
        result['qualification'] += [record(form, 'exact', next(values)) for _ in range(8)]
        for kind in ('missing', 'money', 'wrong_operation', 'assistant_source'):
            result['qualification'].append(record(form, kind, next(values)))
        for kind, count in (('exact', 14), ('paraphrase', 6), ('layout', 4)):
            result['final'] += [record(form, kind, next(values), variant=i) for i in range(count)]
        for kind in ('missing', 'money', 'wrong_operation', 'conflict', 'conflict_between_formats', 'assistant_source'):
            for _ in range(2):
                pair = next(values)
                other = next(values) if kind.startswith('conflict') else None
                result['final'].append(record(form, kind, pair, other))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    arguments = parser.parse_args()
    result = build(read_values(arguments.archive))
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
