"""Frozen multilingual NLI cross-encoder for single, affirmative operations.

Pinned MIT checkpoint: https://huggingface.co/MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli
The tokenizer, config and model bytes come from its onnx/ directory. Weights
stay frozen. The only fit is a two-threshold selection on historical TRAIN.
"""
from functools import lru_cache
import gzip
import hashlib
import json
import os
from pathlib import Path

from learner import _frame

REVISION = '0a71e92a985b6e1ad1828cf67ce9c459639c1dca'
FILES = {
    'model.onnx': '79f8cda2b1230585a95ea0514a6f1bd21c5c986ba0529bb3261213a3e195fa6e',
    'tokenizer.json': 'b2116c05e7305eea30394284760789681c5b3440dd4cd9a8c77539da68f9e8a6',
    'config.json': 'e08b628c4d46c8601dda34ddb40a7857e5961870b80482382d420db6dfb6e00e',
}
TEXT = ('copy', 'reverse', 'uppercase', 'lowercase')
MATH = ('add', 'subtract', 'multiply')
ACTIONS = {
    'copy': 'ricevere il testo fornito esattamente come è scritto',
    'reverse': "ricevere il testo fornito con tutti i caratteri in ordine inverso",
    'uppercase': 'ricevere tutto il testo fornito convertito in lettere maiuscole',
    'lowercase': 'ricevere tutto il testo fornito convertito in lettere minuscole',
    'add': 'ricevere la somma dei due numeri forniti',
    'subtract': 'ricevere il primo numero fornito meno il secondo numero',
    'multiply': 'ricevere il prodotto dei due numeri forniti',
}
REJECT = {
    'explain': 'una spiegazione di un metodo o di una frase',
    'extra': 'di fare anche una seconda operazione sul risultato',
    'conditional': 'di fare il calcolo o la trasformazione soltanto a una condizione',
    'ability': 'se questa operazione è possibile, senza richiederne il risultato',
    'count': 'di contare oppure verificare una proprietà del testo',
    'subset': 'di modificare solo una parte del testo',
    'reversed_subtraction': 'di sottrarre il primo numero dal secondo',
    'negation': 'di evitare questa operazione',
}
TRAIN_FILE = (Path(__file__).resolve().parent.parent /
              'semantic-routing-20261116/evidence/precommitted/TRAIN.json.gz')


class CrossEncoder:
    def __init__(self, cache):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        root = Path(cache)
        for file, digest in FILES.items():
            with (root / file).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                    raise ValueError(f'incorrect NLI artifact: {file}')
        labels = json.loads((root / 'config.json').read_bytes())['id2label']
        if labels != {'0': 'entailment', '1': 'neutral', '2': 'contradiction'}:
            raise ValueError('unexpected label order')
        self.np = np
        self.tokenizer = Tokenizer.from_file(str(root / 'tokenizer.json'))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(root / 'model.onnx'), opts,
                                           providers=['CPUExecutionProvider'])

    @lru_cache(maxsize=512)
    def scores(self, frame, group):
        # Operation-independent masked premise within each type. The object
        # family never appears in either member of a sentence pair.
        slot = 'il testo fornito' if group == TEXT else 'i due numeri forniti'
        query = frame.replace('@', slot).replace('di il ', 'del ').replace('di i ', 'dei ')
        premise = f'La richiesta dell’utente è «{query}».'
        hypotheses = {op: 'L’utente chiede di ' + ACTIONS[op] + '.' for op in group}
        hypotheses.update({key: 'L’utente chiede ' + value + '.' for key, value in REJECT.items()
                           if not (group == TEXT and key == 'reversed_subtraction')
                           and not (group == MATH and key in ('count', 'subset'))})
        # Each hypothesis is tokenized as the SECOND sequence (including XLM-R
        # separator tokens); the output is a full three-way softmax.
        pairs = [self.tokenizer.encode(premise, h) for h in hypotheses.values()]
        width = max(len(p.ids) for p in pairs)
        if width > 192:
            return {}
        ids = self.np.full((len(pairs), width), 1, dtype=self.np.int64)
        mask = self.np.zeros_like(ids)
        for i, p in enumerate(pairs):
            ids[i, :len(p.ids)] = p.ids
            mask[i, :len(p.ids)] = 1
        logits, = self.session.run(['logits'], {'input_ids': ids, 'attention_mask': mask})
        if not self.np.isfinite(logits).all():
            raise ValueError('non-finite NLI prediction')
        prob = self.np.exp(logits - logits.max(axis=1, keepdims=True))
        return dict(zip(hypotheses, map(float, prob[:, 0] / prob.sum(axis=1)), strict=True))


def _rank(scores, op):
    if op not in scores:
        return (-1., -1.)
    own = scores[op]
    return own, own - max(value for key, value in scores.items() if key != op)


def expand_train_cases(trusted):
    if any((frame, op, not label) in trusted for frame, op, label in trusted):
        raise ValueError('inconsistent trusted TRAIN targets')
    cases = set(trusted)
    for frame, op, positive in trusted:
        if positive:
            group = TEXT if op in TEXT else MATH
            # An object-independent request can be supported for several
            # operations. Synthesize alternatives only outside that support.
            cases.update((frame, other, False) for other in group
                         if (frame, other, True) not in trusted)
    return cases


class Selector:
    def __init__(self, families, cross):
        self.families = tuple(families)
        self.cross = cross
        with gzip.open(TRAIN_FILE, 'rt') as stream:
            data = json.load(stream)
        if data['split'] != 'TRAIN':
            raise ValueError('wrong calibration split')
        trusted = set()
        for row in data['rows']:
            bound = _frame(row['query'], self.families)
            if bound is None:
                continue
            family, frame = bound
            if family['id'] != row['source_lesson_id']:
                continue
            op = family['operation']
            positive = row['lesson_id'] == family['id']
            trusted.add((frame, op, positive))
        cases = expand_train_cases(trusted)
        ranks = []
        for frame, op, positive in sorted(cases):
            scores = cross.scores(frame, TEXT if op in TEXT else MATH)
            ranks.append((*_rank(scores, op), positive))
        options = []
        for floor in (.15, .25, .35, .45, .55, .65, .75, .85, .95):
            for margin in (.01, .03, .05, .10, .20, .30):
                accepted = [pos for score, difference, pos in ranks if score >= floor and difference >= margin]
                if all(accepted):
                    options.append((len(accepted), floor, margin))
        correct, self.floor, self.margin = max(options, default=(0, 2., 2.))
        self.calibration = {'nli_revision': REVISION, 'files': FILES, 'calibration_cases': len(ranks),
                            'supported_cases': sum(rank[2] for rank in ranks),
                            'calibration_correct': correct, 'calibration_false': 0,
                            'entailment_floor': self.floor, 'margin': self.margin}

    @lru_cache(maxsize=512)
    def __call__(self, query):
        bound = _frame(query, self.families)
        if bound is None:
            return None
        family, frame = bound
        op = family['operation']
        scores = self.cross.scores(frame, TEXT if op in TEXT else MATH)
        score, difference = _rank(scores, op)
        return family['id'] if score >= self.floor and difference >= self.margin else None


def fit(families):
    cache = os.environ.get('AIRI_NLI_CACHE')
    if not cache:
        raise ValueError('set AIRI_NLI_CACHE to the pinned local model directory')
    return Selector(families, CrossEncoder(cache))
