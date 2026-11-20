"""Compare fixed-budget selection and exact compositional transfer in Airi."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unicodedata

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
RUNTIME = RESEARCH.parent / 'assistant-runtime'
ARCHIVE = RESEARCH / 'semantic-routing-20261116/evidence'
for directory in (RUNTIME, RESEARCH / 'intent-20261120', RESEARCH / 'typed-routing-20261118'):
    sys.path.insert(0, str(directory))

import intent_grammar
import query_router
import typed_router
from exact_aliases import DEFAULT as ALIASES, AliasRouter
from flexible_skills import LearnedQueryRouter
from learn import BUNDLED
from pool import POOL
from query_learning import bindings, case
from skill_bank import SkillBank, _route, canonical, digest
from skill_validation import StableGate

SEEDS = tuple(range(20261120, 20261128))
BUDGETS = (8, 16, 32)
SELECTORS = ('random', 'tfidf', 'v5', 'grammar')
V5 = RESEARCH / 'typed-routing-20261118/evidence/candidate.json.gz'
LEXICAL = ARCHIVE / 'first-run/candidate.json.gz'


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(name):
    return json.loads(gzip.decompress((ARCHIVE / name).read_bytes()))


def normalize(s):
    return unicodedata.normalize('NFC', s).casefold().replace('’', "'")


def article(family):
    text = family['object_phrase']
    return "l'" if text.startswith("l'") else text.split()[0]


def mask(query, families):
    query = normalize(query)
    if '@' in query or not query.isprintable():
        return None
    matches = []
    for family in families:
        phrase = normalize(family['object_phrase'])
        count = query.count(phrase)
        if count > 1:
            return None
        if count:
            before, after = query.split(phrase)
            if (before and before[-1].isalnum()) or (after and after[0].isalnum()):
                continue
            matches.append((family, before + '@' + after))
    return matches[0] if len(matches) == 1 else None


def key(family, frame):
    return family['operation'], article(family), frame


def score_pool(requests, families, directory):
    """Rankers receive proposal identifiers and text only."""
    timings = {}
    confidence = {'random': {p['id']: (0, 0) for p in requests}}
    for name, path, backend in (('tfidf', LEXICAL, query_router), ('v5', V5, typed_router)):
        start = time.perf_counter()
        local = directory / (name + '.json')
        local.write_bytes(gzip.decompress(path.read_bytes()))
        model = backend.read_model(local)
        predictor = backend.compile_model(model).predict if name == 'v5' else lambda q: backend.predict(model, q)
        ranked = {}
        for request in requests:
            prediction = predictor(request['query'])
            ranked[request['id']] = (1, prediction['margin']) if prediction else (0, 0)
        confidence[name] = ranked
        timings[name] = time.perf_counter() - start
    start = time.perf_counter()
    confidence['grammar'] = {p['id']: (int(intent_grammar.predict(p['query'], families) is not None), 0)
                             for p in requests}
    timings['grammar'] = time.perf_counter() - start
    return confidence, timings


def tie(seed, request_id):
    return hashlib.sha256(f'{seed}:{request_id}'.encode()).hexdigest()


def inputs():
    return {str(path.relative_to(RESEARCH.parent)): fingerprint(path) for path in (
        HERE / 'trial.py', HERE / 'pool.py', HERE / 'holdout.py', V5,
        Path(intent_grammar.__file__), Path(typed_router.__file__), Path(query_router.__file__), RUNTIME / 'skill_programs.py',
        RUNTIME / 'flexible_skills.py', RUNTIME / 'query_learning.py', RUNTIME / 'exact_aliases.py', LEXICAL)}


def freeze(out):
    out.mkdir(parents=True, exist_ok=False)
    families = archive('precommitted/FAMILIES.json.gz')['families']
    bank = out / 'bank.json'
    bank.write_bytes(BUNDLED.read_bytes())
    aliases = out / 'aliases.json'
    aliases.write_bytes(ALIASES.read_bytes())
    request_text = [{'id': row['id'], 'query': row['query']} for row in POOL]
    if len(request_text) != len({row['id'] for row in request_text}):
        raise ValueError('duplicate proposal id')
    with tempfile.TemporaryDirectory() as temporary:
        confidence, timings = score_pool(request_text, families, Path(temporary))
    rankings = {}
    for name in SELECTORS:
        rankings[name] = [sorted(confidence[name], key=lambda i: (
            -confidence[name][i][0], -confidence[name][i][1], tie(seed, i))) for seed in SEEDS]
    document = dict(inputs=inputs(), bank=fingerprint(bank), aliases=fingerprint(aliases),
                    family_digest=digest(families), proposal_digest=digest(POOL),
                    budgets=BUDGETS, seeds=SEEDS, scoring_seconds=timings, rankings=rankings)
    (out / 'rankings.json').write_bytes(canonical({'payload': document, 'sha256': digest(document)}) + b'\n')
    print('Frozen rankings', digest(document), 'scoring seconds', timings)


class FrameBackend:
    @staticmethod
    def read_model(model):
        return json.loads(model.read_bytes())

    @staticmethod
    def compile_model(model):
        keys = {tuple(row) for row in model['keys']}

        class Predictor:
            def predict(self, query):
                match = mask(query, model['families'])
                if match is None:
                    return None
                family, frame = match
                if key(family, frame) in keys:
                    return {'lesson_id': family['id'], 'operation': family['operation']}
                return None

        return Predictor()


def selection(identifiers, rows, frames):
    """Spend each of the requested independent reference labels exactly once."""
    installed = set()
    kept = []
    for identifier in identifiers:
        family, frame = frames[identifier]
        if rows[identifier]['intent'] == family['operation']:
            installed.add(key(family, frame))
            kept.append(identifier)
    return installed, kept


def recombinations(rows, frames, families):
    """Keep the original operation while varying object, or deliberately mismatch it."""
    positives, negatives = {}, {}
    originals = {normalize(row['query']) for row in rows.values()}
    for identifier, row in rows.items():
        source, frame = frames[identifier]
        if row['intent'] != source['operation']:
            continue
        for destination in families:
            if source['id'] == destination['id'] or article(source) != article(destination):
                continue
            query = frame.replace('@', destination['object_phrase'])
            if normalize(query) in originals:
                continue
            data = (destination, query)
            identity = destination['id'], normalize(query)
            if source['operation'] == destination['operation']:
                positives[identity] = data
            else:
                negatives[identity] = data
    return positives, {i: item for i, item in negatives.items() if i not in positives}


def compose(bank, alias, router):
    state = bank._read()[0]

    def predict(messages):
        first = _route(state['entries'], messages)
        if first is not None:
            return first
        owned, exact = alias.match(messages)
        return exact if owned else (router.predict(messages) if router else None)

    return state, predict


def score(predict, rows, offset, baseline_cases=None):
    counts = Counter(positive=0, correct=0, wrong_positive=0, negative=0, false_positive=0, new_correct=0)
    for i, (family, query, positive) in enumerate(rows):
        for variant in (0, 1):
            messages, answer = case(family, query, i + offset, variant)
            result = predict(messages)
            counts['positive' if positive else 'negative'] += 1
            if result is not None:
                if positive and result['answer'] == answer:
                    counts['correct'] += 1
                    if baseline_cases is None or (i, variant) not in baseline_cases:
                        counts['new_correct'] += 1
                else:
                    counts['wrong_positive' if positive else 'false_positive'] += 1
    return dict(counts)


def audit(predict, state):
    gate = StableGate()
    historical = gate(predict)
    stored = [row for entry in state['entries'] for row in entry['train'] + entry['qualification']]
    failed = 0
    for row in stored:
        prediction = predict(row['messages'])
        if (prediction['answer'] if prediction else None) != row['answer']:
            failed += 1
    return dict(passed=historical and failed == 0, historical=gate.last_report['cases'],
                stored=len(stored), stored_failures=failed)


def qualify(out):
    file = out / 'rankings.json'
    frozen = json.loads(file.read_bytes())
    design = frozen['payload']
    bank = SkillBank(out / 'bank.json')
    aliases = AliasRouter(bank, out / 'aliases.json')
    families = archive('precommitted/FAMILIES.json.gz')['families']
    if (frozen['sha256'] != digest(design) or design['inputs'] != inputs()
            or design['bank'] != fingerprint(bank.path) or design['aliases'] != fingerprint(aliases.path)
            or design['proposal_digest'] != digest(POOL) or design['family_digest'] != digest(families)):
        raise ValueError('inputs differ from frozen ranking')
    endpoint = out / 'RESULTS.json'
    if endpoint.exists():
        raise ValueError('choose a fresh freeze for a new evaluation')

    from holdout import dataset
    withheld = dataset()
    definitions = {f['id']: f for f in families}
    challenge = [(definitions[r['source_lesson_id']], r['query'], r['lesson_id'] is not None)
                 for r in withheld['rows']]
    pool = {row['id']: row for row in POOL}
    frames = {i: mask(row['query'], families) for i, row in pool.items()}
    if any(item is None or item[0]['id'] != pool[i]['family'] for i, item in frames.items()):
        raise ValueError('pool object ambiguity')
    positives, negatives = recombinations(pool, frames, families)
    derived = [(family, query, positive) for positive, pairs in ((True, positives), (False, negatives))
               for family, query in pairs.values()]
    state, base = compose(bank, aliases, None)
    already_correct = set()
    for i, (family, query, positive) in enumerate(derived):
        for variant in (0, 1):
            messages, answer = case(family, query, 55000 + i, variant)
            result = base(messages)
            if positive and result and result['answer'] == answer:
                already_correct.add((i, variant))
    results = {'freeze_sha256': frozen['sha256'], 'pool_size': len(pool),
               'pool_positive': sum(row['intent'] is not None for row in POOL),
               'scoring_seconds': design['scoring_seconds'], 'budgets': {},
               'independent_baseline': score(base, challenge, withheld['value_offset']),
               'recombined_baseline': score(base, derived, 55000)}

    for budget in BUDGETS:
        budget_report = {}
        for name in SELECTORS:
            counts = []
            first = None
            for ranking in design['rankings'][name]:
                keys, kept = selection(ranking[:budget], pool, frames)
                applicable = sum(key(family, mask(query, families)[1]) in keys
                                 for family, query in positives.values())
                counts.append(dict(accepted=len(kept), frames=len(keys), recombined_pairs=applicable))
                if first is None and budget == 16:
                    with tempfile.TemporaryDirectory() as temporary:
                        path = Path(temporary) / 'model.json'
                        path.write_bytes(canonical(dict(keys=sorted(keys), bindings=bindings(bank, families),
                                                        families=families)))
                        router = LearnedQueryRouter(bank, path, frozen=True, backend=FrameBackend)
                    _, composed = compose(bank, aliases, router)
                    first = dict(recombined=score(composed, derived, 55000, already_correct),
                                 independent=score(composed, challenge, withheld['value_offset']),
                                 retention=audit(composed, state))
                    first['passed'] = (
                        first['independent']['correct'] * 2 >= first['independent']['positive']
                        and all(first[which]['false_positive'] == first[which]['wrong_positive'] == 0
                                for which in ('recombined', 'independent'))
                        and first['retention']['passed'])
            budget_report[name] = dict(repetitions=counts, fixed_seed_execution=first,
                                       mean={k: sum(row[k] for row in counts) / len(counts) for k in counts[0]})
        results['budgets'][str(budget)] = budget_report
    endpoint.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(results['budgets']['16'], ensure_ascii=False))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase', choices=('freeze', 'qualify'))
    cli.add_argument('output', type=Path)
    args = cli.parse_args()
    (freeze if args.phase == 'freeze' else qualify)(args.output)
