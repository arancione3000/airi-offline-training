"""Incrementally transfer reviewed, literal TRAIN requests without allocating experts."""
import argparse
from collections import defaultdict
import fcntl
import gzip
import json
import os
from pathlib import Path
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
# HERE is scripts/research-training/<experiment>.
RUNTIME = HERE.parent.parent / 'assistant-runtime'
sys.path.insert(0, str(RUNTIME))
from exact_aliases import DEFAULT, MAX_ALIASES, read_aliases, resolve_alias
from flexible_skills import DEFAULT_MODEL, LearnedQueryRouter, _semantic_router
from learn import BUNDLED
from query_learning import bindings, case
from skill_bank import SkillBank, RegistryError, _route, canonical, digest
from skill_validation import StableGate

CORPUS = HERE.parent / 'semantic-routing-20261116/evidence/precommitted'
SEMANTIC_RETENTION = DEFAULT_MODEL.with_name('retention.json.gz')


def read_semantic_retention(path=SEMANTIC_RETENTION):
    return json.loads(gzip.decompress(path.read_bytes())) if path.exists() else []


def write_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(canonical(value) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        tmp.unlink(missing_ok=True)


def catalog(rows, approved, entries, retained=()):
    labels = defaultdict(set)
    for row in rows:
        labels[row['query']].add(row['lesson_id'])
    installed = {entry['program']['query'] for entry in entries}
    bound = {b['lesson_id']: b['program_id'] for b in approved}
    result = {}
    for row in rows:
        family, query = row['lesson_id'], row['query']
        if (family not in bound or (query in installed and query not in retained)
                or row['source_lesson_id'] != family
                or labels[query] != {family} or row['condition'] != 'supported_paraphrase'):
            continue
        result.setdefault(query, dict(lesson_id=family, program_id=bound[family], train_id=row['id']))
    # Interleave families so each cycle can cover several existing procedures.
    ranks = defaultdict(int)
    ordered = []
    for query, binding in result.items():
        family = binding['lesson_id']
        ordered.append((ranks[family], query, binding))
        ranks[family] += 1
    return {query: binding for _, query, binding in sorted(ordered)}


def audit(state, aliases, families, negatives, seed, *, semantic=None, semantic_checks=()):
    definitions = {family['id']: family for family in families}
    # The independently defined oracle bounds operands to the existing DSL domain.
    offset = 1000 + seed % 80000
    entries = state['entries']

    def composed(messages):
        result = _route(entries, messages)
        if result is not None:
            return result
        binding = aliases.get(messages[-1]['content']) if messages else None
        if binding is not None:
            return resolve_alias(state, messages, binding)
        return semantic.predict(messages) if semantic is not None else None

    # An exact expert can later take priority for a retained alias. It must
    # implement the same program semantics and reproduce its source evidence.
    by_id = {entry['id']: entry['program'] for entry in entries}
    positives = rejected = 0
    for i, (query, binding) in enumerate(sorted(aliases.items())):
        family = definitions[binding['lesson_id']]
        expected_program = by_id[binding['program_id']]
        for entry in entries:
            program = entry['program']
            if program['query'] == query and any(program[k] != expected_program[k]
                                                  for k in ('operation', 'source')):
                raise RegistryError('conflicting exact expert claims retained alias')
        other = families[(families.index(family) + 1) % len(families)]
        for variant in (0, 1):
            messages, answer = case(family, query, i + offset, variant)
            result = composed(messages)
            expected_route = resolve_alias(state, messages, binding)
            if (not result or not expected_route or result['answer'] != answer
                    or any(result.get(k) != expected_route[k] for k in ('operation', 'source', 'provenance'))):
                raise RegistryError('alias positive or bound expert failed: ' + binding['train_id'])
            positives += 1
        source, question = messages
        competitor = case(other, query, i + offset, 0)[0][0]
        conflicts = case(family, query, i + offset + 1, 0)[0][0]
        # Even a numerically correct replacement is vetoed if it now answers
        # ambiguous mixed-source messages that the admitted alias rejected.
        for unsafe in ([question], [competitor, question], [source, competitor, question],
                       [source, conflicts, question], [{**source, 'role': 'assistant'}, question],
                       [{**source, 'speaker_id': 'A'}, {**question, 'speaker_id': 'B'}],
                       [source, {**question, 'content': 'Non farlo. ' + query}]):
            if composed(unsafe) is not None:
                raise RegistryError('alias source/intent boundary failed: ' + binding['train_id'])
            rejected += 1
    for i, row in enumerate(negatives):
        family = definitions[row['source_lesson_id']]
        if composed(case(family, row['query'], i + offset, 0)[0]) is not None:
            raise RegistryError('unsupported TRAIN request intercepted')
        rejected += 1

    stored = 0
    for entry in entries:
        for row in entry['train'] + entry['qualification']:
            result = composed(row['messages'])
            if (result['answer'] if result else None) != row['answer']:
                raise RegistryError('stored contract failed')
            stored += 1
    historical = StableGate()
    if not historical(composed):
        raise RegistryError('historical contract failed')
    for row in semantic_checks:
        prediction = composed(row['messages'])
        if prediction is None and row.get('allow_abstention', False):
            continue
        answer = prediction['answer'] if prediction is not None else None
        if answer != row['expected']:
            raise RegistryError('qualified semantic behavior would regress or answer incorrectly')
    report = dict(passed=True, positive_total=positives, negative_total=rejected,
                  stored_contracts=stored, historical_cases=historical.last_report['cases'])
    if semantic_checks:
        report['semantic_checks'] = len(semantic_checks)
    return report


def read_corpus():
    rows = json.loads(gzip.decompress((CORPUS / 'TRAIN.json.gz').read_bytes()))['rows']
    families = json.loads(gzip.decompress((CORPUS / 'FAMILIES.json.gz').read_bytes()))['families']
    return rows, families


def audit_candidate(state, aliases, semantic, checks, *, seed):
    rows, families = read_corpus()
    return audit(state, aliases, families, [r for r in rows if r['lesson_id'] is None],
                 seed, semantic=semantic, semantic_checks=checks)


def semantic_state():
    """Capture inputs so the admission and final write refer to the same data."""
    return tuple(path.read_bytes() if path.exists() else None
                 for path in (DEFAULT_MODEL, SEMANTIC_RETENTION))


def cycle(bank_path, state_path, *, limit, seed):
    if not 1 <= limit <= 32 or not 0 <= seed < 2**64:
        raise ValueError('invalid limit or seed')
    bank = SkillBank(bank_path)
    state = bank._read()[0]
    rows, families = read_corpus()
    old = read_aliases(state_path)
    candidates = catalog(rows, bindings(bank, families), state['entries'], old)
    if any(candidates.get(query) != binding for query, binding in old.items()):
        raise RegistryError('previous alias lacks its original trusted binding')
    remaining = [q for q in candidates if q not in old]
    selected = remaining[:min(limit, MAX_ALIASES - len(old))]
    proposal = {**old, **{q: candidates[q] for q in selected}}
    installed = semantic_state()
    semantic = None
    checks = read_semantic_retention(SEMANTIC_RETENTION)
    if installed[0] is not None:
        runtime_router = _semantic_router(bank)
        if runtime_router is None:
            raise RegistryError('installed semantic model is unavailable')
        semantic = LearnedQueryRouter(bank, runtime_router.path, frozen=True,
                                       backend=runtime_router.backend)
        if semantic.snapshot != state:
            raise RegistryError('concurrent bank update; retry with latest state')
        if not checks:
            raise RegistryError('installed semantic model lacks qualified cases')
    elif checks:
        raise RegistryError('qualified semantic cases lack their installed model')
    audited = audit(state, proposal, families, [r for r in rows if r['lesson_id'] is None],
                    seed, semantic=semantic, semantic_checks=checks)
    if selected:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        lock = os.open(state_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with bank._lock():
                if (bank._read()[0] != state or read_aliases(state_path) != old
                        or semantic_state() != installed):
                    raise RegistryError('concurrent update; retry with latest state')
                write_atomic(state_path, dict(schema=1, aliases=proposal, sha256=digest(proposal)))
        finally:
            os.close(lock)
    return dict(status='updated' if selected else 'audited', activated=len(selected),
                total_aliases=len(proposal), remaining=len(remaining)-len(selected),
                audit=audited, seed=seed, bank_sha256=digest(state), aliases_sha256=digest(proposal),
                train_sha256=digest(rows), scope='exact reviewed TRAIN questions on independent values')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, default=BUNDLED)
    parser.add_argument('--state', type=Path, default=DEFAULT)
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--seed', type=int, default=time.time_ns())
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.bank.resolve() == args.state.resolve() or (args.report and args.report.resolve() in
                                                       (args.state.resolve(), args.bank.resolve())):
        parser.error('bank, state, and report require distinct paths')
    result = cycle(args.bank, args.state, limit=args.limit, seed=args.seed)
    if args.report:
        write_atomic(args.report, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
