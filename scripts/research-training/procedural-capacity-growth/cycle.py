"""One bounded automatic update from a pinned, append-only arithmetic task manifest."""
import argparse
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import sys
from tempfile import NamedTemporaryFile

HERE = Path(__file__).resolve().parent
VERIFICATION = HERE.parent / 'incremental-procedure-verification'
sys.path.insert(0, str(VERIFICATION))
from fixtures import INPUTS, scenarios
from verifier import MAX_MAGNITUDE, MAX_NODES, OPERATIONS, Verifier, ordered_nodes


MAX_BYTES = 2 * 1024 * 1024
MAX_TASKS = 16
MAX_ATTEMPTS = 64


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def digest(value):
    return sha256(encode(value)).hexdigest()


def read(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('snapshot must be a regular file')
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError('snapshot too large')
    return raw


def write(path, value):
    raw = encode({'payload': value, 'sha256': digest(value)}) + b'\n'
    if len(raw) > MAX_BYTES:
        raise ValueError('snapshot exceeds its bound')
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def trusted_tasks(path, pin):
    raw = read(path)
    if sha256(raw).hexdigest() != pin:
        raise ValueError('reference manifest differs from the independently supplied digest')
    manifest = json.loads(raw)
    if (type(manifest) is not dict or set(manifest) != {'schema', 'tasks'}
            or type(manifest['schema']) is not int or manifest['schema'] != 1):
        raise ValueError('invalid reference manifest')
    rows = manifest['tasks']
    if type(rows) is not dict or len(rows) > MAX_TASKS:
        raise ValueError('too many reference tasks')
    tasks = {}
    for name, values in rows.items():
        if (not re.fullmatch(r'[a-z][a-z0-9_]{0,47}', name) or type(values) is not list
                or len(values) != len(INPUTS) or any(type(x) is not int or abs(x) > MAX_MAGNITUDE for x in values)):
            raise ValueError('invalid reference task')
        tasks[name] = tuple(values)
    return tasks


def namespace(additions, output, index):
    if output not in additions or not 1 <= len(additions) <= 7:
        raise ValueError('invalid proposed extension')
    names = {name: f'growth_{index}_{i}' for i, name in enumerate(sorted(additions))}
    graph = {names[name]: {'op': recipe['op'], 'deps': [names.get(dep, dep) for dep in recipe['deps']]}
             for name, recipe in additions.items()}
    return graph, names[output]


def stage(engine, graph, additions, output, expected):
    if (type(additions) is not dict or type(output) is not str
            or not 1 <= len(additions) <= 7 or set(graph) & set(additions) or output not in additions):
        raise ValueError('proposal must append new nodes')
    updated = {**graph, **additions}
    tables = {**engine.tables, 'a': tuple(a for a, _ in INPUTS), 'b': tuple(b for _, b in INPUTS)}
    reference = dict(engine.tables)
    for name, (op, deps) in ordered_nodes(updated).items():
        if name in additions:
            function = OPERATIONS[op][1]
            try:
                values = tuple(map(function, *(tables[dep] for dep in deps))) if deps else (function(),) * len(INPUTS)
            except ZeroDivisionError:
                return False
            if any(abs(v) > MAX_MAGNITUDE for v in values):
                return False
            tables[name] = values
            reference[name] = expected if name == output else values
    return engine.update(updated, reference)['status'] == 'accepted'


LEGACY_CYCLE_SHA256 = '4cbd5af7e29fc4dc93883c71b60326b563842f7c07f7d1176decd70b934e5c07'


def engine_fingerprint(*, legacy=False):
    source_files = (Path(__file__), HERE / 'synthesis_growth.py', VERIFICATION / 'synthesis.py',
                    VERIFICATION / 'verifier.py', VERIFICATION / 'fixtures.py')
    hashes = [sha256(path.read_bytes()).hexdigest() for path in source_files]
    if legacy:
        hashes[0] = LEGACY_CYCLE_SHA256
    return digest(hashes)


def restore_state(path, engine_id, *, required=False):
    try:
        raw = read(path)
    except FileNotFoundError:
        if required:
            raise
        return {'schema': 1, 'engine_sha256': engine_id, 'entries': []}
    document = json.loads(raw)
    if type(document) is not dict or set(document) != {'payload', 'sha256'}:
        raise ValueError('invalid growth state')
    state = document['payload']
    if (digest(state) != document['sha256'] or type(state) is not dict
            or set(state) != {'schema', 'engine_sha256', 'entries'} or state['schema'] != 1
            or state['engine_sha256'] not in (engine_id, engine_fingerprint(legacy=True)) or type(state['entries']) is not list
            or len(state['entries']) > MAX_ATTEMPTS):
        raise ValueError('state checksum, schema or engine mismatch')
    return state


def requalify(state, tasks):
    """Replay the append-only journal against its independently pinned references."""
    _, graph, reference, _ = next(scenarios())
    engine = Verifier(INPUTS)
    if engine.update(graph, reference)['status'] != 'accepted':
        raise ValueError('starting procedures failed validation')
    approved, attempted = {}, set()
    for row in state['entries']:
        if type(row) is not dict or row.get('status') not in ('accepted', 'rejected'):
            raise ValueError('invalid learning record')
        fields = {'task', 'reference_sha256', 'state_sha256', 'status'}
        if row['status'] == 'accepted':
            fields |= {'additions', 'output'}
        if set(row) != fields or type(row['task']) is not str:
            raise ValueError('invalid learning record fields')
        name = row['task']
        context = digest(graph)
        if (name in approved or (name, context) in attempted or name not in tasks
                or row['reference_sha256'] != digest(tasks[name]) or row['state_sha256'] != context):
            raise ValueError('invalid learning sequence or existing task reference changed')
        attempted.add((name, context))
        if row['status'] == 'accepted':
            if not stage(engine, graph, row['additions'], row['output'], tasks[name]):
                raise ValueError('previously stored procedure failed requalification')
            graph.update(row['additions'])
            approved[name] = row['output']
    return engine, graph, approved, attempted


def cycle(state_path, tasks_path, pin):
    from synthesis_growth import GrowingLearner

    if state_path.resolve() == tasks_path.resolve():
        raise ValueError('state and reference require separate paths')
    tasks = trusted_tasks(tasks_path, pin)
    engine_id = engine_fingerprint()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_name(state_path.name + '.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = restore_state(state_path, engine_id)
        engine, graph, approved, attempted = requalify(state, tasks)
        context = digest(graph)
        remaining = sorted(name for name in tasks.keys() - approved.keys() if (name, context) not in attempted)
        if not remaining:
            return {'status': 'audited', 'attempts': len(attempted), 'approved': len(approved),
                    'approved_nodes': len(graph), 'pending': 0}
        if len(attempted) == MAX_ATTEMPTS:
            return {'status': 'attempt_limit', 'approved': len(approved), 'pending': len(remaining)}

        name = remaining[0]
        evidence = tasks[name]
        record = {'task': name, 'reference_sha256': digest(evidence), 'state_sha256': context, 'status': 'rejected'}

        def qualify(additions, output, _intermediate_tables):
            proposed, result_node = namespace(additions, output, len(attempted))
            if len(graph) + len(proposed) > MAX_NODES or not stage(engine, graph, proposed, result_node, evidence):
                return False
            graph.update(proposed)
            record.update(status='accepted', additions=proposed, output=result_node)
            return True

        trial = GrowingLearner(graph, engine.tables).learn(evidence.__getitem__, qualify, max_depth=2)
        state['entries'].append(record)
        state['engine_sha256'] = engine_id
        write(state_path, state)
        attempted.add((name, context))
        if record['status'] == 'accepted':
            approved[name] = record['output']
        next_context = digest(graph)
        pending = sum((task, next_context) not in attempted for task in tasks.keys() - approved.keys())
        return {**record, 'approved': len(approved), 'approved_nodes': len(graph), 'pending': pending, 'trial': trial}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reference', type=Path, required=True)
    cli.add_argument('--reference-sha256', required=True)
    cli.add_argument('--state', type=Path, required=True)
    cli.add_argument('--drain', action='store_true', help='Work until the current trusted manifest has no eligible task')
    args = cli.parse_args()
    if not args.drain:
        print(json.dumps(cycle(args.state, args.reference, args.reference_sha256)))
        return
    for _ in range(MAX_ATTEMPTS + 1):
        result = cycle(args.state, args.reference, args.reference_sha256)
        print(json.dumps(result), flush=True)
        if result['status'] in ('audited', 'attempt_limit') or not result['pending']:
            return
    raise RuntimeError('growth queue did not converge within its attempt bound')


if __name__ == '__main__':
    main()
