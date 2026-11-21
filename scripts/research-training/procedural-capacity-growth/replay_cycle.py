"""Reproduce autonomous procedural acquisition across separate process restarts."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys

from cycle import INPUTS, HERE


def manifest(path, tasks):
    contents = {'schema': 1, 'tasks': {task: [f(a, b) for a, b in INPUTS]
                                       for task, f in tasks.items()}}
    raw = json.dumps(contents, sort_keys=True, separators=(',', ':')).encode()
    path.write_bytes(raw)
    return sha256(raw).hexdigest()


def run(directory):
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = directory / 'state.json'

    def invoke(source, digest):
        args = [sys.executable, str(HERE / 'cycle.py'), '--reference', str(source),
                '--reference-sha256', digest, '--state', str(snapshot)]
        return json.loads(subprocess.check_output(args))

    first = directory / 'first.json'
    initial_tasks = {'squared_sum': lambda a, b: (a + b) ** 2}
    first_pin = manifest(first, initial_tasks)
    initial = invoke(first, first_pin)
    empty = invoke(first, first_pin)
    if initial['status'] != 'accepted' or empty['status'] != 'audited':
        raise AssertionError('first acquisition and idle replay differed from expectation')

    extended = directory / 'extended.json'
    extended_pin = manifest(extended, {**initial_tasks, 'gcd': math.gcd,
                                       'minimum_magnitude': lambda a, b: min(abs(a), abs(b))})
    events = [initial, empty]
    for expected in [('gcd', 'rejected'), ('minimum_magnitude', 'accepted'), ('gcd', 'rejected')]:
        report = invoke(extended, extended_pin)
        if (report['task'], report['status']) != expected:
            raise AssertionError('unexpected acquisition or retry')
        events.append(report)
    before = snapshot.read_bytes()
    finished = invoke(extended, extended_pin)
    if (finished['status'] != 'audited' or finished['approved'] != 2
            or finished['attempts'] != 4 or finished['pending'] != 0 or snapshot.read_bytes() != before):
        raise AssertionError('restart or idle run changed the approved graph')
    result = {'events': [*events, finished], 'registry_sha256': sha256(before).hexdigest()}
    (directory / 'report.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    args = cli.parse_args()
    run(args.output)
