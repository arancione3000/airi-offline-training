"""Paired CPU cost of source validation around the same fixed classifier output."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from time import process_time_ns


HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parents[1] / 'assistant-runtime'
sys.path.insert(0, str(RUNTIME))
from flexible_skills import LearnedQueryRouter
from learn import BUNDLED
from query_learning import bindings, case
from skill_bank import SkillBank


BASE = '55f835f203beb1e3d67be7bef9bd76b8734dda38'


def measure(previous, current):
    with gzip.open(HERE.parent / 'semantic-routing-20261116/evidence/precommitted/FAMILIES.json.gz', 'rt') as stream:
        families = json.load(stream)['families']
    family = families[0]
    bank = SkillBank(BUNDLED)
    model = {'bindings': bindings(bank, families)}

    class Backend:
        @staticmethod
        def read_model(path):
            return model

        @staticmethod
        def predict(model, query):
            return {'lesson_id': family['id']}

    routers = [cls(bank, 'unused', frozen=True, backend=Backend) for cls in (previous, current)]
    messages, expected = case(family, 'Richiesta di prova', 6251, 1)
    report = {}
    for turns in (2, 64):
        history = [{'role': 'user', 'content': f'Messaggio di contesto {i}.'} for i in range(turns - 2)]
        history.extend(messages)
        answers = [router.predict(history) for router in routers]
        if answers[0] != answers[1] or not answers[1] or answers[1]['answer'] != expected:
            raise RuntimeError('unequal response or provenance')
        samples = [[], []]
        for rep in range(100):
            for arm in ((0, 1) if rep % 2 == 0 else (1, 0)):
                start = process_time_ns()
                result = routers[arm].predict(history)
                samples[arm].append((process_time_ns() - start) / 1e6)
                if result != answers[arm]:
                    raise RuntimeError('unstable response')
        report[turns] = {
            key: {'cpu_ms_p50': round(statistics.median(values), 4),
                  'cpu_ms_p95': round(statistics.quantiles(values, n=20, method='inclusive')[-1], 4)}
            for key, values in zip(('before', 'after'), samples, strict=True)}
    return {'base_commit': BASE, 'service': 'frozen source check with constant classifier output',
            'pairs_per_history': 100, 'experts': len(bank._read()[0]['entries']),
            'bound_programs': len(model['bindings']), 'turns': report}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    args = cli.parse_args()
    path = 'scripts/assistant-runtime/flexible_skills.py'
    with tempfile.TemporaryDirectory() as work:
        old = Path(work) / 'previous.py'
        old.write_bytes(subprocess.check_output(['git', 'show', f'{BASE}:{path}'], cwd=RUNTIME))
        spec = importlib.util.spec_from_file_location('previous', old)
        previous = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(previous)
        result = measure(previous.LearnedQueryRouter, LearnedQueryRouter)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
