"""Compare warm and cold assistant routing with the same immutable artifacts."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/assistant-runtime'))

from assistant import AiriAssistant
from exact_aliases import DEFAULT, AliasRouter, read_aliases
from flexible_skills import _QuestionRoutes
from learn import BUNDLED
from skill_bank import SkillBank


def measure(baseline, bank, aliases, seed_rows):
    def assistant(cls):
        registry = cls(bank)
        return AiriAssistant(lambda _: 'fallback', registry, _QuestionRoutes(AliasRouter(registry, aliases), None))

    classes = {'before': baseline, 'after': SkillBank}
    warm = {name: assistant(cls) for name, cls in classes.items()}
    timings = {name: {'cold': [], 'warm': []} for name in classes}
    for phase, rows in (('cold', seed_rows[:9]), ('warm', seed_rows * 10)):
        for i, messages in enumerate(rows):
            variants = list(classes) if i % 2 == 0 else list(reversed(classes))
            results = {}
            for name in variants:
                entry = assistant(classes[name]) if phase == 'cold' else warm[name]
                started = time.perf_counter_ns()
                results[name] = entry.respond(messages)
                timings[name][phase].append((time.perf_counter_ns() - started) / 1e6)
            if results['before'] != results['after']:
                raise AssertionError('routing changed')
    return {name: {phase: {'turns': len(values), 'p50_ms': statistics.median(values),
                           'p95_ms': sorted(values)[(95 * len(values) + 99) // 100 - 1]}
                   for phase, values in phases.items()} for name, phases in timings.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline_ref')
    parser.add_argument('output', type=Path)
    parser.add_argument('--bank', type=Path, default=BUNDLED)
    parser.add_argument('--aliases', type=Path, default=DEFAULT)
    args = parser.parse_args()
    payload, _ = SkillBank(args.bank)._read()
    aliases = read_aliases(args.aliases)
    entries = {e['id']: e for e in payload['entries']}
    requests = []
    for query, row in sorted(aliases.items())[:6]:
        messages = next(r['messages'] for r in entries[row['program_id']]['qualification'] if r['answer'] is not None)
        requests.extend((messages, [*messages[:-1], {**messages[-1], 'content': query}],
                         [{**messages[-1], 'content': query}]))
    if len(requests) != 18:
        raise ValueError('benchmark requires at least six approved aliases')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'baseline.py'
        path.write_bytes(subprocess.check_output(['git', 'show',
            args.baseline_ref + ':scripts/assistant-runtime/skill_bank.py'], cwd=ROOT))
        spec = importlib.util.spec_from_file_location('baseline_bank', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report = dict(baseline_ref=args.baseline_ref, bank_revision=payload['revision'],
                      aliases=len(aliases), measurements=measure(module.SkillBank, args.bank, args.aliases, requests))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
