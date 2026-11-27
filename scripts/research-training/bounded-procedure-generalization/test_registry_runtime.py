"""Regression checks for the experiment's SQLite admission and execution contract."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
from threading import Barrier
import unittest

from learner import execute_program, validate_program
from registry import ProgramRegistry, RegistryError


PROGRAM = {'schema': 1, 'steps': [{'op': 'strip'}, {'op': 'lower'}]}
TRAIN = [{'input': ' ALFA', 'expected': 'alfa'}, {'input': '\tBeta ', 'expected': 'beta'}]
CHECKS = [
    {'input': ' GAMMA ', 'expected': 'gamma'},
    {'input': 'Dèlta ', 'expected': 'dèlta'},
    {'input': ' ÊTA ', 'expected': 'êta'},
    {'input': [4, 7], 'expected': None},
]


class RegistryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'test.sqlite'

    def open(self, runtime='test-v1'):
        return ProgramRegistry(self.path, runtime_id=runtime,
                               validate_program=validate_program, execute_program=execute_program)

    def test_rejected_candidate_is_never_executable_after_reopen(self):
        bank = self.open()
        checks = [{**CHECKS[0], 'expected': 'incorretto'}, *CHECKS[1:]]
        result = bank.acquire('normalizzazione', PROGRAM, TRAIN, checks, expected_revision=0)
        self.assertEqual(result, {'status': 'rejected', 'reason': 'candidate_mismatch', 'revision': 0})
        self.assertEqual(self.open().assess('normalizzazione', ' SIGMA ')['status'], 'unknown_skill')

    def test_new_admission_replays_previous_cases_and_preserves_provenance(self):
        bank = self.open()
        first = bank.acquire('primo', PROGRAM, TRAIN, CHECKS, expected_revision=0)
        self.assertEqual(first['status'], 'activated')
        reopened = self.open()
        second = reopened.acquire('secondo', PROGRAM, TRAIN, CHECKS, expected_revision=1)
        self.assertEqual(second['status'], 'activated')
        self.assertEqual(second['replayed_cases'], len(CHECKS))
        result = self.open().assess('primo', ' SIGMA ')
        self.assertEqual(result['answer'], 'sigma')
        self.assertEqual(result['revision'], 2)
        self.assertEqual(result['provenance'], first['provenance'])
        self.assertEqual(self.open().assess('primo', [4, 7])['status'], 'review')

    def test_only_one_writer_can_activate_with_same_expected_revision(self):
        self.open()
        ready = Barrier(2)

        def attempt(task):
            bank = self.open()
            ready.wait(timeout=5)
            return task, bank.acquire(task, PROGRAM, TRAIN, CHECKS, expected_revision=0)

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = dict(pool.map(attempt, ('alfa', 'beta')))
        self.assertCountEqual([result['status'] for result in results.values()], ['activated', 'rejected'])
        bank = self.open()
        for task, result in results.items():
            current = bank.assess(task, ' SIGMA ')
            self.assertEqual(current['revision'], 1)
            if result['status'] == 'activated':
                self.assertEqual(current['answer'], 'sigma')
            else:
                self.assertEqual(result['reason'], 'stale_revision')
                self.assertEqual(current['status'], 'unknown_skill')

    def test_runtime_binding_and_digest_corruption_fail_closed(self):
        bank = self.open()
        self.assertEqual(bank.acquire('primo', PROGRAM, TRAIN, CHECKS)['status'], 'activated')
        with self.assertRaisesRegex(RegistryError, 'different runtime'):
            self.open('test-v2')
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE bounded_registry SET payload=payload || ' ' WHERE id=1")
        with self.assertRaisesRegex(RegistryError, 'digest mismatch'):
            self.open()


if __name__ == '__main__':
    unittest.main()
