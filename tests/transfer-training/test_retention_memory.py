"""Persistence and coverage of certified TRAIN memory, without native imports."""
import copy
import json
import random
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/transfer-training'))

from success_bank import catalog, row_key
from retention_memory import (empty_memory, validate_memory, migrate_memory,
                              refresh, observe, sample_rows, summary)


FINGERPRINT = 'experiment-one'
CHAMPION = 'a' * 64
NEXT_CHAMPION = 'b' * 64


def row(index, language='en', family='foundation-copy', bucket=0, history=0):
    answer = 'value' + str(index)
    messages = [dict(role='user', content=f'{language} {family} literal {index}')]
    for exchange in range(history):
        messages += [dict(role='assistant', content='Recorded.'),
                     dict(role='user', content=f'Pause {exchange}.')]
    messages += [dict(role='assistant', content=answer)]
    return dict(messages=messages, expected=answer, language=language,
                family=family, length_bucket=bucket, history_exchanges=history,
                provenance='deterministic exact TRAIN solver')


def certificate(correct=True, margin=None, nll=0.1):
    margin = (3.0 if correct else -0.2) if margin is None else margin
    return dict(correct=correct, complete=True, includes_eos=True,
                token_count=7, min_margin=margin, mean_nll=nll)


def correct(_):
    return certificate()


def diverse_rows():
    rows = []
    for language in ('it', 'en'):
        for family in ('context-name', 'context-location', 'context-topic',
                       'context-correction', 'foundation-copy', 'foundation-recall'):
            for bucket in range(3):
                for history in range(2):
                    rows.append(row(len(rows), language, family, bucket, history))
    return rows


class RetentionMemoryTests(unittest.TestCase):
    def build(self, rows=None, capacity=1024, limit=256):
        eligible = catalog(rows or diverse_rows(), [])
        memory = refresh(CHAMPION, eligible, correct,
                         empty_memory(FINGERPRINT, capacity=capacity), probe_limit=limit)
        return memory, eligible

    def test_import_keeps_native_torch_dependency_lazy(self):
        directory = str(Path(__file__).resolve().parents[2] / 'scripts/transfer-training')
        program = ('import sys; sys.path.insert(0, ' + repr(directory) + '); '
                   'import retention_memory; assert "torch" not in sys.modules')
        result = subprocess.run([sys.executable, '-c', program], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_json_restart_and_exact_champion_validation(self):
        memory, eligible = self.build()
        restored = json.loads(json.dumps(memory, allow_nan=False))
        validate_memory(restored, eligible, FINGERPRINT, CHAMPION)
        self.assertEqual(restored, memory)
        self.assertEqual(restored['base_champion_sha256'], CHAMPION)
        self.assertRaises(ValueError, validate_memory, restored, eligible,
                          FINGERPRINT, NEXT_CHAMPION)
        self.assertRaises(ValueError, validate_memory, restored, eligible, 'wrong')
        self.assertGreaterEqual(len(summary(restored)), 1)

    def test_context_copy_and_recall_have_no_family_whitelist(self):
        memory, eligible = self.build()
        actual = [eligible[key] for key in memory['entries']]
        self.assertEqual({r['language'] for r in actual}, {'it', 'en'})
        self.assertEqual({r['family'] for r in actual},
                         {r['family'] for r in eligible.values()})
        self.assertEqual({r['length_bucket'] for r in actual}, {0, 1, 2})
        self.assertEqual({r['history_exchanges'] for r in actual}, {0, 1})

    def test_bounded_probe_rotates_and_eventually_visits_every_eligible_row(self):
        eligible = catalog(diverse_rows(), [])
        memory = empty_memory(FINGERPRINT)
        visits = []
        def certify(r):
            visits.append(row_key(r))
            return certificate(correct=False)
        for _ in range(24):
            start = len(visits)
            memory = refresh(CHAMPION, eligible, certify, memory, probe_limit=8)
            self.assertLessEqual(len(visits) - start, 8)
        self.assertEqual(set(visits), set(eligible))
        self.assertGreater(memory['probe_cursor'], 0)

    def test_capacity_never_evicts_certified_knowledge(self):
        rows = diverse_rows()
        eligible = catalog(rows[:8], [])
        memory = refresh(CHAMPION, eligible, correct,
                         empty_memory(FINGERPRINT, capacity=8), probe_limit=128)
        old_keys = set(memory['entries'])
        all_eligible = catalog(rows, [])
        memory = refresh(NEXT_CHAMPION, all_eligible, correct, memory, probe_limit=128)
        self.assertEqual(set(memory['entries']), old_keys)
        self.assertEqual(len(memory['entries']), 8)
        self.assertTrue(memory['capacity_exhausted'])
        self.assertGreater(memory['new_skipped'], 0)

    def test_refresh_recertifies_every_old_row_and_fails_atomically(self):
        memory, eligible = self.build(limit=128)
        before = copy.deepcopy(memory)
        lost = next(iter(memory['entries']))
        def certify(r):
            return certificate(correct=row_key(r) != lost)
        self.assertRaises(ValueError, refresh, NEXT_CHAMPION, eligible, certify,
                          memory, probe_limit=1)
        self.assertEqual(memory, before)
        visited = []
        def count(r):
            visited.append(row_key(r))
            return certificate()
        updated = refresh(NEXT_CHAMPION, eligible, count, memory, probe_limit=1)
        self.assertTrue(set(memory['entries']) <= set(visited))
        self.assertEqual(set(updated['entries']), set(memory['entries']))
        self.assertEqual(memory, before)

    def test_migration_preserves_all_entries_and_rejects_removed_training_rows(self):
        memory, eligible = self.build()
        migrated = migrate_memory(memory, eligible, FINGERPRINT, 'experiment-two', CHAMPION)
        validate_memory(migrated, eligible, 'experiment-two', CHAMPION)
        self.assertEqual(set(migrated['entries']), set(memory['entries']))
        self.assertEqual(memory['fingerprint'], FINGERPRINT)
        missing = dict(eligible)
        missing.pop(next(iter(memory['entries'])))
        self.assertRaises(ValueError, migrate_memory, memory, missing,
                          FINGERPRINT, 'experiment-two', CHAMPION)
        self.assertRaises(ValueError, migrate_memory, memory, eligible,
                          'incorrect-source', 'experiment-two', CHAMPION)

    def test_exact_eligible_row_and_provenance_cannot_be_replaced_on_reload(self):
        memory, eligible = self.build()
        key = next(iter(memory['entries']))
        replaced = copy.deepcopy(eligible)
        replaced[key]['provenance'] = 'student generated answer'
        self.assertRaises(ValueError, validate_memory, memory, replaced, FINGERPRINT, CHAMPION)
        forged = copy.deepcopy(memory)
        forged['entries'][key]['row']['messages'][0]['content'] = 'foreign or evaluation prompt'
        self.assertRaises(ValueError, validate_memory, forged, eligible, FINGERPRINT, CHAMPION)

    def test_only_explicit_eligible_catalog_can_register_a_row(self):
        a, b = row(101), row(102)
        eligible = catalog([a, b], [b])
        self.assertNotIn(row_key(b), eligible)
        memory = refresh(CHAMPION, eligible, correct,
                         empty_memory(FINGERPRINT), probe_limit=128)
        self.assertEqual(set(memory['entries']), {row_key(a)})
        foreign = dict(eligible)
        bad = row(103)
        bad['provenance'] = 'student generated answer'
        foreign[row_key(bad)] = bad
        self.assertRaises(ValueError, refresh, CHAMPION, foreign, correct,
                          memory, probe_limit=128)

    def test_incomplete_or_nonfinite_certificates_cannot_be_registered(self):
        eligible = catalog([row(100)], [])
        for observation in (True, {**certificate(), 'complete': False},
                            {**certificate(), 'includes_eos': False},
                            {**certificate(), 'token_count': 0},
                            {**certificate(), 'min_margin': float('nan')},
                            {**certificate(), 'mean_nll': float('inf')}):
            memory = empty_memory(FINGERPRINT)
            before = copy.deepcopy(memory)
            self.assertRaises(ValueError, refresh, CHAMPION, eligible,
                              lambda r, value=observation: value, memory, probe_limit=1)
            self.assertEqual(memory, before)

    def test_sampler_restarts_exactly_and_rotates_coverage(self):
        memory, eligible = self.build()
        restored = json.loads(json.dumps(memory))
        rng = random.Random(510)
        resumed = random.Random(510)
        seen = set()
        for _ in range(80):
            sampled = sample_rows(memory, eligible, rng, per_language=2, uniform_fraction=.25)
            again = sample_rows(restored, eligible, resumed, per_language=2, uniform_fraction=.25)
            self.assertEqual(sampled, again)
            self.assertEqual(len(sampled), 4)
            self.assertEqual({r['language'] for r in sampled}, {'it', 'en'})
            self.assertTrue(all(eligible[row_key(r)] == r for r in sampled))
            seen.update((r['language'], r['family'], r['length_bucket'], r['history_exchanges'])
                        for r in sampled)
        expected = {(r['language'], r['family'], r['length_bucket'], r['history_exchanges'])
                    for r in eligible.values()}
        self.assertEqual(seen, expected)
        self.assertEqual(memory, restored)

    def test_observed_current_degradation_gets_sampling_priority(self):
        # One coverage stratum per language removes a confound from family rotation.
        rows = [row(i, language) for language in ('it', 'en') for i in range(8)]
        memory, eligible = self.build(rows)
        hard = {language: row_key(next(r for r in rows if r['language'] == language))
                for language in ('it', 'en')}
        def degraded(r):
            return certificate(correct=False, margin=-2.0, nll=3.0) if row_key(r) in hard.values() else certificate()
        observed = observe(memory, eligible, degraded)
        if observed is not None:
            memory = observed
        draws = sample_rows(memory, eligible, random.Random(2), per_language=2, uniform_fraction=0.)
        keys = [row_key(r) for r in draws]
        self.assertTrue(set(hard.values()) <= set(keys))

    def test_finite_capacity_and_probe_configuration(self):
        for bad in (0, -1, 8193, True, 2.5):
            self.assertRaises(ValueError, empty_memory, FINGERPRINT, bad)
        eligible = catalog([row(1)], [])
        for bad in (-1, True, 8193, 2.5):
            self.assertRaises(ValueError, refresh, CHAMPION, eligible, correct,
                              empty_memory(FINGERPRINT), probe_limit=bad)

    def test_unprobed_foreign_catalog_rows_are_rejected(self):
        memory, eligible = self.build()
        bad = row(999)
        bad['provenance'] = 'student generated answer'
        forged = dict(eligible); forged[row_key(bad)] = bad
        before = copy.deepcopy(memory)
        self.assertRaises(ValueError, refresh, NEXT_CHAMPION, forged, correct,
                          memory, probe_limit=0)
        self.assertEqual(memory, before)

    def test_boolean_schema_is_not_a_version_number(self):
        memory, eligible = self.build()
        memory['schema'] = True
        self.assertRaises(ValueError, validate_memory, memory, eligible, FINGERPRINT)

    def test_single_replay_slot_cannot_starve_coverage_when_one_row_degrades(self):
        memory, eligible = self.build()
        hard = {language: next(key for key, value in memory['entries'].items()
                              if value['row']['language'] == language)
                for language in ('it', 'en')}
        memory = observe(memory, eligible, lambda r: certificate(correct=row_key(r) not in hard.values()))
        seen = set()
        for _ in range(160):
            sampled = sample_rows(memory, eligible, random.Random(44),
                                  per_language=1, uniform_fraction=0.)
            seen.update((r['language'], r['family'], r['length_bucket'], r['history_exchanges'])
                        for r in sampled)
        expected = {(r['language'], r['family'], r['length_bucket'], r['history_exchanges'])
                    for r in eligible.values()}
        self.assertEqual(seen, expected)


if __name__ == '__main__':
    unittest.main()
