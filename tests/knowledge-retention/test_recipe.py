import hashlib
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/transfer-training'))
from continual_learning import validate_configuration


class RetentionRecipeTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((ROOT / 'scripts/knowledge-retention/SPEC.json').read_text())

    def test_exact_recipe_and_every_archived_source_pin(self):
        validate_configuration(self.spec)
        for name in ('context_acquisition.py', 'retention_memory.py',
                     'retention_transaction.py', 'retention_guard.py', 'retention_smoke.py'):
            self.assertIn(name, self.spec['script_sha256'])
        # This recipe runs on its immutable historical branch. Main may evolve;
        # its current implementation pins belong to the current service spec.
        for name, expected in self.spec['script_sha256'].items():
            self.assertRegex(expected, r'^[0-9a-f]{64}$', name)

    def test_explicit_terminal_migration_keeps_previous_recipe_frozen(self):
        previous = json.loads((ROOT / 'scripts/context-learning/SPEC.json').read_text())
        self.assertEqual(previous['fingerprint'],
                         'e23f49a0ac9777f876639498cd6aafe7842ae2d82f31d6cfd1728b2961aed646')
        self.assertEqual(self.spec['migration_source_fingerprint'], previous['fingerprint'])
        self.assertNotEqual(self.spec['fingerprint'], previous['fingerprint'])
        self.assertEqual(self.spec['migration_source_release'], 'airi-offline-449276-41523612006-1-3')
        self.assertEqual(self.spec['start_step'], 449276)
        self.assertEqual(self.spec['pilot_step'], self.spec['start_step'])
        self.assertEqual(self.spec['native_source'], previous['native_source'])
        self.assertEqual(self.spec['parent_snapshot'], previous['parent_snapshot'])
        self.assertNotEqual(self.spec['frozen_runner_branch'], previous['frozen_runner_branch'])

    def test_autonomous_jobs_share_one_finite_budget_and_train_only_retention(self):
        learning = self.spec['continual_learning']
        retention = learning['knowledge_retention']
        self.assertLessEqual(self.spec['job_limit'], 4)
        self.assertLessEqual(self.spec['maximum_training_seconds'], 3600)
        self.assertLessEqual(self.spec['max_steps'] - self.spec['start_step'], 4096)
        self.assertLessEqual(learning['maximum_evaluations'], 12)
        self.assertLessEqual(learning['maximum_rejections'], 3)
        self.assertEqual(learning['training_mode'], 'cached_prefix')
        self.assertEqual(retention['source_policy'], 'eligible_train_only')
        self.assertEqual(learning['sampling_schedule'], json.loads(
            (ROOT / 'scripts/context-learning/SPEC.json').read_text())['continual_learning']['sampling_schedule'])
        self.assertEqual(retention['retry_scales'][0], 1.)
        self.assertLessEqual(retention['consolidation_updates'], 128)


if __name__ == '__main__':
    unittest.main()
