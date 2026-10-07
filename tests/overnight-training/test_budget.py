import importlib.util,sys,unittest,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'scripts/overnight-training'
spec=importlib.util.spec_from_file_location('night_run',HERE/'run.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
class BudgetTests(unittest.TestCase):
    def counters(self,seconds):
        return dict(elapsed_training_seconds=seconds,phase=2,completed_steps=21399,conversation_updates=100,stagnant_rounds=999)
    def test_minimum_fresh_compute(self):
        self.assertTrue(run.needs_training(self.counters(35999.999)))
        self.assertFalse(run.needs_training(self.counters(36000)))
    def test_stagnation_does_not_stop_budget(self):
        self.assertTrue(run.needs_training(self.counters(0)))
    def test_invalid_durations(self):
        for value in [-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):run.needs_training(self.counters(value))
    def test_counter_integrity(self):
        c=self.counters(10);c['completed_steps']+=1
        with self.assertRaises(ValueError):run.needs_training(c)
    def test_frozen_scripts(self):
        for name,digest in run.SPEC['script_sha256'].items():
            self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).hexdigest(),digest)
        raw=dict(run.SPEC);fingerprint=raw.pop('fingerprint')
        self.assertEqual(hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest(),fingerprint)
    def test_release_prediction(self):
        import os
        sys.path.insert(0,str(HERE));import transport
        os.environ['GITHUB_RUN_ID']='123';os.environ['GITHUB_RUN_ATTEMPT']='1'
        pub=transport.ReleasePublisher(Path('/tmp/unused-night'))
        self.assertEqual(pub.release_tag(21399),'airi-offline-21399-123-1-1')
