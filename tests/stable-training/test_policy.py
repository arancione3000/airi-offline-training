import json,hashlib,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'scripts/stable-training'
sys.path.insert(0,str(HERE))
import run,policy
class StableTests(unittest.TestCase):
    def metrics(self):return dict(legacy_correct=10,exact_correct=10,nll=dict(it=1.,en=1.),repetition=.01)
    def test_fluency_gain_cannot_hide_old_response_regression(self):
        b=self.metrics();a=self.metrics();a.update(legacy_correct=9,exact_correct=15,nll=dict(it=.7,en=.7))
        self.assertFalse(policy.improved(b,a))
    def test_both_languages_and_total_responses_are_protected(self):
        b=self.metrics();a=self.metrics();a['nll']=dict(it=.8,en=1.1)
        self.assertFalse(policy.improved(b,a))
        a['nll']=dict(it=.8,en=.8);a['exact_correct']=9
        self.assertFalse(policy.improved(b,a))
        a['exact_correct']=11;self.assertTrue(policy.improved(b,a))
    def test_repetition_rejects_apparent_gain(self):
        b=self.metrics();a=self.metrics();a.update(nll=dict(it=.8,en=.8),repetition=.6)
        self.assertFalse(policy.improved(b,a))
    def test_recovery_and_finite_budget(self):
        self.assertFalse(policy.recover_best(1));self.assertTrue(policy.recover_best(2))
        c=dict(elapsed_training_seconds=21599,phase=2,completed_steps=27444,conversation_updates=1)
        self.assertTrue(run.needs_training(c));c['elapsed_training_seconds']=21600
        self.assertFalse(run.needs_training(c))
    def test_scripts_and_spec_are_frozen(self):
        for name,digest in run.SPEC['script_sha256'].items():self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).hexdigest(),digest)
        d=dict(run.SPEC);fingerprint=d.pop('fingerprint')
        self.assertEqual(hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest(),fingerprint)
