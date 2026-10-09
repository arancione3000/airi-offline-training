import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/focused-training'))
from diagnostics import diagnostic_rows,diagnostic_phase
from foundation_data import build_foundation
from policy import reward,improved
import run
class DiagnosticTests(unittest.TestCase):
 def test_probe_is_training_only_and_balanced(self):
  train,test=build_foundation();rows=diagnostic_rows(train)
  self.assertEqual(len(rows),16)
  self.assertTrue(all(r in train for r in rows));self.assertTrue(all(r not in test for r in rows))
  self.assertEqual(sum(r['language']=='it' for r in rows),8)
  self.assertEqual(sum(r['family']=='foundation-copy' for r in rows),8)
 def test_adaptation_is_bounded_and_does_not_oscillate(self):
  self.assertEqual(diagnostic_phase(0,4095,0,16),0)
  self.assertEqual(diagnostic_phase(0,4096,0,16),1)
  self.assertEqual(diagnostic_phase(1,8192,12,16),2)
  self.assertEqual(diagnostic_phase(2,10000,0,16),2)
  self.assertRaises(ValueError,diagnostic_phase,0,100,17,16)
 def test_in_sample_score_cannot_promote(self):
  m=dict(context_rates={'it:name':0.,'en:name':0.},context_pair_rate=0.,legacy_correct=14,dialogue_correct=0,foundation_correct=0,foundation_probes=16,repetition=0.,nll=dict(it=1.,en=1.))
  changed={**m,'training_diagnostic':{'correct':16,'probes':16}}
  self.assertEqual(reward(m),reward(changed));self.assertFalse(improved(m,changed))
 def test_exact_finite_compute_budget(self):
  c=dict(elapsed_training_seconds=run.SPEC['maximum_training_seconds']-1,phase=0,completed_steps=run.SPEC['start_step'],conversation_updates=0)
  self.assertTrue(run.needs_training(c));c['elapsed_training_seconds']+=1
  self.assertFalse(run.needs_training(c))
if __name__=='__main__':unittest.main()
