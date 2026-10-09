from pathlib import Path
import sys,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/noon-training'))
from foundation_data import build_foundation
from policy import improved
from test_policy import metrics
import run
class FoundationTests(unittest.TestCase):
 def test_fixed_heldout_split_and_answers(self):
  train,test=build_foundation()
  self.assertEqual(len(train),884);self.assertEqual(len(test),16)
  first=lambda r:r['messages'][0]['content'].casefold()
  self.assertFalse({first(r) for r in train}&{first(r) for r in test})
  self.assertEqual({r['language'] for r in test},{'it','en'})
  self.assertTrue(all(r['messages'][-1]['content']==r['expected'] for r in train+test))
 def test_regression_on_new_skill_is_rejected(self):
  a=metrics();b=metrics();a['foundation_correct']=4;b['foundation_correct']=3
  b['nll']=dict(it=.8,en=.8);self.assertFalse(improved(a,b))
 def test_absolute_cutoff_and_nonfinite_clock_budget(self):
  counters=dict(elapsed_training_seconds=1.,phase=0,completed_steps=run.SPEC['start_step'],conversation_updates=0)
  with patch.object(run.time,'time',return_value=run.SPEC['training_cutoff_epoch']-1):self.assertTrue(run.needs_training(counters))
  with patch.object(run.time,'time',return_value=run.SPEC['training_cutoff_epoch']):self.assertFalse(run.needs_training(counters))
  counters['elapsed_training_seconds']=float('nan')
  self.assertRaises(ValueError,run.needs_training,counters)
if __name__=='__main__':unittest.main()
