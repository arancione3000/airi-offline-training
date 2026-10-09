import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/transfer-training'))
from composition_data import build_composition,sampling_pools,SEALED_VALUES,PROTECTED_VALUES
from foundation_data import build_foundation
from diagnostics import diagnostic_rows,recipe_phase
from curriculum import norm
from policy import reward,improved
import run

class TransferTests(unittest.TestCase):
 def test_values_and_prompts_of_final_set_are_sealed(self):
  train,test=build_composition();self.assertEqual(len(train),12288);self.assertEqual(len(test),64)
  targets={norm(r['expected']) for r in train}
  self.assertFalse(targets & {norm(v) for v in SEALED_VALUES+PROTECTED_VALUES})
  prompts=lambda rows:{json.dumps(r['messages'][:-1],sort_keys=True) for r in rows}
  self.assertFalse(prompts(train)&prompts(test))
  self.assertEqual(build_composition(),(train,test))
 def test_tasks_have_balanced_composition_lengths_and_languages(self):
  train,_=build_composition();pools=sampling_pools(train)
  self.assertEqual({len(v) for v in pools.values()},{6})
  self.assertEqual({len(rows) for v in pools.values() for rows in v.values()},{1024})
  self.assertTrue(any(len(r['messages'])>4 for r in train))
  self.assertTrue(any(r['messages'][0]['role']=='system' for r in train))
 def test_probe_is_training_only_and_broad(self):
  train,test=build_composition();rows=diagnostic_rows(train)
  self.assertEqual(len(rows),24)
  self.assertTrue(all(r in train and r not in test for r in rows))
  self.assertEqual(sum(r['language']=='it' for r in rows),12)
 def test_schedule_is_compute_based_and_bounded(self):
  self.assertEqual(recipe_phase(14399),0);self.assertEqual(recipe_phase(14400),1)
  self.assertEqual(recipe_phase(28799),1);self.assertEqual(recipe_phase(28800),2)
  for value in (-1,float('nan'),float('inf')):self.assertRaises(ValueError,recipe_phase,value)
 def test_training_and_final_scores_cannot_select_champion(self):
  m=dict(context_rates={'it:name':0.,'en:name':0.},context_pair_rate=0.,legacy_correct=14,dialogue_correct=0,foundation_correct=0,foundation_probes=16,repetition=0.,nll=dict(it=1.,en=1.))
  changed={**m,'training_diagnostic':{'correct':24,'probes':24},'sealed_transfer':{'correct':64,'probes':64}}
  self.assertEqual(reward(m),reward(changed));self.assertFalse(improved(m,changed))
  self.assertFalse(improved(m,{**changed,'legacy_correct':13,'foundation_correct':16}))
 def test_exact_finite_compute_budget(self):
  c=dict(elapsed_training_seconds=43199,phase=2,completed_steps=run.SPEC['start_step'],conversation_updates=0)
  self.assertTrue(run.needs_training(c));c['elapsed_training_seconds']+=1
  self.assertFalse(run.needs_training(c))
 def test_blocked_examples_do_not_enter_training(self):
  train,test=build_composition();blocked=[norm(train[0]['messages'][-2]['content'])]
  filtered,_=build_composition(blocked)
  self.assertTrue(all(norm(m['content']) not in blocked for r in filtered for m in r['messages']))
if __name__=='__main__':unittest.main()
