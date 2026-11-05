import copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/transfer-training'))
import continuous_training as service

class EpochTests(unittest.TestCase):
 def setUp(self):
  self.spec=json.loads((ROOT/'scripts/continuous-training/SPEC.json').read_text())
  self.c=dict(completed_steps=450364,conversation_updates=1088,elapsed_training_seconds=478.,phase=0,
   learning_search=dict(evaluations=4,accepted_updates=0,rejected_trials=1,consecutive_rejections=1,stopped=False,history=[]),
   retention_consolidation_updates=64,retention_transaction=dict(attempted_updates=1094,committed_updates=1045,rejected_updates=49),
   morph=dict(revision=2,weights_digest='a'*64),exploration_pending=False,supervised_tokens=123,
   retention_memory=dict(rows=518),training_registry={'known':'kept'},success_bank={'donor':'kept'})
 def init(self):service.initialize(self.c,self.spec,'a'*64)
 def test_migration_never_resets_lifetime_or_knowledge(self):
  before=copy.deepcopy(self.c);self.init()
  for key in ('completed_steps','conversation_updates','elapsed_training_seconds','retention_transaction','retention_memory','training_registry','success_bank','supervised_tokens'):
   self.assertEqual(self.c[key],before[key])
  self.assertEqual(service.limits(self.c,self.spec),dict(update=5184,seconds=4078.,evaluations=16,consolidation=64))
 def test_many_epochs_no_end_date_and_bounded_audit(self):
  self.init();now=1000
  for epoch in range(70):
   bound=service.limits(self.c,self.spec)
   self.c['conversation_updates']=bound['update'];self.c['completed_steps']=449276+bound['update']
   self.c['learning_search']['evaluations']=bound['evaluations']
   self.c['elapsed_training_seconds']=bound['seconds']
   self.assertFalse(service.can_train(self.c,self.spec))
   service.close_epoch(self.c,self.spec,'a'*64,now)
   frozen=copy.deepcopy(self.c['continuous_training'])
   service.close_epoch(self.c,self.spec,'a'*64,now+1)
   self.assertEqual(self.c['continuous_training'],frozen)
   if frozen['not_before']>now:
    with self.assertRaises(ValueError):service.open_epoch(self.c,self.spec,'a'*64,now)
   now=frozen['not_before'];self.assertTrue(service.open_epoch(self.c,self.spec,'a'*64,now))
   self.assertTrue(service.can_train(self.c,self.spec));self.assertEqual(self.c['continuous_training']['epoch'],epoch+1)
   self.assertEqual(self.c['retention_memory']['rows'],518)
  self.assertLessEqual(len(self.c['continuous_training']['history']),32)
  self.assertEqual(self.c['retention_transaction']['attempted_updates'],1094)
 def test_unapproved_candidate_cannot_open_or_close_epoch(self):
  with self.assertRaises(ValueError):service.initialize(self.c,self.spec,'b'*64)
  self.init();self.c['exploration_pending']=True
  with self.assertRaises(ValueError):service.close_epoch(self.c,self.spec,'a'*64,10)
 def test_profile_rotation_and_lifetime_decisions(self):
  self.init();a=service.profile_index(self.c,self.spec)
  self.c['conversation_updates']+=16;self.c['completed_steps']+=16
  service.close_epoch(self.c,self.spec,'a'*64,100)
  service.open_epoch(self.c,self.spec,'a'*64,self.c['continuous_training']['not_before'])
  self.assertNotEqual(service.profile_index(self.c,self.spec),a)
  self.assertEqual(service.decision_configuration(self.c,self.spec)['maximum_evaluations'],16)
 def test_corrupt_or_regressed_epoch_is_rejected(self):
  self.init();self.c['elapsed_training_seconds']=0
  with self.assertRaises(ValueError):service.validate_state(self.c,self.spec)
 def test_registry_and_histories_have_finite_caps(self):
  self.init();self.c['reward_history']=[{'x':n} for n in range(100)]
  self.c['conversation_updates']+=16;self.c['completed_steps']+=16
  service.close_epoch(self.c,self.spec,'a'*64,100)
  self.assertEqual(len(self.c['reward_history']),64)
  self.assertEqual(self.c['continuous_training']['receipts']['reward_history']['removed'],36)
  self.assertEqual(self.spec['continuous_training']['registry_capacity'],8192)

if __name__=='__main__':unittest.main()
