"""Promotion failures must not erase every short learning trajectory."""
import copy
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/transfer-training'))
from continual_learning import record_decision,evaluation_due
from run import selection_needed


class ExplorationTests(unittest.TestCase):
    def test_large_evaluation_interval_survives_consolidation_offset(self):
        counters=dict(conversation_updates=4160,last_development_update=4160)
        counters['conversation_updates']=5120
        self.assertFalse(evaluation_due(counters,1024))
        counters['conversation_updates']=5184
        self.assertTrue(evaluation_due(counters,1024))
    def setUp(self):
        self.cfg=dict(evaluation_updates=64,trial_updates=512,
                      maximum_rejections=3,maximum_evaluations=48)
        self.state=dict(evaluations=0,consecutive_rejections=0,accepted_updates=0,stopped=False)

    def test_pending_trial_survives_serialized_job_boundary(self):
        for step in range(64,512,64):
            decision=record_decision(self.state,self.cfg,step,False)
            self.assertTrue(decision['exploration_pending'])
            self.assertFalse(decision['rollback'])
            self.assertEqual(self.state['consecutive_rejections'],0)
            self.state=json.loads(json.dumps(self.state))
        decision=record_decision(self.state,self.cfg,512,False)
        self.assertTrue(decision['rollback'])
        self.assertEqual(self.state['consecutive_rejections'],1)
        self.assertFalse(self.state['stopped'])
        self.assertTrue(record_decision(self.state,self.cfg,576,False)['exploration_pending'])

    def test_improvement_commits_and_starts_new_horizon(self):
        record_decision(self.state,self.cfg,64,False)
        decision=record_decision(self.state,self.cfg,128,True)
        self.assertFalse(decision['rollback'])
        self.assertFalse(decision['exploration_pending'])
        self.assertEqual(self.state['trial_start_update'],128)
        self.assertEqual(self.state['accepted_updates'],1)
        self.assertTrue(record_decision(self.state,self.cfg,512,False)['exploration_pending'])

    def test_compute_or_evaluation_limit_restores_champion(self):
        self.assertTrue(record_decision(self.state,self.cfg,16,False,force=True)['rollback'])
        self.cfg['maximum_evaluations']=2
        decision=record_decision(self.state,self.cfg,32,False)
        self.assertTrue(decision['rollback'])
        self.assertTrue(self.state['stopped'])
        self.assertFalse(decision['exploration_pending'])

    def test_same_step_finalization_cannot_leave_unapproved_candidate(self):
        c=dict(completed_steps=64,selected_at_step=64,exploration_pending=True)
        self.assertTrue(selection_needed(c,force=True))
        c['exploration_pending']=False
        self.assertFalse(selection_needed(c,force=True))

    def test_legacy_spec_keeps_previous_finite_behavior(self):
        self.cfg.pop('trial_updates')
        self.assertTrue(record_decision(self.state,self.cfg,64,False)['rollback'])

    def test_completed_trials_eventually_stop(self):
        for step in (512,1024,1536):
            self.assertTrue(record_decision(self.state,self.cfg,step,False)['rollback'])
        self.assertTrue(self.state['stopped'])
        self.assertEqual(self.state['rejected_trials'],3)

    def test_invalid_ledger_rejected_without_mutation(self):
        self.state['trial_start_update']=64
        before=copy.deepcopy(self.state)
        with self.assertRaises(ValueError):record_decision(self.state,self.cfg,32,False)
        self.assertEqual(self.state,before)


if __name__=='__main__':unittest.main()
