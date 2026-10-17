import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/continual-training'))
import control as c


class Fake:
    def __init__(self):
        self.calls=[];self.disabled=False;self.active=[];self.head='a'*40;self.log=''
        self.runs=[dict(id=12,status='completed',conclusion='success',head_branch=c.BRANCH,
                        display_title=c.TITLE,head_sha=self.head)]
        self.state={'fingerprint':c.SPEC['fingerprint'],'completed_steps':c.SPEC['start_step']+16,
                    'elapsed_training_seconds':60.,'maximum_training_seconds':c.SPEC['maximum_training_seconds'],
                    'continue':True,'job_index':0,'model_id':'airi-native','revision':0,
                    'search':{'stopped':False},'production_qualified':False,'live_promoted':False}
        self.raw={};self.release=None;self.rebuild()
    def rebuild(self):
        step=self.state['completed_steps'];name=f'AIRI_TRAINING_CHECKPOINT_{step}.json'
        state=json.dumps(self.state).encode()
        descriptor=dict(name='AIRI_RUN_STATE.json',bytes=len(state),sha256=hashlib.sha256(state).hexdigest())
        archives=[dict(name=f'AIRI_TRAINING_{step}_{part}.zip',bytes=10,sha256=part[0].lower().encode().hex().ljust(64,'a')) for part in ('MODEL','OPTIMIZER')]
        index=json.dumps(dict(fingerprint=c.SPEC['fingerprint'],completed_steps=step,archives=archives,
                              run_state=descriptor,production_qualified=False,live_promoted=False)).encode()
        self.raw={name:index,'AIRI_RUN_STATE.json':state}
        assets=[dict(name=n,id=i,size=len(b),digest='sha256:'+hashlib.sha256(b).hexdigest()) for i,(n,b) in enumerate(self.raw.items())]
        assets += [dict(name=a['name'],size=a['bytes'],digest='sha256:'+a['sha256']) for a in archives]
        self.release=dict(tag_name=f'airi-offline-{step}-12-1-1',target_commitish=self.head,draft=False,prerelease=True,assets=assets)
    def get(self,path):
        if path.startswith('actions/workflows/'):return {'workflow_runs':self.runs}
        if path.startswith('git/ref/'):return {'object':{'sha':self.head}}
        if path.startswith('releases?'):return [self.release] if self.release else []
        if path.startswith('actions/runs?'):return {'workflow_runs':self.active}
        raise AssertionError(path)
    def asset(self,item):return self.raw[item['name']]
    def logs(self,run):return self.log
    def post(self,path,payload):self.calls.append((path,payload))
    def disable(self):self.disabled=True


class ControlTests(unittest.TestCase):
    def test_resume_exact_committed_checkpoint(self):
        client=Fake();result=c.tick(client)
        self.assertEqual(result['action'],'resumed_verified_checkpoint')
        self.assertEqual(client.calls[0][1]['inputs'],{'checkpoint_release':client.release['tag_name'],'continuation':'1'})
    def test_terminal_budget_or_stagnation_disables_controller(self):
        for stopped,seconds in [(True,100.),(False,c.SPEC['maximum_training_seconds']+.1)]:
            client=Fake();client.state.update({'continue':False,'elapsed_training_seconds':seconds})
            client.state['search']['stopped']=stopped;client.rebuild()
            self.assertEqual(c.tick(client)['action'],'finite_experiment_stopped');self.assertTrue(client.disabled)
            self.assertFalse(client.calls)
    def test_active_training_cancellation_and_changed_branch_are_respected(self):
        client=Fake();client.runs[0]['status']='in_progress'
        self.assertEqual(c.tick(client)['action'],'active_job_retained');self.assertFalse(client.calls)
        client=Fake();client.runs[0]['conclusion']='cancelled'
        self.assertEqual(c.tick(client)['action'],'cancellation_respected');self.assertTrue(client.disabled)
        client=Fake();client.head='b'*40
        self.assertEqual(c.tick(client)['action'],'changed_branch_stopped');self.assertFalse(client.calls)
    def test_corruption_wrong_fingerprint_nan_and_stop_contradiction_fail_closed(self):
        client=Fake();client.raw['AIRI_RUN_STATE.json']+=b' '
        self.assertRaises(ValueError,c.tick,client)
        for change in [{'fingerprint':'b'*64},{'elapsed_training_seconds':float('nan')},{'continue':False}]:
            client=Fake();client.state.update(change);client.rebuild();self.assertRaises(ValueError,c.tick,client)
        client=Fake();client.release['assets'][-1]['digest']='sha256:'+'b'*64
        self.assertRaises(ValueError,c.tick,client)
    def test_only_bounded_transient_failure_can_resume(self):
        client=Fake();client.runs[0]['conclusion']='failure';client.log='gh: HTTP 503 service unavailable'
        self.assertEqual(c.tick(client)['action'],'resumed_verified_checkpoint')
        client.runs*=3
        self.assertEqual(c.tick(client)['action'],'nontransient_or_retry_limit_stopped')
        client=Fake();client.runs[0]['conclusion']='failure';client.log='digest mismatch'
        self.assertEqual(c.tick(client)['action'],'nontransient_or_retry_limit_stopped');self.assertFalse(client.calls)
    def test_other_training_and_initial_preflight_do_not_start_duplicate_jobs(self):
        client=Fake();client.active=[dict(status='queued',path='.github/workflows/airi-human-background.yml')]
        self.assertEqual(c.tick(client)['action'],'waiting_for_other_training');self.assertFalse(client.calls)
        client=Fake();client.runs[0]['display_title']='AIRI continual learning preflight'
        self.assertEqual(c.tick(client)['action'],'awaiting_initial_start');self.assertFalse(client.calls)
    def test_training_workflow_has_one_owner_for_continuation(self):
        root=Path(__file__).resolve().parents[2]
        workflow=(root/'.github/workflows/airi-continual-background.yml').read_text()
        self.assertNotIn('/dispatches',workflow)
        self.assertIn('airi-offline-background-training',workflow)
        self.assertIn('--learning-smoke-only',workflow)


if __name__=='__main__':unittest.main()
