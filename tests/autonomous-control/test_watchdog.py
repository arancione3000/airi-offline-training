import copy
import importlib.util
import unittest
from pathlib import Path

path=Path(__file__).resolve().parents[2]/'scripts/autonomous-control/watchdog.py'
spec=importlib.util.spec_from_file_location('watchdog',path)
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)

class FakeClient:
    def __init__(self):
        self.run={'id':10,'path':'.github/workflows/'+w.WORKFLOW,'head_branch':w.BRANCH,
                  'head_sha':w.COMMIT,'status':'completed','conclusion':'failure','run_attempt':1}
        self.latest=10;self.active=[];self.sha=w.COMMIT;self.log='gh: HTTP 500: Internal Server Error'
        self.retried=[]
    def get(self,path):
        if path.startswith('git/'):return {'object':{'sha':self.sha}}
        if path.startswith('actions/workflows/'):return {'workflow_runs':[{**self.run,'id':self.latest}]}
        return copy.deepcopy(self.run)
    def pages(self,path,key):
        if key=='jobs':return iter([{'id':11,'conclusion':'failure'}])
        return iter(self.active)
    def logs(self,job):return self.log
    def rerun(self,run):self.retried.append(run)

class WatchdogTests(unittest.TestCase):
    def test_transient_failure_retries_same_run(self):
        client=FakeClient();self.assertEqual(w.recover(client,10)['action'],'retry submitted');self.assertEqual(client.retried,[10])
    def test_integrity_error_never_retried_even_with_http_500(self):
        client=FakeClient();client.log+='\nfingerprint mismatch';self.assertEqual(w.recover(client,10)['action'],'none')
    def test_success_quality_stop_and_user_cancellation_never_retried(self):
        for conclusion in ['success','cancelled']:
            client=FakeClient();client.run['conclusion']=conclusion;self.assertEqual(w.recover(client,10)['action'],'none')
    def test_active_training_prevents_duplicate(self):
        client=FakeClient();client.active=[{'path':next(iter(w.TRAINING_PATHS))}];self.assertEqual(w.recover(client,10)['action'],'none')
    def test_superseded_and_changed_ref_not_retried(self):
        for key,value in [('latest',12),('sha','other')]:
            client=FakeClient();setattr(client,key,value);self.assertEqual(w.recover(client,10)['action'],'none')
    def test_retry_budget_and_dry_run(self):
        client=FakeClient();client.run['run_attempt']=3;self.assertEqual(w.recover(client,10)['action'],'none')
        client=FakeClient();self.assertEqual(w.recover(client,10,dry_run=True)['action'],'would retry');self.assertEqual(client.retried,[])
    def test_wrong_experiment_rejected(self):
        client=FakeClient();client.run['head_sha']='old';self.assertEqual(w.recover(client,10)['action'],'none')
    def test_workflow_paths_with_ref_suffix(self):
        client=FakeClient();client.run['path']+='@'+w.BRANCH
        self.assertEqual(w.recover(client,10,dry_run=True)['action'],'would retry')
        client.active=[{'path':'.github/workflows/'+w.WORKFLOW+'@'+w.BRANCH}]
        self.assertEqual(w.recover(client,10)['action'],'none')
    def test_afternoon_deadline_prevents_restart(self):
        from unittest.mock import patch
        client=FakeClient()
        workflow='airi-afternoon-background.yml'
        branch,sha,cutoff=w.EXPERIMENTS[workflow]
        client.run.update(path='.github/workflows/'+workflow,head_branch=branch,head_sha=sha)
        client.sha=sha
        with patch.object(w.time,'time',return_value=cutoff+1):
            self.assertEqual(w.recover(client,10)['action'],'none')
            self.assertEqual(client.retried,[])
        with patch.object(w.time,'time',return_value=cutoff-1):
            self.assertEqual(w.recover(client,10)['action'],'retry submitted')

    def test_deterministic_code_error_is_not_transient(self):
        self.assertFalse(w.transient('ValueError: checkpoint missing'))
        for code in [429,500,502,503,504]:self.assertTrue(w.transient(f'HTTP {code}'))

if __name__=='__main__':unittest.main()
