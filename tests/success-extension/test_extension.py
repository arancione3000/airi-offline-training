import copy
import importlib.util
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('success_extension',ROOT/'scripts/success-extension/extend.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)

def fixture():
    step=400000
    run=dict(id=123,run_attempt=1,head_sha=e.SOURCE_SHA,head_branch=e.SOURCE_BRANCH,status='completed',conclusion='success')
    release=dict(tag_name='airi-offline-400000-123-1-1',draft=False,prerelease=True,target_commitish=e.SOURCE_SHA,
                 assets=[dict(name='FINAL_RESULT.json',id=1),dict(name='AIRI_TRAINING_CHECKPOINT_400000.json',id=2)])
    result=dict(counters=dict(elapsed_training_seconds=180000.2,completed_steps=step),next_action=None,
                maximum_training_seconds=180000,production_qualified=False,live_promoted=False)
    index=dict(fingerprint=e.SOURCE_FP,completed_steps=step,production_qualified=False,live_promoted=False,
               archives=[dict(name=f'AIRI_TRAINING_{step}_{part}.zip',sha256='a'*64,bytes=123) for part in ['MODEL','OPTIMIZER']])
    return result,index,release,run

class Fake:
    def __init__(self):
        self.result,self.index,self.release,run=fixture();self.source=[run];self.target=[];self.active=[];self.calls=[]
        self.checks=[dict(id=10,display_title='AIRI success transfer preflight',head_sha=e.TEMPLATE_SHA,status='completed',conclusion='success')]
        self.target_sha='b'*40
        self.target_spec={'maximum_training_seconds':3600,'migration_source_release':self.release['tag_name'],
                          'extension':{'request':e.REQUEST,'source_sha':e.SOURCE_SHA}}
    def branch(self):return {'object':{'sha':self.target_sha}} if self.target else None
    def pages(self,path,key=None):
        if 'branch='+e.TEMPLATE_BRANCH in path:return iter(copy.deepcopy(self.checks))
        if 'branch='+e.SOURCE_BRANCH in path:return iter(copy.deepcopy(self.source))
        if 'branch='+e.BRANCH in path:return iter(copy.deepcopy(self.target))
        if path=='releases':return iter([copy.deepcopy(self.release)])
        if 'status=' in path:return iter(copy.deepcopy(self.active))
        raise AssertionError(path)
    def api(self,path,payload=None):
        self.calls.append((path,payload))
        if path.startswith('git/commits/') and payload is None:return {'tree':{'sha':'tree'}}
        if path=='git/trees':return {'sha':'tree2'}
        if path=='git/commits':return {'sha':self.target_sha}
    def content(self,path,ref):
        if ref==self.target_sha:return json.dumps(self.target_spec).encode()
        if path.endswith('SPEC.json'):return json.dumps({'script_sha256':{}}).encode()
        return b''
    def asset(self,item):return copy.deepcopy(self.result if item['id']==1 else self.index)

class ControllerTests(unittest.TestCase):
    def test_source_and_destination_match_authorized_deployment(self):
        self.assertEqual(e.SOURCE_BRANCH,'dialogue-human-extra50h-20261013')
        self.assertEqual(e.BRANCH,'dialogue-success-transfer-20261015')
        self.assertNotEqual(e.SOURCE_BRANCH,e.BRANCH)
        self.assertEqual(e.SOURCE_WORKFLOW,'airi-human-background.yml')
        self.assertEqual(e.WORKFLOW,'airi-success-background.yml')

    def test_source_completion_rejects_invalid_provenance_and_partial_budget(self):
        for change in ['partial','nan','infinite','continued','fp','duplicate','wrongsha','live','draft']:
            result,index,release,run=fixture()
            if change=='partial':result['counters']['elapsed_training_seconds']=179999
            if change=='nan':result['counters']['elapsed_training_seconds']=float('nan')
            if change=='infinite':result['counters']['elapsed_training_seconds']=float('inf')
            if change=='continued':result['next_action']={}
            if change=='fp':index['fingerprint']='wrong'
            if change=='duplicate':index['archives'][1]=index['archives'][0]
            if change=='wrongsha':run['head_sha']='wrong'
            if change=='live':result['live_promoted']=True
            if change=='draft':release['draft']=True
            with self.subTest(change=change),self.assertRaises(ValueError):e.validate_terminal(result,index,release,run)

    def test_waits_for_real_preflight_and_existing_fifty_hours(self):
        c=Fake();c.checks=[]
        self.assertEqual(e.tick(c)['action'],'waiting_for_real_native_preflight')
        c=Fake();c.source[0].update(status='in_progress',conclusion=None)
        self.assertEqual(e.tick(c)['action'],'waiting_for_current_cycle')
        self.assertFalse(c.calls)

    def test_failed_or_foreign_preflight_cannot_release_training(self):
        for change in ['failure','sha']:
            c=Fake()
            if change=='failure':c.checks[0]['conclusion']='failure'
            else:c.checks[0]['head_sha']='wrong'
            self.assertEqual(e.tick(c)['action'],'waiting_for_real_native_preflight')

    def test_branch_creation_uses_verified_terminal_and_frozen_new_code(self):
        c=Fake()
        with patch.object(e,'build_files',return_value={e.PREFIX+'SPEC.json':'{}'}):
            self.assertEqual(e.tick(c)['action'],'dispatched')
        commit=next(payload for path,payload in c.calls if path=='git/commits')
        self.assertEqual(commit['parents'],[e.TEMPLATE_SHA])
        self.assertIn(('actions/workflows/'+e.WORKFLOW+'/dispatches',{'ref':e.BRANCH,'inputs':{'checkpoint_release':c.release['tag_name'],'continuation':'0'}}),c.calls)

    def test_running_target_is_idempotent_and_global_slot_is_respected(self):
        c=Fake();c.target=[dict(id=200,head_sha=c.target_sha,status='in_progress',conclusion=None)]
        self.assertEqual(e.tick(c)['action'],'already_started');self.assertFalse(c.calls)
        c=Fake();c.active=[{'path':'.github/workflows/airi-human-background.yml'}]
        with patch.object(e,'build_files',return_value={e.PREFIX+'SPEC.json':'{}'}):
            self.assertEqual(e.tick(c)['action'],'waiting_for_training_slot')
        self.assertFalse(any(p.endswith('/dispatches') for p,_ in c.calls))

    def test_cancel_is_respected_and_completed_pilot_disables_controller(self):
        c=Fake();c.target=[dict(id=200,head_sha=c.target_sha,status='completed',conclusion='cancelled')]
        self.assertEqual(e.tick(c)['action'],'intentional_cancellation_respected')
        self.assertTrue(c.calls[-1][0].endswith('/disable'))
        c=Fake();c.target=[dict(id=123,run_attempt=1,head_sha=c.target_sha,status='completed',conclusion='success')]
        c.result.update(maximum_training_seconds=3600);c.result['counters']['elapsed_training_seconds']=3600.1
        self.assertEqual(e.tick(c)['action'],'selective_transfer_pilot_completed_controller_disabled')
        self.assertTrue(c.calls[-1][0].endswith('/disable'))

    def test_actual_template_generation_keeps_runner_hashes_and_new_budget(self):
        root=Path(os.environ['AIRI_SUCCESS_TEMPLATE'])
        t={e.PREFIX+'SPEC.json':(root/e.PREFIX/'SPEC.json').read_bytes()}
        spec=json.loads(t[e.PREFIX+'SPEC.json'])
        for name in spec['script_sha256']:t[e.PREFIX+name]=(root/e.PREFIX/name).read_bytes()
        # The frozen source has its own checked script set; this fixture exercises
        # the real template bytes without asking GitHub for a fabricated source.
        source_root=Path(os.environ['AIRI_FROZEN_SOURCE'])
        source={e.PREFIX+'SPEC.json':(source_root/e.PREFIX/'SPEC.json').read_bytes()}
        source_spec=json.loads(source[e.PREFIX+'SPEC.json'])
        for name in source_spec['script_sha256']:source[e.PREFIX+name]=(source_root/e.PREFIX/name).read_bytes()
        result,index,release,run=fixture()
        generated=e.build_files(source,result,index,release,run,template=t)
        after=json.loads(generated[e.PREFIX+'SPEC.json'])
        self.assertEqual(after['start_step'],400000);self.assertEqual(after['maximum_training_seconds'],3600)
        self.assertEqual(after['migration_source_fingerprint'],e.SOURCE_FP)
        self.assertEqual(after['script_sha256'],spec['script_sha256'])
        self.assertEqual(after['extension']['template_sha'],e.TEMPLATE_SHA)
        changed=dict(t);changed[e.PREFIX+'success_bank.py']+=b'\n# changed'
        self.assertRaises(ValueError,e.build_files,source,result,index,release,run,changed)

if __name__=='__main__':unittest.main()
