import base64,copy,hashlib,importlib.util,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
loader=importlib.util.spec_from_file_location('controller',ROOT/'scripts/continuous-training/controller.py')
module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
FP='f'*64;SHA='a'*40
class Fake:
 def __init__(self):
  self.c=dict(schema=1,enabled=True,halted_reason=None,runner_branch='frozen',runner_sha=SHA,fingerprint=FP,
              seed_release='airi-offline-450364-12-1-1',origin_step=449276,chunk_seconds=180,dispatch=None)
  self.version=0;self.posts=[];self.run_values=[];self.run=None;self.branch=SHA;self.retries=[]
 def control(self):return copy.deepcopy(self.c),str(self.version)
 def write(self,c,sha):
  if sha!=str(self.version):raise RuntimeError('CAS conflict')
  self.c=copy.deepcopy(c);self.version+=1
 def dispatch(self,branch,inputs):self.posts.append((branch,inputs))
 def runs(self):return self.run_values
 def get(self,path):return {'object':{'sha':self.branch}} if path.startswith('git/') else self.run
 def checkpoint(self,run_id):return self.saved,self.index,self.raw
 def logs(self,run):return self.log
 def rerun(self,run):self.retries.append(run['id'])
 def finished(self,job=0):
  request=module.key(FP,self.c['seed_release'],job)
  self.c['dispatch']=dict(key=request,source_release=self.c['seed_release'],job_index=job,run_id=42,requested_at=0,request_attempts=1)
  self.run=dict(id=42,status='completed',conclusion='success',head_sha=SHA,run_attempt=1)
  self.saved='airi-offline-450620-42-1-2'
  state=dict(fingerprint=FP,runner_sha=SHA,job_index=job,completed_steps=450620,production_qualified=False,live_promoted=False,
             continuous_training=dict(not_before=0,epoch=0),**{'continue':True})
  self.raw=json.dumps(state).encode()
  self.index=dict(fingerprint=FP,completed_steps=450620,run_state=dict(name='AIRI_RUN_STATE.json',bytes=len(self.raw),sha256=hashlib.sha256(self.raw).hexdigest()),
                  archives=[dict(name=f'AIRI_TRAINING_450620_{part}.zip',bytes=10,sha256='b'*64) for part in ('MODEL','OPTIMIZER')])

class ControllerTests(unittest.TestCase):
 def test_bootstrap_claim_then_duplicate_cannot_write(self):
  f=Fake();module.tick(f,100)
  self.assertEqual(len(f.posts),1);d=f.c['dispatch']
  args=dict(runner_sha=SHA,source=d['source_release'],job=0,request=d['key'])
  self.assertTrue(module.claim(f,run_id=42,**args));self.assertFalse(module.claim(f,run_id=43,**args))
 def test_controller_crash_after_claim_does_not_double_dispatch(self):
  f=Fake();module.tick(f,100)
  self.assertEqual(module.tick(f,101)['action'],'idle');self.assertEqual(len(f.posts),1)
  f.run_values=[dict(id=42,head_sha=SHA,display_title='AIRI continuous '+f.c['dispatch']['key'],status='completed')]
  self.assertEqual(module.tick(f,102)['action'],'recovered dispatch receipt');self.assertEqual(f.c['dispatch']['run_id'],42)
 def test_no_global_job_or_step_cap_and_monotonic_continuation(self):
  f=Fake();f.finished(job=100000)
  result=module.tick(f,100)
  self.assertEqual(result['job_index'],100001);self.assertEqual(f.posts[0][1]['checkpoint_release'],f.saved)
 def test_cooldown_is_durable_not_a_service_shutdown(self):
  f=Fake();f.finished();state=json.loads(f.raw);state['continuous_training']['not_before']=1000
  f.raw=json.dumps(state).encode();f.index['run_state'].update(bytes=len(f.raw),sha256=hashlib.sha256(f.raw).hexdigest())
  self.assertEqual(module.tick(f,999)['action'],'idle');self.assertTrue(f.c['enabled']);self.assertFalse(f.posts)
  self.assertEqual(module.tick(f,1000)['action'],'dispatched')
 def test_integrity_failure_halts_without_dispatch(self):
  f=Fake();f.finished();f.raw+=b' '
  self.assertEqual(module.tick(f,100)['action'],'halted');self.assertFalse(f.posts)
 def test_program_failure_never_retried_as_network_error(self):
  f=Fake();f.finished();f.run['conclusion']='failure';f.log='gh: HTTP502\nValueError: conflicting registration proof'
  self.assertEqual(module.tick(f,100)['action'],'halted');self.assertFalse(f.retries)
 def test_transient_retry_is_claimed_once_and_bounded(self):
  f=Fake();f.finished();f.run['conclusion']='failure';f.log='gh: HTTP502 BadGateway'
  self.assertEqual(module.tick(f,100)['action'],'retry transient failure')
  self.assertEqual(module.tick(f,101)['action'],'idle');self.assertEqual(f.retries,[42])
  f.run['run_attempt']=3;self.assertEqual(module.tick(f,102)['action'],'halted')
 def test_active_job_and_disabled_service_do_not_dispatch(self):
  f=Fake();f.run_values=[dict(status='in_progress',head_branch='frozen')]
  self.assertEqual(module.tick(f,100)['action'],'idle');self.assertFalse(f.posts)
  f.c['enabled']=False;self.assertEqual(module.tick(f,200)['action'],'idle')
 def test_frozen_branch_movement_halts(self):
  f=Fake();f.branch='d'*40;self.assertEqual(module.tick(f,100)['action'],'halted');self.assertFalse(f.posts)
 def test_wrong_checkpoint_job_and_duplicate_archives_refused(self):
  for mutate in ('job','archives'):
   f=Fake();f.finished()
   if mutate=='job':f.c['dispatch']['job_index']=2
   else:f.index['archives'][1]=f.index['archives'][0]
   self.assertEqual(module.tick(f,100)['action'],'halted')

if __name__=='__main__':unittest.main()
