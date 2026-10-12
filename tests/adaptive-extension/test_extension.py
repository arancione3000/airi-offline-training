import copy,importlib.util,json,math,unittest
from pathlib import Path
path=Path(__file__).resolve().parents[2]/'scripts/adaptive-extension/extend.py'
spec=importlib.util.spec_from_file_location('extension',path);e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)

class Fake:
 def __init__(self):
  self.source=[dict(id=20,run_attempt=1,path=e.TRAINING_PATH,head_branch=e.SOURCE_BRANCH,head_sha=e.SOURCE_SHA,status='in_progress',conclusion=None)]
  self.target=[];self.active=[];self.calls=[]
  self.result=dict(counters=dict(elapsed_training_seconds=14400.1,completed_steps=150000),next_action=None,maximum_training_seconds=14400,production_qualified=False,live_promoted=False)
  self.index=dict(fingerprint=e.SOURCE_FP,completed_steps=150000,archives=[{},{}])
  self.release=dict(tag_name='airi-offline-150000-20-1-1',target_commitish=e.SOURCE_SHA,draft=False,prerelease=True,assets=[dict(name='FINAL_RESULT.json',id=1,size=1),dict(name='AIRI_TRAINING_CHECKPOINT_150000.json',id=2,size=1)])
 def api(self,path,payload=None,method='POST'):
  self.calls.append((path,payload))
  if path.startswith('git/ref/heads/'):return {'object':{'sha':e.TARGET_SHA}}
  return None
 def pages(self,path,key):
  if f'branch={e.SOURCE_BRANCH}' in path:return iter(copy.deepcopy(self.source))
  if f'branch={e.TARGET_BRANCH}' in path:return iter(copy.deepcopy(self.target))
  if path=='releases':return iter([copy.deepcopy(self.release)])
  if 'status=' in path:return iter(copy.deepcopy(self.active))
  raise AssertionError(path)
 def asset(self,item):return copy.deepcopy(self.result if item['id']==1 else self.index)

class ExtensionTests(unittest.TestCase):
 def ready(self):
  c=Fake();c.source[0].update(status='completed',conclusion='success');return c
 def test_waits_for_current_chain(self):
  c=Fake();self.assertEqual(e.tick(c)['action'],'waiting_for_current_cycle');self.assertFalse(any(p.endswith('/dispatches') for p,_ in c.calls))
 def test_requires_exact_frozen_branch(self):
  c=Fake();old=e.TARGET_SHA
  c.api=lambda path,payload=None,method='POST':{'object':{'sha':'wrong'}} if path.startswith('git/ref/') else None
  with self.assertRaises(ValueError):e.tick(c)
 def test_terminal_result_is_strictly_validated(self):
  for change in ('elapsed','nan','continuing','fingerprint','archives','live'):
   c=self.ready()
   if change=='elapsed':c.result['counters']['elapsed_training_seconds']=14399
   if change=='nan':c.result['counters']['elapsed_training_seconds']=math.nan
   if change=='continuing':c.result['next_action']={'release':'x'}
   if change=='fingerprint':c.index['fingerprint']='wrong'
   if change=='archives':c.index['archives'].pop()
   if change=='live':c.result['live_promoted']=True
   with self.subTest(change=change),self.assertRaises(ValueError):e.tick(c)
 def test_waits_for_free_training_slot(self):
  c=self.ready();c.active=[{'path':e.TRAINING_PATH+'@main'}]
  self.assertEqual(e.tick(c)['action'],'waiting_for_training_slot')
 def test_dispatches_once_from_retained_champion(self):
  c=self.ready();result=e.tick(c);self.assertEqual(result['action'],'dispatched')
  self.assertIn((f'actions/workflows/{e.TARGET_WORKFLOW}/dispatches',{'ref':e.TARGET_BRANCH,'inputs':{'checkpoint_release':e.SEED_RELEASE,'continuation':'0'}}),c.calls)
 def test_existing_target_run_prevents_duplicate(self):
  c=self.ready();c.target=[{'id':30,'status':'in_progress'}]
  self.assertEqual(e.tick(c),{'action':'already_started','run_id':30})

if __name__=='__main__':unittest.main()
