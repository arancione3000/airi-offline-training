from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/autonomous-control'))
import start_evolving as s
class Client:
 def __init__(self):self.sha=s.COMMIT;self.runs=[];self.active=[]
 def get(self,path):return {'object':{'sha':self.sha}}
 def pages(self,path,key):
  if key=='jobs':return iter([{'steps':[{'name':'Verify real native optimizer recovery','conclusion':'success'}]}])
  if path.startswith('actions/workflows/'):return iter(self.runs)
  return iter(self.active)
class StartTests(unittest.TestCase):
 def verified(self):
  c=Client();c.runs=[dict(id=1,head_sha=s.COMMIT,display_title='AIRI context preflight',conclusion='success')];return c
 def test_waits_for_preflight(self):
  dispatched=[];self.assertEqual(s.start(Client(),lambda:dispatched.append(1))['action'],'none');self.assertEqual(dispatched,[])
 def test_verified_idle_starts(self):
  dispatched=[];self.assertEqual(s.start(self.verified(),lambda:dispatched.append(1))['action'],'started');self.assertEqual(dispatched,[1])
 def test_prior_training_waits(self):
  c=self.verified();c.active=[dict(path='.github/workflows/airi-stable-background.yml')]
  self.assertEqual(s.start(c,lambda:self.fail('unexpected dispatch'))['action'],'none')
 def test_training_latch_blocks_restart_even_failed(self):
  c=self.verified();c.runs.append(dict(id=2,head_sha=s.COMMIT,display_title='AIRI context training',conclusion='failure'))
  self.assertEqual(s.start(c,lambda:self.fail('unexpected dispatch'))['action'],'none')
 def test_changed_frozen_sha_blocks(self):
  c=self.verified();c.sha='different';self.assertEqual(s.start(c,lambda:self.fail('unexpected dispatch'))['action'],'none')
if __name__=='__main__':unittest.main()
