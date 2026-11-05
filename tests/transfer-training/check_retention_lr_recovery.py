"""Native proof that tentative LR growth cannot bypass exact Adam/guard rollback."""
import copy,json,random,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/transfer-training'))
import torch
from retention_transaction import transactional_update,_equal
CFG=dict(accepted_streak=16,cooldown_updates=8,growth_factor=2.)

class RecoveryTests(unittest.TestCase):
 def setup(self):
  torch.manual_seed(77);m=torch.nn.Linear(1,1,bias=False);o=torch.optim.Adam(m.parameters(),lr=.01)
  for _ in range(3):o.zero_grad();m(torch.ones(1,1)).square().sum().backward();o.step()
  state=dict(schema=1,lr_scale=.125,attempted_updates=0,committed_updates=0,rejected_updates=0,
             recovery=dict(streak=16,cooldown=0,growth_attempts=0,growth_commits=0))
  return m,o,state
 def step(self,m,o):
  o.zero_grad();loss=(m(torch.ones(1,1))+torch.rand(())*.01).square().sum();loss.backward();o.step()
  return dict(loss=float(loss.detach()))
 def test_failed_growth_retries_original_scale_exactly(self):
  m,o,state=self.setup();other=copy.deepcopy(m);oo=torch.optim.Adam(other.parameters(),lr=.01);oo.load_state_dict(copy.deepcopy(o.state_dict()))
  stream=torch.get_rng_state().clone();oo.param_groups[0]['lr']=.00125;self.step(other,oo);oo.param_groups[0]['lr']=.01;expected=torch.get_rng_state().clone();torch.set_rng_state(stream)
  result=transactional_update(m,o,lambda:self.step(m,o),lambda:dict(accepted=o.param_groups[0]['lr']<=.00125),state=state,recovery=CFG)
  self.assertEqual(result['attempts'],2);self.assertEqual(result['scale'],.125)
  self.assertTrue(_equal(o.state_dict(),oo.state_dict()));self.assertTrue(torch.equal(m.weight,other.weight))
  self.assertTrue(torch.equal(expected,torch.get_rng_state()));self.assertEqual(state['recovery']['cooldown'],8)
  self.assertEqual(state['recovery']['growth_commits'],0)
 def test_successful_growth_and_json_resume(self):
  m,o,state=self.setup();r=transactional_update(m,o,lambda:self.step(m,o),lambda:dict(accepted=True),state=state,recovery=CFG)
  self.assertEqual(r['scale'],.25);self.assertEqual(state['recovery']['growth_commits'],1)
  clone=copy.deepcopy(m);opt=torch.optim.Adam(clone.parameters(),lr=.01);opt.load_state_dict(copy.deepcopy(o.state_dict()));saved=json.loads(json.dumps(state));stream=torch.get_rng_state().clone()
  a=transactional_update(m,o,lambda:self.step(m,o),lambda:dict(accepted=True),state=state,recovery=CFG)
  expected=torch.get_rng_state().clone();torch.set_rng_state(stream)
  b=transactional_update(clone,opt,lambda:self.step(clone,opt),lambda:dict(accepted=True),state=saved,recovery=CFG)
  self.assertEqual(a,b);self.assertEqual(saved,state);self.assertTrue(_equal(o.state_dict(),opt.state_dict()));self.assertTrue(torch.equal(expected,torch.get_rng_state()))
 def test_default_does_not_grow_and_invalid_config_fails_before_update(self):
  m,o,state=self.setup();legacy={k:v for k,v in state.items() if k!='recovery'}
  result=transactional_update(m,o,lambda:self.step(m,o),lambda:dict(accepted=True),state=legacy)
  self.assertEqual(result['scale'],.125);self.assertNotIn('recovery',legacy)
  with self.assertRaises(ValueError):transactional_update(m,o,lambda:self.step(m,o),lambda:dict(accepted=True),recovery={**CFG,'accepted_streak':True})

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
