from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/noon-training'))
from policy import improved,unlocked_stage
from context_curriculum import build_context

def metrics():
    return dict(nll=dict(it=1.,en=1.),legacy_correct=13,dialogue_correct=0,context_rates={f'{l}:{t}':0. for l in ('it','en') for t in ('name','location','topic','correction')},context_pair_rate=0.,repetition=0.)

class PolicyTests(unittest.TestCase):
    def test_generic_answer_cannot_unlock(self):
        a=metrics();a['legacy_correct']=28
        self.assertEqual(unlocked_stage(a,0),0)
    def test_context_gain_and_regression(self):
        a=metrics();b=metrics();b['context_rates']={k:.5 for k in b['context_rates']};b['context_pair_rate']=.5
        self.assertTrue(improved(a,b));self.assertEqual(unlocked_stage(b,0),1)
        b['legacy_correct']=12;self.assertFalse(improved(a,b))
    def test_one_language_failure_blocks_expansion(self):
        a=metrics();a['context_rates']={k:1. for k in a['context_rates']};a['context_pair_rate']=1.
        a['context_rates']['en:correction']=0.;self.assertEqual(unlocked_stage(a,0),0)
    def test_repetition_and_nan_rejected(self):
        a=metrics();b=metrics();b['nll']=dict(it=.8,en=.8)
        self.assertTrue(improved(a,b));b['repetition']=.3;self.assertFalse(improved(a,b))
        b['repetition']=0.;b['nll']['it']=float('nan');self.assertFalse(improved(a,b))
    def test_dataset_disjoint_counterfactuals_and_scale(self):
        train,test=build_context()
        self.assertEqual(len(train),2048);self.assertEqual(len(test),32)
        first=lambda r:r['messages'][0]['content'].casefold()
        self.assertFalse({first(r) for r in train}&{first(r) for r in test})
        for pair in {r['pair_id'] for r in test}:
            rows=[r for r in test if r['pair_id']==pair]
            self.assertEqual(len(rows),2)
            self.assertEqual(rows[0]['messages'][-2],rows[1]['messages'][-2])
            self.assertNotEqual(rows[0]['expected'],rows[1]['expected'])
        self.assertEqual([sum(r['stage']<=s for r in train) for s in range(3)],[256,1024,2048])
    def test_protected_prompt_removed(self):
        train,_=build_context();prompt=train[0]['messages'][0]['content']
        filtered,_=build_context([prompt.casefold()])
        self.assertFalse(any(r['messages'][0]['content']==prompt for r in filtered))

if __name__=='__main__':unittest.main()
