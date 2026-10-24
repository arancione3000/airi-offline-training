import hashlib,json,sys,unittest
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/transfer-training'))
from human_data import quality_message,context_size,load_corpus
from curriculum import norm
from policy import improved
import run

class HumanTests(unittest.TestCase):
 def corpus(self):
  p=ROOT/'data/oasst2';m=json.loads((p/'MANIFEST.json').read_text());return load_corpus(p,m['pins'])
 def test_hashes_tree_prompts_and_targets_disjoint(self):
  c=self.corpus()
  for split in c:
   for other in c:
    if split==other:continue
    ids=lambda rows:{r['provenance']['tree_id'] for r in rows}
    prompts=lambda rows:{norm(m['content']) for r in rows for m in r['messages'] if m['role']=='user'}
    self.assertFalse(ids(c[split])&ids(c[other]));self.assertFalse(prompts(c[split])&prompts(c[other]))
  reserved={norm(m['content']) for split in ('development','final') for r in c[split] for m in r['messages']}
  self.assertFalse({norm(m['content']) for r in c['train'] for m in r['messages']} & reserved)
 def test_complete_short_human_examples_and_multiturn(self):
  c=self.corpus();self.assertTrue(any(len(r['messages'])>2 for r in c['train']))
  self.assertTrue(all(context_size(r['messages'])<=1024 and r['provenance']['synthetic'] is False for rows in c.values() for r in rows))
 def test_synthetic_deleted_low_quality_or_unranked_rejected(self):
  r=dict(review_result=True,lang='it',role='assistant',rank=0,text='Una risposta umana.',labels={'quality':{'value':.8,'count':3}})
  self.assertTrue(quality_message(r))
  for change in ({'synthetic':True},{'deleted':True},{'model_name':'teacher'},{'rank':1},{'review_result':False},{'labels':{}}):self.assertFalse(quality_message({**r,**change}))
 def test_archived_corpus_names_do_not_shadow_original_splits(self):
  names=[p.name for p in (ROOT/'data/oasst2').glob('*')]+['train-1024.jsonl','development.jsonl','test.jsonl','CONVERSATION_TEST.json','AUTHORED_CURRICULUM.json']
  self.assertEqual(len(names),len(set(names)))
 def test_runtime_overlap_is_removed(self):
  c=self.corpus();text=c['train'][0]['messages'][0]['content'];p=ROOT/'data/oasst2';pins=json.loads((p/'MANIFEST.json').read_text())['pins']
  filtered=load_corpus(p,pins,[norm(text)])
  self.assertTrue(all(norm(m['content'])!=norm(text) for rows in filtered.values() for r in rows for m in r['messages']))
 def test_reviewed_replacement_can_reuse_old_training_but_never_evaluation(self):
  c=self.corpus();row=c['train'][0];text=norm(row['messages'][0]['content'])
  p=ROOT/'data/oasst2';pins=json.loads((p/'MANIFEST.json').read_text())['pins']
  legacy=load_corpus(p,pins,[text]);reviewed=load_corpus(p,pins,[text],training_blocked=[])
  self.assertNotIn(row,legacy['train']);self.assertIn(row,reviewed['train'])
  self.assertEqual(legacy['development'],reviewed['development']);self.assertEqual(legacy['final'],reviewed['final'])
  reserved={norm(m['content']) for split in ('development','final') for r in reviewed[split] for m in r['messages']}
  self.assertFalse({norm(m['content']) for r in reviewed['train'] for m in r['messages']} & reserved)
 def test_sampling_actually_reaches_human_data_both_languages(self):
  for focus in run.SAMPLING_SCHEDULES:
   self.assertEqual(Counter(run.sampling_group(i,focus) for i in range(16))['human-dialogue'],4)
   self.assertEqual({'it' if i%3!=1 else 'en' for i in range(48) if run.sampling_group(i,focus)=='human-dialogue'},{'it','en'})
 def test_human_gain_cannot_mask_regression_or_read_final(self):
  m=dict(context_rates={'it:name':0.,'en:name':0.},context_pair_rate=0.,legacy_correct=14,dialogue_correct=2,foundation_correct=4,foundation_probes=16,repetition=0.,nll={'it':1.,'en':1.},human_nll={'en':2.})
  self.assertTrue(improved(m,{**m,'human_nll':{'en':1.97}}))
  self.assertFalse(improved(m,{**m,'human_nll':{'en':1.5},'legacy_correct':13}))
  self.assertFalse(improved(m,{**m,'human_nll':{'en':2.05},'dialogue_correct':3}))
  self.assertFalse(improved(m,{**m,'human_final':{'score':100}}))

if __name__=='__main__':unittest.main()
