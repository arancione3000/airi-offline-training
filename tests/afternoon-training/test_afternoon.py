import importlib.util,json,hashlib,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'scripts/afternoon-training'
sys.path.insert(0,str(HERE))
import run,curriculum,extra_dialogues
class AfternoonTests(unittest.TestCase):
    def c(self):return dict(elapsed_training_seconds=0.,phase=2,completed_steps=27443,conversation_updates=0)
    def test_absolute_deadline(self):
        deadline=run.SPEC['training_cutoff_epoch']
        self.assertTrue(run.needs_training(self.c(),deadline-1))
        self.assertFalse(run.needs_training(self.c(),deadline))
        self.assertFalse(run.needs_training(self.c(),deadline+1))
        from datetime import datetime,timezone
        self.assertEqual(datetime.fromtimestamp(deadline,timezone.utc).isoformat(),'2026-10-08T11:40:00+00:00')
        self.assertEqual(run.SPEC['report_deadline_epoch']-deadline,1200)
    def test_invalid_training_counter(self):
        for v in [-1,float('nan'),float('inf')]:
            c=self.c();c['elapsed_training_seconds']=v
            with self.assertRaises(ValueError):run.needs_training(c,0)
    def test_validation_disjoint_and_expanded(self):
        _,old=curriculum.build();train,val=extra_dialogues.build_extra(old_validation=old)
        self.assertEqual(len(train),400)
        self.assertEqual(len(val),16)
        self.assertEqual({r['language'] for r in train},{'it','en'})
        prompts={curriculum.norm(r['messages'][0]['content']) for r in train}
        self.assertFalse(prompts&{curriculum.norm(r['messages'][0]['content']) for r in old+val})
        keys=lambda rows:{json.dumps(r['messages'][:-1],sort_keys=True) for r in rows}
        self.assertFalse(keys(train)&keys(old+val))
    def test_protected_text_all_roles(self):
        blocked={'bene, grazie! e tu come stai?','sono rita e mi piace il colore viola.'}
        train,val=extra_dialogues.build_extra(blocked)
        for r in train+val:self.assertFalse(blocked&{curriculum.norm(m['content']) for m in r['messages']})
    def test_memory_answers_depend_on_history(self):
        train,val=extra_dialogues.build_extra()
        for r in train+val:
            if 'memory' not in r['family']:continue
            self.assertEqual(len(r['messages']),4)
            self.assertIn(r['expected'],r['messages'][0]['content'])
            self.assertNotEqual(r['expected'],r['messages'][1]['content'])
    def test_source_pinned(self):
        for name,digest in run.SPEC['script_sha256'].items():self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).hexdigest(),digest)
        d=dict(run.SPEC);fingerprint=d.pop('fingerprint')
        self.assertEqual(hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest(),fingerprint)
