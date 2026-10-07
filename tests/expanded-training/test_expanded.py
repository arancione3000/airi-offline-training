import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2]/'scripts/expanded-training'


def load(name):
    spec = importlib.util.spec_from_file_location('expanded_'+name,SOURCE/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bg, policy, corpus = load('background'), load('policy'), load('corpus')


class ExpandedTests(unittest.TestCase):
    def test_recipe_and_sources(self):
        base = {k:v for k,v in bg.SPEC.items() if k not in ['fingerprint','parameter_count']}
        self.assertEqual(bg.hashlib.sha256(json.dumps(base,sort_keys=True).encode()).hexdigest(),bg.FINGERPRINT)
        for name,sha in bg.SPEC['script_sha256'].items():
            self.assertEqual(bg.digest(SOURCE/name),sha)
        self.assertEqual(bg.TARGET,policy.TARGET)
        self.assertEqual(bg.SPEC['pilot_step'],policy.PILOT)

    def test_corpus_disjoint_and_complete(self):
        class Tokenizer:
            def encode(self,text,**kwargs):
                return list(text.encode())+[1,2]
        rows=[('0','protected sentence','a'),('1','new sentence one','a'),
              ('2','NEW SENTENCE ONE','b'),('3','new sentence two','b'),
              ('4','new sentence three','c')]
        train,heldout=corpus.select(rows,'it',{'protected sentence'},Tokenizer(),train_count=2,heldout_count=1)
        self.assertEqual([x['id'] for x in heldout],['1'])
        self.assertEqual([x['id'] for x in train],['3','4'])
        self.assertEqual(train[1]['author'],'c')
        with self.assertRaises(ValueError):
            corpus.select(rows,'it',set(),Tokenizer(),train_count=10,heldout_count=1)

    def test_corpus_integrity_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ['CORPUS_TRAIN.jsonl','CORPUS_HELDOUT.jsonl']:
                (root/name).write_text('native sentence')
            manifest={'tokenizer_digest':'native','files':{p.name:bg.digest(p) for p in root.iterdir()}}
            (root/'CORPUS_MANIFEST.json').write_text(json.dumps(manifest))
            corpus.verify(root,'native')
            with self.assertRaises(ValueError):corpus.verify(root,'changed tokenizer')
            (root/'CORPUS_TRAIN.jsonl').write_text('corrupted')
            with self.assertRaises(ValueError):corpus.verify(root,'native')

    def test_raw_deck_resume_and_no_replacement_first_epoch(self):
        full=policy.DeckSampler()
        series=[full.index('it',i,37) for i in range(111)]
        first=policy.DeckSampler()
        partial=[first.index('it',i,37) for i in range(51)]
        restored=policy.DeckSampler()
        self.assertEqual(partial+[restored.index('it',i,37) for i in range(51,111)],series)
        self.assertEqual(len(set(series[:37])),37)

    def test_schedule_and_finite_bounds(self):
        groups=[policy.group_for(step) for step in range(policy.START,policy.PILOT)]
        self.assertEqual(groups.count('corpus-it'),256)
        self.assertEqual(groups.count('corpus-en'),128)
        self.assertEqual(groups.count('dialogue-it'),64)
        self.assertEqual(groups.count('dialogue-en'),64)
        for step in [policy.START-1,policy.TARGET]:
            with self.assertRaises(ValueError):policy.group_for(step)

    def test_pilot_requires_improvement_and_all_gates(self):
        before={'language':{'language_nll':2.},'corpus':{'it':2.,'en':2.},
                'development':{lang:{'nll_per_byte':2.} for lang in ['it','en']}}
        after=json.loads(json.dumps(before))
        after['language']['language_nll']=1.99
        after['corpus']={'it':1.9,'en':1.9}
        self.assertTrue(policy.pilot_decision(before,after,(True,'ok'),(True,'ok'))['passed'])
        self.assertFalse(policy.pilot_decision(before,after,(False,'regression'),(True,'ok'))['passed'])
        self.assertFalse(policy.pilot_decision(before,after,(True,'ok'),(False,'repetition'))['passed'])
        after['development']['it']['nll_per_byte']=2.001
        self.assertFalse(policy.pilot_decision(before,after,(True,'ok'),(True,'ok'))['passed'])

    def test_seed_migration_requires_exact_final_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'weights').write_text('native weights')
            manifest={'fingerprint':bg.LEGACY_FINGERPRINT,'completed_steps':17203,'files':{'weights':bg.digest(root/'weights')}}
            (root/'checkpoint.json').write_text(json.dumps(manifest))
            self.assertEqual(bg.verify_checkpoint(root,bg.LEGACY_FINGERPRINT)['completed_steps'],17203)
            with self.assertRaises(ValueError):bg.verify_checkpoint(root)
            manifest['completed_steps']=17204
            (root/'checkpoint.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):bg.verify_checkpoint(root,bg.LEGACY_FINGERPRINT)

    def test_pilot_checkpoint_is_not_interrupted_before_assessment(self):
        save=bg.bounded_save(lambda *a,**k:{'completed_steps':policy.PILOT},
                             lambda path:self.fail('interrupted pilot gate'),started=0,seconds=60,clock=lambda:100)
        self.assertEqual(save(None,None,None,None,None,Path('checkpoint'))['completed_steps'],policy.PILOT)

    def test_workflow_only_continues_after_remote_success(self):
        workflow=(SOURCE.parents[1]/'.github/workflows/airi-expanded-background.yml').read_text()
        self.assertIn("if: steps.train.outputs.completed == 'false'",workflow)
        self.assertIn('group: airi-offline-background-training',workflow)
        self.assertIn('next_count >= 64',workflow)
        self.assertNotIn('always()',workflow)
        self.assertIn('airi-expanded-background.yml/dispatches',workflow)


if __name__=='__main__':unittest.main()
