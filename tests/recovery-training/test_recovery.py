import ast
import importlib.util
import json
import random
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

SOURCE=Path(__file__).resolve().parents[2]/'scripts/recovery-training'
def load(name):
    spec=importlib.util.spec_from_file_location(name,SOURCE/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
bg=load('background');policy=load('recovery_policy')

class RecoveryTests(unittest.TestCase):
    def test_recipe_hashes_and_finite_budget(self):
        for name,expected in bg.SPEC['script_sha256'].items():
            self.assertEqual(bg.digest(SOURCE/name),expected)
        base={k:v for k,v in bg.SPEC.items() if k not in ['fingerprint','parameter_count']}
        self.assertEqual(bg.hashlib.sha256(json.dumps(base,sort_keys=True).encode()).hexdigest(),bg.FINGERPRINT)
        self.assertEqual(bg.TARGET,policy.START_STEP+policy.EXTRA_UPDATES)

    def test_runtime_recipe_matches_published_fingerprint(self):
        tree=ast.parse((SOURCE/'run.py').read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='spec' for t in n.targets))
        rows=[{'provenance':{'supervised_tokens':0}} for _ in range(10486)]
        rows[0]['provenance']['supervised_tokens']=3472648
        env={'EXPECTED_PARENT':bg.SPEC['parent_sha256'],'EXPECTED_DATA':bg.SPEC['dataset_sha256'],
             'reference':SimpleNamespace(tokenizer=SimpleNamespace(digest=bg.SPEC['tokenizer_digest'])),
             'torch':SimpleNamespace(__version__=bg.SPEC['torch_version']),
             'args':SimpleNamespace(max_steps=bg.TARGET),'dialogue':[None]*10486,'rows':rows,
             'anchors':[None]*61,'replay':{g:[None]*n for g,n in bg.SPEC['replay_rows'].items()},
             'Path':Path,'__file__':str(SOURCE/'run.py'),'ROOT':SOURCE.parent,
             'sha256_file':lambda p:bg.digest(SOURCE/Path(p).name)}
        runtime_spec=eval(compile(ast.Expression(node.value),'<recipe>','eval'),env)
        self.assertEqual(bg.hashlib.sha256(json.dumps(runtime_spec,sort_keys=True).encode()).hexdigest(),bg.FINGERPRINT)

    def test_migration_keeps_adam_state_and_only_changes_lr(self):
        state={'momentum':[1,2]}
        opt=SimpleNamespace(param_groups=[{'lr':1e-5,'weight_decay':.01}],state=state)
        policy.migrate_optimizer(opt,{'completed_steps':13107})
        self.assertIs(opt.state,state)
        self.assertEqual(opt.param_groups,[{'lr':2e-6,'weight_decay':.01}])
        with self.assertRaises(ValueError):policy.migrate_optimizer(opt,{'completed_steps':13106})

    def test_balance_and_exact_sampler_resume(self):
        language={'it':[0,1],'en':[2,3]};dialogue=['it1','it2','en1','en2']
        replay={'language-it':['ri1','ri2'],'language-en':['re'],'domains':['rd']}
        def series(rng,counters,start,end):
            return [policy.choose_batch(i,counters,rng,language,dialogue,replay) for i in range(start,end)]
        full_c={'replay_updates':2621};full=series(random.Random(91),full_c,13107,13131)
        rng=random.Random(91);split_c={'replay_updates':2621}
        first=series(rng,split_c,13107,13119)
        resumed_rng=random.Random();resumed_rng.setstate(rng.getstate())
        resumed_c=json.loads(json.dumps(split_c))
        self.assertEqual(first+series(resumed_rng,resumed_c,13119,13131),full)
        self.assertEqual(resumed_c,full_c)
        self.assertEqual(sum(g=='dialogue-it' for g,b in full),6)
        self.assertEqual(sum(g=='dialogue-en' for g,b in full),6)
        self.assertEqual(sum(g in replay for g,b in full),12)
        self.assertEqual(full_c['recovery_replay_updates'],12)
        self.assertEqual(full_c['replay_updates'],2633)
        with self.assertRaises(ValueError):series(rng,split_c,bg.TARGET,bg.TARGET+1)

    def test_old_fingerprint_only_exact_completed_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'file').write_text('native weights')
            m={'fingerprint':bg.LEGACY_FINGERPRINT,'completed_steps':13107,'files':{'file':bg.digest(p/'file')}}
            (p/'checkpoint.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):bg.verify_checkpoint(p)
            self.assertEqual(bg.verify_checkpoint(p,bg.LEGACY_FINGERPRINT)['completed_steps'],13107)
            m['completed_steps']=13108;(p/'checkpoint.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):bg.verify_checkpoint(p,bg.LEGACY_FINGERPRINT)

    def test_autonomous_dispatch_after_committed_checkpoint_only(self):
        wf=(SOURCE.parents[1]/'.github/workflows/airi-recovery-background.yml').read_text()
        self.assertIn("if: steps.train.outputs.completed == 'false'",wf)
        self.assertIn('airi-recovery-background.yml/dispatches',wf)
        self.assertIn('group: airi-offline-background-training',wf)
        self.assertIn('next_count >= 64',wf)
        self.assertIn('scripts/recovery-training/background.py',wf)
        self.assertNotIn('always()',wf)

if __name__=='__main__':unittest.main()
