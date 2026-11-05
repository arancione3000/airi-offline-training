import ast
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/transfer-training'))
import run as runner
from context_curriculum import build_context
from context_acquisition import build_context_acquisition,context_value_pairs,matched_context_pair
from continual_learning import validate_configuration
from success_bank import catalog,prompt_key,row_key


class RetentionOrchestrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources,_=build_context()
        cls.values=build_context_acquisition(max_pairs=8)
        cls.source=cls.sources[0]
        cls.value_pair=context_value_pairs(cls.values)[(cls.source['language'],cls.source['family'])][0]

    def registry(self):
        registry=runner.empty_training_registry('recipe')
        pair=matched_context_pair(self.source,self.value_pair)
        rows=runner.register_context_pair(registry,self.source,pair)
        return registry,rows

    def validate(self,registry,**kwargs):
        return runner.validate_training_registry(registry,self.sources,self.values,[],fingerprint='recipe',**kwargs)

    def test_materialized_rows_have_exact_reconstructible_train_proofs(self):
        registry,rows=self.registry()
        restored=json.loads(json.dumps(registry))
        self.assertEqual(self.validate(restored),rows)
        self.assertEqual(catalog(rows,[]),{row_key(row):row for row in rows})
        self.assertEqual(rows[0]['messages'][:-1],matched_context_pair(self.source,self.value_pair)[0]['messages'][:-1])
        self.assertEqual({entry['source_row_key'] for entry in restored['entries'].values()},{row_key(self.source)})
        self.assertEqual({entry['counterfactual_index'] for entry in restored['entries'].values()},{0,1})

    def test_row_metadata_and_source_value_proofs_cannot_be_forged(self):
        registry,_=self.registry()
        for name,value in [('source_row_key','0'*64),('value_group','missing'),('counterfactual_index',9)]:
            bad=copy.deepcopy(registry);next(iter(bad['entries'].values()))[name]=value
            with self.assertRaises(ValueError):self.validate(bad)
        bad=copy.deepcopy(registry)
        next(iter(bad['entries'].values()))['row']['stage']=999
        with self.assertRaises(ValueError):self.validate(bad)
        bad=copy.deepcopy(registry)
        next(iter(bad['entries'].values()))['row']['provenance']='authored lie'
        with self.assertRaises(ValueError):self.validate(bad)

    def test_restored_registry_rejects_protected_history_and_unregistered_source(self):
        registry,rows=self.registry()
        with self.assertRaises(ValueError):
            self.validate(registry,protected_histories={prompt_key(rows[0])})
        with self.assertRaises(ValueError):
            runner.validate_training_registry(registry,[],self.values,[],fingerprint='recipe')
        self.assertEqual(runner.validate_training_registry(runner.empty_training_registry('recipe'),[],[],[],fingerprint='recipe'),[])

    def test_final_consolidation_has_its_own_quota_within_global_budget(self):
        spec=dict(start_step=100,max_steps=120,maximum_training_seconds=100.,recipes=[{}])
        counters=dict(completed_steps=110,conversation_updates=10,elapsed_training_seconds=50.,phase=0,
                      learning_search=dict(stopped=True),retention_consolidation_updates=0)
        with patch.object(runner,'SPEC',spec),patch.object(runner,'RETENTION',{'version':1}):
            self.assertFalse(runner.needs_training(counters))
            self.assertTrue(runner.consolidation_allowed(counters,{'consolidation_updates':4}))
            self.assertTrue(counters['learning_search']['stopped'])
            for change in [dict(retention_consolidation_updates=4),dict(elapsed_training_seconds=100.),
                           dict(completed_steps=120,conversation_updates=20)]:
                self.assertFalse(runner.consolidation_allowed({**counters,**change},{'consolidation_updates':4}))
            with self.assertRaises(ValueError):
                runner.consolidation_allowed({**counters,'retention_consolidation_updates':-1},{'consolidation_updates':4})

    def test_early_rollbacks_cannot_exhaust_terminal_consolidation_reservation(self):
        spec=dict(start_step=100,max_steps=1000,maximum_training_seconds=100.,recipes=[{}])
        original=dict(completed_steps=100,conversation_updates=0,elapsed_training_seconds=1.,phase=0,
                      learning_search=dict(stopped=False),retention_consolidation_updates=0)
        with patch.object(runner,'SPEC',spec):
            for quota in (1,3,128):
                counters=copy.deepcopy(original);config={'consolidation_updates':quota}
                while runner.consolidation_allowed(counters,config):
                    counters['retention_consolidation_updates']+=1
                    counters['conversation_updates']+=1;counters['completed_steps']+=1
                self.assertEqual(counters['retention_consolidation_updates'],quota//2)
                self.assertFalse(counters['learning_search']['stopped'])
                # Resume carries spent credit; another early rollback cannot
                # recreate it, and terminal use never restarts the search.
                counters=json.loads(json.dumps(counters))
                self.assertFalse(runner.consolidation_allowed(counters,config))
                counters['learning_search']['stopped']=True
                terminal=0
                while runner.consolidation_allowed(counters,config):
                    counters['retention_consolidation_updates']+=1;terminal+=1
                    counters['conversation_updates']+=1;counters['completed_steps']+=1
                self.assertEqual(terminal,quota-quota//2)
                self.assertEqual(counters['retention_consolidation_updates'],quota)
                self.assertTrue(counters['learning_search']['stopped'])

    def test_opt_in_recipe_is_strict_and_pins_all_retention_implementations(self):
        source=json.loads((ROOT/'scripts/continual-training/SPEC.json').read_text())
        config=dict(version=1,capacity=1024,refresh_probe_limit=128,prompt_kl_weight=8.,max_prompt_kl=.01,
                    max_nll_regression=.02,guard_examples_per_language=4,retry_scales=[1.,.5,.25,.125],
                    consolidation_updates=128,source_policy='eligible_train_only')
        def checked(change=None,mode='cached_prefix',missing=None):
            spec=copy.deepcopy(source)
            spec['continual_learning'].update(training_mode=mode,human_training_scope='reviewed_only',feature_cache_mebibytes=64,protected_head_rank=0,
                knowledge_retention={**config,**(change or {})})
            for name in ('retention_memory.py','retention_transaction.py','retention_guard.py'):spec['script_sha256'][name]='a'*64
            if missing:spec['script_sha256'].pop(missing)
            spec.pop('fingerprint');spec['fingerprint']=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            return spec
        self.assertTrue(validate_configuration(checked()))
        for change in [dict(version=True),dict(capacity=False),dict(source_policy='evaluation'),dict(extra=1),
                       dict(retry_scales=[1.,1.]),dict(retry_scales=[True]),dict(max_prompt_kl=float('nan')),
                       dict(consolidation_updates=129),dict(guard_examples_per_language=0)]:
            with self.assertRaises(ValueError):validate_configuration(checked(change))
        with self.assertRaises(ValueError):validate_configuration(checked(mode='cached_head'))
        with self.assertRaises(ValueError):validate_configuration(checked(missing='retention_guard.py'))

    def test_all_continual_and_archive_optimizer_sites_use_the_transaction_wrapper(self):
        tree=ast.parse(Path(runner.__file__).read_text())
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
               and node.func.id in {'continual_step','distill_step'}]
        self.assertEqual(len(calls),3)
        wrapped=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                 and node.func.id=='protected_update']
        self.assertEqual(len(wrapped),3)
        for node in wrapped:
            self.assertIsInstance(node.args[0],ast.Lambda)
            self.assertIn(node.args[0].body,calls)

    def test_nonfinite_rolled_back_transfer_keeps_recipe_diagnostics(self):
        options=dict(retention_weight=8.,champion_kl_weight=16.,memories=['it','en'])
        self.assertEqual(runner.transfer_retention_statistics({},options),
                         dict(retention_weight=8.,champion_kl_weight=16.,retention_examples=2))


if __name__=='__main__':unittest.main()
