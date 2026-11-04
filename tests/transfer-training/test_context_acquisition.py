import copy
import hashlib
import json
import importlib.util
from pathlib import Path
import string
import sys
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/transfer-training'))
from context_acquisition import (build_context_acquisition, acquisition_audit,
                                 TEMPLATES, MAX_PAIRS, context_skeletons,
                                 context_value_pairs, matched_context_pair,
                                 acquisition_curriculum_digest,
                                 validate_acquisition_configuration, paired_batch)
from composition_data import PROTECTED_VALUES, SEALED_VALUES, PREVIOUS_SEALED_VALUES
from curriculum import norm, build
from context_curriculum import build_context
from foundation_data import build_foundation


class ContextAcquisitionTests(unittest.TestCase):
    def test_legacy_probe_cannot_attest_to_an_acquisition_recipe_it_does_not_run(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            spec = folder / 'recipe.json'
            spec.write_text(json.dumps({'continual_learning': {'context_acquisition': {}}}))
            result = subprocess.run(
                [sys.executable, str(root / 'scripts/transfer-training/learning_probe.py'),
                 '--checkpoint', str(folder / 'missing-model'),
                 '--archive', str(folder / 'missing-archive'),
                 '--reference', str(folder / 'missing-reference'),
                 '--native-replay', str(folder / 'missing-replay'),
                 '--baseline-result', str(folder / 'missing-baseline.json'),
                 '--output', str(folder / 'result.json'), '--spec', str(spec)],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('does not support matched context acquisition', result.stderr)
            self.assertFalse((folder / 'result.json').exists())

    def test_deterministic_bounded_inventory_and_broad_values(self):
        rows = build_context_acquisition()
        self.assertEqual(len(rows), 2048)
        self.assertEqual(rows, build_context_acquisition())
        audit = acquisition_audit(rows)
        self.assertEqual(audit['counterfactual_pairs'], MAX_PAIRS)
        self.assertTrue(audit['all_groups_have_two_rows'])
        self.assertEqual(audit['languages'], {'en': 1024, 'it': 1024})
        self.assertEqual(set(audit['length_buckets']), {0, 1, 2, 3})
        self.assertTrue(set(string.ascii_uppercase) <= set(audit['initials']))
        self.assertGreater(audit['unique_values'], 1000)
        self.assertTrue(any(any(c.isdigit() for c in value) for value in {r['expected'] for r in rows}))
        self.assertTrue(any(' ' in r['expected'] for r in rows))
        self.assertNotEqual(audit['sha256'], acquisition_audit(build_context_acquisition(seed=20261104))['sha256'])

    def test_each_pair_changes_only_source_value_and_gold_answer(self):
        rows = build_context_acquisition(max_pairs=64)
        for left, right in zip(rows[::2], rows[1::2]):
            self.assertEqual(left['acquisition_group'], right['acquisition_group'])
            self.assertEqual((left['counterfactual_index'], right['counterfactual_index']), (0, 1))
            self.assertNotEqual(left['expected'][0], right['expected'][0])
            self.assertEqual(left['messages'][1:-1], right['messages'][1:-1])
            self.assertEqual(left['messages'][-1]['content'], left['expected'])
            self.assertEqual(right['messages'][-1]['content'], right['expected'])
            self.assertEqual(left['messages'][0]['content'].replace(left['expected'], '{VALUE}'),
                             right['messages'][0]['content'].replace(right['expected'], '{VALUE}'))
            self.assertNotIn(left['expected'], left['messages'][-2]['content'])
            self.assertNotIn(right['expected'], right['messages'][-2]['content'])

    def test_existing_reserved_values_and_exact_blocked_text_never_supply_labels(self):
        old = build_context_acquisition(max_pairs=64)
        banned_value = old[0]['expected']
        banned_text = old[10]['messages'][0]['content']
        blocked = [banned_text.swapcase() + '  ']
        reserved = [banned_value.swapcase()]
        saved = copy.deepcopy((blocked, reserved))
        rows = build_context_acquisition(blocked, reserved, max_pairs=64)
        self.assertEqual((blocked, reserved), saved)
        forbidden = {norm(x) for x in PROTECTED_VALUES + SEALED_VALUES + PREVIOUS_SEALED_VALUES}
        forbidden.add(norm(banned_value))
        self.assertFalse({norm(r['expected']) for r in rows} & forbidden)
        self.assertFalse(any(norm(m['content']) == norm(banned_text) for r in rows for m in r['messages']))

    def test_existing_heldout_sets_are_unchanged_and_not_new_contexts(self):
        _, legacy = build()
        _, contexts = build_context()
        _, foundations = build_foundation()
        heldout = legacy + contexts + foundations
        before = hashlib.sha256(json.dumps(heldout, sort_keys=True).encode()).hexdigest()
        blocked = {norm(m['content']) for r in heldout for m in r['messages']}
        rows = build_context_acquisition(blocked, max_pairs=128)
        self.assertFalse(any(norm(m['content']) in blocked for r in rows for m in r['messages']))
        self.assertEqual(before, hashlib.sha256(json.dumps(heldout, sort_keys=True).encode()).hexdigest())
        prompts = {json.dumps(r['messages'][:-1], sort_keys=True) for r in rows}
        self.assertFalse(prompts & {json.dumps(r['messages'][:-1], sort_keys=True) for r in heldout})

    def test_invalid_or_exhausted_configuration_fails_finitely(self):
        for seed in (-1, True, 1.5, 2 ** 63):
            self.assertRaises(ValueError, build_context_acquisition, seed=seed)
        for pairs in (0, -1, True, 1.5, MAX_PAIRS + 1):
            self.assertRaises(ValueError, build_context_acquisition, max_pairs=pairs)
        questions = [question for by_task in TEMPLATES.values()
                     for variants in by_task.values() for _, question in variants]
        self.assertRaises(ValueError, build_context_acquisition, questions, max_pairs=1)

    def test_matched_pairs_keep_original_train_history_and_all_heldout_sets_unchanged(self):
        from success_bank import prompt_key
        train, heldout = build_context()
        before = copy.deepcopy((train, heldout))
        skeletons = context_skeletons([r for r in train if r['stage'] == 0])
        values = context_value_pairs(build_context_acquisition(max_pairs=64))
        protected = {prompt_key(r) for r in heldout}
        # These common fragments were already TRAIN and may overlap heldout;
        # forbidding whole histories must not prohibit valid original wording.
        blocked = {norm(m['content']) for r in heldout for m in r['messages']}
        for key, groups in skeletons.items():
            for group in groups:
                source = group[0]
                pair = matched_context_pair(source, values[key][0], blocked=blocked,
                             reserved_values={r['expected'] for r in heldout},
                             protected_histories=protected)
                self.assertEqual(pair[0]['messages'][1:-1], source['messages'][1:-1])
                self.assertEqual(pair[1]['messages'][1:-1], source['messages'][1:-1])
                for member in pair:
                    self.assertEqual(member['messages'][0]['content'],
                                     source['messages'][0]['content'].replace(source['expected'], member['expected']))
                    self.assertEqual(member['messages'][-1]['content'], member['expected'])
                    self.assertNotIn(prompt_key(member), protected)
                    self.assertEqual(member['stage'], source['stage'])
                self.assertNotEqual(pair[0]['expected'], pair[1]['expected'])
        self.assertEqual((train, heldout), before)

    def test_matched_pair_refuses_reserved_values_or_whole_protected_history(self):
        from success_bank import prompt_key
        train, _ = build_context()
        source = train[0]
        groups = context_value_pairs(build_context_acquisition(max_pairs=64))
        values = groups[(source['language'], source['family'])][0]
        pair = matched_context_pair(source, values)
        self.assertRaises(ValueError, matched_context_pair, source, values,
                          reserved_values=[values[0]['expected'].swapcase()])
        self.assertRaises(ValueError, matched_context_pair, source, values,
                          protected_histories=[prompt_key(pair[0])])
        self.assertRaises(ValueError, matched_context_pair, source, values,
                          blocked=[pair[0]['messages'][0]['content'].swapcase()])
        self.assertRaises(ValueError, matched_context_pair, source, values,
                          blocked=[values[0]['expected'].swapcase()])
        wrong = copy.deepcopy(source);wrong['family'] = 'extra-dialogue'
        self.assertRaises(ValueError, matched_context_pair, wrong, values)
        mixed = copy.deepcopy(values);mixed[1]['acquisition_group'] = 'unrelated'
        self.assertRaises(ValueError, matched_context_pair, source, mixed)

    def test_opt_in_recipe_is_bounded_and_bound_to_saved_curriculum_identity(self):
        from continual_learning import validate_configuration
        from curriculum import digest
        root = Path(__file__).resolve().parents[2]
        source = json.loads((root/'scripts/context-learning/SPEC.json').read_text())
        cfg = dict(version=1, seed=20261103, max_pairs=1024,
                   template_policy='matched_train', paired_batch=True)
        def checked(change=None, mode='cached_prefix'):
            spec = copy.deepcopy(source)
            spec['continual_learning'].update(training_mode=mode, context_acquisition={**cfg, **(change or {})})
            spec['script_sha256']['context_acquisition.py'] = hashlib.sha256(
                (root/'scripts/transfer-training/context_acquisition.py').read_bytes()).hexdigest()
            spec.pop('fingerprint')
            spec['fingerprint'] = hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            return spec
        self.assertEqual(validate_configuration(checked())['context_acquisition'], cfg)
        for change in (dict(version=True), dict(version=2), dict(seed=True), dict(seed=-1),
                       dict(max_pairs=7), dict(max_pairs=1025), dict(paired_batch=False),
                       dict(template_policy='evaluation'), dict(extra='typo')):
            self.assertRaises(ValueError, validate_configuration, checked(change))
        self.assertRaises(ValueError, validate_configuration, checked(mode='cached_head'))
        unpinned = checked();del unpinned['script_sha256']['context_acquisition.py']
        unpinned.pop('fingerprint')
        unpinned['fingerprint'] = hashlib.sha256(json.dumps(unpinned,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.assertRaises(ValueError, validate_configuration, unpinned)
        self.assertRaises(ValueError, validate_acquisition_configuration, None)
        base = [{'messages': [{'role':'user','content':'original'}]}]
        values = build_context_acquisition(max_pairs=8)
        self.assertEqual(acquisition_curriculum_digest(base), digest(base))
        identity = acquisition_curriculum_digest(base,cfg,values)
        saved = json.loads(json.dumps(dict(base=base,configuration=cfg,values=values)))
        self.assertEqual(identity, acquisition_curriculum_digest(saved['base'],saved['configuration'],saved['values']))
        self.assertNotEqual(identity, acquisition_curriculum_digest(base,{**cfg,'seed':20261104},values))
        changed = copy.deepcopy(values);changed[0]['expected'] = 'Changed'
        self.assertNotEqual(identity, acquisition_curriculum_digest(base,cfg,changed))
        self.assertRaises(ValueError, acquisition_curriculum_digest, base, None, values)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'native tensor dependency is optional in orchestration checks')
    def test_paired_causal_batch_padding_and_supervised_token_accounting(self):
        import torch
        from generalist_lm.model import GeneralistLMConfig
        from generalist_lm.runtime import GeneralistRuntime
        from generalist_lm.training import causal_training_objective
        from continual_learning import continual_step
        from prefix_tuning import configure_prefix
        from generalist_lm.tokenizer import PAD
        torch.set_num_threads(2)
        short_ids = torch.tensor([[1, 12, 20, 30, 2]])
        long_ids = torch.tensor([[1, 12, 20, 31, 32, 33, 2]])
        short_labels = short_ids.clone();short_labels[:, :3] = -100
        long_labels = long_ids.clone();long_labels[:, :3] = -100
        original = [(short_ids.clone(),short_labels.clone()),(long_ids.clone(),long_labels.clone())]
        batch = paired_batch([(short_ids,short_labels),(long_ids,long_labels)])
        self.assertEqual(tuple(batch[0].shape),(2,7))
        self.assertTrue(bool((batch[0][0,5:]==PAD).all()))
        self.assertTrue(bool((batch[1][0,5:]==-100).all()))
        self.assertEqual(int((batch[1][:,1:]!=-100).sum()),6)
        for supplied,saved in zip(((short_ids,short_labels),(long_ids,long_labels)),original):
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(supplied,saved)))
        runtime = GeneralistRuntime.fresh(GeneralistLMConfig(d_model=16,n_layers=2,n_heads=2,d_ff=32,context_length=16))
        with torch.no_grad():
            batched = runtime.model(batch[0])['logits']
            torch.testing.assert_close(batched[0,:5],runtime.model(short_ids)['logits'][0],atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(batched[1],runtime.model(long_ids)['logits'][0],atol=1e-6,rtol=1e-5)
            # Every prediction after the short row's EOS targets ignored padding.
            changed_padding = batched.clone();changed_padding[0,4:] += 100 * torch.randn_like(changed_padding[0,4:])
            objective = lambda logits: causal_training_objective(logits,batch[1],batch[0],
                           repetition_unlikelihood_weight=.05,eos_loss_weight=1.2)[0]
            torch.testing.assert_close(objective(batched),objective(changed_padding),atol=0,rtol=0)
        champion = copy.deepcopy(runtime.model);reference = copy.deepcopy(runtime.model)
        for model in (champion,reference):
            for parameter in model.parameters():parameter.requires_grad_(False)
        cache = configure_prefix(runtime.model,cache_bytes=100000)
        with torch.no_grad():
            torch.testing.assert_close(runtime.model(batch[0])['logits'],batched,atol=1e-6,rtol=1e-5)
        frozen = {name:value.detach().clone() for name,value in runtime.model.named_parameters()
                  if not value.requires_grad}
        attention_before = runtime.model.blocks[-1].attn.out.weight.detach().clone()
        class CountedAdam(torch.optim.AdamW):
            calls = 0
            def step(self,*args,**kwargs):
                self.calls += 1
                return super().step(*args,**kwargs)
        optimizer = CountedAdam([value for value in runtime.model.parameters() if value.requires_grad],lr=1e-4)
        memory = (short_ids,short_labels)
        result = continual_step(runtime.model,champion,reference,optimizer,batch,[memory],memory)
        self.assertEqual(optimizer.calls,1)
        self.assertEqual(result['supervised_tokens'],6+2+2)
        self.assertGreater(cache.hits,0)
        self.assertFalse(torch.equal(runtime.model.blocks[-1].attn.out.weight,attention_before))
        for name,value in runtime.model.named_parameters():
            if name in frozen:self.assertTrue(torch.equal(value,frozen[name]))
        self.assertTrue(all(value.grad is None for model in (champion,reference) for value in model.parameters()))
        self.assertRaises(ValueError,paired_batch,[memory])


if __name__ == '__main__':
    unittest.main()
