"""Native-dependency checks for prefix configuration, interpolation and migration."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/transfer-training'))

import torch
from generalist_lm.model import GeneralistLMConfig
from generalist_lm.runtime import GeneralistRuntime
from continual_learning import try_interpolations, validate_configuration
from fast_head import configure_head
from prefix_tuning import configure_prefix, native_evaluation
from session import restore_session, save_session


class PrefixIntegrationTests(unittest.TestCase):
    @staticmethod
    def runtime():
        return GeneralistRuntime.fresh(GeneralistLMConfig(d_model=32, n_layers=2,
                                                         n_heads=2, d_ff=64,
                                                         context_length=64))

    @staticmethod
    def checked_configuration(**change):
        root = Path(__file__).resolve().parents[2]
        spec = json.loads((root / 'scripts/continual-training/SPEC.json').read_text())
        spec['continual_learning'].update(training_mode='cached_prefix',
                                         human_training_scope='reviewed_only',
                                         feature_cache_mebibytes=128,
                                         protected_head_rank=0)
        spec['continual_learning'].update(change)
        spec.pop('fingerprint')
        spec['fingerprint'] = hashlib.sha256(json.dumps(spec, sort_keys=True,
                                                       separators=(',', ':')).encode()).hexdigest()
        return spec

    def test_prefix_configuration_is_bounded_reviewed_and_never_head_projected(self):
        self.assertEqual(validate_configuration(self.checked_configuration())['training_mode'],
                         'cached_prefix')
        for values in (dict(feature_cache_mebibytes=True),
                       dict(feature_cache_mebibytes=1024),
                       dict(feature_cache_mebibytes=15),
                       dict(human_training_scope='legacy_and_reviewed'),
                       dict(protected_head_rank=8)):
            with self.subTest(values=values):
                self.assertRaises(ValueError, validate_configuration,
                                  self.checked_configuration(**values))

    def test_rejected_interpolation_restores_tail_and_preserves_cached_prefix(self):
        torch.manual_seed(37)
        champion = self.runtime()
        candidate = self.runtime()
        candidate.model.load_state_dict(champion.model.state_dict())
        cache = configure_prefix(candidate.model, cache_bytes=100000)
        ids = torch.tensor([[1, 4, 18, 33, 5]])
        with torch.no_grad():
            candidate.model.blocks[-1].attn.out.weight.add_(.02)
            candidate.model.lm_head.weight.add_(.01)
        original = copy.deepcopy(candidate.model.state_dict())
        candidate.model(ids)
        versions = cache.versions
        entry = next(iter(cache.entries.values()))
        probes = []

        def evaluate():
            candidate.model(ids)
            probes.append(candidate.model.blocks[-1].attn.out.weight.detach().clone())
            return 'rejected'

        result = try_interpolations(candidate.model, champion.model,
                                    evaluate=evaluate, acceptable=lambda value: False)
        self.assertIsNone(result['alpha'])
        self.assertEqual(result['probes'], 2)
        self.assertEqual(len(probes), 2)
        self.assertTrue(all(torch.equal(value, candidate.model.state_dict()[name])
                            for name, value in original.items()))
        self.assertEqual(cache.versions, versions)
        self.assertEqual(cache.invalidations, 0)
        self.assertIs(next(iter(cache.entries.values())), entry)
        self.assertEqual(cache.hits, 2)
        self.assertTrue(torch.equal(candidate.model(ids)['logits'], cache.original(ids)['logits']))

    def test_accepted_interpolation_changes_only_declared_tail(self):
        champion = self.runtime()
        candidate = self.runtime()
        candidate.model.load_state_dict(champion.model.state_dict())
        cache = configure_prefix(candidate.model)
        with torch.no_grad():
            for name, parameter in candidate.model.named_parameters():
                if parameter.requires_grad:
                    parameter.add_(.1)
        original = copy.deepcopy(candidate.model.state_dict())
        base = champion.model.state_dict()
        result = try_interpolations(candidate.model, champion.model,
                                    evaluate=lambda: 'passes unchanged gates',
                                    acceptable=lambda value: True, alphas=(.25,))
        self.assertEqual(result['alpha'], .25)
        for name, value in candidate.model.state_dict().items():
            if name in cache.trainable_names:
                self.assertTrue(torch.equal(value, torch.lerp(base[name], original[name], .25)))
            else:
                self.assertTrue(torch.equal(value, original[name]))
        self.assertEqual(cache.invalidations, 0)

    def test_frozen_weight_mismatch_rejects_before_any_evaluation_or_mutation(self):
        champion = self.runtime()
        candidate = self.runtime()
        candidate.model.load_state_dict(champion.model.state_dict())
        configure_prefix(candidate.model)
        with torch.no_grad():
            candidate.model.blocks[0].ff.up.weight.add_(.01)
        original = copy.deepcopy(candidate.model.state_dict())
        probes = []
        self.assertRaises(ValueError, try_interpolations, candidate.model, champion.model,
                          evaluate=lambda: probes.append(True), acceptable=lambda _: True)
        self.assertFalse(probes)
        self.assertTrue(all(torch.equal(value, candidate.model.state_dict()[name])
                            for name, value in original.items()))

    def test_explicit_head_to_prefix_migration_starts_new_adam_then_resumes_it(self):
        torch.set_num_threads(2)
        runtime = self.runtime()
        configure_head(runtime.model)
        head_adam = torch.optim.AdamW([p for p in runtime.model.parameters() if p.requires_grad], lr=.001)
        ids = torch.tensor([[1, 4, 18, 33, 5]])
        head_adam.zero_grad()
        runtime.model(ids, ids)['loss'].backward()
        head_adam.step()
        sampler = random.Random(8)

        def factory(model):
            configure_prefix(model, cache_bytes=10000)
            return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'head'
            save_session(runtime, head_adam, sampler, sampler, dict(completed_steps=1),
                         source, fingerprint='head-seed')
            # The incompatible scope must never silently load the old Adam.
            self.assertRaises(ValueError, restore_session, source, factory,
                              fingerprint='head-seed')
            migrated, adam, _ = restore_session(source, factory, fingerprint='head-seed',
                                                load_optimizer=False)
            self.assertFalse(adam.state)
            self.assertTrue(all(torch.equal(value, migrated.model.state_dict()[name])
                                for name, value in runtime.model.state_dict().items()))
            adam.zero_grad()
            migrated.model(ids, ids)['loss'].backward()
            adam.step()
            destination = Path(directory) / 'prefix'
            save_session(migrated, adam, sampler, sampler, dict(completed_steps=2),
                         destination, fingerprint='prefix-descendant')
            resumed, resumed_adam, _ = restore_session(destination, factory,
                                                       fingerprint='prefix-descendant')
            for rt, optimizer in ((migrated, adam), (resumed, resumed_adam)):
                optimizer.zero_grad()
                rt.model(ids, ids)['loss'].backward()
                optimizer.step()
            self.assertTrue(all(torch.equal(value, resumed.model.state_dict()[name])
                                for name, value in migrated.model.state_dict().items()))

    def test_historical_native_metrics_can_replace_context_and_restore_cache_guard(self):
        runtime = self.runtime()
        cache = configure_prefix(runtime.model, cache_bytes=10000)
        ids = torch.tensor([[1, 4, 18, 33, 5]])
        expected = runtime.model(ids)['logits'].detach()
        entry = next(iter(cache.entries.values()))
        wrapper = runtime.model.forward
        old_config, old_model_config = runtime.config, runtime.model.config

        def historical_metrics(*, fail=False):
            # Exact temporary replacement pattern used by common_metrics.
            runtime.config = replace(old_config, context_length=128)
            runtime.model.config = replace(old_model_config, context_length=128)
            try:
                if fail:
                    raise RuntimeError('evaluation failed')
                return runtime.model(ids)['logits'].detach()
            finally:
                runtime.config = old_config
                runtime.model.config = old_model_config

        # Training continues to reject an unannounced configuration change.
        self.assertRaises(RuntimeError, historical_metrics)
        with native_evaluation(runtime.model):
            self.assertTrue(torch.equal(historical_metrics(), expected))
            self.assertEqual(runtime.model.forward, cache.original)
        self.assertEqual(runtime.model.forward, wrapper)
        self.assertTrue(torch.equal(runtime.model(ids)['logits'], expected))
        self.assertEqual(cache.invalidations, 0)
        self.assertIs(next(iter(cache.entries.values())), entry)
        with self.assertRaisesRegex(RuntimeError, 'evaluation failed'):
            with native_evaluation(runtime.model):
                historical_metrics(fail=True)
        self.assertEqual(runtime.model.forward, wrapper)
        self.assertIs(runtime.config, old_config)
        self.assertIs(runtime.model.config, old_model_config)
        self.assertTrue(torch.equal(runtime.model(ids)['logits'], expected))
        self.assertEqual(cache.invalidations, 0)


if __name__ == '__main__':
    unittest.main()
