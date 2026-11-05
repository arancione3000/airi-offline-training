"""Native checks for full-prompt preservation and the opt-in objective."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/transfer-training'))
import torch
from continual_learning import continual_step
from generalist_lm.training import reference_kl_loss
from retention_guard import full_sequence_kl, make_retention_guard
from success_transfer import distill_step


class PositionLogits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        values = torch.zeros(4, 8)
        values[0, 3] = 6; values[1, 4] = 6; values[2, 2] = 6
        self.values = torch.nn.Parameter(values)

    def forward(self, ids):
        return {'logits': self.values[:ids.shape[1]].unsqueeze(0).expand(ids.shape[0], -1, -1)}


class RetentionGuardTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model = PositionLogits()
        self.teacher = copy.deepcopy(self.model).eval()
        for p in self.teacher.parameters(): p.requires_grad_(False)
        self.ids = torch.tensor([[1, 3, 4, 2]])
        self.labels = torch.tensor([[-100, -100, 4, 2]])
        self.batch = (self.ids, self.labels)

    def test_prompt_regression_is_detected_even_when_every_old_answer_is_unchanged(self):
        guard = make_retention_guard(self.model, self.teacher, [self.batch], max_prompt_kl=.01)
        with torch.no_grad(): self.model.values[0, 7] = 12
        target_kl, _ = reference_kl_loss(self.model(self.ids)['logits'], self.teacher(self.ids)['logits'], self.labels)
        self.assertEqual(float(target_kl), 0.)
        outcome = guard()
        self.assertFalse(outcome['accepted'])
        self.assertTrue(any('full_sequence_kl' in r for r in outcome['reasons']))
        self.assertEqual(outcome['previously_correct_tokens'], outcome['preserved_tokens'])
        self.assertTrue(guard.before_observations[0]['correct'])
        self.assertTrue(outcome['observations'][0]['correct'])

    def test_old_correct_token_cannot_be_traded_for_a_better_mean(self):
        guard = make_retention_guard(self.model, self.teacher, [self.batch],
                                     max_prompt_kl=100, max_nll_regression=100)
        with torch.no_grad(): self.model.values[1, 5] = 7
        outcome = guard()
        self.assertFalse(outcome['accepted'])
        self.assertFalse(outcome['observations'][0]['correct'])
        self.assertTrue(guard.before_observations[0]['correct'])

    def test_already_degraded_memory_can_recover_without_a_rejection_deadlock(self):
        with torch.no_grad(): self.model.values[1, 5] = 8
        guard = make_retention_guard(self.model, self.teacher, [self.batch],
                                     max_prompt_kl=0, max_nll_regression=0)
        with torch.no_grad(): self.model.values[1, 5] = 7
        self.assertTrue(guard()['accepted'])  # Still wrong, but improving; EOS preserved.

    def test_padding_is_not_teacher_information(self):
        ids = torch.tensor([[1, 3, 4, 2, 0, 0]])
        old = torch.zeros(1, 6, 8); changed = old.clone()
        changed[:, 3:, 7] = 100
        kl, count = full_sequence_kl(changed, old, ids)
        self.assertEqual(count, 3)
        self.assertEqual(float(kl), 0.)

    def test_nonfinite_proposal_fails_closed_and_modes_are_restored(self):
        self.model.train()
        guard = make_retention_guard(self.model, self.teacher, [self.batch])
        self.assertTrue(self.model.training)
        with torch.no_grad(): self.model.values[0, 0] = float('nan')
        self.assertFalse(guard()['accepted'])
        self.assertTrue(self.model.training)

    def test_complete_certified_teacher_is_required(self):
        invalid = self.labels.clone(); invalid[0, -1] = -100
        self.assertRaises(ValueError, make_retention_guard, self.model, self.teacher, [(self.ids, invalid)])
        with torch.no_grad(): self.teacher.values[1, 5] = 12
        self.assertRaises(ValueError, make_retention_guard, self.model, self.teacher, [self.batch])

    def test_prompt_loss_trains_previously_unprotected_positions_and_zero_is_legacy(self):
        baseline = PositionLogits()
        with torch.no_grad(): baseline.values[0, 7] = 8
        plain, explicit_zero, protected = [copy.deepcopy(baseline) for _ in range(3)]
        for model, options in ((plain, {}), (explicit_zero, {'prompt_kl_weight': 0}),
                               (protected, {'prompt_kl_weight': 8})):
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0)
            stats = continual_step(model, self.teacher, self.teacher, optimizer,
                                    self.batch, [self.batch], self.batch,
                                    native_weight=0, **options)
            self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))
        self.assertTrue(torch.equal(plain.values, explicit_zero.values))
        self.assertTrue(torch.equal(plain.values[0], baseline.values[0]))
        self.assertLess(float(protected.values[0, 7].detach()), float(baseline.values[0, 7].detach()))
        self.assertEqual(stats['prompt_reference_tokens'], 3)

    def test_donor_consolidation_protects_prompt_and_zero_preserves_legacy(self):
        baseline = PositionLogits()
        with torch.no_grad(): baseline.values[0, 7] = 8
        plain, explicit_zero, protected = [copy.deepcopy(baseline) for _ in range(3)]
        entry = {'positions': [{'position': 2, 'target': 4,
                                'probabilities': [0, 0, 0, 0, 1, 0, 0, 0]}]}
        for model, options in ((plain, {}), (explicit_zero, {'prompt_kl_weight': 0}),
                               (protected, {'prompt_kl_weight': 8})):
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0)
            stats = distill_step(model, self.teacher, optimizer, entry,
                                 self.batch, self.batch, memories=[self.batch],
                                 lr_multiplier=1, lr_cap=.01, **options)
            self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))
            self.assertEqual(optimizer.param_groups[0]['lr'], .01)
        self.assertTrue(torch.equal(plain.values, explicit_zero.values))
        self.assertTrue(torch.equal(plain.values[0], baseline.values[0]))
        self.assertLess(float(protected.values[0, 7].detach()), float(baseline.values[0, 7].detach()))
        self.assertEqual(stats['prompt_reference_tokens'], 3)


if __name__ == '__main__':
    unittest.main()
