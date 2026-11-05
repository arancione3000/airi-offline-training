"""TRAIN-only functional retention, including the question and conversation.

These sentinels constrain individual updates. They neither consume evaluation
labels nor replace independent development/adoption gates. A sampled sentinel
does not guarantee preservation on every unseen formulation.
"""
import math


def full_sequence_kl(logits, teacher_logits, ids):
    """Compare every real next-token distribution; padding is never a target."""
    from generalist_lm.tokenizer import PAD
    from generalist_lm.training import reference_kl_loss
    labels = ids.clone()
    labels[ids == PAD] = -100
    return reference_kl_loss(logits, teacher_logits, labels)


def _frame(model, batch):
    import torch
    from torch.nn import functional as F
    from generalist_lm.tokenizer import EOS
    ids, labels = batch
    if (ids.ndim != 2 or ids.shape[0] != 1 or labels.shape != ids.shape
            or ids.shape[1] < 2 or ids.dtype != torch.long or labels.dtype != torch.long):
        raise ValueError('retention guard requires a complete single TRAIN example')
    targets = labels[:, 1:]
    mask = targets != -100
    if (not bool(mask.any()) or int(targets[0, -1]) != EOS
            or not bool((ids[:, 1:][mask] == targets[mask]).all())):
        raise ValueError('retention guard requires gold TRAIN targets including final EOS')
    logits = model(ids)['logits'].float()
    if logits.shape[:2] != ids.shape:
        raise ValueError('invalid native retention logits')
    if not bool(torch.isfinite(logits).all()):
        return None
    scores = logits[:, :-1][mask]
    gold = targets[mask]
    correct = scores.argmax(-1) == gold
    nll = float(F.cross_entropy(scores, gold))
    alternatives = scores.clone()
    alternatives.scatter_(1, gold[:, None], -float('inf'))
    margin = float((scores.gather(1, gold[:, None]).squeeze(1) - alternatives.max(-1).values).min())
    return {'logits': logits.detach(), 'correct': correct.detach(), 'nll': nll,
            'certificate': dict(correct=bool(correct.all()), complete=True, includes_eos=True,
                                token_count=int(mask.sum()), min_margin=margin, mean_nll=nll)}


def make_retention_guard(model, teacher, batches, *, max_prompt_kl=.01,
                         max_nll_regression=.02):
    """Capture a pre-proposal TRAIN baseline and return a read-only step guard.

    Previously correct target tokens cannot flip. Full-distribution KL and
    answer NLL are bounded against the frozen champion. If a sampled example
    was already degraded by an earlier unobserved update, the current proposal
    may repair it over several steps, but may not worsen its baseline metrics.
    Every protected whole answer must still be correct before final adoption.
    """
    import torch
    from success_transfer import frozen_independent
    frozen_independent(model, teacher)
    if not batches:
        raise ValueError('TRAIN sentinels cannot be empty')
    for value in (max_prompt_kl, max_nll_regression):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError('invalid finite retention guard threshold')

    def measure_baseline():
        frames = []
        modes = [(module, module.training) for m in (model, teacher) for module in m.modules()]
        try:
            model.eval(); teacher.eval()
            with torch.no_grad():
                for batch in batches:
                    old = _frame(teacher, batch)
                    before = _frame(model, batch)
                    if old is None or before is None:
                        raise ValueError('nonfinite pre-proposal retention evidence')
                    if not bool(old['correct'].all()):
                        raise ValueError('sentinel is not a certified champion TRAIN answer')
                    kl, count = full_sequence_kl(before['logits'], old['logits'], batch[0])
                    divergence = float(kl)
                    if not count or not math.isfinite(divergence):
                        raise ValueError('invalid full-sequence retention baseline')
                    frames.append((batch, old, before, divergence))
        finally:
            for module, training in modes:
                module.training = training
        return frames

    frames = measure_baseline()

    def guard():
        reasons = []
        observations = []
        nll_deltas, divergences, preserved, previously_correct = [], [], 0, 0
        modes = [(module, module.training) for module in model.modules()]
        try:
            model.eval()
            with torch.no_grad():
                for index, (batch, old, before, before_kl) in enumerate(frames):
                    after = _frame(model, batch)
                    if after is None:
                        return dict(accepted=False, reason='nonfinite_retention_logits', rows=len(frames))
                    observations.append(after['certificate'])
                    prior_correct = before['correct']
                    previously_correct += int(prior_correct.sum())
                    preserved += int((after['correct'] & prior_correct).sum())
                    if not bool(after['correct'][prior_correct].all()):
                        reasons.append(f'TRAIN:{index}:previously_correct_token_lost')
                    kl, count = full_sequence_kl(after['logits'], old['logits'], batch[0])
                    divergence = float(kl)
                    if not count or not math.isfinite(divergence):
                        return dict(accepted=False, reason='nonfinite_retention_kl', rows=len(frames))
                    divergences.append(divergence)
                    nll_deltas.append(after['nll'] - old['nll'])
                    if divergence > max(max_prompt_kl, before_kl) + 1e-12:
                        reasons.append(f'TRAIN:{index}:full_sequence_kl')
                    if after['nll'] > max(old['nll'] + max_nll_regression, before['nll']) + 1e-12:
                        reasons.append(f'TRAIN:{index}:answer_nll')
        finally:
            for module, training in modes:
                module.training = training
        return dict(accepted=not reasons, reasons=reasons, rows=len(frames),
                    observations=observations,
                    max_full_sequence_kl=max(divergences),
                    max_answer_nll_increase=max(nll_deltas),
                    previously_correct_tokens=previously_correct,
                    preserved_tokens=preserved,
                    scope='sampled certified TRAIN sentinels; independent adoption gates remain required')
    guard.before_observations = [before['certificate'] for _, _, before, _ in frames]
    return guard


def retention_guard(model, teacher, batches, **kwargs):
    return make_retention_guard(model, teacher, batches, **kwargs)()
