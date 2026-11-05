"""Ephemeral 32-unit preflight of the actual persistent-retention training path.

Only the caller's audited eligible TRAIN catalog grants labels. This helper
never calls development evaluation, adopts a champion or publishes a checkpoint.
Its temporary checkpoint additionally proves one exact continuation unit on two
branches; those two validation exposures are reported separately from training.
"""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import time

import torch
from torch.nn import functional as F

from continual_learning import continual_step, retention_rows, verified_memories
from fast_head import install_feature_cache
from generalist_lm.runtime import GeneralistRuntime
from generalist_lm.tokenizer import EOS
from prefix_tuning import configure_prefix, native_evaluation
from retention_guard import make_retention_guard
from retention_memory import (empty_memory, make_certifier, observe, refresh,
                              sample_rows, summary, validate_memory)
from retention_transaction import transactional_update
from session import restore_session, save_session
from success_bank import catalog, row_key


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def real_retention_smoke(runtime, checkpoint, reference, eligible, human_rows,
                         encode, anchors, evaluate, state_digest, config,
                         learning, fingerprint):
    """Exercise real ledger replay, prompt KL, guarded retries and exact resume.

    The caller restricts this path to its explicitly pinned migration seed.
    Existing retention sessions must use their authenticated ledger/champion
    rather than reconstructing knowledge from a possibly degraded candidate.
    ``human_rows`` and ``evaluate`` are API-compatible with older smoke helpers;
    neither can add labels or held-out feedback to this preflight.
    """
    if not hasattr(runtime.model, '_airi_prefix_features'):
        raise ValueError('retention preflight requires the real configured native prefix')
    if not anchors:
        raise ValueError('retention preflight requires native TRAIN replay anchors')
    checkpoint = Path(checkpoint)
    profile = learning['profiles'][0]
    cache_bytes = learning['feature_cache_mebibytes'] * 1024 * 1024
    # Loading/configuring frozen teachers must not consume the training stream.
    stream = torch.get_rng_state().clone()
    try:
        champion = GeneralistRuntime.from_checkpoint(checkpoint / 'model')
        champion.model.eval()
        for parameter in champion.model.parameters(): parameter.requires_grad_(False)
        install_feature_cache(champion.model, cache_bytes=min(cache_bytes, 64 * 1024 * 1024))
    finally:
        torch.set_rng_state(stream)
    protected_digest = state_digest(champion.model)
    reference_digest = state_digest(reference.model)
    certifier = make_certifier(champion.model, encode, EOS)
    old_rows = verified_memories(champion, retention_rows(list(eligible.values())), encode)
    if len(old_rows) > config['capacity']:
        raise ValueError('preflight capacity cannot hold every legacy certified TRAIN answer')
    memory = refresh(protected_digest, catalog(old_rows, []), certifier,
                     empty_memory(fingerprint, capacity=config['capacity']), probe_limit=len(old_rows))
    memory = refresh(protected_digest, eligible, certifier, memory,
                     probe_limit=config['refresh_probe_limit'])
    if {entry['row']['language'] for entry in memory['entries'].values()} != {'en', 'it'}:
        raise ValueError('preflight requires actual certified TRAIN answers in both languages')
    initial_rows = len(memory['entries'])
    print(json.dumps(dict(phase='retention preflight TRAIN ledger initialized',
                          protected_rows=initial_rows, legacy_certified_rows=len(old_rows),
                          retention_languages=summary(memory)['languages'])), flush=True)
    # Materialized context examples are not created by this bounded preflight.
    # Persist the same authenticated empty registry format as the real session.
    registry = dict(schema=1, fingerprint=fingerprint, source_fingerprints=[fingerprint], entries={})
    tasks = sorted((row for row in eligible.values() if row['family'] == 'extra-dialogue'),
                   key=lambda row: (len(row['expected']), row_key(row)))
    if not tasks:
        raise ValueError('retention preflight needs an eligible authored dialogue TRAIN task')

    def nll(model, batch):
        modes = [(module, module.training) for module in model.modules()]
        try:
            model.eval()
            with torch.no_grad():
                logits = model(batch[0])['logits'][:, :-1].float()
                return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[1][:, 1:].reshape(-1)))
        finally:
            for module, mode in modes: module.training = mode

    # Fixed TRAIN-only selection, bounded to sixteen short authored tasks.
    task = max(tasks[:16], key=lambda row: nll(champion.model, encode(row)))
    batch = encode(task)
    initial = nll(runtime.model, batch)
    rng, anchor_rng = random.Random(42071), random.Random(42072)
    counters = dict(completed_steps=0, supervised_tokens=0, committed_supervised_tokens=0,
                    reference_tokens=0, committed_reference_tokens=0,
                    attempted_prompt_reference_tokens=0, committed_prompt_reference_tokens=0,
                    retention_transaction=dict(schema=1, lr_scale=1., attempted_updates=0,
                                               committed_updates=0, rejected_updates=0))

    def factory(model):
        configure_prefix(model, cache_bytes=cache_bytes)
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=profile['learning_rate'], weight_decay=.01)

    optimizer = factory(runtime.model)

    def update(rt, opt, bank, progress, sampler, replay_sampler):
        memories = [encode(row) for row in sample_rows(bank, eligible, replay_sampler,
                    per_language=learning.get('retention_examples_per_language', 4))]
        sentinels = sample_rows(bank, eligible, replay_sampler,
                               per_language=config['guard_examples_per_language'])
        guard = make_retention_guard(rt.model, champion.model, [encode(row) for row in sentinels],
                                    max_prompt_kl=config['max_prompt_kl'],
                                    max_nll_regression=config['max_nll_regression'])
        anchor = anchors[sampler.randrange(len(anchors))]
        proposed = []
        def propose():
            stats = continual_step(rt.model, champion.model, reference.model, opt, batch,
                                   memories, anchor, retention_weight=profile['retention_weight'],
                                   champion_kl_weight=profile['champion_kl_weight'],
                                   prompt_kl_weight=config['prompt_kl_weight'])
            proposed.append(stats)
            return stats
        result = transactional_update(rt.model, opt, propose, guard,
                    state=progress['retention_transaction'], scales=config['retry_scales'],
                    sampler_rngs=(sampler, replay_sampler))
        observations = result['guard']['observations'] if result['accepted'] else guard.before_observations
        if len(observations) != len(sentinels):
            raise AssertionError('preflight lost post-transaction certificate evidence')
        evidence = {row_key(row): value for row, value in zip(sentinels, observations)}
        bank = observe(bank, eligible, lambda row: evidence[row_key(row)], keys=list(evidence))
        progress['completed_steps'] += 1
        for field in ('supervised_tokens', 'reference_tokens'):
            progress[field] += sum(int(stats[field]) for stats in proposed)
            progress['committed_' + field] += int(result['stats'].get(field, 0)) if result['accepted'] else 0
        progress['attempted_prompt_reference_tokens'] += sum(int(stats['prompt_reference_tokens']) for stats in proposed)
        progress['committed_prompt_reference_tokens'] += int(result['stats'].get('prompt_reference_tokens', 0)) if result['accepted'] else 0
        return bank, result

    tick = time.monotonic()
    for unit in range(32):
        memory, last_result = update(runtime, optimizer, memory, counters, rng, anchor_rng)
        if (unit + 1) % 8 == 0:
            print(json.dumps(dict(phase='retention preflight TRAIN progress', units=unit + 1,
                                  **counters['retention_transaction'])), flush=True)
    training_seconds = time.monotonic() - tick
    final = nll(runtime.model, batch)
    training_counters = deepcopy(counters)
    memory = observe(memory, eligible, make_certifier(runtime.model, encode, EOS))
    lost = [key for key, entry in memory['entries'].items() if not entry['observation']['correct']]
    if lost:
        raise AssertionError('retention preflight lost previously certified TRAIN answers: ' + str(len(lost)))
    if training_counters['retention_transaction']['committed_updates'] and not final < initial:
        raise AssertionError('guarded native objective did not lower its selected TRAIN task NLL')
    if not training_counters['retention_transaction']['committed_updates']:
        raise AssertionError('retention preflight committed no real optimizer update')
    validate_memory(memory, eligible, fingerprint, protected_digest)

    # Native checkpoint export is checked before the extra continuation proof.
    # Load one restored model only; no full frozen-model deepcopy is required.
    runtime.model.eval()
    with torch.no_grad(): exported_logits = runtime.model(batch[0])['logits'].detach().clone()
    prompt = task['messages'][:-1]
    exported_chat = runtime.chat(prompt, temperature=0, max_new_tokens=32)
    with tempfile.TemporaryDirectory(prefix='airi-retention-preflight-') as temporary:
        path = Path(temporary) / 'checkpoint'
        save_session(runtime, optimizer, rng, anchor_rng, counters, path, fingerprint=fingerprint,
                     retention_memory=memory, training_registry=registry)
        # First branch has one physical continuation exposure (possibly retries).
        memory, expected_result = update(runtime, optimizer, memory, counters, rng, anchor_rng)
        expected_torch = torch.get_rng_state().clone()
        restored, restored_optimizer, payload = restore_session(path, factory,
                                    fingerprint=fingerprint, require_retention=True)
        if payload['training_registry'] != registry or payload['retention_memory']['entries'].keys() != memory['entries'].keys():
            raise AssertionError('native session lost authenticated registry or protected membership')
        restored.model.eval()
        with torch.no_grad(), native_evaluation(restored.model):
            if not torch.equal(exported_logits, restored.model(batch[0])['logits']):
                raise AssertionError('native exported retention logits changed')
            if exported_chat != restored.chat(prompt, temperature=0, max_new_tokens=32):
                raise AssertionError('native exported retention chat changed')
        # Inference validation must not affect the resumed training RNG.
        torch.set_rng_state(payload['torch_rng'])
        resumed_rng, resumed_anchor_rng = random.Random(), random.Random()
        resumed_rng.setstate(payload['sampler_rng']); resumed_anchor_rng.setstate(payload['anchor_rng'])
        resumed_memory, resumed_result = update(restored, restored_optimizer, payload['retention_memory'],
                        payload['counters'], resumed_rng, resumed_anchor_rng)
        if (not all(torch.equal(value, restored.model.state_dict()[name]) for name, value in runtime.model.state_dict().items())
                or not _equal(optimizer.state_dict(), restored_optimizer.state_dict())
                or any(not _equal(left.grad, right.grad)
                       for left, right in zip(runtime.model.parameters(), restored.model.parameters()))
                or memory != resumed_memory or counters != payload['counters']
                or expected_result != resumed_result or rng.getstate() != resumed_rng.getstate()
                or anchor_rng.getstate() != resumed_anchor_rng.getstate()
                or not torch.equal(expected_torch, torch.get_rng_state())):
            raise AssertionError('retention model/Adam/ledger/adaptation/sampler continuation was not exact')
    if (state_digest(champion.model) != protected_digest or state_digest(reference.model) != reference_digest
            or any(p.grad is not None for teacher in (champion.model, reference.model) for p in teacher.parameters())):
        raise AssertionError('frozen native champion or original reference changed during preflight')
    adaptive = training_counters['retention_transaction']
    report = dict(phase='real native persistent retention preflight PASS', units=32,
                  physical_attempts=adaptive['attempted_updates'], committed_updates=adaptive['committed_updates'],
                  rejected_attempts=adaptive['rejected_updates'], lr_scale=adaptive['lr_scale'],
                  initial_training_task_nll=initial, final_training_task_nll=final,
                  training_retention_rows=initial_rows, retention_languages=summary(memory)['languages'],
                  full_bank_lost=0, training_seconds=training_seconds,
                  attempted_supervised_tokens=training_counters['supervised_tokens'],
                  committed_supervised_tokens=training_counters['committed_supervised_tokens'],
                  attempted_prompt_reference_tokens=training_counters['attempted_prompt_reference_tokens'],
                  committed_prompt_reference_tokens=training_counters['committed_prompt_reference_tokens'],
                  resume_proof_units_per_branch=1, resume_proof_branches=2,
                  resume_proof_physical_attempts_per_branch=expected_result['attempts'],
                  resume_proof_committed_per_branch=int(expected_result['accepted']),
                  champion_unchanged=True, reference_unchanged=True, resume_exact=True,
                  native_export_exact=True, development_evaluations=0, live_promoted=False,
                  scope='ephemeral TRAIN task/retention/integrity check; no held-out improvement or adoption claim')
    print(json.dumps(report, allow_nan=False), flush=True)
    return report
