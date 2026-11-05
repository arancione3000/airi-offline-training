"""Real native-prefix Adam transactions: retry, exact rollback, resume and guards."""
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/transfer-training'))
import torch
from generalist_lm.model import GeneralistLMConfig
from generalist_lm.runtime import GeneralistRuntime
from prefix_tuning import configure_prefix_tuning
from retention_transaction import transactional_update


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def fresh():
    torch.manual_seed(704)
    model = GeneralistRuntime.fresh(GeneralistLMConfig(
        d_model=24, n_layers=2, n_heads=2, d_ff=48, context_length=32,
        position_encoding='rope', attention_type='gqa', n_kv_heads=1,
        norm_type='rmsnorm')).model
    configure_prefix_tuning(model, cache_bytes=8192, cache_entries=8)
    adam = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.003)
    return model, adam, random.Random(771)


def proposal(model, adam, sampler):
    # Both checkpointed RNG sources affect the actual gradients, so equality of
    # models/moments after retry is a meaningful trajectory test.
    token = 12 + sampler.randrange(12)
    ids = torch.tensor([[1, 6, token, 22, 31, 2]])
    labels = ids.clone(); labels[:, :3] = -100
    adam.zero_grad(set_to_none=True)
    loss = model(ids, labels)['loss'] * (1 + .05 * torch.rand(()))
    loss.backward()
    adam.step()
    return {'loss': float(loss.detach()), 'token': token}


def snapshot(model, adam, sampler):
    return {'model': deepcopy(model.state_dict()), 'adam': deepcopy(adam.state_dict()),
            'grads': [None if p.grad is None else p.grad.detach().clone()
                      for p in model.parameters()],
            'torch_rng': torch.get_rng_state().clone(), 'sampler': sampler.getstate(),
            'modes': [m.training for m in model.modules()]}


def assert_snapshot(before, model, adam, sampler):
    assert equal(before, snapshot(model, adam, sampler)), 'writable trajectory was not restored exactly'


def restore(model, adam, sampler, saved):
    model.load_state_dict(saved['model']); adam.load_state_dict(deepcopy(saved['adam']))
    for p, grad in zip(model.parameters(), saved['grads']):
        p.grad = None if grad is None else grad.clone()
    torch.set_rng_state(saved['torch_rng']); sampler.setstate(saved['sampler'])
    for m, mode in zip(model.modules(), saved['modes']): m.training = mode


def run_checks():
    torch.set_num_threads(2)
    model, adam, sampler = fresh()
    # Preexisting Adam moments and gradients must survive; an empty fresh Adam
    # would not expose accidental moment resets.
    proposal(model, adam, sampler)
    model.train(); model.blocks[0].eval()
    before = snapshot(model, adam, sampler)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    state = {}
    calls = 0
    def guard():
        nonlocal calls
        calls += 1
        # A read-only guard may use randomness/eval without perturbing training.
        torch.rand(4); sampler.random(); model.eval()
        return {'accepted': calls == 2, 'margin': .5}
    result = transactional_update(model, adam, lambda: proposal(model, adam, sampler), guard,
                                  state=state, scales=(1, .5), sampler_rngs=(sampler,))
    actual = snapshot(model, adam, sampler)
    assert result['accepted'] and result['attempts'] == 2 and result['scale'] == .5
    assert state == dict(schema=1, lr_scale=.5, attempted_updates=2, committed_updates=1, rejected_updates=1)
    restore(model, adam, sampler, before)
    adam.param_groups[0]['lr'] *= .5
    proposal(model, adam, sampler)
    adam.param_groups[0]['lr'] *= 2
    assert_snapshot(actual, model, adam, sampler)
    assert all(torch.equal(value, dict(model.named_parameters())[name]) for name, value in frozen.items())
    assert all(int(v['step']) == 2 for v in adam.state.values()), 'Adam moments reset or rejected step survived'

    # All rejected attempts, including their final grads/moments/RNG, disappear.
    before = snapshot(model, adam, sampler)
    rejected = transactional_update(model, adam, lambda: proposal(model, adam, sampler),
                                    lambda: {'accepted': False}, state=state,
                                    scales=(1, .5, .25), sampler_rngs=(sampler,))
    assert not rejected['accepted'] and rejected['attempts'] == 3
    assert_snapshot(before, model, adam, sampler)
    assert state['attempted_updates'] == 5 and state['committed_updates'] == 1 and state['rejected_updates'] == 4

    # Proposal and guard exceptions also restore the exact preexisting state.
    def broken_proposal():
        proposal(model, adam, sampler)
        raise LookupError('proposal failure')
    def broken_guard():
        torch.rand(2); sampler.random(); model.eval()
        raise LookupError('guard failure')
    for propose, check in ((broken_proposal, lambda: {'accepted': True}),
                           (lambda: proposal(model, adam, sampler), broken_guard)):
        before = snapshot(model, adam, sampler)
        exception_state = {}
        try:
            transactional_update(model, adam, propose, check, state=exception_state,
                                 sampler_rngs=(sampler,))
        except LookupError: pass
        else: raise AssertionError('callback exception was hidden')
        assert_snapshot(before, model, adam, sampler)
        assert exception_state['attempted_updates'] == 1
        assert exception_state['committed_updates'] == 0 and exception_state['rejected_updates'] == 1

    # Exactly one step is enforced even when a callback reports plausible stats.
    for count in (0, 2):
        before = snapshot(model, adam, sampler)
        def bad_steps():
            for _ in range(count): proposal(model, adam, sampler)
            return {'loss': 1.0}
        try:
            transactional_update(model, adam, bad_steps, lambda: {'accepted': True}, sampler_rngs=(sampler,))
        except RuntimeError as error: assert 'exactly one' in str(error)
        else: raise AssertionError('invalid step count committed')
        assert_snapshot(before, model, adam, sampler)

    # Guard mutation is a programming error; ordinary candidate corruption is a
    # rejection. Both paths fail closed and roll back trainable storage.
    trainable = next(p for p in model.parameters() if p.requires_grad)
    def mutating_guard():
        with torch.no_grad(): trainable.add_(.01)
        return {'accepted': True}
    before = snapshot(model, adam, sampler)
    try:
        transactional_update(model, adam, lambda: proposal(model, adam, sampler), mutating_guard,
                             sampler_rngs=(sampler,))
    except RuntimeError as error: assert 'must not mutate' in str(error)
    else: raise AssertionError('mutating guard committed')
    assert_snapshot(before, model, adam, sampler)
    for corruption in ('weights', 'grads', 'stats', 'guard'):
        before = snapshot(model, adam, sampler)
        def corrupt_proposal():
            result = proposal(model, adam, sampler)
            if corruption == 'weights':
                with torch.no_grad(): trainable.flatten()[0] = float('nan')
            if corruption == 'grads': trainable.grad.flatten()[0] = float('nan')
            if corruption == 'stats': result['loss'] = float('inf')
            return result
        result = transactional_update(model, adam, corrupt_proposal,
                                      lambda: {'accepted': True, 'metric': float('nan')} if corruption == 'guard'
                                      else {'accepted': True}, scales=(1,), sampler_rngs=(sampler,))
        assert not result['accepted']
        json.dumps(result, allow_nan=False)
        assert_snapshot(before, model, adam, sampler)

    # JSON transaction metadata plus native model/Adam/RNG checkpoint resume.
    state = json.loads(json.dumps(state, allow_nan=False))
    saved = snapshot(model, adam, sampler)
    buffer = io.BytesIO(); torch.save(saved, buffer); buffer.seek(0)
    saved = torch.load(buffer, weights_only=False)
    outcome = transactional_update(model, adam, lambda: proposal(model, adam, sampler),
                                   lambda: {'accepted': True}, state=state, sampler_rngs=(sampler,))
    expected = snapshot(model, adam, sampler); expected_state = deepcopy(state)
    # Independent model instance makes accidental in-memory state visible.
    other, other_adam, other_sampler = fresh()
    restore(other, other_adam, other_sampler, saved)
    previous_state = deepcopy(state)
    previous_state['attempted_updates'] -= 1; previous_state['committed_updates'] -= 1
    resumed = transactional_update(other, other_adam, lambda: proposal(other, other_adam, other_sampler),
                                   lambda: {'accepted': True}, state=previous_state,
                                   sampler_rngs=(other_sampler,))
    assert_snapshot(expected, other, other_adam, other_sampler)
    assert previous_state == expected_state and resumed == outcome

    # Corrupted JSON counters/schema cannot restart a supposedly valid ledger.
    for bad_state in ({'schema': True}, {'committed_updates': 1},
                      {'attempted_updates': 1, 'rejected_updates': 2},
                      {'lr_scale': float('nan')}):
        before = snapshot(model, adam, sampler)
        try:
            transactional_update(model, adam, lambda: proposal(model, adam, sampler),
                                 lambda: {'accepted': True}, state=bad_state,
                                 sampler_rngs=(sampler,))
        except ValueError: pass
        else: raise AssertionError('invalid JSON transaction state accepted')
        assert_snapshot(before, model, adam, sampler)

    # Frozen storage is never silently accepted. Contract violations cannot be
    # rolled back without copying the giant frozen prefix: discard this model.
    for mutator in ('proposal', 'guard'):
        other, other_adam, other_sampler = fresh()
        frozen_parameter = next(p for p in other.parameters() if not p.requires_grad)
        def frozen_mutation():
            with torch.no_grad(): frozen_parameter.add_(.01)
        def proposed():
            result = proposal(other, other_adam, other_sampler)
            if mutator == 'proposal': frozen_mutation()
            return result
        def guarded():
            if mutator == 'guard': frozen_mutation()
            return {'accepted': True}
        try:
            transactional_update(other, other_adam, proposed, guarded)
        except RuntimeError as error: assert 'discard this model' in str(error)
        else: raise AssertionError('frozen-prefix mutation committed')
    print(json.dumps({'retention_transaction_native': 'PASS', 'real_prefix_adam_lr_retry_exact': True,
                      'rejection_exception_rng_moments_restored': True, 'json_resume_exact': True,
                      'finite_single_step_and_read_only_guard_enforced': True}))


if __name__ == '__main__': run_checks()
