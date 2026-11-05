"""Bounded, reversible CPU Adam proposals above an immutable native prefix.

``propose_fn()`` makes exactly one optimizer step and returns JSON statistics.
``guard_fn()`` reads TRAIN-only retention anchors and returns a JSON mapping
with a literal bool ``accepted`` field. A guard must not mutate parameters,
gradients or optimizer state. Evaluation mode and RNG use are isolated from
the accepted training trajectory. No evaluation labels or adoption policy are
part of this module.

Only trainable parameter/gradient tensors and optimizer state are copied. The
frozen prefix is not copied: callbacks must leave it and model buffers intact.
Version checks detect violations of that contract; callers must discard a
model after a callback illegally changes frozen storage. Ordinary rejection
and exceptions restore every writable training tensor, Adam and sampler RNG.
"""
from copy import deepcopy
import math
import random

import torch


MIN_LR_SCALE = 1 / 64
_STATE_DEFAULTS = dict(schema=1, lr_scale=1.0, attempted_updates=0,
                       committed_updates=0, rejected_updates=0)


def _finite_json(value):
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(_finite_json(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite_json(item)
                   for key, item in value.items())
    return False


def _state_values(state):
    if state is not None and not isinstance(state, dict):
        raise TypeError('transaction state must be a JSON mapping')
    result = dict(_STATE_DEFAULTS)
    if state is not None:
        result.update(state)
    if (not _finite_json(result) or type(result['schema']) is not int
            or result['schema'] != 1):
        raise ValueError('invalid transaction state schema or nonfinite state')
    scale = result['lr_scale']
    if (isinstance(scale, bool) or not isinstance(scale, (int, float))
            or not MIN_LR_SCALE <= scale <= 1):
        raise ValueError('state lr_scale must be between 1/64 and 1')
    for key in ('attempted_updates', 'committed_updates', 'rejected_updates'):
        if type(result[key]) is not int or result[key] < 0:
            raise ValueError(f'invalid transaction counter: {key}')
    if result['committed_updates'] + result['rejected_updates'] != result['attempted_updates']:
        raise ValueError('transaction counters do not account for every physical attempt')
    return result


def _equal(left, right):
    """Exact structural equality for small writable optimizer state."""
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _tensor_signature(tensors):
    return tuple((name, id(value), value._version, value.requires_grad,
                  tuple(value.shape), value.dtype, value.device)
                 for name, value in tensors)


def _finite_tensors(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tensors(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tensors(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def transactional_update(model, optimizer, propose_fn, guard_fn, *, state=None,
                         scales=(1, .5, .25, .125), sampler_rngs=(), recovery=None):
    """Commit the first guarded real Adam proposal, or restore the old state.

    Retry factors are fixed before the first attempt. Each is multiplied by the
    checkpointed ``state['lr_scale']`` and floored at 1/64; duplicate floor
    attempts are removed. Successful scale is retained for the next call. An
    entirely rejected call retains its smallest attempted scale. Scale never
    grows implicitly. Parameter group base LRs are restored on every return.

    The supplied JSON state records physical proposal calls separately from
    committed updates. It contains no token counters. Exceptions count physical
    calls and roll back writable model/optimizer/RNG state before re-raising.
    ``stats`` is the final proposal's statistics; ``guard`` is its guard result.
    Nonfinite proposals or guard diagnostics fail closed as ordinary rejection.
    """
    adaptive = _state_values(state)
    recovered = False
    if recovery is not None:
        if (not isinstance(recovery,dict) or set(recovery)!={'accepted_streak','cooldown_updates','growth_factor'}
                or type(recovery['accepted_streak']) is not int or not 16<=recovery['accepted_streak']<=1024
                or type(recovery['cooldown_updates']) is not int or not 1<=recovery['cooldown_updates']<=1024
                or isinstance(recovery['growth_factor'],bool) or recovery['growth_factor']!=2.):
            raise ValueError('invalid guarded learning-rate recovery configuration')
        progress=deepcopy(adaptive.get('recovery',dict(streak=0,cooldown=0,growth_attempts=0,growth_commits=0)))
        if (set(progress)!={'streak','cooldown','growth_attempts','growth_commits'}
                or any(type(v) is not int or v<0 for v in progress.values())
                or progress['growth_commits']>progress['growth_attempts']):
            raise ValueError('invalid persisted learning-rate recovery state')
        recovered=(progress['cooldown']==0 and progress['streak']>=recovery['accepted_streak']
                   and adaptive['lr_scale']<1.)
        adaptive['recovery']=progress
    starting_scale=min(1.,adaptive['lr_scale']*2.) if recovered else adaptive['lr_scale']
    factors = tuple(scales)
    if (not factors or len(factors) > 16
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) or not 0 < x <= 1 for x in factors)
            or any(a <= b for a, b in zip(factors, factors[1:]))):
        raise ValueError('scales must contain 1..16 finite strictly decreasing factors in (0, 1]')
    attempts_scales = tuple(dict.fromkeys(max(MIN_LR_SCALE, starting_scale * factor)
                                         for factor in factors))
    parameters = tuple(model.named_parameters())
    trainable = tuple((name, parameter) for name, parameter in parameters if parameter.requires_grad)
    if not trainable or any(p.device.type != 'cpu' for _, p in parameters):
        raise ValueError('transaction requires CPU model parameters and a trainable subset')
    owned = [p for group in optimizer.param_groups for p in group['params']]
    trainable_ids = {id(p) for _, p in trainable}
    if (not owned or len({id(p) for p in owned}) != len(owned)
            or any(id(p) not in trainable_ids for p in owned)):
        raise ValueError('optimizer must own a nonempty unique subset of trainable model parameters')
    base_lrs = [group['lr'] for group in optimizer.param_groups]
    if any(isinstance(lr, bool) or not isinstance(lr, (int, float))
           or not math.isfinite(lr) or lr <= 0 for lr in base_lrs):
        raise ValueError('transaction requires finite positive scalar base learning rates')
    rngs = tuple(sampler_rngs)
    if (any(not isinstance(rng, random.Random) for rng in rngs)
            or len({id(rng) for rng in rngs}) != len(rngs)):
        raise ValueError('sampler_rngs must be distinct random.Random instances')
    frozen = tuple((name, p) for name, p in parameters if not p.requires_grad)
    immutable_signature = _tensor_signature(frozen + tuple(model.named_buffers()))
    scope = tuple((name, id(p), p.requires_grad) for name, p in parameters)
    weights = tuple(p.detach().clone() for _, p in trainable)
    gradients = tuple(None if p.grad is None else p.grad.detach().clone() for _, p in trainable)
    optimizer_state = deepcopy(optimizer.state_dict())
    rng_state = torch.get_rng_state().clone()
    sampler_states = tuple(deepcopy(rng.getstate()) for rng in rngs)
    modes = tuple((module, module.training) for module in model.modules())
    attempts = steps = 0
    stats, guard, chosen_scale = {}, {}, None

    def immutable_check():
        current = tuple(model.named_parameters())
        if tuple((name, id(p), p.requires_grad) for name, p in current) != scope:
            raise RuntimeError('proposal changed model parameter identity or trainable scope')
        if _tensor_signature(frozen + tuple(model.named_buffers())) != immutable_signature:
            raise RuntimeError('callback modified frozen parameters or buffers; discard this model')
        current_owned = [p for group in optimizer.param_groups for p in group['params']]
        if tuple(map(id, current_owned)) != tuple(map(id, owned)):
            raise RuntimeError('callback changed optimizer parameter ownership')

    def restore_rng(torch_state, python_states):
        torch.set_rng_state(torch_state)
        for rng, saved in zip(rngs, python_states):
            rng.setstate(saved)

    def restore():
        with torch.no_grad():
            for (_, p), saved, grad in zip(trainable, weights, gradients):
                p.copy_(saved)
                p.grad = None if grad is None else grad.clone()
        optimizer.load_state_dict(deepcopy(optimizer_state))
        restore_rng(rng_state, sampler_states)
        for module, training in modes:
            module.training = training

    def record(committed, scale):
        adaptive['attempted_updates'] += attempts
        adaptive['committed_updates'] += int(committed)
        adaptive['rejected_updates'] += attempts - int(committed)
        if scale is not None:
            adaptive['lr_scale'] = float(scale)
        if recovery is not None:
            progress=adaptive['recovery']
            progress['growth_attempts']+=int(recovered and attempts>0)
            if committed and attempts==1:
                if recovered:
                    progress['growth_commits']+=1;progress['streak']=0
                    progress['cooldown']=recovery['cooldown_updates']
                elif progress['cooldown']:
                    progress['cooldown']-=1;progress['streak']=0
                else:progress['streak']=min(recovery['accepted_streak'],progress['streak']+1)
            else:
                progress['streak']=0;progress['cooldown']=recovery['cooldown_updates']
        if state is not None:
            state.update(adaptive)

    def count_step(_optimizer, _args, _kwargs):
        nonlocal steps
        steps += 1

    hook = optimizer.register_step_post_hook(count_step)
    try:
        for scale in attempts_scales:
            restore()
            for group, lr in zip(optimizer.param_groups, base_lrs):
                group['lr'] = lr * scale
            attempts += 1
            steps = 0
            raw_stats = propose_fn()
            if steps != 1:
                raise RuntimeError('propose_fn must execute exactly one optimizer.step()')
            immutable_check()
            stats = {} if raw_stats is None else raw_stats
            if not isinstance(stats, dict):
                raise TypeError('propose_fn must return a JSON statistics mapping or None')
            finite_proposal = (_finite_json(stats)
                               and all(_finite_tensors(p) for _, p in trainable)
                               and all(_finite_tensors(p.grad) for _, p in trainable)
                               and _finite_tensors(optimizer.state_dict()))
            if not finite_proposal:
                guard = dict(accepted=False, reason='nonfinite_proposal')
                stats = {}  # Never put NaN/Infinity in checkpoint/report output.
            else:
                candidate_signature = _tensor_signature(trainable)
                candidate_grads = tuple(None if p.grad is None else p.grad.detach().clone()
                                        for _, p in trainable)
                candidate_optimizer = deepcopy(optimizer.state_dict())
                candidate_rng = torch.get_rng_state().clone()
                candidate_samplers = tuple(deepcopy(rng.getstate()) for rng in rngs)
                candidate_modes = tuple((module, module.training) for module in model.modules())
                try:
                    guard = guard_fn()
                    immutable_check()
                    if (steps != 1 or _tensor_signature(trainable) != candidate_signature
                            or not _equal(candidate_optimizer, optimizer.state_dict())
                            or any(not _equal(before, p.grad)
                                   for before, (_, p) in zip(candidate_grads, trainable))):
                        raise RuntimeError('guard_fn must not mutate trainable parameters, gradients or optimizer')
                finally:
                    restore_rng(candidate_rng, candidate_samplers)
                    for module, training in candidate_modes:
                        module.training = training
                if not isinstance(guard, dict) or type(guard.get('accepted')) is not bool:
                    raise TypeError('guard_fn must return a JSON mapping with a boolean accepted field')
                if not _finite_json(guard):
                    guard = dict(accepted=False, reason='nonfinite_guard')
            chosen_scale = scale
            if guard['accepted']:
                for group, lr in zip(optimizer.param_groups, base_lrs):
                    group['lr'] = lr
                record(True, scale)
                return dict(accepted=True, attempts=attempts, scale=scale,
                            stats=stats, guard=guard)
        restore()
        record(False, chosen_scale)
        return dict(accepted=False, attempts=attempts, scale=chosen_scale,
                    stats=stats, guard=guard)
    except BaseException:
        restore()
        record(False, chosen_scale)
        raise
    finally:
        hook.remove()
