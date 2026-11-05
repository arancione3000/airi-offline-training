"""Actual runner donor callbacks retain smaller real Adam steps under an LR cap."""
import ast
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/transfer-training'))
import torch
from generalist_lm.model import GeneralistLMConfig
from generalist_lm.runtime import GeneralistRuntime
import run as runner
from retention_transaction import transactional_update
from success_transfer import distill_step


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def snapshot(model, optimizer):
    return dict(model=copy.deepcopy(model.state_dict()), adam=copy.deepcopy(optimizer.state_dict()),
                grads=[None if p.grad is None else p.grad.detach().clone() for p in model.parameters()],
                rng=torch.get_rng_state().clone(), modes=[m.training for m in model.modules()])


def restore(model, optimizer, saved):
    model.load_state_dict(saved['model']); optimizer.load_state_dict(copy.deepcopy(saved['adam']))
    for parameter, gradient in zip(model.parameters(), saved['grads']):
        parameter.grad = None if gradient is None else gradient.clone()
    torch.set_rng_state(saved['rng'])
    for module, training in zip(model.modules(), saved['modes']):
        module.training = training


def callbacks():
    """Compile the real two donor lambdas and scale adapter from the runner.

    Keeping their production syntax means omitting scaled caps at either call
    site reproduces the failure, without running network publication or evals.
    """
    tree = ast.parse(Path(runner.__file__).read_text())
    donors = [node.args[0] for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == 'protected_update' and node.args
              and isinstance(node.args[0], ast.Lambda)
              and isinstance(node.args[0].body, ast.Call)
              and isinstance(node.args[0].body.func, ast.Name)
              and node.args[0].body.func.id == 'distill_step']
    assert len(donors) == 2, 'both ordinary donor replay and consolidation must be exercised'
    adapters = [node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == 'record_proposal']
    assert len(adapters) == 1
    return donors, adapters[0]


def check(donor, adapter, *, multiplier, cap, initial_scale):
    torch.manual_seed(401)
    model = GeneralistRuntime.fresh(GeneralistLMConfig(
        d_model=24, n_layers=1, n_heads=2, d_ff=48, context_length=32)).model
    teacher = copy.deepcopy(model)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    base_lr = 3e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    ids = torch.tensor([[1, 6, 12, 22, 31, 2]])
    labels = ids.clone(); labels[:, :3] = -100
    batch = (ids, labels)
    probabilities = [0.] * model.config.vocab_size
    probabilities[22] = 1.
    entry = dict(row={}, positions=[dict(position=3, target=22, probabilities=probabilities)])
    options = dict(memories=[batch], retention_weight=.5, champion_kl_weight=1.)
    # Real preexisting Adam moments and gradients expose accidental reset/reuse.
    distill_step(model, teacher, optimizer, entry, batch, batch,
                 lr_multiplier=multiplier, lr_cap=cap, **options)
    before = snapshot(model, optimizer)
    scope = dict(runtime=SimpleNamespace(model=model), protected_champion=SimpleNamespace(model=teacher),
                 opt=optimizer, entry=entry, lesson=entry, encode=lambda row: batch, batch=batch,
                 anchor=batch, transfer_anchor=batch, options=options,
                 transfer_options=lambda prepared: prepared, distill_step=distill_step,
                 SPEC=dict(success_transfer=dict(lr_multiplier=multiplier, lr_cap=cap)))
    scope['propose'] = eval(compile(ast.Expression(copy.deepcopy(donor)), runner.__file__, 'eval'), scope)
    scope.update(base_lrs=[base_lr], proposed=[])
    module = ast.Module(body=[copy.deepcopy(adapter)], type_ignores=[])
    exec(compile(module, runner.__file__, 'exec'), scope)
    actual_lrs = []
    hook = optimizer.register_step_pre_hook(lambda opt, args, kwargs: actual_lrs.append(opt.param_groups[0]['lr']))
    calls = 0
    def guard():
        nonlocal calls
        calls += 1
        return dict(accepted=calls == 2)
    state = dict(schema=1, lr_scale=initial_scale, attempted_updates=0,
                 committed_updates=0, rejected_updates=0)
    try:
        outcome = transactional_update(model, optimizer, scope['record_proposal'], guard,
                                       state=state, scales=(1., .5))
    finally:
        hook.remove()
    expected_lr = min(cap, base_lr * multiplier)
    assert actual_lrs == [expected_lr * initial_scale, expected_lr * initial_scale * .5], actual_lrs
    assert outcome['accepted'] and outcome['attempts'] == 2
    assert state == dict(schema=1, lr_scale=initial_scale * .5, attempted_updates=2,
                         committed_updates=1, rejected_updates=1)
    accepted = snapshot(model, optimizer)
    restore(model, optimizer, before)
    accepted_scale = initial_scale * .5
    optimizer.param_groups[0]['lr'] = base_lr * accepted_scale
    distill_step(model, teacher, optimizer, entry, batch, batch,
                 lr_multiplier=multiplier, lr_cap=cap * accepted_scale, **options)
    optimizer.param_groups[0]['lr'] = base_lr
    assert equal(accepted, snapshot(model, optimizer)), 'retry changed the accepted Adam trajectory'
    assert len(scope['proposed']) == 2


def run_checks():
    torch.set_num_threads(2)
    donors, adapter = callbacks()
    for donor in donors:
        check(donor, adapter, multiplier=5., cap=1e-6, initial_scale=1.)
        check(donor, adapter, multiplier=1., cap=1e-4, initial_scale=.25)
    diagnostics = runner.transfer_retention_statistics({}, dict(
        retention_weight=4., champion_kl_weight=8., memories=['it', 'en']))
    assert diagnostics == dict(retention_weight=4., champion_kl_weight=8., retention_examples=2)
    print(json.dumps(dict(retention_capped_distill_native='PASS', donor_and_consolidation=True,
                          capped_and_uncapped_actual_lrs=True, adaptive_scale=True,
                          accepted_adam_trajectory_exact=True, nonfinite_diagnostics_fallback=True)))


if __name__ == '__main__':
    run_checks()
