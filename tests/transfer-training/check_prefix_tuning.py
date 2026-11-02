"""Actual native attention gradients, cache boundaries, export and Adam resume."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/transfer-training'))
import torch
from generalist_lm.model import GeneralistLMConfig
from generalist_lm.runtime import GeneralistRuntime
from prefix_tuning import configure_prefix_tuning, prefix_summary
from session import save_session, restore_session


def run_checks():
    torch.set_num_threads(2)
    torch.manual_seed(713)
    cfg = GeneralistLMConfig(d_model=32, n_layers=3, n_heads=2, d_ff=64,
                            context_length=64, position_encoding='rope',
                            attention_type='gqa', n_kv_heads=1, norm_type='rmsnorm')
    runtime = GeneralistRuntime.fresh(cfg)
    ids = torch.tensor([[1, 4, 18, 25, 33, 5]])
    labels = ids.clone()
    labels[:, :3] = -100
    raw = runtime.model(ids, labels)
    before = copy.deepcopy(runtime.model.state_dict())
    rng = torch.get_rng_state()
    cache = configure_prefix_tuning(runtime.model, cache_bytes=1600, cache_entries=2)
    assert torch.equal(rng, torch.get_rng_state()), 'configuration consumed checkpoint RNG'
    assert all(torch.equal(value, runtime.model.state_dict()[name]) for name, value in before.items())
    assert torch.equal(raw['logits'], runtime.model(ids, labels)['logits'])
    assert torch.equal(raw['loss'], runtime.model(ids, labels)['loss'])
    assert cache.hits == 1 and cache.bytes <= cache.max_bytes
    # A copied configured model must never evaluate the source model's weights.
    cloned = copy.deepcopy(runtime.model)
    clone_cache = cloned._airi_prefix_features
    with torch.no_grad():
        cloned.lm_head.weight[10].add_(.02)
    assert clone_cache.model is cloned
    assert torch.equal(cloned(ids)['logits'], clone_cache.original(ids)['logits'])
    assert not torch.equal(cloned(ids)['logits'], runtime.model(ids)['logits'])
    trainable = {name for name, p in runtime.model.named_parameters() if p.requires_grad}
    assert 'blocks.2.attn.q_proj.weight' in trainable
    assert 'blocks.2.attn.out.weight' in trainable
    assert 'blocks.2.attn_norm.weight' in trainable
    assert 'lm_head.weight' in trainable
    assert not any('.ff.' in name for name in trainable)
    assert not any(name.startswith(('blocks.0.', 'blocks.1.', 'token_embedding.')) for name in trainable)
    frozen = {name: value.clone() for name, value in runtime.model.state_dict().items()
              if name not in trainable}
    optimizer = torch.optim.AdamW([p for p in runtime.model.parameters() if p.requires_grad], lr=.002)

    def step(rt, opt):
        opt.zero_grad(set_to_none=True)
        loss = rt.model(ids, labels)['loss']
        loss.backward()
        assert rt.model.blocks[-1].attn.q_proj.weight.grad.abs().sum() > 0
        assert rt.model.blocks[-1].attn.out.weight.grad.abs().sum() > 0
        opt.step()
        return float(loss.detach())

    losses = [step(runtime, optimizer) for _ in range(16)]
    assert losses[-1] < losses[0]
    assert all(torch.equal(value, runtime.model.state_dict()[name]) for name, value in frozen.items())
    assert not torch.equal(before['blocks.2.attn.q_proj.weight'], runtime.model.blocks[-1].attn.q_proj.weight)
    assert torch.equal(runtime.model(ids)['logits'], cache.original(ids)['logits'])
    assert cache.invalidations == 0, 'tail updates invalidated the immutable prefix'
    for n in range(5):
        other = ids.clone()
        other[0, -1] = 20 + n
        assert torch.equal(runtime.model(other)['logits'], cache.original(other)['logits'])
        assert cache.bytes <= 1600 and len(cache.entries) <= 2
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'checkpoint'
        sampler = random.Random(7)
        save_session(runtime, optimizer, sampler, sampler, dict(completed_steps=16), path,
                     fingerprint='prefix-native-check')
        native = GeneralistRuntime.from_checkpoint(path / 'model')
        assert torch.equal(runtime.model(ids)['logits'], native.model(ids)['logits'])
        for use_cache in (False, True):
            assert torch.equal(runtime.model.generate(ids, max_new_tokens=8, eos_token_id=None, use_cache=use_cache),
                               native.model.generate(ids, max_new_tokens=8, eos_token_id=None, use_cache=use_cache))

        def factory(model):
            configure_prefix_tuning(model, cache_bytes=1600, cache_entries=2)
            return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.002)

        resumed, adam, _ = restore_session(path, factory, fingerprint='prefix-native-check')
        for _ in range(3):
            step(runtime, optimizer)
            step(resumed, adam)
        assert all(torch.equal(value, resumed.model.state_dict()[name])
                   for name, value in runtime.model.state_dict().items())
        assert all(int(state['step']) == 19 for state in adam.state.values())
    old_invalidations = cache.invalidations
    with torch.no_grad():
        runtime.model.token_embedding.weight.add_(.01)
    assert torch.equal(runtime.model(ids)['logits'], cache.original(ids)['logits'])
    assert cache.invalidations == old_invalidations + 1
    runtime.model.blocks[0].attn.q_proj.weight.requires_grad_(True)
    try:
        runtime.model(ids)
    except RuntimeError:
        pass
    else:
        raise AssertionError('trainable prefix was silently cached')
    runtime.model.blocks[0].attn.q_proj.weight.requires_grad_(False)
    runtime.config.norm_placement = 'post'
    try:
        runtime.model(ids)
    except RuntimeError:
        pass
    else:
        raise AssertionError('changed prefix math reused cached activations')
    runtime.config.norm_placement = 'pre'
    # Full native positional handling, including buffers, is reused on cache fill.
    for position in ('learned', 'sinusoidal', 'rope'):
        rt = GeneralistRuntime.fresh(GeneralistLMConfig(d_model=32, n_layers=2, n_heads=2,
                                                       d_ff=64, context_length=64,
                                                       position_encoding=position))
        exact = rt.model(ids)['logits']
        configured = configure_prefix_tuning(rt.model, cache_bytes=0)
        assert torch.equal(exact, rt.model(ids)['logits'])
        assert configured.bytes == 0 and not configured.entries
    print(json.dumps(dict(prefix_tuning_native='PASS', attention_receives_gradients=True,
                          frozen_prefix_and_ffn_unchanged=True, native_export_exact=True,
                          kv_generation_exact=True, optimizer_resume_exact=True,
                          independent_copied_model=True,
                          cache_bounded=True, strategy=prefix_summary(runtime.model))))


if __name__ == '__main__':
    run_checks()
