"""Native proof for isolated final-FFN-output plasticity; no capability claim."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/transfer-training'))
import torch
from generalist_lm.model import GeneralistLMConfig
from generalist_lm.runtime import GeneralistRuntime
from prefix_tuning import configure_prefix_tuning,prefix_summary
from session import save_session,restore_session


def exact(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


def run_checks():
    torch.set_num_threads(2);torch.manual_seed(827)
    cfg=GeneralistLMConfig(d_model=32,n_layers=3,n_heads=2,d_ff=64,context_length=64,
                           position_encoding='rope',attention_type='gqa',n_kv_heads=1,
                           norm_type='layernorm',bias=True)
    runtime=GeneralistRuntime.fresh(cfg)
    baseline=copy.deepcopy(runtime.model)
    ids=torch.tensor([[1,4,18,25,33,5]])
    labels=ids.clone();labels[:,:3]=-100
    initial=runtime.model(ids,labels)
    before=copy.deepcopy(runtime.model.state_dict());stream=torch.get_rng_state().clone()
    default=configure_prefix_tuning(baseline,cache_bytes=16384)
    cache=configure_prefix_tuning(runtime.model,cache_bytes=16384,train_ffn_output=True)
    assert torch.equal(stream,torch.get_rng_state())
    assert all(torch.equal(v,runtime.model.state_dict()[n]) for n,v in before.items())
    assert torch.equal(initial['logits'],runtime.model(ids,labels)['logits'])
    assert torch.equal(initial['loss'],runtime.model(ids,labels)['loss'])
    old={n for n,p in baseline.named_parameters() if p.requires_grad}
    new={n for n,p in runtime.model.named_parameters() if p.requires_grad}
    assert new-old=={'blocks.2.ff.down.weight','blocks.2.ff.down.bias'}
    assert prefix_summary(baseline)['mode']=='cached_native_final_attention'
    assert prefix_summary(runtime.model)['mode']=='cached_native_final_attention_ffn_output'
    assert configure_prefix_tuning(runtime.model,train_ffn_output=True) is cache
    for changed in (False,1,None):
        try:configure_prefix_tuning(runtime.model,train_ffn_output=changed)
        except ValueError:pass
        else:raise AssertionError('changed/invalid FFN scope silently accepted')
    frozen={n:p.detach().clone() for n,p in runtime.model.named_parameters() if not p.requires_grad}
    opt=torch.optim.AdamW([p for p in runtime.model.parameters() if p.requires_grad],lr=.002)
    def step(rt,optimizer):
        optimizer.zero_grad(set_to_none=True)
        loss=rt.model(ids,labels)['loss'];loss.backward()
        down=rt.model.blocks[-1].ff.down
        assert down.weight.grad is not None and float(down.weight.grad.abs().sum())>0
        assert down.bias.grad is not None and float(down.bias.grad.abs().sum())>0
        assert all(p.grad is None for n,p in rt.model.named_parameters() if not p.requires_grad)
        optimizer.step()
        return float(loss.detach())
    losses=[step(runtime,opt) for _ in range(8)]
    assert losses[-1]<losses[0]
    assert not torch.equal(before['blocks.2.ff.down.weight'],runtime.model.blocks[-1].ff.down.weight)
    assert all(torch.equal(v,runtime.model.state_dict()[n]) for n,v in frozen.items())
    assert cache.invalidations==0 and cache.bytes<=cache.max_bytes
    assert torch.equal(runtime.model(ids)['logits'],cache.original(ids)['logits'])
    with tempfile.TemporaryDirectory(prefix='airi-ffn-plasticity-') as directory:
        path=Path(directory)/'checkpoint';sampler=random.Random(8)
        save_session(runtime,opt,sampler,sampler,dict(completed_steps=8),path,fingerprint='ffn-output-native')
        native=GeneralistRuntime.from_checkpoint(path/'model')
        assert torch.equal(runtime.model(ids)['logits'],native.model(ids)['logits'])
        for kv in (False,True):
            assert torch.equal(runtime.model.generate(ids,max_new_tokens=8,eos_token_id=None,use_cache=kv),
                               native.model.generate(ids,max_new_tokens=8,eos_token_id=None,use_cache=kv))
        def factory(model):
            configure_prefix_tuning(model,cache_bytes=16384,train_ffn_output=True)
            return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.002)
        resumed,adam,_=restore_session(path,factory,fingerprint='ffn-output-native')
        step(runtime,opt);step(resumed,adam)
        assert exact(runtime.model.state_dict(),resumed.model.state_dict())
        assert exact(opt.state_dict(),adam.state_dict())
    print(json.dumps(dict(prefix_plasticity_native='PASS',default_unchanged=True,
                          only_final_down_added=True,initial_logits_and_rng_exact=True,
                          down_receives_gradients=True,frozen_parameters_unchanged=True,
                          native_export_and_kv_exact=True,adam_resume_exact=True,
                          cache_not_invalidated=True,trainable_parameters=prefix_summary(runtime.model)['trainable_parameters'])))

if __name__=='__main__':run_checks()
