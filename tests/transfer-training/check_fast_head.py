"""Exact cached-head updates, bounded storage and native checkpoint portability."""
import copy,hashlib,json,random,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/transfer-training'))
import torch
from generalist_lm.runtime import GeneralistRuntime
from generalist_lm.model import GeneralistLMConfig
from fast_head import configure_head,install_feature_cache,head_summary
from session import save_session,restore_session

torch.set_num_threads(2);torch.manual_seed(912)
runtime=GeneralistRuntime.fresh(GeneralistLMConfig(d_model=32,n_layers=1,n_heads=2,d_ff=64,context_length=64))
ids=torch.tensor([[1,4,18,25,33,5]])
original=copy.deepcopy(runtime.model.state_dict())
raw=runtime.model(ids)['logits'].detach();rng=torch.get_rng_state()
cache=configure_head(runtime.model,cache_bytes=10000)
assert torch.equal(rng,torch.get_rng_state()),'configuration perturbed checkpoint RNG'
assert not runtime.config.tie_embeddings
assert runtime.model.lm_head.weight is not runtime.model.token_embedding.weight
assert all(torch.equal(v,runtime.model.state_dict()[k]) for k,v in original.items())
assert torch.equal(raw,runtime.model(ids)['logits'])
assert torch.equal(raw,runtime.model(ids)['logits']) and cache.hits==1
protected={k:v.clone() for k,v in runtime.model.state_dict().items() if k!='lm_head.weight'}
optimizer=torch.optim.AdamW([p for p in runtime.model.parameters() if p.requires_grad],lr=.003)
def step(model,opt):
    opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(ids)['logits'][:,-1],torch.tensor([45]));loss.backward();opt.step()
    return float(loss.detach())
losses=[step(runtime.model,optimizer) for _ in range(12)]
assert losses[-1]<losses[0]
assert all(torch.equal(v,runtime.model.state_dict()[k]) for k,v in protected.items())
assert set(dict(runtime.model.named_parameters()).keys())-{'lm_head.weight'}
assert sum(p.requires_grad for p in runtime.model.parameters())==1
assert torch.equal(runtime.model(ids)['logits'],cache.original(ids)['logits'])
assert cache.hits>=12 and cache.bytes<=cache.max_bytes
from continual_learning import try_interpolations
old_champion=GeneralistRuntime.fresh(GeneralistLMConfig(d_model=32,n_layers=1,n_heads=2,d_ff=64,context_length=64))
old_champion.model.load_state_dict(original)
saved_head=runtime.model.lm_head.weight.detach().clone();versions=cache.versions
result=try_interpolations(runtime.model,old_champion.model,evaluate=lambda:None,acceptable=lambda x:False)
assert result['alpha'] is None and torch.equal(saved_head,runtime.model.lm_head.weight)
runtime.model(ids)
assert versions==cache.versions and cache.invalidations==0,'head merge discarded valid frozen-body features'

with tempfile.TemporaryDirectory() as directory:
    p=Path(directory)/'checkpoint';r=random.Random(2);ar=random.Random(3)
    save_session(runtime,optimizer,r,ar,dict(completed_steps=12),p,fingerprint='fast-head-native')
    info=json.loads((p/'model/metadata.json').read_text())
    assert info['parameters']==sum(q.numel() for q in runtime.model.parameters())
    assert info['trainable_parameters']==runtime.config.d_model*runtime.config.vocab_size
    native=GeneralistRuntime.from_checkpoint(p/'model')
    assert torch.equal(native.model(ids)['logits'],runtime.model(ids)['logits'])
    for use_cache in [False,True]:
        assert torch.equal(native.model.generate(ids,max_new_tokens=8,eos_token_id=None,use_cache=use_cache),
                           runtime.model.generate(ids,max_new_tokens=8,eos_token_id=None,use_cache=use_cache))
    def factory(model):
        configure_head(model,cache_bytes=10000)
        return torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],lr=.003)
    resumed,adam,_=restore_session(p,factory,fingerprint='fast-head-native')
    for _ in range(3):step(runtime.model,optimizer);step(resumed.model,adam)
    assert all(torch.equal(v,resumed.model.state_dict()[k]) for k,v in runtime.model.state_dict().items())
    assert all(int(v['step'])==15 for v in adam.state.values())

# Modifying a frozen body through a legitimate load invalidates cached features.
before=cache.invalidations
with torch.no_grad():runtime.model.position_embedding.weight.add_(.01)
assert torch.equal(runtime.model(ids)['logits'],cache.original(ids)['logits'])
assert cache.invalidations==before+1
runtime.model.position_embedding.weight.requires_grad_(True)
try:runtime.model(ids)
except RuntimeError:pass
else:raise AssertionError('trainable backbone entered frozen feature cache')
runtime.model.position_embedding.weight.requires_grad_(False)

# Frozen teachers may cache without untying their native embeddings.
teacher=GeneralistRuntime.fresh(GeneralistLMConfig(d_model=32,n_layers=1,n_heads=2,d_ff=64,context_length=64))
for q in teacher.model.parameters():q.requires_grad_(False)
raw=teacher.model(ids)['logits'];frozen=install_feature_cache(teacher.model,cache_bytes=1000)
assert teacher.model.lm_head.weight is teacher.model.token_embedding.weight
for n in range(8):
    x=ids.clone();x[0,-1]=n+20
    assert torch.equal(teacher.model(x)['logits'],frozen.original(x)['logits'])
    assert frozen.bytes<=1000
print(json.dumps(dict(fast_head_native='PASS',native_export_exact=True,kv_generation_exact=True,
                     optimizer_resume_exact=True,body_frozen=True,cache_bounded=True,
                     trainable_parameters=head_summary(runtime.model)['trainable_parameters'])))

from fast_head import set_projection,attach_projection
projected=GeneralistRuntime.fresh(GeneralistLMConfig(d_model=32,n_layers=1,n_heads=2,d_ff=64,context_length=64))
configure_head(projected.model)
base_weight=projected.model.lm_head.weight.detach().clone()
features=torch.eye(32)[:4]
set_projection(projected.model,base_weight,features)
adam=attach_projection(torch.optim.AdamW([projected.model.lm_head.weight],lr=.02),projected.model)
protected_logits=features@base_weight.T
acquisition=torch.eye(32)[8:9];target=torch.tensor([45])
observed=[]
for _ in range(32):
    adam.zero_grad();loss=torch.nn.functional.cross_entropy(projected.model.lm_head(acquisition),target)
    loss.backward();adam.step();observed.append(float(loss.detach()))
    assert torch.allclose(features@projected.model.lm_head.weight.T,protected_logits,atol=1e-7,rtol=0)
assert observed[-1]<observed[0]
assert not torch.equal(projected.model.lm_head.weight,base_weight)
with tempfile.TemporaryDirectory() as directory:
    p=Path(directory)/'p';r=random.Random(4)
    save_session(projected,adam,r,r,dict(completed_steps=32),p,fingerprint='projected-native')
    def factory(model):
        configure_head(model);set_projection(model,base_weight,features)
        return attach_projection(torch.optim.AdamW([model.lm_head.weight],lr=.02),model)
    resumed,ro,_=restore_session(p,factory,fingerprint='projected-native')
    for model,opt in [(projected.model,adam),(resumed.model,ro)]:
        opt.zero_grad();torch.nn.functional.cross_entropy(model.lm_head(acquisition),target).backward();opt.step()
    assert torch.equal(projected.model.lm_head.weight,resumed.model.lm_head.weight)
    assert torch.allclose(features@resumed.model.lm_head.weight.T,protected_logits,atol=1e-7,rtol=0)
print('Projected head learns a new direction, preserves protected logits, and resumes Adam exactly.')
