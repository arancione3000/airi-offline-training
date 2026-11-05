"""Real native model/Adam plus hash-committed retention and TRAIN registry recovery."""
import copy
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/transfer-training'))
import torch
from generalist_lm.model import CausalTransformerLM,GeneralistLMConfig
from generalist_lm.runtime import GeneralistRuntime
from session import save_session,restore_session,train_step
from retention_memory import empty_memory,refresh
from context_curriculum import build_context
from context_acquisition import build_context_acquisition,context_value_pairs,matched_context_pair
from success_bank import catalog
from run import empty_training_registry,register_context_pair,validate_training_registry

torch.set_num_threads(2);torch.manual_seed(31)
config=GeneralistLMConfig(d_model=32,n_heads=2,n_layers=1,d_ff=64,context_length=32)
runtime=GeneralistRuntime(CausalTransformerLM(config),config)
reference=GeneralistRuntime(CausalTransformerLM(config),config)
for parameter in reference.model.parameters():parameter.requires_grad_(False)
factory=lambda model:torch.optim.AdamW(model.parameters(),lr=3e-6)
optimizer=factory(runtime.model)
ids=torch.tensor([[2,12,13,14,15,1]]);labels=ids.clone();labels[:,:2]=-100
train_step(runtime.model,reference.model,optimizer,ids,labels,ids,labels)

sources,_=build_context();values=build_context_acquisition(max_pairs=8)
source=sources[0];pool=context_value_pairs(values)[(source['language'],source['family'])][0]
registry=empty_training_registry('retention-session')
rows=register_context_pair(registry,source,matched_context_pair(source,pool))
# Syntactically complete synthetic certificates isolate persistence from the
# separate native certifier/guard tests, which verify the actual causal logits.
certify=lambda row:dict(correct=True,complete=True,includes_eos=True,token_count=3,min_margin=.1,mean_nll=.2)
memory=refresh('a'*64,catalog(rows,[]),certify,empty_memory('retention-session'),probe_limit=2)
memory['sampling_cursors']={'persisted-stratum':9}
counters=dict(completed_steps=5,conversation_updates=1,retention_consolidation_updates=3,
    retention_transaction=dict(schema=1,lr_scale=.25,attempted_updates=3,committed_updates=1,rejected_updates=2))
rng=random.Random(42);anchor_rng=random.Random(43)

with tempfile.TemporaryDirectory() as temporary:
    path=Path(temporary)/'checkpoint'
    manifest=save_session(runtime,optimizer,rng,anchor_rng,counters,path,fingerprint='retention-session',
                           retention_memory=memory,training_registry=registry)
    assert {'retention_memory.json','training_registry.json'}<=manifest['files'].keys()
    restored,restored_optimizer,payload=restore_session(path,factory,fingerprint='retention-session',require_retention=True)
    assert payload['retention_memory']==memory and payload['training_registry']==registry
    assert payload['counters']==counters
    assert validate_training_registry(payload['training_registry'],sources,values,[],fingerprint='retention-session')==rows
    for missing in ('resume.pt','optimizer.pt','model/model.pt','model/config.json'):
        bad=copy.deepcopy(manifest);bad['files'].pop(missing)
        (path/'checkpoint.json').write_text(json.dumps(bad))
        try:restore_session(path,factory,fingerprint='retention-session',require_retention=True)
        except ValueError as error:assert 'mandatory' in str(error)
        else:raise AssertionError('omitted mandatory state was trusted: '+missing)
    bad=copy.deepcopy(manifest);bad['completed_steps']=999
    (path/'checkpoint.json').write_text(json.dumps(bad))
    try:restore_session(path,factory,fingerprint='retention-session',require_retention=True)
    except ValueError as error:assert 'completed_steps' in str(error)
    else:raise AssertionError('manifest/resume step mismatch was trusted')
    (path/'checkpoint.json').write_text(json.dumps(manifest))
    train_step(runtime.model,reference.model,optimizer,ids,labels,ids,labels)
    train_step(restored.model,reference.model,restored_optimizer,ids,labels,ids,labels)
    assert all(torch.equal(a,b) for a,b in zip(runtime.model.parameters(),restored.model.parameters()))
    assert optimizer.state_dict()['param_groups']==restored_optimizer.state_dict()['param_groups']
    for key,state in optimizer.state_dict()['state'].items():
        for name,value in state.items():
            other=restored_optimizer.state_dict()['state'][key][name]
            assert torch.equal(value,other) if isinstance(value,torch.Tensor) else value==other

    original=(path/'retention_memory.json').read_text()
    (path/'retention_memory.json').write_text(original+' ')
    try:restore_session(path,factory,fingerprint='retention-session',require_retention=True)
    except ValueError:pass
    else:raise AssertionError('tampered retention certificate file was trusted')
    (path/'retention_memory.json').write_text(original)
    registry_text=(path/'training_registry.json').read_text()
    (path/'training_registry.json').unlink()
    try:restore_session(path,factory,fingerprint='retention-session',require_retention=True)
    except (ValueError,FileNotFoundError):pass
    else:raise AssertionError('missing materialized TRAIN registry silently recreated')
    (path/'training_registry.json').write_text(registry_text)
    bad=copy.deepcopy(manifest);bad['files'].pop('training_registry.json')
    (path/'checkpoint.json').write_text(json.dumps(bad))
    try:restore_session(path,factory,fingerprint='retention-session',require_retention=True)
    except ValueError:pass
    else:raise AssertionError('uncommitted TRAIN registry silently trusted')

    legacy=Path(temporary)/'legacy'
    save_session(runtime,optimizer,rng,anchor_rng,counters,legacy,fingerprint='legacy')
    restore_session(legacy,factory,fingerprint='legacy')
    try:restore_session(legacy,factory,fingerprint='legacy',require_retention=True)
    except ValueError:pass
    else:raise AssertionError('same-recipe retention resume accepted legacy files')

print('Native retention session: exact memory/registry/adaptive-state and Adam trajectory recovered; tampering, missing and uncommitted files rejected; legacy default preserved.')
