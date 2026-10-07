"""Offline experiment helpers; native objectives and checkpoint formats."""
from pathlib import Path
import hashlib,json,os
import torch
from generalist_lm.training import causal_training_objective,reference_kl_loss
from generalist_lm.runtime import GeneralistRuntime
from generalist_lm.bootstrap_training import _save_optimizer_checkpoint,_load_optimizer_checkpoint

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def atomic_json(path,payload):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False))
    tmp.replace(path)

def train_step(model,reference,optimizer,ids,labels,anchor_ids,anchor_labels,*,precision='fp32',kl_weight=.25):
    student_storage={p.untyped_storage().data_ptr() for p in model.parameters()}
    if any(p.untyped_storage().data_ptr() in student_storage for p in reference.parameters()):
        raise ValueError('reference must have independent parameter storage')
    model.train();reference.eval();optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=(precision=='bf16')):
        logits=model(ids)['logits']
        ce,objective=causal_training_objective(logits,labels,ids,repetition_unlikelihood_weight=.05,eos_loss_weight=1.2)
        anchor_student=model(anchor_ids)['logits']
        with torch.no_grad():anchor_reference=reference(anchor_ids)['logits']
        kl,count=reference_kl_loss(anchor_student,anchor_reference,anchor_labels)
        loss=ce+kl_weight*kl
    if not bool(torch.isfinite(loss)):raise RuntimeError('non-finite training loss; no optimizer update')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
    optimizer.step()
    return {'loss':float(loss.detach()),'ce':float(ce.detach()),'kl':float(kl.detach()),
            'supervised_tokens':int((labels[:,1:]!=-100).sum()),'reference_tokens':count,
            'gradient_norm_before_clipping':float(norm),'objective':objective}

def save_session(runtime,optimizer,rng,anchor_rng,counters,path,*,fingerprint):
    path=Path(path)
    if path.exists():raise FileExistsError(f'checkpoint already exists: {path.name}')
    tmp=path.with_name(path.name+'.partial');tmp.mkdir(parents=True,exist_ok=False)
    runtime.save_checkpoint(tmp/'model',metadata={
        'production_qualified':False,'role':'offline-conversation-training-checkpoint',
        'fingerprint':fingerprint,'attempted_supervised_tokens':counters.get('supervised_tokens',0),
        'accepted_live_tokens':0,'completed_steps':counters.get('completed_steps',0),
    })
    _save_optimizer_checkpoint(optimizer,tmp/'optimizer.pt')
    torch.save({'sampler_rng':rng.getstate(),'anchor_rng':anchor_rng.getstate(),
                'torch_rng':torch.get_rng_state(),'counters':counters,'fingerprint':fingerprint},tmp/'resume.pt')
    manifest={'fingerprint':fingerprint,'completed_steps':counters.get('completed_steps',0),
              'files':{str(p.relative_to(tmp)):sha256_file(p) for p in tmp.rglob('*') if p.is_file()}}
    atomic_json(tmp/'checkpoint.json',manifest)
    tmp.replace(path)
    return manifest

def restore_session(path,optimizer_factory,*,fingerprint):
    path=Path(path);manifest=json.loads((path/'checkpoint.json').read_text())
    if manifest['fingerprint']!=fingerprint:raise ValueError('checkpoint provenance mismatch')
    for name,digest in manifest['files'].items():
        p=path/name
        if not p.resolve().is_relative_to(path.resolve()):raise ValueError('unsafe checkpoint path')
        if sha256_file(p)!=digest:raise ValueError(f'checkpoint digest mismatch: {name}')
    payload=torch.load(path/'resume.pt',map_location='cpu',weights_only=True)
    if payload['fingerprint']!=fingerprint:raise ValueError('resume provenance mismatch')
    runtime=GeneralistRuntime.from_checkpoint(path/'model')
    optimizer=optimizer_factory(runtime.model)
    _load_optimizer_checkpoint(optimizer,path/'optimizer.pt')
    torch.set_rng_state(payload['torch_rng'])
    return runtime,optimizer,payload
