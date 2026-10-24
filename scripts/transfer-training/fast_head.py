"""CPU head tuning with bounded caches of immutable native body activations.

No new inference implementation is needed: checkpoints contain an ordinary,
untied native lm_head. Cache entries never contain predictions or labels.
"""
from collections import OrderedDict
from types import MethodType
import torch


class FrozenFeatureCache:
    def __init__(self, model, *, max_bytes=256*1024*1024, max_entries=4096):
        if not isinstance(max_bytes,int) or isinstance(max_bytes,bool) or not 0<=max_bytes<=512*1024*1024:
            raise ValueError('invalid bounded feature cache')
        self.model=model;self.max_bytes=max_bytes;self.max_entries=max_entries
        self.entries=OrderedDict();self.bytes=0;self.hits=0;self.misses=0;self.invalidations=0
        self.body=[p for name,p in model.named_parameters() if name!='lm_head.weight']
        self.versions=tuple(p._version for p in self.body)
        self.original=model.forward;self.capture=False;self.captured=None
        self.hook=model.lm_head.register_forward_pre_hook(self._capture)

    def _capture(self, module, args):
        if self.capture:self.captured=args[0].detach().clone()

    def _check(self):
        if any(p.requires_grad for p in self.body):
            raise RuntimeError('cached native body must remain frozen')
        versions=tuple(p._version for p in self.body)
        if versions!=self.versions:
            self.entries.clear();self.bytes=0;self.versions=versions;self.invalidations+=1

    def forward(self, ids, labels=None, *, past_key_values=None, use_cache=False):
        self._check()
        # Native KV-cache generation remains the exact original code path.
        if labels is not None or past_key_values is not None or use_cache:
            return self.original(ids,labels,past_key_values=past_key_values,use_cache=use_cache)
        key=(str(ids.dtype),tuple(ids.shape),ids.detach().contiguous().cpu().numpy().tobytes())
        if key in self.entries:
            self.hits+=1;features=self.entries.pop(key);self.entries[key]=features
            return dict(logits=self.model.lm_head(features),loss=None,past_key_values=None)
        self.misses+=1;self.capture=True;self.captured=None
        try:result=self.original(ids)
        finally:self.capture=False
        features=self.captured;self.captured=None
        if features is None:raise RuntimeError('native head activation was not captured')
        size=features.numel()*features.element_size()+len(key[2])
        if size<=self.max_bytes and self.max_entries:
            while self.entries and (self.bytes+size>self.max_bytes or len(self.entries)>=self.max_entries):
                old_key,old=self.entries.popitem(last=False)
                self.bytes-=old.numel()*old.element_size()+len(old_key[2])
            self.entries[key]=features;self.bytes+=size
        return result

    def summary(self):
        return dict(hits=self.hits,misses=self.misses,bytes=self.bytes,entries=len(self.entries),
                    invalidations=self.invalidations,max_bytes=self.max_bytes)


def configure_head(model, *, cache_bytes=256*1024*1024):
    """Expand tied output weights into a separate, initially identical head."""
    existing=getattr(model,'_airi_head_features',None)
    if existing is not None:
        existing._check();return existing
    if float(model.config.dropout)!=0 or model.lm_head.weight.device.type!='cpu':
        raise ValueError('cached head tuning requires deterministic CPU inference')
    if model.lm_head.weight is model.token_embedding.weight:
        # Do not perturb checkpoint RNG merely by allocating an identical head.
        with torch.random.fork_rng(devices=[]):
            head=torch.nn.Linear(model.config.d_model,model.config.vocab_size,bias=False)
        head=head.to(dtype=model.lm_head.weight.dtype,device=model.lm_head.weight.device)
        with torch.no_grad():head.weight.copy_(model.lm_head.weight)
        model.lm_head=head
    model.config.tie_embeddings=False
    for p in model.parameters():p.requires_grad_(False)
    model.lm_head.weight.requires_grad_(True)
    return install_feature_cache(model,cache_bytes=cache_bytes)


def install_feature_cache(model, *, cache_bytes=128*1024*1024):
    """Cache a frozen backbone, including the unchanged frozen teachers."""
    existing=getattr(model,'_airi_head_features',None)
    if existing is not None:
        existing._check();return existing
    if float(model.config.dropout)!=0 or model.lm_head.weight.device.type!='cpu':
        raise ValueError('feature cache requires deterministic CPU inference')
    cache=FrozenFeatureCache(model,max_bytes=cache_bytes)
    cache._check()
    model._airi_head_features=cache
    def forward(this,ids,labels=None,*,past_key_values=None,use_cache=False):
        return cache.forward(ids,labels,past_key_values=past_key_values,use_cache=use_cache)
    model.forward=MethodType(forward,model)
    return cache


def head_summary(model):
    cache=getattr(model,'_airi_head_features',None)
    projection=getattr(model,'_airi_head_projection',None)
    return dict(mode='cached_native_head',
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                total_parameters=sum(p.numel() for p in model.parameters()),
                protected_feature_rank=projection['rank'] if projection else 0,
                feature_cache=cache.summary() if cache else None)


def set_projection(model,base_weight,features):
    """Constrain the head delta to leave selected TRAINING features unchanged."""
    if features.ndim!=2 or features.shape[1]!=model.config.d_model or not 1<=features.shape[0]<=model.config.d_model//2:
        raise ValueError('invalid bounded head protection features')
    if not bool(torch.isfinite(features).all()) or base_weight.shape!=model.lm_head.weight.shape:
        raise ValueError('invalid native head protection')
    with torch.no_grad():
        _,singular,vh=torch.linalg.svd(features.double(),full_matrices=False)
        rank=int((singular>max(1e-10,float(singular.max())*1e-8)).sum())
        if rank==0:raise ValueError('empty protected feature span')
        basis=vh[:rank]
        projector=torch.eye(features.shape[1],dtype=torch.float64)-basis.T@basis
    model._airi_head_projection=dict(base=base_weight.detach().double().clone(),projector=projector,rank=rank)
    return rank


def configure_projection(model,champion,rows,encode,*,rank):
    """Select fragile certified training tokens, never held-out examples."""
    if not isinstance(rank,int) or isinstance(rank,bool) or not 1<=rank<=model.config.d_model//2:
        raise ValueError('invalid protected training rank')
    groups={};capture=[]
    hook=champion.lm_head.register_forward_pre_hook(lambda module,args:capture.append(args[0].detach()))
    try:
        with torch.no_grad():
            for row in rows:
                ids,labels=encode(row);capture.clear();logits=champion(ids)['logits'][0,:-1]
                targets=labels[0,1:];positions=(targets!=-100).nonzero().flatten()
                gold=targets[positions];scores=logits[positions]
                if not bool((scores.argmax(-1)==gold).all()):raise ValueError('uncertified training row in head protection')
                alternatives=scores.clone();alternatives[torch.arange(len(positions)),gold]=-float('inf')
                margins=scores[torch.arange(len(positions)),gold]-alternatives.max(-1).values
                index=int(margins.argmin());position=int(positions[index])
                family=row['family'];intent=row.get('expected','') if family in {'extra-dialogue','foundation-dialogue'} else ''
                key=(row['language'],family,intent)
                candidate=(float(margins[index]),capture[-1][0,position].clone())
                if key not in groups or candidate[0]<groups[key][0]:groups[key]=candidate
    finally:hook.remove()
    # Balance both languages and conversational/other skills. Within each
    # stratum protect the weakest certified training groups first.
    buckets={}
    for key,value in groups.items():
        kind='dialogue' if key[1] in {'extra-dialogue','foundation-dialogue'} else 'other'
        buckets.setdefault((key[0],kind),[]).append(value)
    for values in buckets.values():values.sort(key=lambda x:x[0])
    chosen=[]
    for index in range(rank):
        for key in sorted(buckets):
            if index<len(buckets[key]):chosen.append(buckets[key][index])
            if len(chosen)==rank:break
        if len(chosen)==rank:break
    return set_projection(model,champion.lm_head.weight,torch.stack([x[1] for x in chosen]))


def attach_projection(optimizer,model):
    """Projected Adam; optimizer moments persist, inference stays ordinary native."""
    def before(opt,args,kwargs):
        p=getattr(model,'_airi_head_projection',None)
        if p is None or not bool(torch.isfinite(p['projector']).all()):
            raise RuntimeError('head protection must be initialized before optimizer update')
    def after(opt,args,kwargs):
        p=model._airi_head_projection
        with torch.no_grad():
            weight=model.lm_head.weight
            weight.copy_((p['base']+(weight.double()-p['base'])@p['projector']).to(weight.dtype))
    optimizer.register_step_pre_hook(before);optimizer.register_step_post_hook(after)
    return optimizer
