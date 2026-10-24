"""Ephemeral verification of the repaired objective on a real native checkpoint."""
import copy
import json
import random
import torch
from torch.nn import functional as F
from generalist_lm.runtime import GeneralistRuntime
from continual_learning import continual_step, retention_rows, verified_memories, rejection_reasons,cache_tokenizer


def real_learning_smoke(runtime, inherited_optimizer, checkpoint, reference, eligible,
                        human_rows, encode, anchors, evaluate, state_digest):
    if hasattr(runtime.model,'_airi_head_features'):
        from fast_head_smoke import real_head_smoke
        return real_head_smoke(runtime,checkpoint,reference,eligible,human_rows,encode,anchors,evaluate,state_digest)
    champion=GeneralistRuntime.from_checkpoint(checkpoint/'model')
    cache_tokenizer(champion.tokenizer)
    champion.model.eval()
    for p in champion.model.parameters():p.requires_grad_(False)
    protected_digest=state_digest(champion.model)
    memory=verified_memories(champion,retention_rows(list(eligible.values())),encode)
    memories=[encode(next(r for r in memory if r['language']==lang)) for lang in ('it','en')]
    # A training task chosen without reading any development or final answer.
    tasks=sorted((r for r in eligible.values() if r['family']=='extra-dialogue'),
                 key=lambda r:(len(r['expected']),r['messages'][0]['content']))
    if not tasks:raise AssertionError('reviewed authored dialogue was excluded from the lesson catalog')
    def task_loss(row):
        ids,labels=encode(row)
        with torch.no_grad():
            logits=champion.model(ids)['logits'][:,:-1]
            return float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1)))
    row=max(tasks[:16],key=task_loss);batch=encode(row)
    def loss():
        runtime.model.eval()
        with torch.no_grad():
            logits=runtime.model(batch[0])['logits'][:,:-1]
            return float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch[1][:,1:].reshape(-1)))
    before=evaluate(runtime);initial=loss()
    optimizer=torch.optim.AdamW(runtime.model.parameters(),lr=1e-5,weight_decay=.01)
    for _ in range(32):
        continual_step(runtime.model,champion.model,reference.model,optimizer,batch,memories,anchors[0],
                       retention_weight=2.,champion_kl_weight=2.)
    final=loss();after=evaluate(runtime)
    print(json.dumps({'phase':'real objective training task measured','initial_nll':initial,'final_nll':final}),flush=True)
    if not final<initial:raise AssertionError('repaired native training objective did not learn its training task')
    if state_digest(champion.model)!=protected_digest or any(p.grad is not None for p in champion.model.parameters()):
        raise AssertionError('protected real champion changed')
    print(json.dumps({'phase':'real native continual objective PASS',
                      'parameters':sum(p.numel() for p in runtime.model.parameters()),
                      'optimizer_updates':32,'initial_training_task_nll':initial,'final_training_task_nll':final,
                      'development_before':{k:before[k] for k in ('legacy_correct','dialogue_correct','foundation_correct','human_nll')},
                      'development_after':{k:after[k] for k in ('legacy_correct','dialogue_correct','foundation_correct','human_nll')},
                      'development_rejection_reasons':rejection_reasons(before,after),
                      'training_retention_rows':len(memory),'champion_unchanged':True,
                      'scope':'ephemeral optimizer/integrity experiment; training-task gain is not general conversation',
                      'live_promoted':False}),flush=True)
