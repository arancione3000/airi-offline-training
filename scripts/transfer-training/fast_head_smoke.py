"""Measured native CPU optimizer comparison; never publishes its trained weights."""
import gc,json,os,random,tempfile,time
from pathlib import Path
import torch
from torch.nn import functional as F
from generalist_lm.runtime import GeneralistRuntime
from continual_learning import continual_step,retention_rows,verified_memories
from fast_head import configure_head,install_feature_cache,head_summary,configure_projection,attach_projection
from session import save_session


def real_head_smoke(runtime,checkpoint,reference,eligible,human_rows,encode,anchors,evaluate,state_digest):
    champion=GeneralistRuntime.from_checkpoint(checkpoint/'model')
    for p in champion.model.parameters():p.requires_grad_(False)
    champion.model.eval();protected=state_digest(champion.model)
    install_feature_cache(champion.model,cache_bytes=64*1024*1024)
    memory=verified_memories(champion,retention_rows(list(eligible.values())),encode)
    rank=getattr(runtime.model,'_airi_projection_rank_requested',0)
    if rank:configure_projection(runtime.model,champion.model,memory,encode,rank=rank)
    memories=[encode(next(r for r in memory if r['language']==lang)) for lang in ('it','en')]
    # One reviewed training example, chosen without looking at any held-out text.
    candidates=[r for r in human_rows if 32<=encode(r)[0].shape[1]<=192]
    if not candidates:raise AssertionError('no bounded reviewed benchmark task')
    row=sorted(candidates,key=lambda r:json.dumps(r['messages'],sort_keys=True))[len(candidates)//2]
    batch=encode(row);anchor=anchors[0]
    def loss(rt):
        with torch.no_grad():
            logits=rt.model(batch[0])['logits'][:,:-1]
            return float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch[1][:,1:].reshape(-1)))
    initial=loss(champion)
    full=GeneralistRuntime.from_checkpoint(checkpoint/'model')
    full_opt=torch.optim.AdamW(full.model.parameters(),lr=1e-4,weight_decay=.01)
    tick=time.monotonic()
    for _ in range(8):
        continual_step(full.model,champion.model,reference.model,full_opt,batch,memories,anchor,
                       retention_weight=16,champion_kl_weight=16)
    full_seconds=time.monotonic()-tick
    del full_opt,full;gc.collect()
    optimizer=torch.optim.AdamW([p for p in runtime.model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    if rank:attach_projection(optimizer,runtime.model)
    tick=time.monotonic()
    for _ in range(8):
        continual_step(runtime.model,champion.model,reference.model,optimizer,batch,memories,anchor,
                       retention_weight=16,champion_kl_weight=16)
    head_seconds=time.monotonic()-tick
    # The first eight include cold acquisition/replay feature fills. Repeating
    # the SAME batch separately measures amortized cache behavior.
    tick=time.monotonic()
    for _ in range(24):
        continual_step(runtime.model,champion.model,reference.model,optimizer,batch,memories,anchor,
                       retention_weight=16,champion_kl_weight=16)
    warm_seconds=time.monotonic()-tick;final=loss(runtime)
    if not final<initial:raise AssertionError('real cached head did not learn reviewed training task')
    if state_digest(champion.model)!=protected:raise AssertionError('frozen champion changed')
    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory)/'checkpoint';r=random.Random(1)
        save_session(runtime,optimizer,r,r,dict(completed_steps=32),p,fingerprint='ephemeral-cached-head-smoke')
        native=GeneralistRuntime.from_checkpoint(p/'model')
        native.model.eval();runtime.model.eval()
        with torch.no_grad():
            if not torch.equal(native.model(batch[0])['logits'],runtime.model(batch[0])['logits']):
                raise AssertionError('native exported logits differ from cached-head training')
        prompt=row['messages'][:-1]
        if native.chat(prompt,temperature=0,max_new_tokens=64)!=runtime.chat(prompt,temperature=0,max_new_tokens=64):
            raise AssertionError('native exported chat differs from cached-head training')
    print(json.dumps(dict(phase='real native cached head benchmark PASS',
        visible_cpus=os.cpu_count(),torch_threads=torch.get_num_threads(),strategy=head_summary(runtime.model),
        full_8_updates_seconds=full_seconds,head_8_updates_seconds=head_seconds,
        head_warm_24_updates_seconds=warm_seconds,
        cold_inclusive_optimizer_speedup=full_seconds/head_seconds,
        repeated_batch_optimizer_speedup=(full_seconds/8)/(warm_seconds/24),
        training_task_initial_nll=initial,training_task_final_nll=final,
        native_export_exact=True,champion_unchanged=True,
        scope='same bounded reviewed training batch,objective and CPU; optimizer timing excludes load/evaluation; not end-to-end throughput or general conversation',
        live_promoted=False)),flush=True)
