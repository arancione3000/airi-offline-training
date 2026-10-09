"""One native 1024-context dialogue pass. Offline; never promotes weights."""
import argparse,gc,hashlib,json,os,random,resource,shutil,time
from dataclasses import replace
from pathlib import Path
import torch
from generalist_lm.runtime import GeneralistRuntime,checkpoint_model_files
from generalist_lm.training import SFTExample,encode_sft_example,load_sft_jsonl,nll_stats_on_examples
from generalist_lm.research_cycle import _transfer_compatible_weights,_research_eligible
from generalist_lm.bootstrap_data import load_bootstrap_replay
from generalist_lm.curriculum import train_rows
from generalist_lm.phase5_diagnostics import protected_bootstrap_texts,evaluate_phase5_language,degeneration_gate
from generalist_lm.lineage_migration import evaluate_lineage_runtime
from session import atomic_json,sha256_file,train_step,save_session,restore_session
from persist import publish
from recovery_policy import choose_batch

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'recovered/evidence/dialogue-anchors'
PARENT=ROOT/'context-trial/generalist-state/bootstrap-data/candidate'
OUT=ROOT/'long-training/results'
EXPECTED_DATA='4cd65b21228375b7abaf81da5e32e977dc1182f0a73b942dc81c34a38a698d91'
EXPECTED_PARENT='998530839d9722316b59cb28ff1ef68a7863f517de73fa8fef8ce9677a24b85e'

def optimizer(model):return torch.optim.AdamW(model.parameters(),lr=2e-6,weight_decay=.01)
def norm(text):return ' '.join(text.strip().casefold().split())

def select_dialogues(rows,blocked):
    # Exclude exact normalized held-out text in every role, including user prompts.
    return [r for r in rows if not any(norm(m['content']) in blocked for m in r['messages'])]

def prepare(tokenizer):
    assert sha256_file(DATA/'data/train-1024.jsonl')==EXPECTED_DATA
    rows=[json.loads(x) for x in (DATA/'data/train-1024.jsonl').read_text().splitlines()]
    assert len(rows)==10564
    heldout=load_sft_jsonl(DATA/'data/development.jsonl')+load_sft_jsonl(DATA/'data/test.jsonl')
    blocked=protected_bootstrap_texts()|{norm(m['content']) for x in heldout for m in x.messages}
    def eligible(x,length):
        return (not any(norm(m['content']) in blocked for m in x.messages)
                and len(tokenizer.serialize_messages(x.messages[:-1],add_generation_prompt=True))
                +len(tokenizer.encode(x.messages[-1]['content'],eos=True))<=length)
    rows=select_dialogues(rows,blocked)
    dialogues=[SFTExample.from_dict(x) for x in rows]
    assert all(eligible(x,1024) for x in dialogues)
    groups={'language-it':[],'language-en':[],'domains':[]}
    replay=load_bootstrap_replay(PARENT.parent)
    for doc in replay.documents:
        if doc.source.startswith('tatoeba-it-'):group='language-it';prompt='Ripeti questa frase: '+doc.text
        elif doc.source.startswith('tatoeba-en-'):group='language-en';prompt='Repeat this sentence: '+doc.text
        else:continue
        row=SFTExample([{'role':'user','content':prompt},{'role':'assistant','content':doc.text}])
        if eligible(row,1024):groups[group].append(row)
    curriculum=[r for r in train_rows() if eligible(r.sft(),128)]
    anchors=[r.sft() for r in curriculum]
    groups['domains']=[r.sft() for r in curriculum if r.domain!='language']
    selector=random.Random(71026)
    for name,values in groups.items():
        assert values,name
        if len(values)>512:groups[name]=selector.sample(values,512)
    assert anchors
    def encode(values,length):
        return [tuple(v.unsqueeze(0) for v in encode_sft_example(x,tokenizer,length,require_complete=True)) for x in values]
    deck=list(range(len(dialogues)));random.Random(71025).shuffle(deck)
    return encode(dialogues,1024),{g:encode(v,1024) for g,v in groups.items()},encode(anchors,128),deck,rows

def common_metrics(runtime):
    # Comparable target budget; restore real context after every evaluation.
    old=runtime.config;old_model=runtime.model.config
    runtime.config=replace(old,context_length=128);runtime.model.config=replace(old_model,context_length=128)
    try:
        development={lang:nll_stats_on_examples(runtime.model,runtime.tokenizer,
                     [SFTExample.from_dict(json.loads(x)) for x in (DATA/'data/development.jsonl').read_text().splitlines()
                      if json.loads(x)['provenance']['language']==lang]) for lang in ['it','en']}
        return {'language':evaluate_phase5_language(runtime),'domains':evaluate_lineage_runtime(runtime,cycle=1),
                'development':development,'metric_context':128,'model_context':old.context_length}
    finally:runtime.config=old;runtime.model.config=old_model

def state_digest(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    args=argparse.ArgumentParser();args.add_argument('--resume',type=Path);args.add_argument('--max-steps',type=int,default=17203)
    args=args.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'RUNNING.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    started=time.monotonic();counters={}
    try:
        torch.set_num_threads(2)
        assert sha256_file(PARENT/'model.pt')==EXPECTED_PARENT
        parent_files=checkpoint_model_files(PARENT,verify=True)
        parent_hashes={str(p):sha256_file(p) for p in parent_files}
        reference=GeneralistRuntime.from_checkpoint(PARENT);teacher_digest=state_digest(reference.model)
        for p in reference.model.parameters():p.requires_grad_(False)
        dialogue,replay,anchors,deck,rows=prepare(reference.tokenizer)
        assert args.max_steps==17203
        spec={'parent_sha256':EXPECTED_PARENT,'parent_snapshot':'e949157c88a582b7e52ca46001afcf8da8ddd22e',
              'dataset_sha256':EXPECTED_DATA,'context_length':1024,'tokenizer_digest':reference.tokenizer.digest,
              'torch_version':torch.__version__,'precision':'fp32','batch_size':1,'learning_rate':2e-6,
              'weight_decay':.01,'gradient_clip':1.,'kl_weight':4.,'unlikelihood_weight':.2,'eos_weight':2.0,
              'seed':71025,'max_steps':args.max_steps,'dialogue_schedule':'recovery: 25% Italian dialogue, 25% English dialogue, 50% replay; seeded sampling with replacement',
              'replay_policy':'cycle language-it, language-en, domains; training-only; 50% updates, not token quota',
              'recovery_start_step':13107,'recovery_updates':4096,'migration_source_release':'airi-offline-13107-37560538801-1-4',
              'source_rows':10564,'rows':len(dialogue),'excluded_prompt_overlap_rows':10564-len(dialogue),
              'dialogue_supervised_tokens_per_pass':sum(r['provenance']['supervised_tokens'] for r in rows),
              'anchors':len(anchors),'replay_rows':{g:len(v) for g,v in replay.items()},
              'script_sha256':{p.name:sha256_file(p) for p in [Path(__file__),ROOT/'long-training/session.py',ROOT/'long-training/persist.py',ROOT/'long-training/recovery_policy.py']},
              'promotion':'offline only; production_qualified false; human conversation assessment still required'}
        fingerprint=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
        rng=random.Random(71025);anchor_rng=random.Random(71026)
        by_language={lang:[i for i,r in enumerate(rows) if r['provenance']['language']==lang] for lang in ['it','en']}
        assert all(by_language.values())
        if args.resume:
            runtime,opt,payload=restore_session(args.resume,optimizer,fingerprint=fingerprint)
            counters=payload['counters'];rng.setstate(payload['sampler_rng']);anchor_rng.setstate(payload['anchor_rng'])
        else:
            raise ValueError('recovery requires a verified native checkpoint')
        if counters['completed_steps']<13107:raise ValueError('recovery starts at step 13107')
        counters.setdefault('recovery_replay_updates',0)
        atomic_json(OUT/'SPEC.json',{**spec,'fingerprint':fingerprint,'parameter_count':sum(p.numel() for p in runtime.model.parameters())})
        extras=[OUT/'SPEC.json',Path(__file__),ROOT/'long-training/session.py',ROOT/'long-training/persist.py',ROOT/'long-training/recovery_policy.py',
                DATA/'data/train-1024.jsonl',DATA/'data/development.jsonl',DATA/'data/test.jsonl',DATA/'CONVERSATION_TEST.json']
        def checkpoint(durable=False):
            step=counters['completed_steps'];path=OUT/f'checkpoint-{step:05d}'
            if not path.exists():save_session(runtime,opt,rng,anchor_rng,counters,path,fingerprint=fingerprint)
            if durable:
                publish(path,OUT/'archives',extra_files=extras)
                atomic_json(OUT/'DURABILITY.json',{'completed_steps':step,'status':'saved externally','production_qualified':False})
            existing=sorted(OUT.glob('checkpoint-*'))
            for previous in existing[:-2]:shutil.rmtree(previous)
            return path
        if not args.resume:checkpoint(durable=True)
        groups=list(replay);last=time.monotonic()
        while counters['completed_steps']<args.max_steps:
            step=counters['completed_steps']
            group,batch=choose_batch(step,counters,rng,by_language,dialogue,replay)
            anchor=anchors[anchor_rng.randrange(len(anchors))]
            tick=time.monotonic();stats=train_step(runtime.model,reference.model,opt,*batch,*anchor)
            counters['elapsed_training_seconds']+=time.monotonic()-tick
            counters['completed_steps']+=1;counters['supervised_tokens']+=stats['supervised_tokens']
            counters['reference_tokens']+=stats['reference_tokens']
            counters['tokens_by_group'][group]=counters['tokens_by_group'].get(group,0)+stats['supervised_tokens']
            if (step+1)%16==0 or step+1==args.max_steps:
                progress={**counters,'status':'training','target_steps':args.max_steps,'last_update':stats,
                          'eta_training_seconds':counters['elapsed_training_seconds']/counters['completed_steps']*(args.max_steps-counters['completed_steps']),
                          'peak_rss_gb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'live_promoted':False}
                atomic_json(OUT/'PROGRESS.json',progress)
                print(json.dumps({k:progress[k] for k in ['completed_steps','supervised_tokens','eta_training_seconds','peak_rss_gb']}),flush=True)
            if (step+1)%16==0:checkpoint(durable=(step+1==128 or (step+1)%1024==0))
        final=checkpoint(durable=True)
        assert state_digest(reference.model)==teacher_digest and all(p.grad is None for p in reference.model.parameters())
        assert all(sha256_file(p)==digest for p,digest in parent_hashes.items())
        atomic_json(OUT/'PROGRESS.json',{**counters,'status':'evaluating','live_promoted':False})
        before=common_metrics(reference);after=common_metrics(runtime)
        retention=_research_eligible(before['domains'],after['domains'],minimum_loss_gain=.01,max_domain_regression=.03)
        degeneration=degeneration_gate(before['language'],after['language'],max_repetition_regression=.04,max_entropy_collapse_fraction=.70)
        dev_ok=all(after['development'][g]['nll_per_byte']<=before['development'][g]['nll_per_byte'] for g in ['it','en'])
        lang_ok=after['language']['language_nll']<=before['language']['language_nll']
        prompts=json.loads((DATA/'CONVERSATION_TEST.json').read_text())['prompts']
        generations=[{'prompt':p,'response':runtime.chat([{'role':'user','content':p}],max_new_tokens=48)} for p in prompts]
        atomic_json(OUT/'NEW_GENERATIONS.json',generations)
        with torch.no_grad():long_logits=runtime.model(torch.full((1,1024),runtime.tokenizer.encode('a')[0],dtype=torch.long))['logits']
        result={'counters':counters,'before':before,'after':after,'retention_gate':list(retention),'degeneration_gate':list(degeneration),
                'development_nonregression':dev_ok,'language_nonregression':lang_ok,
                'metric_gates_passed':bool(retention[0] and degeneration[0] and dev_ok and lang_ok),
                'conversation_assessment':'pending; raw generations saved; no conversational success claimed',
                'long_logits_shape':list(long_logits.shape),'long_logits_finite':bool(torch.isfinite(long_logits).all()),
                'parent_unchanged':True,'reference_unchanged':True,'production_qualified':False,'live_promoted':False}
        atomic_json(OUT/'FINAL_RESULT.json',result)
        report=OUT/'AIRI_LONG_TRAINING_REPORT.md'
        report.write_text('Prova nativa completata. Nessuna promozione live.\n\n'+json.dumps(result,ensure_ascii=False,indent=2))
        from persist import upload
        upload([report,OUT/'FINAL_RESULT.json',OUT/'NEW_GENERATIONS.json'])
        atomic_json(OUT/'PROGRESS.json',{**counters,'status':'completed; manual conversation assessment required','live_promoted':False})
    except Exception as exc:
        # Keep the last verified snapshot; do not automatically retrain after failure.
        atomic_json(OUT/'ERROR.json',{'type':type(exc).__name__,'message':str(exc),'counters':counters,'live_promoted':False})
        raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
