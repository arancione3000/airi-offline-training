"""Conservative native dialogue continuation with automatic offline rollback."""
import argparse
import gc
import hashlib
import importlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE/'SPEC.json').read_text())

def validate_budget(counters):
    value=counters['elapsed_training_seconds']
    import math
    if not isinstance(value,(float,int)) or not math.isfinite(value) or value<0:
        raise ValueError('invalid training duration')
    if not 0<=counters['phase']<len(SPEC['recipes']):
        raise ValueError('invalid recipe phase')
    if counters['completed_steps']!=SPEC['start_step']+counters['conversation_updates']:
        raise ValueError('update counters mismatch')

def needs_training(counters):
    validate_budget(counters)
    return counters['elapsed_training_seconds']<SPEC['maximum_training_seconds']

def main():
    import torch
    from torch.nn import functional as F
    from generalist_lm.runtime import GeneralistRuntime
    from generalist_lm.training import SFTExample, encode_sft_example
    from session import atomic_json, sha256_file, save_session, restore_session, train_step
    from curriculum import build, digest, norm, exact_correct, repetitive, pilot_eligible, choose_winner
    import transport
    import legacy_run

    parser=argparse.ArgumentParser()
    parser.add_argument('--release',required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=10800)
    args=parser.parse_args()
    if not 60<=args.seconds<=10800:
        raise ValueError('invalid finite search budget')
    for name,expected in SPEC['script_sha256'].items():
        if sha256_file(HERE/name)!=expected:
            raise ValueError('conversation experiment script changed: '+name)
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    transport.SPEC=SPEC
    transport.FINGERPRINT=SPEC['fingerprint']
    transport.TARGET=SPEC['max_steps']
    transport.LEGACY_FINGERPRINT=SPEC['migration_source_fingerprint']
    transport.SEED_TAG=SPEC['migration_source_release']
    transport.SOURCE_FINGERPRINTS={
        transport.SEED_TAG: transport.LEGACY_FINGERPRINT,
    }
    # Strict exact migration; transport's old numeric bounds are replaced here.
    def verify(path,expected_fingerprint=SPEC['fingerprint']):
        manifest=json.loads((path/'checkpoint.json').read_text())
        if manifest['fingerprint']!=expected_fingerprint:
            raise ValueError('checkpoint fingerprint mismatch')
        step=manifest['completed_steps']
        if not SPEC['start_step']<=step<=SPEC['max_steps']:
            raise ValueError('checkpoint outside finite conversation experiment')
        if expected_fingerprint==transport.LEGACY_FINGERPRINT and step!=SPEC['start_step']:
            raise ValueError('only the exact verified pilot checkpoint can migrate')
        if not manifest['files']:
            raise ValueError('empty checkpoint')
        for name,expected in manifest['files'].items():
            path_to_file=path/name
            if not path_to_file.resolve().is_relative_to(path.resolve()) or sha256_file(path_to_file)!=expected:
                raise ValueError('checkpoint digest mismatch: '+name)
        return manifest
    transport.verify_checkpoint=verify
    checkpoint=transport.load_release(args.release,root)
    data=root/'recovered/evidence/dialogue-anchors';(data/'data').mkdir(parents=True,exist_ok=True)
    for name in ['train-1024.jsonl','development.jsonl','test.jsonl']:
        shutil.copy2(root/'restored/experiment'/name,data/'data'/name)
    shutil.copy2(root/'restored/experiment/CONVERSATION_TEST.json',data/'CONVERSATION_TEST.json')
    legacy_run.ROOT=root;legacy_run.DATA=data
    legacy_run.PARENT=root/'context-trial/generalist-state/bootstrap-data/candidate'
    if sha256_file(legacy_run.PARENT/'model.pt')!=legacy_run.EXPECTED_PARENT:
        raise ValueError('native reference changed')
    torch.set_num_threads(2)
    reference=GeneralistRuntime.from_checkpoint(legacy_run.PARENT)
    reference_digest=legacy_run.state_digest(reference.model)
    for parameter in reference.model.parameters(): parameter.requires_grad_(False)
    _,replay,anchors,_,human_rows=legacy_run.prepare(reference.tokenizer)
    # This trial deliberately excludes long, multiturm and tiny-answer human rows.
    # Structural filtering cannot certify semantic quality; authored rows carry most updates.
    human_rows=[r for r in human_rows if len(r['messages'])==2
                and 15<=len(r['messages'][-1]['content'])<=240
                and len(reference.tokenizer.serialize_messages(r['messages'][:-1],add_generation_prompt=True))
                    +len(reference.tokenizer.encode(r['messages'][-1]['content'],eos=True))<=512]
    blocked={norm(m['content']) for file in ['development.jsonl','test.jsonl']
             for line in (data/'data'/file).read_text().splitlines()
             for m in json.loads(line)['messages']}
    blocked.update(norm(p) for p in json.loads((data/'CONVERSATION_TEST.json').read_text())['prompts'])
    from extra_dialogues import build_extra
    authored,heldout=build(blocked)
    extra_train,extra_heldout=build_extra(blocked,heldout)
    heldout+=extra_heldout
    from context_curriculum import build_context
    context_train,context_heldout=build_context(blocked)
    extra_train+=context_train
    heldout+=context_heldout
    from foundation_data import build_foundation
    foundation_train,foundation_heldout=build_foundation(blocked)
    extra_train+=foundation_train
    heldout+=foundation_heldout
    from diagnostics import diagnostic_rows,diagnostic_phase
    training_diagnostics=diagnostic_rows(foundation_train)
    authored+=extra_train
    if not extra_train or not extra_heldout:raise ValueError('empty extra curriculum')
    if not authored or not heldout: raise ValueError('empty curriculum')
    all_heldout={norm(m['content']) for r in heldout for m in r['messages']}
    human_rows=[r for r in human_rows if not any(norm(m['content']) in all_heldout for m in r['messages'])]
    def encode(row):
        ids,labels=encode_sft_example(SFTExample.from_dict(row),reference.tokenizer,1024,require_complete=True)
        scored=(labels!=-100).nonzero()
        width=int(scored[:,0].max())+1 if scored.numel() else 0
        if width<2:raise ValueError('empty supervised answer')
        return ids[:width].unsqueeze(0),labels[:width].unsqueeze(0)
    # Validation includes complete history and targets at 1024, no 128-token truncation.
    validation=[encode(r) for r in heldout]
    def evaluate(runtime):
        runtime.model.eval();nll={};samples=[]
        with torch.no_grad():
            for lang in ['it','en']:
                total,count=0.,0
                for row,(ids,labels) in zip(heldout,validation):
                    if row['language']!=lang:continue
                    logits=runtime.model(ids)['logits'][:,:-1,:]
                    total+=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),reduction='sum'))
                    count+=int((labels[:,1:]!=-100).sum())
                if not count:raise ValueError('empty validation language')
                nll[lang]=total/count
            # A fixed small selection covers phrases, unseen sums and unseen names.
            for lang in ['it','en']:
                values=[r for r in heldout if r['language']==lang]
                selected=values[:8]+[r for r in values if r['family']=='addition-unseen-operands'][:4]+[r for r in values if r['family']=='memory-unseen-name']
                selected+=[r for r in values if r['family'].startswith('extra-heldout') or r['family'].startswith('context-') or r['family'].startswith('foundation-heldout')]
                for row in selected:
                    response=runtime.chat(row['messages'][:-1],temperature=0,max_new_tokens=64)
                    samples.append({'messages':row['messages'][:-1],'expected':row['expected'],'response':response,
                                    'correct':exact_correct(response,row['expected']),'family':row['family'],'language':lang,'pair_id':row.get('pair_id')})
        training_samples=[];training_loss=0.;training_tokens=0
        with torch.no_grad():
            for row in training_diagnostics:
                ids,labels=encode(row)
                logits=runtime.model(ids)['logits'][:,:-1,:]
                training_loss+=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),reduction='sum'))
                training_tokens+=int((labels[:,1:]!=-100).sum())
                response=runtime.chat(row['messages'][:-1],temperature=0,max_new_tokens=32)
                training_samples.append({'messages':row['messages'][:-1],'expected':row['expected'],'response':response,'correct':exact_correct(response,row['expected']),'family':row['family'],'language':row['language']})
        context=[s for s in samples if s['family'].startswith('context-')]
        rates={f'{lang}:{task}':sum(s['correct'] for s in context if s['language']==lang and s['family']=='context-'+task)/sum(1 for s in context if s['language']==lang and s['family']=='context-'+task) for lang in ['it','en'] for task in ['name','location','topic','correction']}
        pairs={s['pair_id'] for s in context}
        pair_rate=sum(all(s['correct'] for s in context if s['pair_id']==pair) for pair in pairs)/len(pairs)
        return {'training_diagnostic':{'scope':'in-sample only; excluded from reward and all generalization gates','correct':sum(x['correct'] for x in training_samples),'probes':len(training_samples),'nll':training_loss/training_tokens,'samples':training_samples},'foundation_correct':sum(s['correct'] for s in samples if s['family'].startswith('foundation-')),'foundation_probes':sum(1 for s in samples if s['family'].startswith('foundation-')),'context_rates':rates,'context_pair_rate':pair_rate,'dialogue_correct':sum(s['correct'] for s in samples if s['family'].startswith('extra-')),'nll':nll,'exact_correct':sum(s['correct'] for s in samples),'probes':len(samples),
                'legacy_correct':sum(s['correct'] for s in samples if not s['family'].startswith(('extra-','context-','foundation-'))),
                'repetition':sum(repetitive(s['response']) for s in samples)/len(samples),'samples':samples,
                'scope':'narrow authored heldout tasks; not proof of general conversation'}
    recipe=SPEC['recipes'][0]
    factory=lambda model:torch.optim.AdamW(model.parameters(),lr=recipe['learning_rate'],weight_decay=.01)
    source_fingerprint=transport.SOURCE_FINGERPRINTS.get(args.release,SPEC['fingerprint'])
    runtime,opt,payload=restore_session(checkpoint,factory,fingerprint=source_fingerprint)
    rng,anchor_rng=random.Random(),random.Random()
    rng.setstate(payload['sampler_rng']);anchor_rng.setstate(payload['anchor_rng'])
    counters=payload['counters']
    source_digest=sha256_file(checkpoint/'model/model.pt')
    curriculum_sha=digest(authored+heldout)
    if args.release==transport.SEED_TAG:
        # Preserve the native weights, Adam moments and RNG; only change explicit LR.
        counters={'completed_steps':SPEC['start_step'],'supervised_tokens':0,'reference_tokens':0,
                  'conversation_updates':0,'curriculum_sha256':curriculum_sha,
                  'baseline':evaluate(runtime),'seed_model_sha256':source_digest,'elapsed_training_seconds':0.,
                  'phase':0,'stagnant_rounds':0,'best_release':transport.SEED_TAG,'precision':'fp32',
                  'foundation_examples':len(foundation_train),'foundation_heldout':len(foundation_heldout),'source_attempted_updates':payload['counters'].get('conversation_updates'),'source_training_seconds':payload['counters'].get('elapsed_training_seconds'),'new_authored_examples':len(extra_train),'new_heldout_examples':len(extra_heldout),
                  'diagnostic_phase_changes':[],'curriculum_stage':0,'stage_changes':[],'reward_history':[],'rollbacks':0,'maximum_training_seconds':SPEC['maximum_training_seconds']}
        counters['champion']=counters['baseline']
    else:
        if counters['curriculum_sha256']!=curriculum_sha:
            raise ValueError('curriculum mismatch')
        validate_budget(counters)
    recipe=SPEC['recipes'][counters['phase']]
    for group in opt.param_groups:group['lr']=recipe['learning_rate']
    out=root/'conversation-results';out.mkdir(exist_ok=True)
    dataset=out/'AUTHORED_CURRICULUM.json';atomic_json(dataset,{'train':authored,'heldout':heldout,'sha256':curriculum_sha})
    extras=[HERE/name for name in SPEC['script_sha256']]+[HERE/'SPEC.json',dataset]
    extras+=list((data/'data').glob('*.jsonl'))+[data/'CONVERSATION_TEST.json']
    publisher=transport.ReleasePublisher(root/'releases')
    # Cache the champion independently so ordinary periodic saves cannot delete it.
    if counters['best_release']==args.release:
        best_path=checkpoint;best_fingerprint=source_fingerprint
    else:
        best_path=transport.load_release(counters['best_release'],root/'champion-cache')
        best_fingerprint=transport.SOURCE_FINGERPRINTS.get(counters['best_release'],SPEC['fingerprint'])
    def save(*,champion=False):
        nonlocal best_path,best_fingerprint
        path=out/f'checkpoint-{counters["completed_steps"]:05d}'
        if path.exists():shutil.rmtree(path)
        save_session(runtime,opt,rng,anchor_rng,counters,path,fingerprint=SPEC['fingerprint'])
        publisher.publish(path,extra_files=extras)
        if champion:best_path=path;best_fingerprint=SPEC['fingerprint']
        for old in out.glob('checkpoint-*'):
            if old!=path and old!=best_path:shutil.rmtree(old)
    started=time.monotonic();next_action=None
    last_save_seconds=counters['elapsed_training_seconds']
    authored_by={lang:[r for r in authored if r['language']==lang and not r['family'].startswith(('extra-','context-','foundation-'))] for lang in ['it','en']}
    extra_by={lang:[r for r in extra_train if r['language']==lang] for lang in ['it','en']}
    human_by={lang:[r for r in human_rows if r['provenance']['language']==lang] for lang in ['it','en']}
    if any(not v for v in human_by.values()):raise ValueError('empty filtered human language')
    while needs_training(counters):
        update=counters['conversation_updates'];lang='it' if update%3!=1 else 'en'
        slot=update%8
        if slot==0:
            families=sorted({r['family'] for r in authored_by[lang]})
            family=rng.choice(families)
            batch=encode(rng.choice([r for r in authored_by[lang] if r['family']==family]));group='old-authored-'+lang
        elif slot==1:
            batch=encode(rng.choice([r for r in extra_by[lang] if r['family'].startswith('extra-') or r['family']=='foundation-dialogue']));group='curated-dialogue-'+lang
        elif slot<5:
            families=sorted({r['family'] for r in foundation_train if r['language']==lang and r['family'] in ['foundation-copy','foundation-recall']})
            family=rng.choice(families)
            batch=encode(rng.choice([r for r in foundation_train if r['language']==lang and r['family']==family]));group='foundation-'+lang
        elif slot==5:
            batch=encode(rng.choice([r for r in training_diagnostics if r['language']==lang]));group='in-sample-diagnostic-'+lang
        elif slot<7:
            candidates=[r for r in extra_by[lang] if r['family'].startswith('context-') and r['stage']<=counters['curriculum_stage']]
            family=rng.choice(sorted({r['family'] for r in candidates}))
            batch=encode(rng.choice([r for r in candidates if r['family']==family]));group='verified-context-'+lang
        else:
            group=['language-it','language-en','domains'][(update//8)%3]
            batch=rng.choice(replay[group])
        scored=(batch[1]!=-100).nonzero()
        width=int(scored[:,1].max())+1
        batch=tuple(v[:,:width] for v in batch)
        anchor=rng.choice(anchors)
        if not needs_training(counters):break
        tick=time.monotonic()
        stats=train_step(runtime.model,reference.model,opt,*batch,*anchor,kl_weight=recipe['kl_weight'])
        counters['elapsed_training_seconds']+=time.monotonic()-tick
        counters['conversation_updates']+=1;counters['completed_steps']+=1
        counters['supervised_tokens']+=stats['supervised_tokens'];counters['reference_tokens']+=stats['reference_tokens']
        counters.setdefault('updates_by_group',{})[group]=counters.setdefault('updates_by_group',{}).get(group,0)+1
        update=counters['conversation_updates']
        if update%16==0:
            atomic_json(out/'PROGRESS.json',{**counters,'status':'training','last_update':stats,'live_promoted':False})
            print(json.dumps({'step':counters['completed_steps'],'updates':update,'loss':stats['loss'],'phase':counters['phase']}),flush=True)
            if (update==16 or update%128==0) and needs_training(counters):publisher.progress({**counters,'status':'training','last_update':stats},args.release)
        if update%1024==0:
            from policy import improved as quality_improved,recover_best,reward,unlocked_stage
            assessment=evaluate(runtime);counters['last_assessment']=assessment
            counters['reward_history'].append({'step':counters['completed_steps'],'score':reward(assessment),'context':assessment['context_rates'],'pair_rate':assessment['context_pair_rate']})
            champion=counters['champion']
            improved=quality_improved(champion,assessment)
            counters['stagnant_rounds']=0 if improved else counters['stagnant_rounds']+1
            if improved:
                counters['champion']=assessment
                stage=unlocked_stage(assessment,counters['curriculum_stage'])
                if stage!=counters['curriculum_stage']:
                    counters['curriculum_stage']=stage
                    counters['stage_changes'].append({'step':counters['completed_steps'],'stage':stage})
                # Predict exact next release before serializing so resume retains champion.
                counters['best_release']=publisher.release_tag(counters['completed_steps'])
                save(champion=True);last_save_seconds=counters['elapsed_training_seconds']
            diag=assessment['training_diagnostic']
            phase=diagnostic_phase(counters['phase'],update,diag['correct'],diag['probes'])
            if phase!=counters['phase']:
                counters['phase']=phase
                recipe=SPEC['recipes'][phase]
                for group in opt.param_groups:group['lr']=recipe['learning_rate']
                counters['diagnostic_phase_changes'].append({'step':counters['completed_steps'],'phase':phase,'training_correct':diag['correct'],'heldout_foundation_correct':assessment['foundation_correct']})
                print(json.dumps({'event':'diagnostic recipe adaptation','phase':phase,'training_correct':diag['correct']}),flush=True)
        if counters['elapsed_training_seconds']-last_save_seconds>=3600:
            save();last_save_seconds=counters['elapsed_training_seconds']
        if update%16==0 and time.monotonic()-started>=args.seconds and needs_training(counters):
            save();next_action={'release':publisher.next_release};break
    if legacy_run.state_digest(reference.model)!=reference_digest or any(p.grad is not None for p in reference.model.parameters()):
        raise ValueError('frozen reference changed')
    if next_action is None:
        counters['last_assessment']=evaluate(runtime)
        metrics=legacy_run.common_metrics(runtime)
        original=legacy_run.common_metrics(reference)
        from generalist_lm.research_cycle import _research_eligible
        from generalist_lm.phase5_diagnostics import degeneration_gate
        counters['native_retention_gate']=list(_research_eligible(original['domains'],metrics['domains'],minimum_loss_gain=.01,max_domain_regression=.03))
        counters['native_degeneration_gate']=list(degeneration_gate(original['language'],metrics['language'],max_repetition_regression=.04,max_entropy_collapse_fraction=.70))
        counters['native_metrics']=metrics
        counters['stop_reason']='six hours of focused native diagnostic training completed; best offline candidate retained'
        save()
        atomic_json(out/'FINAL_RESULT.json',{'counters':counters,'next_action':None,
                  'maximum_training_seconds':SPEC['maximum_training_seconds'],
                  'finished_at_epoch':time.time(),
                  'production_qualified':False,'live_promoted':False,
                  'conversation_assessment':'narrow authored tasks; general conversation requires review'})
        publisher.upload([out/'FINAL_RESULT.json'])
    with open(os.environ['GITHUB_OUTPUT'],'a') as output:
        output.write('next_action='+json.dumps(next_action,separators=(',',':'))+'\n')
        output.write('checkpoint_release='+str(publisher.next_release)+'\n')

if __name__=='__main__':main()
