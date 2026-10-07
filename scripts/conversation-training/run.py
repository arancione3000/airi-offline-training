"""Finite autonomous recipe search, followed by validated offline learning."""
import argparse
import gc
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
    parser.add_argument('--ledger-release',default='')
    parser.add_argument('--stage',choices=['pilot','learn'],default='pilot')
    parser.add_argument('--trial',type=int,default=0)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=10800)
    args=parser.parse_args()
    if not 0<=args.trial<len(SPEC['recipes']) or not 60<=args.seconds<=10800:
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
    authored,heldout=build(blocked)
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
                for row in selected:
                    response=runtime.chat(row['messages'][:-1],temperature=0,max_new_tokens=64)
                    samples.append({'messages':row['messages'][:-1],'expected':row['expected'],'response':response,
                                    'correct':exact_correct(response,row['expected']),'family':row['family']})
        return {'nll':nll,'exact_correct':sum(s['correct'] for s in samples),'probes':len(samples),
                'repetition':sum(repetitive(s['response']) for s in samples)/len(samples),'samples':samples,
                'scope':'narrow authored heldout tasks; not proof of general conversation'}
    recipe=SPEC['recipes'][args.trial]
    factory=lambda model:torch.optim.AdamW(model.parameters(),lr=recipe['learning_rate'],weight_decay=.01)
    source_fingerprint=transport.LEGACY_FINGERPRINT if args.release==transport.SEED_TAG else SPEC['fingerprint']
    runtime,opt,payload=restore_session(checkpoint,factory,fingerprint=source_fingerprint)
    rng,anchor_rng=random.Random(),random.Random()
    rng.setstate(payload['sampler_rng']);anchor_rng.setstate(payload['anchor_rng'])
    counters=payload['counters']
    source_digest=sha256_file(checkpoint/'model/model.pt')
    curriculum_sha=digest(authored+heldout)
    if args.release==transport.SEED_TAG:
        # Fresh moments are intentional for the new objective; native seed is retained.
        opt=factory(runtime.model)
        counters={'completed_steps':SPEC['start_step'],'supervised_tokens':0,'reference_tokens':0,
                  'conversation_updates':0,'stage':'pilot','trial':args.trial,'curriculum_sha256':curriculum_sha,
                  'baseline':evaluate(runtime),'seed_model_sha256':source_digest,'elapsed_training_seconds':0.,
                  'recipe':recipe,'precision':'fp32','optimizer_policy':'explicit new AdamW for new curriculum objective'}
    else:
        if counters['trial']!=args.trial or counters['curriculum_sha256']!=curriculum_sha:
            raise ValueError('trial or curriculum mismatch')
        if args.stage=='learn' and counters['stage']=='pilot':
            counters['stage']='learn';counters['champion']=counters['last_assessment'];counters['stagnant_rounds']=0
        elif counters['stage']!=args.stage:raise ValueError('stage mismatch')
    ledger={'records':[]}
    if args.ledger_release:
        ledger_path=root/'ledger';ledger_path.mkdir()
        transport.gh('release','download',transport.check_tag(args.ledger_release),'--repo',os.environ['GITHUB_REPOSITORY'],
                     '--pattern','SEARCH_STATE.json','--dir',ledger_path)
        ledger=json.loads((ledger_path/'SEARCH_STATE.json').read_text())
        if ledger.get('fingerprint')!=SPEC['fingerprint']:raise ValueError('ledger fingerprint mismatch')
    out=root/'conversation-results';out.mkdir(exist_ok=True)
    dataset=out/'AUTHORED_CURRICULUM.json';atomic_json(dataset,{'train':authored,'heldout':heldout,'sha256':curriculum_sha})
    extras=[HERE/name for name in SPEC['script_sha256']]+[HERE/'SPEC.json',dataset]
    extras+=list((data/'data').glob('*.jsonl'))+[data/'CONVERSATION_TEST.json']
    publisher=transport.ReleasePublisher(root/'releases')
    def save():
        path=out/f'checkpoint-{counters["completed_steps"]:05d}'
        if path.exists():shutil.rmtree(path)
        save_session(runtime,opt,rng,anchor_rng,counters,path,fingerprint=SPEC['fingerprint'])
        publisher.publish(path,extra_files=extras)
        for old in out.glob('checkpoint-*'):
            if old!=path:shutil.rmtree(old)
    started=time.monotonic();next_action=None
    limit=SPEC['pilot_updates'] if args.stage=='pilot' else SPEC['pilot_updates']+SPEC['learning_updates']
    authored_by={lang:[r for r in authored if r['language']==lang] for lang in ['it','en']}
    human_by={lang:[r for r in human_rows if r['provenance']['language']==lang] for lang in ['it','en']}
    if any(not v for v in human_by.values()):raise ValueError('empty filtered human language')
    while counters['conversation_updates']<limit:
        update=counters['conversation_updates'];lang='it' if update%3!=1 else 'en'
        slot=update%8
        if slot<4:batch=encode(rng.choice(authored_by[lang]));group='authored-'+lang
        elif slot<7:batch=encode(rng.choice(human_by[lang]));group='human-'+lang
        else:
            group=['language-it','language-en','domains'][(update//8)%3]
            batch=rng.choice(replay[group])
        scored=(batch[1]!=-100).nonzero()
        width=int(scored[:,1].max())+1
        batch=tuple(v[:,:width] for v in batch)
        anchor=rng.choice(anchors);tick=time.monotonic()
        stats=train_step(runtime.model,reference.model,opt,*batch,*anchor,kl_weight=recipe['kl_weight'])
        counters['elapsed_training_seconds']+=time.monotonic()-tick
        counters['conversation_updates']+=1;counters['completed_steps']+=1
        counters['supervised_tokens']+=stats['supervised_tokens'];counters['reference_tokens']+=stats['reference_tokens']
        counters.setdefault('updates_by_group',{})[group]=counters.setdefault('updates_by_group',{}).get(group,0)+1
        update=counters['conversation_updates']
        if update%16==0:
            atomic_json(out/'PROGRESS.json',{**counters,'status':'training','last_update':stats,'live_promoted':False})
            print(json.dumps({'step':counters['completed_steps'],'updates':update,'loss':stats['loss'],'trial':args.trial}),flush=True)
            if update==16 or update%128==0:publisher.progress({**counters,'status':'training','last_update':stats},args.release)
        assessment_due=(args.stage=='pilot' and update==limit) or (args.stage=='learn' and (update-SPEC['pilot_updates'])%512==0)
        if assessment_due:
            assessment=evaluate(runtime);counters['last_assessment']=assessment
            if args.stage=='pilot':break
            champion=counters['champion']
            improved=(assessment['exact_correct']>=champion['exact_correct']
                      and all(assessment['nll'][l]<champion['nll'][l]*.995 for l in ['it','en'])
                      and assessment['repetition']<=max(.2,champion['repetition']))
            counters['stagnant_rounds']=0 if improved else counters['stagnant_rounds']+1
            if improved:
                counters['champion']=assessment
                save();counters['best_release']=publisher.next_release
            if counters['stagnant_rounds']>=3:
                counters['stop_reason']='three validation rounds without measured improvement; best_release retained'
                break
        if update%16==0 and time.monotonic()-started>=args.seconds:
            save();next_action={'release':publisher.next_release,'stage':args.stage,'trial':args.trial,'ledger_release':args.ledger_release};break
    if next_action is None:
        if args.stage=='pilot':
            assessment=counters['last_assessment']
            save()
            record={'trial':args.trial,'recipe':recipe,'before':counters['baseline'],'after':assessment,
                    'eligible':pilot_eligible(counters['baseline'],assessment),'release':publisher.next_release}
            if any(r['trial']==args.trial for r in ledger['records']):raise ValueError('duplicate trial record')
            ledger['records'].append(record)
            ledger['fingerprint']=SPEC['fingerprint']
            atomic_json(out/'SEARCH_STATE.json',ledger);publisher.upload([out/'SEARCH_STATE.json'])
            if args.trial+1<len(SPEC['recipes']):
                next_action={'release':transport.SEED_TAG,'stage':'pilot','trial':args.trial+1,'ledger_release':publisher.next_release}
            else:
                winner=choose_winner(ledger['records'])
                if winner:
                    next_action={'release':winner['release'],'stage':'learn','trial':winner['trial'],'ledger_release':publisher.next_release}
                else:counters['stop_reason']='all recipes failed measured response/validation improvement'
        else:
            # Production gates keep their original definitions and remain offline.
            metrics=legacy_run.common_metrics(runtime)
            original=legacy_run.common_metrics(reference)
            from generalist_lm.research_cycle import _research_eligible
            from generalist_lm.phase5_diagnostics import degeneration_gate
            retained=_research_eligible(original['domains'],metrics['domains'],minimum_loss_gain=.01,max_domain_regression=.03)
            degenerated=degeneration_gate(original['language'],metrics['language'],max_repetition_regression=.04,max_entropy_collapse_fraction=.70)
            counters['native_retention_gate']=list(retained);counters['native_degeneration_gate']=list(degenerated)
            counters['native_metrics']=metrics
            prompts=json.loads((data/'CONVERSATION_TEST.json').read_text())['prompts']
            atomic_json(out/'NEW_GENERATIONS.json',[{'prompt':p,'response':runtime.chat([{'role':'user','content':p}],temperature=0,max_new_tokens=96)} for p in prompts])
            save();publisher.upload([out/'NEW_GENERATIONS.json'])
        if legacy_run.state_digest(reference.model)!=reference_digest or any(p.grad is not None for p in reference.model.parameters()):
            raise ValueError('frozen reference changed')
        atomic_json(out/'FINAL_RESULT.json',{'counters':counters,'next_action':next_action,'production_qualified':False,'live_promoted':False,
                                          'conversation_assessment':'pending manual review; exact curriculum tasks are limited evidence'})
        publisher.upload([out/'FINAL_RESULT.json'])
    with open(os.environ['GITHUB_OUTPUT'],'a') as output:
        output.write('next_action='+json.dumps(next_action,separators=(',',':'))+'\n')
        output.write('checkpoint_release='+str(publisher.next_release)+'\n')

if __name__=='__main__':main()
