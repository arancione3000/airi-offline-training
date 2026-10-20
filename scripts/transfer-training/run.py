"""Bounded compositional training with separate memorization and sealed transfer tests."""
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
SPEC_PATH = Path(os.environ.get('AIRI_EXPERIMENT_SPEC', str(HERE/'SPEC.json'))).resolve()
SPEC = json.loads(SPEC_PATH.read_text())
LEARNING = SPEC.get('continual_learning')

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
    search=counters.get('learning_search',{})
    return (not search.get('stopped',False)
            and counters['elapsed_training_seconds']<SPEC['maximum_training_seconds'])

SAMPLING_SCHEDULES={
    'balanced':('old-authored',)*6+('curated-dialogue',)*6+('composition',)*2+('verified-context','native-replay'),
    'context':('old-authored',)*4+('curated-dialogue',)*4+('composition',)*2+('verified-context',)*4+('native-replay',)*2,
    'dialogue':('old-authored',)*4+('curated-dialogue',)*6+('composition',)*2+('verified-context',)*2+('native-replay',)*2,
    'foundation':('old-authored',)*4+('curated-dialogue',)*3+('composition',)*5+('verified-context',)*2+('native-replay',)*2,
    'retention':('old-authored',)*5+('curated-dialogue',)*4+('composition',)*2+('verified-context',)*2+('native-replay',)*3,
}

# Human dialogue receives four out of sixteen update slots.
for _focus,_schedule in list(SAMPLING_SCHEDULES.items()):
    _slots=list(_schedule)
    for _index in (0,4,8,12):_slots[_index]='human-dialogue'
    SAMPLING_SCHEDULES[_focus]=tuple(_slots)

def sampling_group(update,focus='balanced'):
    if not isinstance(update,int) or isinstance(update,bool) or update<0:
        raise ValueError('invalid update index')
    if focus not in SAMPLING_SCHEDULES:raise ValueError('invalid training focus')
    return SAMPLING_SCHEDULES[focus][update%16]

def selection_needed(counters,*,force=False):
    """Avoid penalizing a candidate twice when evaluation and a boundary coincide."""
    return not (force and not counters.get('exploration_pending',False)
                and counters.get('selected_at_step')==counters.get('completed_steps'))

def success_slot(update,sample_group,has_lessons):
    slots=LEARNING.get('success_transfer_slots',(2,10)) if LEARNING else (2,10)
    return bool(has_lessons and update%16 in slots and sample_group not in {'human-dialogue','native-replay'})

def main():
    import torch
    from torch.nn import functional as F
    from generalist_lm.runtime import GeneralistRuntime
    from generalist_lm.training import SFTExample, encode_sft_example
    from session import atomic_json, sha256_file, save_session, restore_session, restore_champion, train_step
    from curriculum import build, digest, norm, exact_correct, repetitive, pilot_eligible, choose_winner
    import transport
    import legacy_run

    parser=argparse.ArgumentParser()
    parser.add_argument('--release',required=True)
    parser.add_argument('--corpus-audit-only',action='store_true')
    parser.add_argument('--success-smoke-only',action='store_true')
    parser.add_argument('--learning-smoke-only',action='store_true')
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=10800)
    args=parser.parse_args()
    if not 60<=args.seconds<=10800:
        raise ValueError('invalid finite search budget')
    for name,expected in SPEC['script_sha256'].items():
        if sha256_file(HERE/name)!=expected:
            raise ValueError('conversation experiment script changed: '+name)
    if LEARNING:
        from continual_learning import validate_configuration
        validate_configuration(SPEC)
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
    if LEARNING:
        from continual_learning import cache_tokenizer
        cache_tokenizer(reference.tokenizer)
    reference_digest=legacy_run.state_digest(reference.model)
    for parameter in reference.model.parameters(): parameter.requires_grad_(False)
    _,replay,anchors,_,human_rows=legacy_run.prepare(reference.tokenizer,include_dialogue_batches=False)
    previous_human_text={norm(m['content']) for r in human_rows for m in r['messages']}
    # Preserve complete human histories and answers, never truncate.
    human_rows=[r for r in human_rows if 2<=len(r['messages'])<=6
                and 12<=len(r['messages'][-1]['content'].encode('utf-8'))<=480
                and len(reference.tokenizer.serialize_messages(r['messages'][:-1],add_generation_prompt=True))
                    +len(reference.tokenizer.encode(r['messages'][-1]['content'],eos=True))<=1024]
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
    from diagnostics import diagnostic_rows,recipe_phase
    from composition_data import build_composition, sampling_pools
    composition_train,sealed_transfer=build_composition(blocked)
    foundation_train+=composition_train
    extra_train+=composition_train
    training_diagnostics=diagnostic_rows(foundation_train)
    composition_pools=sampling_pools(foundation_train)
    authored+=extra_train
    if not extra_train or not extra_heldout:raise ValueError('empty extra curriculum')
    if not authored or not heldout: raise ValueError('empty curriculum')
    all_heldout={norm(m['content']) for r in heldout for m in r['messages']}
    from human_data import load_corpus
    human_directory=HERE.parents[1]/'data/oasst2'
    corpus=load_corpus(human_directory,SPEC['human_corpus']['pins'],blocked|all_heldout|previous_human_text|{norm(m['content']) for r in authored for m in r['messages']})
    external_development=[]
    for lang in ('it','en'):
        external_development += [r for r in corpus['development'] if r['language']==lang][:16]
    external_final=[]
    for lang in ('it','en'):
        external_final += [r for r in corpus['final'] if r['language']==lang][:8]
    corpus_heldout_text={norm(m['content']) for split in ('development','final') for r in corpus[split] for m in r['messages']}
    human_rows=[r for r in human_rows if not any(norm(m['content']) in all_heldout|corpus_heldout_text for m in r['messages'])]
    for r in human_rows:r['language']=r['provenance']['language']
    human_rows+=corpus['train']
    def encode(row):
        ids,labels=encode_sft_example(SFTExample.from_dict(row),reference.tokenizer,1024,require_complete=True)
        scored=(labels!=-100).nonzero()
        width=int(scored[:,0].max())+1 if scored.numel() else 0
        if width<2:raise ValueError('empty supervised answer')
        return ids[:width].unsqueeze(0),labels[:width].unsqueeze(0)
    # Validation includes complete history and targets at 1024, no 128-token truncation.
    validation=[encode(r) for r in heldout]
    human_validation=[encode(r) for r in external_development]
    if args.corpus_audit_only:
        counts={split:{lang:sum(r['language']==lang for r in rows) for lang in ('it','en')} for split,rows in corpus.items()}
        for row in human_rows:encode(row)
        print(json.dumps({'phase':'verified real corpus audit','source_release':args.release,'human_counts':counts,'combined_human_training_rows':len(human_rows),'combined_multiturn_rows':sum(len(r['messages'])>2 for r in human_rows),'development_used':len(external_development),'final_used':len(external_final),'optimizer_updates':0,'live_promoted':False}),flush=True)
        return
    def evaluate(runtime):
        runtime.model.eval();nll={};samples=[];human_nll={}
        with torch.no_grad():
            for lang in ('it','en'):
                total,count=0.,0
                for row,(ids,labels) in zip(external_development,human_validation):
                    if row['language']!=lang:continue
                    logits=runtime.model(ids)['logits'][:,:-1,:]
                    total+=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),reduction='sum'))
                    count+=int((labels[:,1:]!=-100).sum())
                if count:human_nll[lang]=total/count
            if not human_nll:raise ValueError('no unseen human development examples')
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
        return {'human_nll':human_nll,'human_development_rows':len(external_development),'training_diagnostic':{'scope':'in-sample only; excluded from reward and all generalization gates','correct':sum(x['correct'] for x in training_samples),'probes':len(training_samples),'nll':training_loss/training_tokens,'samples':training_samples},'foundation_correct':sum(s['correct'] for s in samples if s['family'].startswith('foundation-')),'foundation_probes':sum(1 for s in samples if s['family'].startswith('foundation-')),'context_rates':rates,'context_pair_rate':pair_rate,'dialogue_correct':sum(s['correct'] for s in samples if s['family'].startswith('extra-')),'nll':nll,'exact_correct':sum(s['correct'] for s in samples),'probes':len(samples),
                'legacy_correct':sum(s['correct'] for s in samples if not s['family'].startswith(('extra-','context-','foundation-'))),
                'repetition':sum(repetitive(s['response']) for s in samples)/len(samples),'samples':samples,
                'scope':'narrow authored heldout tasks; not proof of general conversation'}
    recipe=SPEC['recipes'][0]
    factory=lambda model:torch.optim.AdamW(model.parameters(),lr=recipe['learning_rate'],weight_decay=.01)
    source_fingerprint=transport.SOURCE_FINGERPRINTS.get(args.release,SPEC['fingerprint'])
    runtime,opt,payload=restore_session(checkpoint,factory,fingerprint=source_fingerprint)
    if LEARNING:cache_tokenizer(runtime.tokenizer)
    rng,anchor_rng=random.Random(),random.Random()
    rng.setstate(payload['sampler_rng']);anchor_rng.setstate(payload['anchor_rng'])
    counters=payload['counters']
    source_digest=sha256_file(checkpoint/'model/model.pt')
    from success_bank import (catalog, empty_bank, validate_bank, probe_rows,
                              summary as bank_summary, retains_correct_answers, migrate_bank)
    eligible = catalog(authored, heldout + sealed_transfer + external_development + external_final)
    if args.learning_smoke_only:
        from continual_smoke import real_learning_smoke
        real_learning_smoke(runtime,opt,checkpoint,reference,eligible,human_rows,
                            encode,anchors,evaluate,legacy_run.state_digest)
        return
    if args.success_smoke_only:
        from success_smoke import real_checkpoint_smoke
        real_checkpoint_smoke(runtime,opt,checkpoint,eligible,encode,anchors,
                              SPEC['fingerprint'],legacy_run.state_digest)
        return
    bank_file = checkpoint/'success_bank.json'
    if source_fingerprint == SPEC['fingerprint'] and not bank_file.exists():
        raise ValueError('resumed success-transfer checkpoint has no committed bank')
    bank = (json.loads(bank_file.read_text()) if bank_file.exists() else empty_bank(SPEC['fingerprint']))
    if bank_file.exists() and source_fingerprint!=SPEC['fingerprint']:
        bank=migrate_bank(bank,eligible,source_fingerprint,SPEC['fingerprint'],runtime.config.vocab_size)
    validate_bank(bank,eligible,SPEC['fingerprint'],runtime.config.vocab_size)
    curriculum_sha=digest(authored+heldout+sealed_transfer+human_rows+external_development+external_final)
    if args.release==transport.SEED_TAG:
        # Preserve the native weights, Adam moments and RNG; only change explicit LR.
        counters={'completed_steps':SPEC['start_step'],'supervised_tokens':0,'reference_tokens':0,
                  'conversation_updates':0,'curriculum_sha256':curriculum_sha,
                  'baseline':evaluate(runtime),'seed_model_sha256':source_digest,'elapsed_training_seconds':0.,
                  'phase':0,'stagnant_rounds':0,'best_release':transport.SEED_TAG,'precision':'fp32',
                  'foundation_examples':len(foundation_train),'foundation_heldout':len(foundation_heldout),'source_attempted_updates':payload['counters'].get('conversation_updates'),'source_training_seconds':payload['counters'].get('elapsed_training_seconds'),'composition_examples':len(composition_train),'sealed_transfer_probes':len(sealed_transfer),'new_authored_examples':len(extra_train),'new_heldout_examples':len(extra_heldout),
                  'diagnostic_phase_changes':[],'curriculum_stage':0,'stage_changes':[],'reward_history':[],'rollbacks':0,'selection_history':[],'human_examples':len(human_rows),'human_corpus_counts':{split:len(rows) for split,rows in corpus.items()},'maximum_training_seconds':SPEC['maximum_training_seconds']}
        counters['champion']=counters['baseline']
    else:
        if counters['curriculum_sha256']!=curriculum_sha:
            raise ValueError('curriculum mismatch')
        validate_budget(counters)
    from policy import (improved as quality_improved,recover_best,recovery_lr_factor,
                        reward,training_focus,unlocked_stage)
    counters.setdefault('rollbacks',0);counters.setdefault('selection_history',[])
    counters.setdefault('training_focus',training_focus(counters['champion']))
    if LEARNING:
        counters.setdefault('learning_search',{'consecutive_rejections':0,'evaluations':0,
                            'accepted_updates':0,'stopped':False,'history':[]})
        if args.release==transport.SEED_TAG:
            # The champion's durable Adam remains in its source checkpoint. A
            # fresh, explicit exploration state avoids inherited stale moments.
            opt=factory(runtime.model)
            counters['learning_search']['optimizer_policy']='preserve candidate weights and Adam within bounded trial; fresh Adam only after rollback or accepted interpolation'
    recipe=SPEC['recipes'][counters['phase']]
    def apply_learning_rate():
        if LEARNING:
            from continual_learning import profile_index
            profile=LEARNING['profiles'][profile_index(counters['learning_search']['consecutive_rejections'],len(LEARNING['profiles']))]
            for parameter_group in opt.param_groups:parameter_group['lr']=profile['learning_rate']
            counters['effective_learning_rate']=profile['learning_rate']
            return
        factor=recovery_lr_factor(counters['rollbacks'],tuple(SPEC.get('recovery_lr_factors',[1.,.5,.25])))
        for parameter_group in opt.param_groups:parameter_group['lr']=recipe['learning_rate']*factor
        counters['effective_learning_rate']=recipe['learning_rate']*factor
    apply_learning_rate()
    out=root/'conversation-results';out.mkdir(exist_ok=True)
    dataset=out/'AUTHORED_CURRICULUM.json';atomic_json(dataset,{'train':authored,'development':heldout,'sealed_transfer':sealed_transfer,'sha256':curriculum_sha})
    extras=[HERE/name for name in SPEC['script_sha256']]+[SPEC_PATH,dataset]
    extras+=list((data/'data').glob('*.jsonl'))+[data/'CONVERSATION_TEST.json']
    extras+=list(human_directory.glob('*'))
    publisher=transport.ReleasePublisher(root/'releases')
    # Cache the champion independently so ordinary periodic saves cannot delete it.
    if counters['best_release']==args.release:
        best_path=checkpoint;best_fingerprint=source_fingerprint
    else:
        best_path=transport.load_release(counters['best_release'],root/'champion-cache')
        best_fingerprint=transport.SOURCE_FINGERPRINTS.get(counters['best_release'],SPEC['fingerprint'])
    from morph_model import inherit_state, adopt, consolidate, model_record
    from model_channel import Client as ModelClient, publish_current
    if 'morph' not in counters:
        counters['morph']=inherit_state(payload['counters'].get('morph'),legacy_run.state_digest(runtime.model))
    def load_protected_champion():
        stream=torch.get_rng_state()
        try:protected = GeneralistRuntime.from_checkpoint(best_path/'model')
        finally:torch.set_rng_state(stream)
        protected.model.eval()
        for parameter in protected.model.parameters(): parameter.requires_grad_(False)
        if LEARNING:cache_tokenizer(protected.tokenizer)
        return protected
    protected_champion = load_protected_champion()
    if LEARNING:
        from continual_learning import retention_rows,verified_memories
        memory_rows=verified_memories(protected_champion,retention_rows(list(eligible.values())),encode)
        memory_by={lang:[encode(r) for r in memory_rows if r['language']==lang] for lang in ('it','en')}
        counters['retention_memory']={'rows':len(memory_rows),'sha256':digest(memory_rows),
                                     'scope':'eligible authored training only; no evaluation answers'}
    def transfer_options():
        if not LEARNING:return {}
        from continual_learning import transfer_replay_options
        return transfer_replay_options(LEARNING,counters['learning_search'],memory_by,anchor_rng)
    def save(*,champion=False):
        nonlocal best_path,best_fingerprint
        path=out/f'checkpoint-{counters["completed_steps"]:05d}'
        if path.exists():shutil.rmtree(path)
        counters['success_bank']=bank_summary(bank)
        save_session(runtime,opt,rng,anchor_rng,counters,path,fingerprint=SPEC['fingerprint'],success_bank=bank)
        if LEARNING:
            state={'fingerprint':SPEC['fingerprint'],'completed_steps':counters['completed_steps'],
                   'elapsed_training_seconds':counters['elapsed_training_seconds'],
                   'maximum_training_seconds':SPEC['maximum_training_seconds'],
                   'continue':needs_training(counters),'job_index':int(os.environ.get('CONTINUATION','0')),
                   'model_id':counters['morph']['model_id'],'revision':counters['morph']['revision'],
                   'search':counters['learning_search'],'production_qualified':False,'live_promoted':False}
            atomic_json(path/'AIRI_RUN_STATE.json',state)
            manifest=json.loads((path/'checkpoint.json').read_text())
            manifest['files']['AIRI_RUN_STATE.json']=sha256_file(path/'AIRI_RUN_STATE.json')
            atomic_json(path/'checkpoint.json',manifest)
        publisher.publish(path,extra_files=extras)
        if champion:
            best_path=path;best_fingerprint=SPEC['fingerprint']
            state=counters['morph']
            if legacy_run.state_digest(runtime.model)!=state['weights_digest']:
                raise ValueError('stable identity must reference the selected champion weights')
            record=model_record(state,release=publisher.next_release,fingerprint=SPEC['fingerprint'],
                                step=counters['completed_steps'],
                                model_files={n:d for n,d in json.loads((path/'checkpoint.json').read_text())['files'].items() if n.startswith('model/')})
            # Publish the pointer only after its immutable checkpoint is complete.
            counters['model_channel']=publish_current(ModelClient(os.environ['GITHUB_REPOSITORY']),record,os.environ['GITHUB_SHA'])
        for old in out.glob('checkpoint-*'):
            if old!=path and old!=best_path:shutil.rmtree(old)
    def select_candidate(reason,*,force=False):
        nonlocal runtime,opt,best_path,best_fingerprint,last_save_seconds,protected_champion,memory_rows,memory_by
        if not selection_needed(counters,force=force):return 'already_selected'
        assessment=evaluate(runtime);counters['last_candidate_assessment']=assessment
        counters['reward_history'].append({'step':counters['completed_steps'],'score':reward(assessment),'context':assessment['context_rates'],'pair_rate':assessment['context_pair_rate']})
        champion=counters['champion']
        accepted=quality_improved(champion,assessment) and retains_correct_answers(champion,assessment)
        rollback_candidate=False
        if LEARNING:
            from continual_learning import rejection_reasons, try_interpolations, record_decision
            search=counters['learning_search']
            report={'step':counters['completed_steps'],'learning_rate':counters['effective_learning_rate'],
                    'reasons':rejection_reasons(champion,assessment),'interpolation':None}
            if not accepted:
                blended=try_interpolations(runtime.model,protected_champion.model,
                    evaluate=lambda:evaluate(runtime),
                    acceptable=lambda value:quality_improved(champion,value) and retains_correct_answers(champion,value),
                    alphas=LEARNING['interpolation_alphas'])
                report['interpolation']=blended['alpha'];report['interpolation_probes']=blended['probes']
                if blended['alpha'] is not None:
                    assessment=blended['assessment'];accepted=True
                    # A merged parameter path has no exact Adam trajectory.
                    opt=factory(runtime.model);apply_learning_rate()
            report['accepted']=accepted
            decision=record_decision(search,LEARNING,counters['conversation_updates'],accepted,force=force)
            report.update(decision)
            rollback_candidate=decision['rollback']
            counters['exploration_pending']=decision['exploration_pending']
            search['history'].append(report);search['history']=search['history'][-64:]
            print(json.dumps({'phase':'independent continual-learning decision',**report}),flush=True)
            if accepted:apply_learning_rate()
        if not accepted:
            # Mine training observations before destructive rollback. Eval rows are
            # deliberately unavailable to harvest() and distill_step().
            from success_transfer import harvest
            donated = harvest(runtime,protected_champion,
                              probe_rows(eligible,bank['harvests'],SPEC['success_transfer']['probe_rows']),
                              encode,bank,eligible,
                              {'step':counters['completed_steps'],'fingerprint':SPEC['fingerprint'],
                               'model_sha256':legacy_run.state_digest(runtime.model),
                               'runner_sha':os.environ.get('GITHUB_SHA','local')})
            counters['success_bank']=bank_summary(bank)
            counters['success_last_harvest']={'step':counters['completed_steps'],'donated_rows':donated}
            print(json.dumps({'phase':'salvaged rejected native candidate',**counters['success_bank']}),flush=True)
        counters['stagnant_rounds']=0 if accepted else counters['stagnant_rounds']+1
        action='continue_candidate'
        if accepted:
            adopt(counters['morph'],legacy_run.state_digest(runtime.model),
                  step=counters['completed_steps'],reason='independently verified candidate')
            counters['champion']=assessment;action='accepted'
        elif (rollback_candidate if LEARNING else force or recover_best(counters['stagnant_rounds'])):
            runtime,opt,_=restore_champion(best_path,factory,rng,anchor_rng,fingerprint=best_fingerprint)
            if LEARNING:opt=factory(runtime.model)
            counters['rollbacks']+=1;counters['stagnant_rounds']=0;action='rolled_back'
            apply_learning_rate()
            def restore_transaction():
                nonlocal runtime,opt
                runtime,opt,_=restore_champion(best_path,factory,rng,anchor_rng,fingerprint=best_fingerprint)
                if LEARNING:opt=factory(runtime.model)
                apply_learning_rate()
            def learn_lesson(entry):
                from success_transfer import distill_step
                tick=time.monotonic()
                stats=distill_step(runtime.model,protected_champion.model,opt,entry,encode(entry['row']),
                                   rng.choice(memory_by[entry['row']['language']]) if LEARNING else rng.choice(anchors),lr_multiplier=SPEC['success_transfer']['lr_multiplier'],
                                   lr_cap=SPEC['success_transfer']['lr_cap'],**transfer_options())
                counters['elapsed_training_seconds']+=time.monotonic()-tick
                counters['conversation_updates']+=1;counters['completed_steps']+=1
                counters['supervised_tokens']+=stats['supervised_tokens'];counters['reference_tokens']+=stats['reference_tokens']
                bank['transfer_updates']+=1
                counters['last_transfer_retention']={k:stats[k] for k in
                    ('retention_weight','champion_kl_weight','retention_examples')}
                group='morph-consolidation-'+entry['row']['language']
                counters.setdefault('updates_by_group',{})[group]=counters.setdefault('updates_by_group',{}).get(group,0)+1
            result=consolidate(counters['morph'],bank,champion,train=learn_lesson,
                               evaluate=lambda:evaluate(runtime),restore=restore_transaction,
                               can_train=lambda:needs_training(counters),
                               can_adopt=lambda a,b:quality_improved(a,b) and retains_correct_answers(a,b),
                               step=lambda:counters['completed_steps'],
                               max_updates=SPEC['morph_model']['updates_per_skill'])
            counters['last_consolidation']={k:v for k,v in result.items() if k!='assessment'}
            if LEARNING:
                # Consolidation is a separate transaction, not part of the next
                # candidate's learning horizon.
                counters['learning_search']['trial_start_update']=counters['conversation_updates']
            if result['action']=='skill_accepted':
                adopt(counters['morph'],legacy_run.state_digest(runtime.model),
                      step=counters['completed_steps'],reason='selective consolidation: '+result['skill'])
                counters['champion']=result['assessment'];action='consolidated'
                if LEARNING:
                    search=counters['learning_search']
                    search['consecutive_rejections']=0;search['accepted_updates']+=1
                    search['stopped']=search['evaluations']>=LEARNING['maximum_evaluations']
                    search['history'][-1]['consolidated']=True
                    apply_learning_rate()
                print(json.dumps({'phase':'single AIRI model acquired verified skill',
                                  'model_id':counters['morph']['model_id'],'revision':counters['morph']['revision'],
                                  'skill':result['skill']}),flush=True)
        if action in {'accepted','consolidated'}:
            stage=unlocked_stage(counters['champion'],counters['curriculum_stage'])
            if stage!=counters['curriculum_stage']:
                counters['curriculum_stage']=stage;counters['stage_changes'].append({'step':counters['completed_steps'],'stage':stage})
        counters['training_focus']=training_focus(counters['champion'])
        counters['last_assessment']=counters['champion']
        counters['selection_history'].append({'step':counters['completed_steps'],'reason':reason,'action':action,'focus':counters['training_focus'],'effective_learning_rate':counters['effective_learning_rate']})
        if action=='continue_candidate':
            counters['selected_at_step']=counters['completed_steps']
        if action in {'accepted','rolled_back','consolidated'}:
            counters['selected_at_step']=counters['completed_steps']
            counters['best_release']=publisher.release_tag(counters['completed_steps'])
            save(champion=True);last_save_seconds=counters['elapsed_training_seconds']
            if action in {'accepted','consolidated'}:
                del protected_champion
                gc.collect()
                protected_champion = load_protected_champion()
                if LEARNING:
                    memory_rows=verified_memories(protected_champion,retention_rows(list(eligible.values())),encode)
                    memory_by={lang:[encode(r) for r in memory_rows if r['language']==lang] for lang in ('it','en')}
                    counters['retention_memory']={'rows':len(memory_rows),'sha256':digest(memory_rows),
                                                 'scope':'certified correct eligible training answers; no evaluation answers'}
        return action
    # Initialize the stable model once; later rollback saves leave its pointer
    # untouched. Updating the same identity never requires a manual checkpoint swap.
    if args.release==transport.SEED_TAG:
        counters['best_release']=publisher.release_tag(counters['completed_steps'])
        save(champion=True)
    started=time.monotonic();next_action=None
    last_save_seconds=counters['elapsed_training_seconds']
    authored_by={lang:[r for r in authored if r['language']==lang and not r['family'].startswith(('extra-','context-','foundation-'))] for lang in ['it','en']}
    extra_by={lang:[r for r in extra_train if r['language']==lang] for lang in ['it','en']}
    human_by={lang:[r for r in human_rows if r['language']==lang] for lang in ['it','en']}
    if any(not v for v in human_by.values()):raise ValueError('empty filtered human language')
    while needs_training(counters):
        update=counters['conversation_updates'];lang='it' if update%3!=1 else 'en'
        sample_group=(LEARNING['sampling_schedule'][update%16] if LEARNING
                      else sampling_group(update,counters['training_focus']))
        if sample_group=='human-dialogue':
            batch=encode(rng.choice(human_by[lang]));group='human-dialogue-'+lang
        elif sample_group=='old-authored':
            families=sorted({r['family'] for r in authored_by[lang]})
            family=rng.choice(families)
            batch=encode(rng.choice([r for r in authored_by[lang] if r['family']==family]));group='old-authored-'+lang
        elif sample_group=='curated-dialogue':
            batch=encode(rng.choice([r for r in extra_by[lang] if r['family'].startswith('extra-') or r['family']=='foundation-dialogue']));group='curated-dialogue-'+lang
        elif sample_group=='composition':
            pools=composition_pools[lang]
            family=rng.choice(sorted(pools))
            batch=encode(rng.choice(pools[family]));group='foundation-'+lang
        elif sample_group=='verified-context':
            candidates=[r for r in extra_by[lang] if r['family'].startswith('context-') and r['stage']<=counters['curriculum_stage']]
            family=rng.choice(sorted({r['family'] for r in candidates}))
            batch=encode(rng.choice([r for r in candidates if r['family']==family]));group='verified-context-'+lang
        else:
            group=['language-it','language-en','domains'][(update//16)%3]
            batch=rng.choice(replay[group])
        lesson=None
        # Preserve all human and native replay slots; at most 2/16 other slots
        # replay archived improvements from any preceding native candidate.
        if success_slot(update,sample_group,bank['entries']):
            keys=sorted(bank['entries'])
            lesson=bank['entries'][keys[(bank['transfer_updates'])%len(keys)]]
            batch=encode(lesson['row']);group='success-transfer-'+lesson['row']['language']
        scored=(batch[1]!=-100).nonzero()
        width=int(scored[:,1].max())+1
        batch=tuple(v[:,:width] for v in batch)
        anchor=rng.choice(anchors)
        if not needs_training(counters):break
        tick=time.monotonic()
        if lesson is None:
            if LEARNING:
                from continual_learning import continual_step,profile_index
                profile=LEARNING['profiles'][profile_index(counters['learning_search']['consecutive_rejections'],len(LEARNING['profiles']))]
                memories=[memory_by[l][anchor_rng.randrange(len(memory_by[l]))] for l in ('it','en')]
                stats=continual_step(runtime.model,protected_champion.model,reference.model,opt,batch,memories,anchor,
                                     retention_weight=profile['retention_weight'],
                                     champion_kl_weight=profile['champion_kl_weight'])
            else:stats=train_step(runtime.model,reference.model,opt,*batch,*anchor,kl_weight=recipe['kl_weight'])
        else:
            from success_transfer import distill_step
            stats=distill_step(runtime.model,protected_champion.model,opt,lesson,batch,
                               rng.choice(memory_by[lesson['row']['language']]) if LEARNING else anchor,
                               lr_multiplier=SPEC['success_transfer']['lr_multiplier'],
                               lr_cap=SPEC['success_transfer']['lr_cap'],**transfer_options())
            bank['transfer_updates']+=1
            counters['last_transfer_retention']={k:stats[k] for k in
                ('retention_weight','champion_kl_weight','retention_examples')}
            counters['success_bank']=bank_summary(bank)
        counters['elapsed_training_seconds']+=time.monotonic()-tick
        counters['conversation_updates']+=1;counters['completed_steps']+=1
        counters['supervised_tokens']+=stats['supervised_tokens'];counters['reference_tokens']+=stats['reference_tokens']
        counters.setdefault('updates_by_group',{})[group]=counters.setdefault('updates_by_group',{}).get(group,0)+1
        update=counters['conversation_updates']
        if update%16==0:
            atomic_json(out/'PROGRESS.json',{**counters,'status':'training','last_update':stats,'live_promoted':False})
            print(json.dumps({'step':counters['completed_steps'],'updates':update,'loss':stats['loss'],'phase':counters['phase']}),flush=True)
            if (update==16 or update%128==0) and needs_training(counters):publisher.progress({**counters,'status':'training','last_update':stats},args.release)
        if update%(LEARNING['evaluation_updates'] if LEARNING else 1024)==0:
            select_candidate('periodic-development')
        # Memorization cannot lower LR. Three fixed phases depend on actual compute,
        # preserved across jobs, rather than correctness on oversampled examples.
        phase=recipe_phase(counters['elapsed_training_seconds'])
        if phase!=counters['phase']:
            counters['phase']=phase;recipe=SPEC['recipes'][phase]
            apply_learning_rate()
            counters['diagnostic_phase_changes'].append({'step':counters['completed_steps'],'phase':phase,'reason':'predeclared compute schedule'})
        if counters['elapsed_training_seconds']-last_save_seconds>=3600:
            save();last_save_seconds=counters['elapsed_training_seconds']
        if update%16==0 and time.monotonic()-started>=args.seconds and needs_training(counters):
            if LEARNING:
                # A job boundary is not evidence of a failed learning trial.
                # Save the unselected candidate and its separate champion ref.
                save()
            else:select_candidate('chunk-boundary',force=True)
            next_action={'release':publisher.next_release};break
    if legacy_run.state_digest(reference.model)!=reference_digest or any(p.grad is not None for p in reference.model.parameters()):
        raise ValueError('frozen reference changed')
    if next_action is None:
        select_candidate('finite-budget-boundary',force=True)
        metrics=legacy_run.common_metrics(runtime)
        original=legacy_run.common_metrics(reference)
        from generalist_lm.research_cycle import _research_eligible
        from generalist_lm.phase5_diagnostics import degeneration_gate
        counters['native_retention_gate']=list(_research_eligible(original['domains'],metrics['domains'],minimum_loss_gain=.01,max_domain_regression=.03))
        counters['native_degeneration_gate']=list(degeneration_gate(original['language'],metrics['language'],max_repetition_regression=.04,max_entropy_collapse_fraction=.70))
        counters['native_metrics']=metrics
        # Run once after training; never expose this set to sampling, LR or selection.
        sealed_samples=[]
        for row in sealed_transfer:
            response=runtime.chat(row['messages'][:-1],temperature=0,max_new_tokens=64)
            sealed_samples.append({'messages':row['messages'][:-1],'expected':row['expected'],'response':response,'correct':exact_correct(response,row['expected']),'family':row['family'],'language':row['language']})
        counters['sealed_transfer']={'correct':sum(r['correct'] for r in sealed_samples),'probes':len(sealed_samples),'samples':sealed_samples,'scope':'final only; excluded from selection and adaptation'}
        counters['human_final']={'samples':[{'messages':r['messages'][:-1],'reference':r['expected'],'response':runtime.chat(r['messages'][:-1],temperature=0,max_new_tokens=128),'language':r['language']} for r in external_final],'scope':'Final-only unseen human conversations; qualitative review required; excluded from selection.'}
        counters['development_scope']='Repeated development checks; not an untouched final test.' 
        counters['stop_reason']=('stagnation/evaluation limit reached; selected champion retained'
                                 if counters.get('learning_search',{}).get('stopped')
                                 else 'finite selective-transfer experiment completed; independent quality gates retained')
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
