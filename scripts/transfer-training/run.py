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
FAST_HEAD = bool(LEARNING and LEARNING.get('training_mode')=='cached_head')
FAST_PREFIX = bool(LEARNING and LEARNING.get('training_mode')=='cached_prefix')
CACHED_TRAINING = FAST_HEAD or FAST_PREFIX
RETENTION = LEARNING.get('knowledge_retention') if LEARNING else None

def empty_training_registry(fingerprint):
    return {'schema':1,'fingerprint':fingerprint,'source_fingerprints':[fingerprint],'entries':{}}

def normalized_context_row(row):
    """Keep exact TRAIN inputs/answers; normalize only audited provenance metadata."""
    import copy
    result=copy.deepcopy(row)
    result['provenance']='deterministic audited matched TRAIN context; original TRAIN template and literal TRAIN value'
    return result

def validate_training_registry(registry,original_rows,value_rows,forbidden,*,fingerprint,
                               blocked=(),reserved_values=(),protected_histories=()):
    """Reconstruct every materialized row from pinned TRAIN sources before trusting it."""
    from success_bank import row_key,catalog
    from context_acquisition import matched_context_pair,context_value_pairs
    if (not isinstance(registry,dict) or set(registry)!={'schema','fingerprint','source_fingerprints','entries'}
            or registry.get('schema')!=1 or registry.get('fingerprint')!=fingerprint
            or not isinstance(registry.get('entries'),dict)):
        raise ValueError('invalid materialized TRAIN registry')
    ancestry=registry.get('source_fingerprints')
    if (not isinstance(ancestry,list) or not ancestry or fingerprint not in ancestry
            or any(not isinstance(x,str) or not x for x in ancestry) or len(set(ancestry))!=len(ancestry)):
        raise ValueError('invalid materialized TRAIN registry ancestry')
    sources={row_key(row):row for row in original_rows}
    groups={rows[0]['acquisition_group']:rows for pairs in context_value_pairs(value_rows).values() for rows in pairs}
    rows=[]
    for key,entry in registry['entries'].items():
        if not isinstance(entry,dict) or set(entry)!={'row','source_row_key','value_group','counterfactual_index'}:
            raise ValueError('invalid materialized TRAIN registration proof')
        source=sources.get(entry['source_row_key']);values=groups.get(entry['value_group'])
        if source is None or values is None or type(entry['counterfactual_index']) is not int:
            raise ValueError('materialized TRAIN row lacks its original source/value pool')
        pair=matched_context_pair(source,values,blocked=blocked,reserved_values=reserved_values,
                                  protected_histories=protected_histories)
        matches=[normalized_context_row(row) for row in pair
                 if row['counterfactual_index']==entry['counterfactual_index']]
        if len(matches)!=1 or entry['row']!=matches[0] or key!=row_key(matches[0]):
            raise ValueError('materialized TRAIN row differs from its reconstructed source')
        rows.append(matches[0])
    if rows and catalog(rows,forbidden)!={row_key(row):row for row in rows}:
        raise ValueError('materialized TRAIN registry contains a protected history')
    return rows

def register_context_pair(registry,source,pair):
    from success_bank import row_key
    rows=[]
    for original in pair:
        row=normalized_context_row(original);key=row_key(row)
        proof={'row':row,'source_row_key':row_key(source),'value_group':row['acquisition_group'],
               'counterfactual_index':row['counterfactual_index']}
        previous=registry['entries'].get(key)
        if previous is not None and previous!=proof:
            raise ValueError('materialized TRAIN row has conflicting registration proof')
        registry['entries'][key]=proof;rows.append(row)
    return rows

def consolidation_allowed(counters,config):
    """Reserve at least half the finite salvage quota until exploration stops."""
    validate_budget(counters)
    quota=config['consolidation_updates']
    spent=counters.get('retention_consolidation_updates',0)
    if type(spent) is not int or not 0<=spent<=quota:
        raise ValueError('invalid persisted retention consolidation quota')
    # Early rollbacks share the first half. The final rejected candidate can
    # still donate lessons after the ordinary search has permanently stopped.
    limit=quota if counters.get('learning_search',{}).get('stopped',False) else quota//2
    return (counters['elapsed_training_seconds']<SPEC['maximum_training_seconds']
            and counters['completed_steps']<SPEC['max_steps']
            and spent<limit)

def transfer_retention_statistics(stats,options):
    """Keep recipe diagnostics even when a nonfinite proposal was rolled back."""
    return {'retention_weight':stats.get('retention_weight',options.get('retention_weight',1.)),
            'champion_kl_weight':stats.get('champion_kl_weight',options.get('champion_kl_weight',1.)),
            'retention_examples':stats.get('retention_examples',len(options.get('memories',[])))}

def training_strategy(model):
    if FAST_PREFIX:
        from prefix_tuning import prefix_summary
        return prefix_summary(model)
    from fast_head import head_summary
    return head_summary(model)

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
            and (not RETENTION or counters['completed_steps']<SPEC['max_steps'])
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

def success_slot(update,sample_group,has_lessons,*,schedule=None):
    from continual_learning import nonstarving_replay_slots
    if schedule is None:
        schedule=LEARNING['sampling_schedule'] if LEARNING else SAMPLING_SCHEDULES['balanced']
    slots=LEARNING.get('success_transfer_slots',(2,10)) if LEARNING else (2,10)
    return bool(has_lessons and schedule[update%16]==sample_group
                and update%16 in nonstarving_replay_slots(schedule,slots))

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
    if CACHED_TRAINING:
        from fast_head import configure_head,install_feature_cache,head_summary
        install_feature_cache(reference.model,cache_bytes=64*1024*1024)
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
    acquisition_blocked=blocked|all_heldout|{norm(m['content']) for r in authored for m in r['messages']}
    reviewed_only=bool(LEARNING and LEARNING.get('human_training_scope')=='reviewed_only')
    corpus=load_corpus(human_directory,SPEC['human_corpus']['pins'],acquisition_blocked|previous_human_text,
                       training_blocked=acquisition_blocked if reviewed_only else None)
    external_development=[]
    for lang in ('it','en'):
        external_development += [r for r in corpus['development'] if r['language']==lang][:16]
    external_final=[]
    for lang in ('it','en'):
        external_final += [r for r in corpus['final'] if r['language']==lang][:8]
    corpus_heldout_text={norm(m['content']) for split in ('development','final') for r in corpus[split] for m in r['messages']}
    human_rows=[r for r in human_rows if not any(norm(m['content']) in all_heldout|corpus_heldout_text for m in r['messages'])]
    for r in human_rows:r['language']=r['provenance']['language']
    human_rows=corpus['train'] if reviewed_only else human_rows+corpus['train']
    context_config=LEARNING.get('context_acquisition') if LEARNING else None
    context_values=[]
    if context_config is not None:
        from context_acquisition import (build_context_acquisition, context_value_pairs,
                                         context_skeletons, acquisition_audit,
                                         matched_context_pair, paired_batch)
        from success_bank import prompt_key
        context_excluded_text=blocked|all_heldout|corpus_heldout_text
        context_reserved={r['expected'] for r in heldout+sealed_transfer}
        context_reserved.update(r['expected'] for split in ('development','final') for r in corpus[split])
        old_validation=[json.loads(line) for file in ['development.jsonl','test.jsonl']
                        for line in (data/'data'/file).read_text().splitlines()]
        context_protected_histories={prompt_key(r) for r in heldout+sealed_transfer
                                     +corpus['development']+corpus['final']+old_validation}
        context_values=build_context_acquisition(context_excluded_text,context_reserved,
                            seed=context_config['seed'],max_pairs=context_config['max_pairs'])
        context_value_index=context_value_pairs(context_values)
        context_skeleton_index=context_skeletons(context_train)
        expected_context_groups={(lang,'context-'+task) for lang in ('it','en')
                                 for task in ('name','location','topic','correction')}
        if set(context_value_index)!=expected_context_groups or set(context_skeleton_index)!=expected_context_groups:
            raise ValueError('incomplete matched TRAIN context acquisition coverage')
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
    def factory(model):
        if FAST_PREFIX:
            from prefix_tuning import configure_prefix
            configure_prefix(model,cache_bytes=LEARNING['feature_cache_mebibytes']*1024*1024)
        if FAST_HEAD:
            configure_head(model,cache_bytes=LEARNING['feature_cache_mebibytes']*1024*1024)
            model._airi_projection_rank_requested=LEARNING.get('protected_head_rank',0)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=recipe['learning_rate'],weight_decay=.01)
        if FAST_HEAD and LEARNING.get('protected_head_rank',0):
            from fast_head import attach_projection
            attach_projection(optimizer,model)
        return optimizer
    source_fingerprint=transport.SOURCE_FINGERPRINTS.get(args.release,SPEC['fingerprint'])
    runtime,opt,payload=restore_session(checkpoint,factory,fingerprint=source_fingerprint,
        load_optimizer=not (CACHED_TRAINING and args.release==transport.SEED_TAG),
        require_retention=bool(RETENTION and source_fingerprint==SPEC['fingerprint']))
    if LEARNING:cache_tokenizer(runtime.tokenizer)
    rng,anchor_rng=random.Random(),random.Random()
    rng.setstate(payload['sampler_rng']);anchor_rng.setstate(payload['anchor_rng'])
    counters=payload['counters']
    source_digest=sha256_file(checkpoint/'model/model.pt')
    from success_bank import (catalog, empty_bank, validate_bank, probe_rows,row_key,
                              summary as bank_summary, retains_correct_answers, migrate_bank)
    forbidden=heldout+sealed_transfer+external_development+external_final
    training_registry=None;retention_memory=None
    if RETENTION:
        # Session verified BOTH files against the immutable manifest before any
        # restored row can grant donor/certificate eligibility.
        training_registry=payload.get('training_registry',empty_training_registry(source_fingerprint))
        registered=validate_training_registry(training_registry,context_train,context_values,forbidden,
            fingerprint=source_fingerprint,
            blocked=context_excluded_text if context_config else (),
            reserved_values=context_reserved if context_config else (),
            protected_histories=context_protected_histories if context_config else ())
        if source_fingerprint!=SPEC['fingerprint']:
            training_registry['fingerprint']=SPEC['fingerprint']
            training_registry['source_fingerprints']=sorted(set(training_registry['source_fingerprints'])|{SPEC['fingerprint']})
        retention_memory=payload.get('retention_memory')
    else:registered=[]
    eligible = catalog(authored+registered,forbidden)
    if args.learning_smoke_only or (CACHED_TRAINING and args.success_smoke_only):
        if RETENTION:
            if args.release!=transport.SEED_TAG or retention_memory is not None:
                raise ValueError('retention preflight requires the exact migration seed; resumed ledgers must use the full runner')
            from retention_smoke import real_retention_smoke
            real_retention_smoke(runtime,checkpoint,reference,eligible,human_rows,
                                 encode,anchors,evaluate,legacy_run.state_digest,
                                 RETENTION,LEARNING,SPEC['fingerprint'])
            return
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
    curriculum_rows=authored+heldout+sealed_transfer+human_rows+external_development+external_final
    if context_config is None:
        curriculum_sha=digest(curriculum_rows)
    else:
        from context_acquisition import acquisition_curriculum_digest
        curriculum_sha=acquisition_curriculum_digest(curriculum_rows,context_config,context_values)
    if args.release==transport.SEED_TAG:
        # Preserve native weights and RNG. A cached-training migration explicitly
        # starts fresh Adam for its exact scope; later resumes restore that Adam.
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
    if CACHED_TRAINING:counters['training_strategy']=training_strategy(runtime.model)
    counters['human_acquisition_scope']='reviewed corpus only' if reviewed_only else 'legacy plus reviewed corpus'
    if context_config is not None:
        counters['context_acquisition']={**acquisition_audit(context_values),
                       'configuration':dict(context_config),
                       'template_policy':'Only values are transplanted into original TRAIN histories; old retention remains separate'}
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
    dataset=out/'AUTHORED_CURRICULUM.json'
    dataset_payload={'train':authored,'development':heldout,'sealed_transfer':sealed_transfer,'sha256':curriculum_sha}
    if context_config is not None:
        dataset_payload.update(context_acquisition=dict(context_config),
                               context_acquisition_values=context_values)
    atomic_json(dataset,dataset_payload)
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
        if CACHED_TRAINING:install_feature_cache(protected.model,cache_bytes=64*1024*1024)
        if LEARNING:cache_tokenizer(protected.tokenizer)
        return protected
    protected_champion = load_protected_champion()
    if LEARNING:
        from continual_learning import retention_rows,verified_memories
        if RETENTION:
            from retention_memory import (empty_memory,validate_memory,migrate_memory,refresh,
                                           make_certifier,sample_rows,observe,summary as memory_summary)
            from generalist_lm.tokenizer import EOS
            champion_digest=legacy_run.state_digest(protected_champion.model)
            certifier=make_certifier(protected_champion.model,encode,EOS)
            if retention_memory is None:
                if source_fingerprint==SPEC['fingerprint']:
                    raise ValueError('resumed retention recipe cannot recreate a missing certificate ledger')
                old_rows=verified_memories(protected_champion,retention_rows(list(eligible.values())),encode)
                if len(old_rows)>RETENTION['capacity']:
                    raise ValueError('retention capacity cannot hold all previous certified TRAIN answers')
                retention_memory=empty_memory(SPEC['fingerprint'],RETENTION['capacity'])
                retention_memory=refresh(champion_digest,catalog(old_rows,forbidden),certifier,
                                         retention_memory,probe_limit=len(old_rows))
                retention_memory=refresh(champion_digest,eligible,certifier,retention_memory,
                                         probe_limit=RETENTION['refresh_probe_limit'])
            elif source_fingerprint!=SPEC['fingerprint']:
                retention_memory=migrate_memory(retention_memory,eligible,source_fingerprint,
                                                 SPEC['fingerprint'],champion_digest)
                if retention_memory['capacity']!=RETENTION['capacity']:
                    raise ValueError('explicit retention migration must preserve certificate capacity')
                retention_memory=refresh(champion_digest,eligible,certifier,retention_memory,
                                         probe_limit=RETENTION['refresh_probe_limit'])
            else:
                validate_memory(retention_memory,eligible,SPEC['fingerprint'],champion_digest)
                if retention_memory['capacity']!=RETENTION['capacity']:
                    raise ValueError('retention capacity differs from resumed recipe')
            memory_rows=[entry['row'] for entry in retention_memory['entries'].values()]
            if {r['language'] for r in memory_rows}!={'it','en'}:
                raise ValueError('persistent retention requires certified TRAIN answers in both languages')
            counters['retention_memory']=memory_summary(retention_memory)
            counters.setdefault('retention_transaction',{'schema':1,'lr_scale':1.,'attempted_updates':0,
                'committed_updates':0,'rejected_updates':0})
            counters.setdefault('retention_consolidation_updates',0)
        else:
            memory_rows=verified_memories(protected_champion,retention_rows(list(eligible.values())),encode)
            counters['retention_memory']={'rows':len(memory_rows),'sha256':digest(memory_rows),
                                         'scope':'eligible authored training only; no evaluation answers'}
        memory_by={lang:[encode(r) for r in memory_rows if r['language']==lang] for lang in ('it','en')}
    def protect_current_head():
        if FAST_HEAD and LEARNING.get('protected_head_rank',0):
            from fast_head import configure_projection
            configure_projection(runtime.model,protected_champion.model,memory_rows,encode,
                                 rank=LEARNING['protected_head_rank'])
    protect_current_head()
    def transfer_options(prepared=None):
        if prepared is not None:return prepared
        if not LEARNING:return {}
        if RETENTION:
            from continual_learning import profile_index
            profile=LEARNING['profiles'][profile_index(counters['learning_search']['consecutive_rejections'],len(LEARNING['profiles']))]
            return dict(memories=[encode(row) for row in sample_rows(retention_memory,eligible,anchor_rng,
                            per_language=LEARNING.get('retention_examples_per_language',1))],
                        retention_weight=profile['retention_weight'],champion_kl_weight=profile['champion_kl_weight'],
                        prompt_kl_weight=RETENTION['prompt_kl_weight'])
        from continual_learning import transfer_replay_options
        return transfer_replay_options(LEARNING,counters['learning_search'],memory_by,anchor_rng)
    def protected_update(propose):
        nonlocal retention_memory
        if not RETENTION:return propose()
        from retention_guard import make_retention_guard
        from retention_transaction import transactional_update
        rows=sample_rows(retention_memory,eligible,anchor_rng,
                         per_language=RETENTION['guard_examples_per_language'])
        guard=make_retention_guard(runtime.model,protected_champion.model,[encode(row) for row in rows],
                   max_prompt_kl=RETENTION['max_prompt_kl'],max_nll_regression=RETENTION['max_nll_regression'])
        base_lrs=[group['lr'] for group in opt.param_groups]
        proposed=[]
        def record_proposal():
            scales=[group['lr']/base for group,base in zip(opt.param_groups,base_lrs)]
            if not scales or any(abs(value-scales[0])>1e-12 for value in scales):
                raise ValueError('retention transaction must scale all optimizer groups equally')
            stats=propose(scales[0]);proposed.append(stats);return stats
        result=transactional_update(runtime.model,opt,record_proposal,guard,
                    state=counters['retention_transaction'],scales=RETENTION['retry_scales'],
                    sampler_rngs=(rng,anchor_rng))
        observations=result['guard']['observations'] if result['accepted'] else guard.before_observations
        if len(observations)!=len(rows):raise ValueError('incomplete post-transaction retention evidence')
        evidence={row_key(row):value for row,value in zip(rows,observations)}
        retention_memory=observe(retention_memory,eligible,
                                 lambda row:evidence[row_key(row)],keys=list(evidence))
        stats=dict(result['stats'])
        # All physical retry work belongs to attempted-token/compute accounting;
        # only the accepted optimizer trajectory counts as committed tokens.
        for name in ('supervised_tokens','reference_tokens'):
            committed=int(stats.get(name,0)) if result['accepted'] else 0
            counters['committed_'+name]=counters.get('committed_'+name,0)+committed
            stats[name]=sum(int(value.get(name,0)) for value in proposed)
        if any('prompt_reference_tokens' in value for value in proposed):
            counters['committed_prompt_reference_tokens']=counters.get('committed_prompt_reference_tokens',0)+(int(stats.get('prompt_reference_tokens',0)) if result['accepted'] else 0)
            stats['prompt_reference_tokens']=sum(int(value.get('prompt_reference_tokens',0)) for value in proposed)
            counters['attempted_prompt_reference_tokens']=counters.get('attempted_prompt_reference_tokens',0)+stats['prompt_reference_tokens']
        stats.setdefault('loss',0.)
        stats.update(committed=result['accepted'],physical_attempts=result['attempts'],
                     retention_guard=result['guard'],retention_lr_scale=result['scale'])
        counters['last_retention_transaction']={k:stats[k] for k in
                       ('committed','physical_attempts','retention_guard','retention_lr_scale')}
        return stats
    def preserves_certificates():
        nonlocal retention_memory
        if not RETENTION:return True
        retention_memory=observe(retention_memory,eligible,make_certifier(runtime.model,encode,EOS))
        lost=[key for key,entry in retention_memory['entries'].items() if not entry['observation']['correct']]
        counters['retention_last_adoption_check']={'protected':len(retention_memory['entries']),
                                                  'lost':len(lost),'accepted':not lost}
        return not lost
    def refresh_champion_memory():
        nonlocal retention_memory,memory_rows,memory_by
        retention_memory=refresh(legacy_run.state_digest(runtime.model),eligible,
                    make_certifier(runtime.model,encode,EOS),retention_memory,
                    probe_limit=RETENTION['refresh_probe_limit'])
        memory_rows=[entry['row'] for entry in retention_memory['entries'].values()]
        memory_by={lang:[encode(row) for row in memory_rows if row['language']==lang] for lang in ('it','en')}
        counters['retention_memory']=memory_summary(retention_memory)
    def save(*,champion=False):
        nonlocal best_path,best_fingerprint
        path=out/f'checkpoint-{counters["completed_steps"]:05d}'
        counters['success_bank']=bank_summary(bank)
        if RETENTION:
            if champion and not preserves_certificates():
                raise ValueError('cannot publish a champion that lost a certified TRAIN answer')
            validate_memory(retention_memory,eligible,SPEC['fingerprint'],
                            legacy_run.state_digest(runtime.model) if champion else None)
            counters['retention_memory']=memory_summary(retention_memory)
        if CACHED_TRAINING:counters['training_strategy']=training_strategy(runtime.model)
        if path.exists():shutil.rmtree(path)
        save_session(runtime,opt,rng,anchor_rng,counters,path,fingerprint=SPEC['fingerprint'],success_bank=bank,
                     retention_memory=retention_memory,training_registry=training_registry)
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
        retained=preserves_certificates()
        accepted=quality_improved(champion,assessment) and retains_correct_answers(champion,assessment) and retained
        rollback_candidate=False
        if LEARNING:
            from continual_learning import rejection_reasons, try_interpolations, record_decision
            search=counters['learning_search']
            report={'step':counters['completed_steps'],'learning_rate':counters['effective_learning_rate'],
                    'reasons':rejection_reasons(champion,assessment),'interpolation':None}
            if not retained:report['reasons'].append('previously certified TRAIN answer lost')
            if not accepted:
                blended=try_interpolations(runtime.model,protected_champion.model,
                    evaluate=lambda:evaluate(runtime),
                    acceptable=lambda value:quality_improved(champion,value) and retains_correct_answers(champion,value) and preserves_certificates(),
                    alphas=LEARNING['interpolation_alphas'])
                report['interpolation']=blended['alpha'];report['interpolation_probes']=blended['probes']
                if blended['alpha'] is not None:
                    assessment=blended['assessment'];accepted=True
                    # A merged parameter path has no exact Adam trajectory.
                    opt=factory(runtime.model);apply_learning_rate()
                elif RETENTION:
                    # Rejected interpolation callbacks observed temporary
                    # weights. Restore drift priorities to the actual candidate.
                    preserves_certificates()
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
            if RETENTION:refresh_champion_memory()
            adopt(counters['morph'],legacy_run.state_digest(runtime.model),
                  step=counters['completed_steps'],reason='independently verified candidate')
            counters['champion']=assessment;action='accepted'
        elif (rollback_candidate if LEARNING else force or recover_best(counters['stagnant_rounds'])):
            runtime,opt,_=restore_champion(best_path,factory,rng,anchor_rng,fingerprint=best_fingerprint,
                         require_retention=bool(RETENTION and best_fingerprint==SPEC['fingerprint']))
            if LEARNING:opt=factory(runtime.model)
            protect_current_head()
            counters['rollbacks']+=1;counters['stagnant_rounds']=0;action='rolled_back'
            apply_learning_rate()
            def restore_transaction():
                nonlocal runtime,opt
                runtime,opt,_=restore_champion(best_path,factory,rng,anchor_rng,fingerprint=best_fingerprint,
                         require_retention=bool(RETENTION and best_fingerprint==SPEC['fingerprint']))
                if LEARNING:opt=factory(runtime.model)
                protect_current_head()
                apply_learning_rate()
            def learn_lesson(entry):
                from success_transfer import distill_step
                tick=time.monotonic()
                anchor=rng.choice(memory_by[entry['row']['language']]) if LEARNING else rng.choice(anchors)
                options=transfer_options()
                stats=protected_update(lambda scale=1.:distill_step(runtime.model,protected_champion.model,opt,entry,encode(entry['row']),
                                   anchor,lr_multiplier=SPEC['success_transfer']['lr_multiplier'],
                                   lr_cap=SPEC['success_transfer']['lr_cap']*scale,**transfer_options(options)))
                counters['elapsed_training_seconds']+=time.monotonic()-tick
                counters['conversation_updates']+=1;counters['completed_steps']+=1
                counters['supervised_tokens']+=stats['supervised_tokens'];counters['reference_tokens']+=stats['reference_tokens']
                bank['transfer_updates']+=1
                if RETENTION:counters['retention_consolidation_updates']+=1
                counters['last_transfer_retention']=transfer_retention_statistics(stats,options)
                group='morph-consolidation-'+entry['row']['language']
                counters.setdefault('updates_by_group',{})[group]=counters.setdefault('updates_by_group',{}).get(group,0)+1
            result=consolidate(counters['morph'],bank,champion,train=learn_lesson,
                               evaluate=lambda:evaluate(runtime),restore=restore_transaction,
                               can_train=lambda:consolidation_allowed(counters,RETENTION) if RETENTION else needs_training(counters),
                               can_adopt=lambda a,b:quality_improved(a,b) and retains_correct_answers(a,b) and preserves_certificates(),
                               step=lambda:counters['completed_steps'],
                               max_updates=SPEC['morph_model']['updates_per_skill'])
            counters['last_consolidation']={k:v for k,v in result.items() if k!='assessment'}
            if LEARNING:
                # Consolidation is a separate transaction, not part of the next
                # candidate's learning horizon.
                counters['learning_search']['trial_start_update']=counters['conversation_updates']
            if result['action']=='skill_accepted':
                if RETENTION:refresh_champion_memory()
                adopt(counters['morph'],legacy_run.state_digest(runtime.model),
                      step=counters['completed_steps'],reason='selective consolidation: '+result['skill'])
                counters['champion']=result['assessment'];action='consolidated'
                if LEARNING:
                    search=counters['learning_search']
                    if not RETENTION:
                        search['consecutive_rejections']=0
                        search['stopped']=search['evaluations']>=LEARNING['maximum_evaluations']
                    search['accepted_updates']+=1
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
        counters['last_development_update']=counters['conversation_updates']
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
                if LEARNING and not RETENTION:
                    memory_rows=verified_memories(protected_champion,retention_rows(list(eligible.values())),encode)
                    memory_by={lang:[encode(r) for r in memory_rows if r['language']==lang] for lang in ('it','en')}
                    counters['retention_memory']={'rows':len(memory_rows),'sha256':digest(memory_rows),
                                                 'scope':'certified correct eligible training answers; no evaluation answers'}
                    protect_current_head()
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
        schedule=(LEARNING['sampling_schedule'] if LEARNING
                  else SAMPLING_SCHEDULES[counters['training_focus']])
        sample_group=schedule[update%16]
        if sample_group=='human-dialogue':
            if LEARNING:
                from continual_learning import acquisition_language
                lang=acquisition_language(LEARNING,human_by,rng,lang)
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
            if context_config is None:
                candidates=[r for r in extra_by[lang] if r['family'].startswith('context-') and r['stage']<=counters['curriculum_stage']]
                family=rng.choice(sorted({r['family'] for r in candidates}))
                batch=encode(rng.choice([r for r in candidates if r['family']==family]))
            else:
                families=sorted(family for (language,family),groups in context_skeleton_index.items()
                                if language==lang and any(rows[0]['stage']<=counters['curriculum_stage'] for rows in groups))
                family=rng.choice(families)
                skeletons=[rows for rows in context_skeleton_index[(lang,family)]
                           if rows[0]['stage']<=counters['curriculum_stage']]
                source=rng.choice(skeletons)[0]
                values=rng.choice(context_value_index[(lang,family)])
                pair=matched_context_pair(source,values,blocked=context_excluded_text,
                         reserved_values=context_reserved,protected_histories=context_protected_histories)
                if RETENTION:
                    pair=register_context_pair(training_registry,source,pair)
                    registered=catalog(pair,forbidden)
                    if len(registered)!=len(pair):raise ValueError('materialized context pair failed TRAIN audit')
                    eligible.update(registered)
                batch=paired_batch([encode(row) for row in pair])
            group='verified-context-'+lang
        else:
            group=['language-it','language-en','domains'][(update//16)%3]
            batch=rng.choice(replay[group])
        lesson=None
        # Preserve human/native replay and at least one acquisition per group;
        # only surplus requested slots replay archived native improvements.
        if success_slot(update,sample_group,bank['entries'],schedule=schedule):
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
                from continual_learning import continual_step,profile_index,retention_memories
                profile=LEARNING['profiles'][profile_index(counters['learning_search']['consecutive_rejections'],len(LEARNING['profiles']))]
                memories=([encode(row) for row in sample_rows(retention_memory,eligible,anchor_rng,
                            per_language=LEARNING.get('retention_examples_per_language',1))] if RETENTION
                          else retention_memories(LEARNING,memory_by,anchor_rng))
                options={'prompt_kl_weight':RETENTION['prompt_kl_weight']} if RETENTION else {}
                stats=protected_update(lambda scale=1.:continual_step(runtime.model,protected_champion.model,reference.model,opt,batch,memories,anchor,
                                     retention_weight=profile['retention_weight'],
                                     champion_kl_weight=profile['champion_kl_weight'],**options))
            else:stats=train_step(runtime.model,reference.model,opt,*batch,*anchor,kl_weight=recipe['kl_weight'])
        else:
            from success_transfer import distill_step
            transfer_anchor=rng.choice(memory_by[lesson['row']['language']]) if LEARNING else anchor
            options=transfer_options()
            stats=protected_update(lambda scale=1.:distill_step(runtime.model,protected_champion.model,opt,lesson,batch,
                               transfer_anchor,
                               lr_multiplier=SPEC['success_transfer']['lr_multiplier'],
                               lr_cap=SPEC['success_transfer']['lr_cap']*scale,**transfer_options(options)))
            bank['transfer_updates']+=1
            counters['last_transfer_retention']=transfer_retention_statistics(stats,options)
            counters['success_bank']=bank_summary(bank)
        counters['elapsed_training_seconds']+=time.monotonic()-tick
        counters['conversation_updates']+=1;counters['completed_steps']+=1
        counters['supervised_tokens']+=stats['supervised_tokens'];counters['reference_tokens']+=stats['reference_tokens']
        counters.setdefault('updates_by_group',{})[group]=counters.setdefault('updates_by_group',{}).get(group,0)+1
        update=counters['conversation_updates']
        if update%16==0:
            if CACHED_TRAINING:counters['training_strategy']=training_strategy(runtime.model)
            atomic_json(out/'PROGRESS.json',{**counters,'status':'training','last_update':stats,'live_promoted':False})
            print(json.dumps({'step':counters['completed_steps'],'updates':update,'loss':stats['loss'],'phase':counters['phase']}),flush=True)
            if (update==16 or update%128==0) and needs_training(counters):publisher.progress({**counters,'status':'training','last_update':stats},args.release)
        from continual_learning import evaluation_due
        if (evaluation_due(counters,LEARNING['evaluation_updates']) if LEARNING else update%1024==0):
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
        if FAST_PREFIX:
            # Original final metrics temporarily use the original 128-token
            # context. Run the native forward while that configuration is
            # active; cached training resumes only after it is restored.
            from prefix_tuning import native_evaluation
            with native_evaluation(runtime.model):
                metrics=legacy_run.common_metrics(runtime)
        else:
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
