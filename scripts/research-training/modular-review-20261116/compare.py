"""Preregistered fresh-value review against the SAME small frozen native model.

This is a bounded system ablation, not an equal-compute Transformer-class test.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
SEED=20261116230519
GOLD_OPERATIONS={
    'parcel_copy':'copy','signal_reverse':'reverse','card_upper':'uppercase',
    'plate_lower':'lowercase','delta_sum':'add','sigma_product':'multiply',
    'auto_v1_remaining':'subtract','auto_v1_contact':'copy',
    'auto_v1_customer_reference':'copy','auto_v1_order_reference':'copy',
    'auto_v1_email_alias':'copy','auto_v1_tracking_reverse':'reverse',
}
PARAPHRASES={
    'parcel_copy':'Mi ricordi il codice del pacco?',
    'signal_reverse':'Restituisci il segnale acquisito al contrario.',
    'card_upper':'Mostra il testo della tessera in maiuscolo.',
    'plate_lower':'Mostra il testo della targa in minuscolo.',
    'delta_sum':'Quanto fa la somma dei due valori delta?',
    'sigma_product':'Quanto fa il prodotto dei due interi sigma?',
}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def independent_answer(operation,value):
    if operation=='copy':return value
    if operation=='reverse':return ''.join(reversed(value))
    if operation=='uppercase':return value.upper()
    if operation=='lowercase':return value.lower()
    a,b=value
    if operation=='add':return str(a+b)
    if operation=='subtract':return str(a-b)
    if operation=='multiply':return str(a*b)
    raise ValueError('unknown authored operation')


def cases_for(entries):
    """Task semantics are authored above; compiler predictions never supply gold."""
    rng=random.Random(SEED);known=[];paraphrased=[];negative=[]
    for index,entry in enumerate(entries):
        task=entry['lesson_id'];operation=GOLD_OPERATIONS[task]
        source=entry['program']['source']
        if operation in ('add','subtract','multiply'):
            value=(rng.randint(1000001,9000000),-rng.randint(1000001,9000000))
            body=str(value[0])+source['separator']+str(value[1])
        else:
            value=f'ÅbßÉz-review-{rng.randrange(10**9):09d}-Qx{index}';body=value
        messages=[{'role':'user','content':source['prefix']+body+source['suffix']},
                  {'role':'user','content':entry['program']['query']}]
        known.append({'id':f'known-{task}','lesson_id':task,'condition':'known_template',
                      'messages':messages,'expected':independent_answer(operation,value)})
        if task in PARAPHRASES:
            paraphrased.append({'id':f'wording-{task}','lesson_id':task,'condition':'new_wording',
                'messages':[dict(messages[0]),{'role':'user','content':PARAPHRASES[task]}],
                'expected':independent_answer(operation,value)})
            negative.append({'id':f'quoted-{task}','lesson_id':task,'condition':'expert_abstention',
                'messages':[{'role':'user','content':'"'+messages[0]['content']+'"'},dict(messages[1])],
                'expected':None})
    assert len(known)==12 and len(paraphrased)==len(negative)==6
    return known+paraphrased+negative


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--record',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('choose an absent output; no endpoint retry')
    args.output.mkdir(parents=True);frozen=args.output/'sources';frozen.mkdir()
    for path in (ROOT/'scripts/assistant-runtime').glob('*.py'):shutil.copy2(path,frozen/path.name)
    (frozen/'skills').mkdir()
    shutil.copy2(ROOT/'scripts/assistant-runtime/skills/stable_contracts.json.gz',frozen/'skills/stable_contracts.json.gz')
    shutil.copy2(args.bank,args.output/'bank.json')
    shutil.copy2(__file__,args.output/'compare-as-run.py')
    sys.path.insert(0,str(frozen))
    from assistant import AiriAssistant
    from context_memory import resolve
    from skill_bank import SkillBank,_route
    from skill_validation import StableGate
    bank=SkillBank(args.output/'bank.json');state,_=bank._read()
    assert state['revision']==12
    cases=cases_for(state['entries'])
    # Endpoint inputs never overlap stored TRAIN/qualification histories or values.
    stored={json.dumps(row['messages'],sort_keys=True,ensure_ascii=False)
            for entry in state['entries'] for row in entry['train']+entry['qualification']}
    assert all(json.dumps(row['messages'],sort_keys=True,ensure_ascii=False) not in stored for row in cases)
    save(args.output/'CASES.json',cases)
    record=json.loads(args.record.read_text())
    prereg={'created_at_utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,
        'bank_sha256':sha(args.output/'bank.json'),'cases_sha256':sha(args.output/'CASES.json'),
        'compiler_sha256':sha(frozen/'skill_programs.py'),
        'source_sha256':{str(path.relative_to(args.output)):sha(path) for path in frozen.rglob('*') if path.is_file()},
        'comparison_source_sha256':sha(__file__),'model_record_sha256':sha(args.record),
        'model_revision':record['revision'],'weights_digest':record['weights_digest'],
        'policy':'24 fixed cases:12 known templates,6 authored paraphrases,6 quoted sources; at most64 greedy tokens; no candidate changes',
        'fewshot':'same stored8TRAIN examples for that expert; check token length, no truncation',
        'negative_scoring':'expert abstention only; no fabricated native abstention accuracy'}
    save(args.output/'PREREGISTRATION.json',prereg)
    gate=StableGate();assert gate(lambda m:_route(state['entries'],m))
    contracts=0
    for entry in state['entries']:
        for row in entry['train']+entry['qualification']:
            result=_route(state['entries'],row['messages'])
            assert (None if result is None else result['answer'])==row['answer']
            contracts+=1
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0,str(ROOT/'scripts/transfer-training'))
    from chat_current import validate_selected_model
    from generalist_lm.runtime import GeneralistRuntime
    model_root=validate_selected_model(args.checkpoint,record)
    runtime=GeneralistRuntime.from_checkpoint(model_root)
    def native(messages):return runtime.chat(messages,temperature=0,max_new_tokens=64)
    # Warm reference correctness uses a prior correctly-scored fixed benchmark row.
    prior=json.loads(gzip.decompress((ROOT/'scripts/research-training/learnability-20261115/evidence/REPORT.json.gz').read_bytes()))['baseline']['samples']
    warm=next(row for row in prior if row['correct']);assert native(warm['messages'])==warm['expected']
    rng=torch.random.get_rng_state().clone();observations=[]
    for index,row in enumerate(cases):
        entry=next(e for e in state['entries'] if e['lesson_id']==row['lesson_id'])
        demos=[]
        for train in entry['train']:
            demos.extend(train['messages']);demos.append({'role':'assistant','content':train['answer']})
        fewshot=demos+row['messages']
        tokens=len(runtime.tokenizer.serialize_messages(fewshot,add_generation_prompt=True))
        if tokens>=runtime.model.config.context_length:raise ValueError('fewshot context would truncate')
        start=time.perf_counter();plain=native(row['messages']);plain_time=time.perf_counter()-start
        start=time.perf_counter();conditioned=native(fewshot);conditioned_time=time.perf_counter()-start
        expert=bank.predict(row['messages'])
        # All abstentions receive exactly the already measured native plain call.
        composed=AiriAssistant(lambda messages:plain,skill_bank=bank).respond(row['messages'])
        observations.append({'case':row,'native_plain':plain,'native_8_train_examples':conditioned,
            'composed':composed,'expert_prediction':expert,'fewshot_tokens':tokens,
            'seconds':{'native_plain':plain_time,'native_in_context':conditioned_time},
            'correct_native_plain':None if row['expected'] is None else plain==row['expected'],
            'correct_native_in_context':None if row['expected'] is None else conditioned==row['expected'],
            'correct_composed':None if row['expected'] is None else composed['answer']==row['expected'],
            'expert_abstained':expert is None})
        print(json.dumps({'completed':index+1,'total':len(cases)}),flush=True)
    assert torch.equal(rng,torch.random.get_rng_state())
    # Validate exact native tensor bytes again; no trained/floating state is exported.
    assert validate_selected_model(args.checkpoint,record)==model_root
    summary={}
    for condition in ('known_template','new_wording'):
        group=[o for o in observations if o['case']['condition']==condition]
        summary[condition]={'count':len(group),**{name:sum(o[key] for o in group) for name,key in
            [('native_plain_correct','correct_native_plain'),('native_8_examples_correct','correct_native_in_context'),('composed_correct','correct_composed')]},
            'expert_abstentions':sum(o['expert_abstained'] for o in group)}
    negatives=[o for o in observations if o['case']['condition']=='expert_abstention']
    report={'complete':True,'preregistration_sha256':sha(args.output/'PREREGISTRATION.json'),
        'bank_revision':state['revision'],'summary':summary,'negative_expert_abstentions':sum(o['expert_abstained'] for o in negatives),
        'negative_count':len(negatives),'stable_gate':gate.last_report,'stored_contracts_passed':contracts,
        'greedy_rng_unchanged':True,'native_weights_revalidated':True,'native_weights_changed':False,
        'scope':'same frozen small Airi native model; bounded system ablation, not general Transformer superiority or equal training compute',
        'observations':observations}
    save(args.output/'REPORT.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='observations'}),flush=True)


if __name__=='__main__':main()
