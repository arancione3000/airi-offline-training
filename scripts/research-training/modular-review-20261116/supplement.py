"""Explicit post-review arithmetic confirmation INSIDE the declared integer range.

The original 24-case endpoint is preserved. This is a separate follow-up, not a
replacement blind result or a change to any expert/compiler.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

ROOT=Path(__file__).resolve().parents[3]
SEED=20261116223647


def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--record',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('choose an absent output')
    a.output.mkdir(parents=True);shutil.copy2(__file__,a.output/'supplement-as-run.py')
    sys.path.insert(0,str(a.first_run/'sources'))
    from assistant import AiriAssistant
    from skill_bank import SkillBank
    bank=SkillBank(a.first_run/'bank.json');state,_=bank._read();rng=random.Random(SEED)
    cases=[]
    for task,operation in (('delta_sum','add'),('sigma_product','multiply'),('auto_v1_remaining','subtract')):
        entry=next(e for e in state['entries'] if e['lesson_id']==task)
        left,right=rng.randint(10001,950000),-rng.randint(10001,950000)
        answer=str(left+right if operation=='add' else left*right if operation=='multiply' else left-right)
        source=entry['program']['source']
        cases.append({'lesson_id':task,'messages':[{'role':'user','content':source['prefix']+str(left)+source['separator']+str(right)+source['suffix']},
            {'role':'user','content':entry['program']['query']}],'expected':answer})
    save(a.output/'CASES.json',cases);record=json.loads(a.record.read_text())
    prereg={'created_at_utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,
        'model_revision':record['revision'],'weights_digest':record['weights_digest'],
        'first_preregistration_sha256':sha(a.first_run/'PREREGISTRATION.json'),
        'cases_sha256':sha(a.output/'CASES.json'),'bank_sha256':sha(a.first_run/'bank.json'),
        'source_sha256':sha(__file__),'scope':'separate post-review confirmation after detecting first endpoint used numeric values outside declared range; no code or expert changes'}
    save(a.output/'PREREGISTRATION.json',prereg)
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0,str(ROOT/'scripts/transfer-training'))
    from chat_current import validate_selected_model
    from generalist_lm.runtime import GeneralistRuntime
    model_root=validate_selected_model(a.checkpoint,record);runtime=GeneralistRuntime.from_checkpoint(model_root)
    def native(messages):return runtime.chat(messages,temperature=0,max_new_tokens=64)
    before=torch.random.get_rng_state().clone();observations=[]
    for case in cases:
        entry=next(e for e in state['entries'] if e['lesson_id']==case['lesson_id']);demos=[]
        for row in entry['train']:demos.extend(row['messages']);demos.append({'role':'assistant','content':row['answer']})
        fewshot=demos+case['messages'];tokens=len(runtime.tokenizer.serialize_messages(fewshot,add_generation_prompt=True))
        assert tokens<runtime.model.config.context_length
        plain=native(case['messages']);conditioned=native(fewshot)
        composed=AiriAssistant(lambda _:plain,skill_bank=bank).respond(case['messages'])
        observations.append({'case':case,'native_plain':plain,'native_8_train_examples':conditioned,'composed':composed,
            'fewshot_tokens':tokens,'plain_correct':plain==case['expected'],'in_context_correct':conditioned==case['expected'],
            'composed_correct':composed['answer']==case['expected']})
        print(json.dumps({'completed':len(observations),'total':3}),flush=True)
    assert torch.equal(before,torch.random.get_rng_state()) and validate_selected_model(a.checkpoint,record)==model_root
    report={'complete':True,'native_weights_revalidated':True,'native_weights_changed':False,
        'greedy_rng_unchanged':True,'bank_revision':state['revision'],'scope':prereg['scope'],
        'summary':{'count':3,'native_plain_correct':sum(o['plain_correct'] for o in observations),
                   'native_8_examples_correct':sum(o['in_context_correct'] for o in observations),
                   'composed_correct':sum(o['composed_correct'] for o in observations)},'observations':observations}
    save(a.output/'REPORT.json',report);print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
