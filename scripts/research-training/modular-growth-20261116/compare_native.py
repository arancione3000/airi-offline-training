"""Fixed small native/in-context/program comparison, not a Transformer-class claim."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/assistant-runtime'))
from assistant import AiriAssistant
from chat import VerifiedNative
from skill_bank import SkillBank


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--curriculum',type=Path,required=True)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--record',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('comparison already recorded')
    import torch
    torch.set_num_threads(2)
    lessons=json.loads((a.curriculum/'LESSONS/TRAIN.json').read_text())['lessons']
    rows=json.loads((a.curriculum/'CASES/FINAL.json').read_text())['cases']
    chosen=[r for lesson in lessons for r in [v for v in rows if v['lesson_id']==lesson['lesson_id']][:4]]
    a.output.mkdir(parents=True)
    (a.output/'CASES.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2)+'\n')
    record=json.loads(a.record.read_text());native=VerifiedNative(cache=a.output/'unused',checkpoint=a.checkpoint,record=record)
    baseline=json.loads(gzip.decompress((ROOT/'scripts/research-training/learnability-20261115/evidence/REPORT.json.gz').read_bytes()))['baseline']['samples']
    warm=next(r for r in baseline if r['correct'])
    assert native(warm['messages'])==warm['expected']
    runtime=native.runtime
    def fallback(messages):return runtime.chat(messages,temperature=0,max_new_tokens=64)
    system=AiriAssistant(fallback,skill_bank=SkillBank(a.bank));observations=[];rng=torch.random.get_rng_state().clone()
    for i,row in enumerate(chosen):
        lesson=next(l for l in lessons if l['lesson_id']==row['lesson_id'])
        demonstrations=[]
        for train in lesson['examples']:
            demonstrations.extend(train['messages']);demonstrations.append({'role':'assistant','content':train['answer']})
        fewshot=demonstrations+row['messages']
        tokens=len(runtime.tokenizer.serialize_messages(fewshot,add_generation_prompt=True))
        if tokens>=runtime.model.config.context_length:raise ValueError('few-shot baseline would truncate; no silent comparison')
        start=time.perf_counter();plain=fallback(row['messages']);plain_time=time.perf_counter()-start
        start=time.perf_counter();conditioned=fallback(fewshot);conditioned_time=time.perf_counter()-start
        start=time.perf_counter();learned=system.respond(row['messages']);learned_time=time.perf_counter()-start
        observations.append({'case':row,'native_plain':plain,'native_8_examples_in_context':conditioned,
            'learned_program':learned,'correct_plain':plain==row['expected_answer'],
            'correct_in_context':conditioned==row['expected_answer'],'correct_program':learned['answer']==row['expected_answer'],
            'fewshot_prompt_tokens':tokens,'seconds':{'plain':plain_time,'in_context':conditioned_time,'program':learned_time}})
        if (i+1)%6==0:print(json.dumps({'phase':'comparison','completed':i+1}),flush=True)
    assert torch.equal(rng,torch.random.get_rng_state())
    result={'scope':'same frozen Airi native reference; learned symbolic programs versus no/8-example context, not an equal-training-compute or general Transformer comparison',
      'model_revision':record['revision'],'weights_digest':record['weights_digest'],'sample_policy':'first four precommitted FINAL cases per lesson, fixed before inference',
      'count':len(chosen),'correct_native_plain':sum(r['correct_plain'] for r in observations),
      'correct_native_in_context':sum(r['correct_in_context'] for r in observations),'correct_program':sum(r['correct_program'] for r in observations),
      'greedy_rng_unchanged':True,'observations':observations,'complete':True}
    (a.output/'REPORT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='observations'}),flush=True)


if __name__=='__main__':main()
