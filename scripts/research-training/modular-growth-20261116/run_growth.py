"""One fixed modular-growth evaluation; independent cases never enter synthesis."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/assistant-runtime'))
from assistant import AiriAssistant
from context_memory import resolve
from skill_bank import SkillBank
from skill_validation import StableGate


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--curriculum',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('choose a fresh output; no rerun selection')
    a.output.mkdir(parents=True)
    inputs={str(path):sha(path) for path in a.curriculum.rglob('*') if path.is_file()}
    code={str(path):sha(path) for path in (ROOT/'scripts/assistant-runtime').glob('*.py')}
    manifest={'inputs':inputs,'code':code,'rules':'single synthesis from TRAIN; qualification accept/reject only; final first observed after activation, never model selection'}
    (a.output/'PREREGISTRATION.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lessons=json.loads((a.curriculum/'LESSONS/TRAIN.json').read_text())['lessons']
    bank=SkillBank(a.output/'bank.json');growth=[];t=time.perf_counter()
    for lesson in lessons:
        gate=StableGate();result=bank.learn(lesson,regression_gate=gate)
        growth.append({'lesson_id':lesson['lesson_id'],'transition':result,'registry':bank.status(),'stable_gate':gate.last_report})
    training_seconds=time.perf_counter()-t
    observations={}
    for split in ('DEV','FINAL','NEGATIVE'):
        rows=json.loads((a.curriculum/f'CASES/{split}.json').read_text())['cases'];values=[]
        for row in rows:
            result=bank.predict(row['messages']);answer=result['answer'] if result is not None else None
            expected=None if row['expected_abstain'] else row['expected_answer']
            if result:
                witness=result['provenance'];text=row['messages'][witness['message_index']]['content']
                assert text[witness['start']:witness['end']]==witness['span']
                assert text[witness['value_start']:witness['value_end']]==witness['value_span']
            values.append({'id':row['id'],'lesson_id':row['lesson_id'],'messages':row['messages'],
                           'expected':expected,'prediction':result,'correct':answer==expected})
        observations[split]={'correct':sum(v['correct'] for v in values),'total':len(values),'observations':values}
    before=bank.status();reload_equal=SkillBank(bank.path).status()==before
    replay=[bank.learn(l)['action'] for l in lessons]
    after=bank.status();assert before==after
    bad=json.loads((a.curriculum/'LESSONS/CORRUPTED.json').read_text())['lesson']
    rejected=bank.learn(bad);assert bank.status()==before
    # Simulate losing the convenience copy; committed snapshots remain authoritative.
    bank.path.write_text('{interrupted-main')
    recovered=SkillBank(bank.path);recovery=recovered.status()
    assert recovery['state_sha256']==before['state_sha256']
    snapshot=next(p for p in bank.snapshots.glob('*.json') if p.name.startswith(f'{before["revision"]:06d}-'))
    bank.path.write_bytes(snapshot.read_bytes())
    assert bank.status()==before
    source=json.loads(gzip.decompress((ROOT/'scripts/research-training/learnability-20261115/evidence/REPORT.json.gz').read_bytes()))['baseline']['samples']
    native_map={json.dumps(r['messages'],sort_keys=True,ensure_ascii=False):r['response'] for r in source}
    assistant=AiriAssistant(lambda m:native_map[json.dumps(m,sort_keys=True,ensure_ascii=False)],skill_bank=bank)
    old=[]
    for row in source:
        result=assistant.respond(row['messages'])
        old.append({'correct':result['answer']==row['expected'],'previous_native_correct':row['correct'],'source':result['source']})
    times=[];messages=observations['FINAL']['observations'][0]['messages']
    for _ in range(30):
        start=time.perf_counter();bank.predict(messages);times.append(time.perf_counter()-start)
    report={'scope':'bounded program induction and persistent expert growth, not neural parameter learning or Transformer superiority',
        'growth':growth,'training_seconds':training_seconds,'endpoints':observations,'registry':before,
        'restart_equal':reload_equal,'replay_actions':replay,'replay_unchanged':before==after,'bad_lesson':rejected,
        'bad_lesson_unchanged':bank.status()==before,'snapshot_recovery':recovery,
        'legacy92':{'correct':sum(r['correct'] for r in old),'native_baseline':23,
                    'native_old_correct_lost':sum(r['previous_native_correct'] and not r['correct'] for r in old),
                    'learned_intercepts':sum(r['source']=='learned_skill' for r in old),
                    'scope':'actual symbolic routes, hash-bound archived native fallback inherited'},
        'latency_seconds':{'median':statistics.median(times),'max':max(times),'samples':len(times)},
        'baselines':{'empty_learned_bank_answers':0,'existing_memory_answers':sum(resolve(r['messages']) is not None for r in observations['FINAL']['observations'])},
        'complete':True,'production_qualified':False,'neural_weights_changed':False,'preregistration_sha256':sha(a.output/'PREREGISTRATION.json')}
    report['all_gates_passed']=all(s['correct']==s['total'] for s in observations.values()) and report['restart_equal'] and all(x=='unchanged' for x in replay) and rejected['action']=='rejected' and all(g['transition']['action']=='activated' and g['stable_gate']['passed'] for g in growth) and report['legacy92']['correct']>=27 and report['legacy92']['native_old_correct_lost']==0
    for path,h in {**inputs,**code}.items():assert sha(path)==h
    (a.output/'REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('growth','endpoints')}))
    if not report['all_gates_passed']:raise SystemExit(1)


if __name__=='__main__':main()
