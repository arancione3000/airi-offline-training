"""Independent operational gates for lexical paraphrase routing.

Only TRAIN is passed to the router learner. Qualification and later endpoints
are accept/reject measurements, not sources of candidate aliases or thresholds.
"""
import operator

from context_memory import resolve
from flexible_skills import LearnedQueryRouter
from skill_bank import _route
from skill_validation import StableGate


def bindings(bank,families):
    definitions={row['id']:row for row in families}
    result=[]
    for entry in bank._read()[0]['entries']:
        family=definitions.get(entry['lesson_id'])
        if family is None:continue
        program=entry['program']
        if any(program[key]!=family[other] for key,other in
               [('query','canonical_query'),('source','source'),('operation','operation')]):
            raise ValueError('family metadata does not match approved program')
        result.append({'lesson_id':entry['lesson_id'],'program_id':entry['id'],'canonical_query':program['query']})
    return result


def case(family,query,index,variant):
    op=family['operation'];source=family['source']
    if op in ('add','subtract','multiply'):
        a=113+index*7+variant*503;b=29+index*3+variant*101
        body=str(a)+source['separator']+str(b)
        answer=str({'add':operator.add,'subtract':operator.sub,'multiply':operator.mul}[op](a,b))
    else:
        body=f'ÁbC-{index}-{variant}-qZé'
        answer={'copy':lambda x:x,'reverse':lambda x:''.join(reversed(x)),
                'uppercase':str.upper,'lowercase':str.lower}[op](body)
    return [{'role':'user','content':source['prefix']+body+source['suffix']},
            {'role':'user','content':query}],answer


def evaluate(bank,model_path,rows,families,*,value_offset=0,backend=None):
    state=bank._read()[0];router=LearnedQueryRouter(bank,model_path,frozen=True,backend=backend)
    definitions={row['id']:row for row in families}
    installed={entry['lesson_id'] for entry in state['entries']}
    observations=[]
    def composed(messages):
        return _route(state['entries'],messages) or router.predict(messages)
    for index,row in enumerate(rows):
        family=definitions[row['source_lesson_id']]
        if family['id'] not in installed:continue
        for variant in (0,1):
            messages,gold=case(family,row['query'],index+value_offset,variant)
            expected=gold if row['lesson_id'] is not None else None
            result=composed(messages);answer=None if result is None else result['answer']
            observations.append({'id':row['id'],'variant':variant,'positive':expected is not None,
                                 'messages':messages,'expected':expected,'prediction':result,'correct':answer==expected})
    positives=[r for r in observations if r['positive']];negatives=[r for r in observations if not r['positive']]
    gate=StableGate();gate(composed)
    old=[]
    for entry in state['entries']:
        for row in entry['train']+entry['qualification']:
            result=composed(row['messages']);answer=None if result is None else result['answer']
            old.append(answer==row['answer'])
    result={'positive_correct':sum(r['correct'] for r in positives),'positive_total':len(positives),
            'negative_total':len(negatives),'false_answers':sum(not r['correct'] for r in negatives),
            'stored_contracts':len(old),'stored_failures':sum(not ok for ok in old),
            'historical_gate':gate.last_report,'observations':observations}
    result['passed']=(bool(positives) and result['positive_correct']>=len(positives)*.5
                      and result['false_answers']==0 and result['stored_failures']==0 and gate.last_report['passed'])
    return result
