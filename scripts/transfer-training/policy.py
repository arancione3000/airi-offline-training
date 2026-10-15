"""Verifier score for offline selection; this is not a learned reward model."""
import math

def reward(m):
    return (.40*sum(m['context_rates'].values())/len(m['context_rates'])+.25*m['context_pair_rate']+.20*m['legacy_correct']/28+.10*m['dialogue_correct']/16+.05*m.get('foundation_correct',0)/max(1,m.get('foundation_probes',16))-.25*m['repetition'])

def improved(before,after):
    if before.get('human_nll'):
        values=after.get('human_nll',{})
        if set(values)!=set(before['human_nll']):return False
        if any(not math.isfinite(v) or v<=0 or v>before['human_nll'][k]*1.02 for k,v in values.items()):return False
    if not all(math.isfinite(x) for x in list(after['nll'].values())+list(after['context_rates'].values())+[after['context_pair_rate'],after['repetition']]):return False
    if after.get('foundation_correct',0)<before.get('foundation_correct',0):return False
    if after['legacy_correct']<before['legacy_correct'] or after['dialogue_correct']<before['dialogue_correct']:return False
    if any(after['context_rates'][k]<v for k,v in before['context_rates'].items()):return False
    if after['context_pair_rate']<before['context_pair_rate']:return False
    if after['repetition']>max(.05,before['repetition']):return False
    if any(after['nll'][k]>v*1.02 for k,v in before['nll'].items()):return False
    human_gain=bool(before.get('human_nll')) and all(after['human_nll'][k]<v*.99 for k,v in before['human_nll'].items())
    return reward(after)>reward(before)+.005 or (reward(after)>=reward(before) and (all(after['nll'][k]<v*.99 for k,v in before['nll'].items()) or human_gain))

def unlocked_stage(m,current):
    threshold=(.5,.75)[min(current,1)]
    if current<2 and min(m['context_rates'].values())>=threshold and m['context_pair_rate']>=threshold:return current+1
    return current

def recover_best(stagnant_rounds):return stagnant_rounds>=2

def training_focus(m):
    """Choose the weakest measured development skill without reading final probes."""
    context=(sum(m['context_rates'].values())/len(m['context_rates'])+m['context_pair_rate'])/2
    skills={
        'context':context,
        'dialogue':m['dialogue_correct']/16,
        'foundation':m.get('foundation_correct',0)/max(1,m.get('foundation_probes',16)),
        'retention':m['legacy_correct']/28,
    }
    return min(skills,key=lambda name:(skills[name],name))

def recovery_lr_factor(rollbacks,factors=(1.0,.5,.25)):
    if not isinstance(rollbacks,int) or isinstance(rollbacks,bool) or rollbacks<0:
        raise ValueError('invalid rollback count')
    if not factors or any(not math.isfinite(x) or not 0<x<=1 for x in factors):
        raise ValueError('invalid recovery learning-rate factors')
    return factors[min(rollbacks,len(factors)-1)]
