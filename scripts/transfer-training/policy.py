"""Verifier score for offline selection; this is not a learned reward model."""
import math

def reward(m):
    return (.40*sum(m['context_rates'].values())/len(m['context_rates'])+.25*m['context_pair_rate']+.20*m['legacy_correct']/28+.10*m['dialogue_correct']/16+.05*m.get('foundation_correct',0)/max(1,m.get('foundation_probes',16))-.25*m['repetition'])

def improved(before,after):
    if not all(math.isfinite(x) for x in list(after['nll'].values())+list(after['context_rates'].values())+[after['context_pair_rate'],after['repetition']]):return False
    if after.get('foundation_correct',0)<before.get('foundation_correct',0):return False
    if after['legacy_correct']<before['legacy_correct'] or after['dialogue_correct']<before['dialogue_correct']:return False
    if any(after['context_rates'][k]<v for k,v in before['context_rates'].items()):return False
    if after['context_pair_rate']<before['context_pair_rate']:return False
    if after['repetition']>max(.05,before['repetition']):return False
    if any(after['nll'][k]>v*1.02 for k,v in before['nll'].items()):return False
    return reward(after)>reward(before)+.005 or (reward(after)>=reward(before) and all(after['nll'][k]<v*.99 for k,v in before['nll'].items()))

def unlocked_stage(m,current):
    threshold=(.5,.75)[min(current,1)]
    if current<2 and min(m['context_rates'].values())>=threshold and m['context_pair_rate']>=threshold:return current+1
    return current

def recover_best(stagnant_rounds):return stagnant_rounds>=2
