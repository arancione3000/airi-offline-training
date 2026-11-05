"""Explicit indefinite service with bounded epochs and monotonic lifetime ledgers.

Closing an unsuccessful epoch selects the existing champion. It never turns a
rejected candidate into a baseline or discards TRAIN certificates or Adam/RNG.
"""
from copy import deepcopy
import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_configuration(spec):
    cfg=spec['continuous_training']
    fields={'version','mode','epoch_updates','epoch_training_seconds','epoch_decisions',
            'minimum_backoff_seconds','maximum_backoff_seconds','history_capacity',
            'registry_capacity','lr_recovery'}
    if not isinstance(cfg,dict) or set(cfg)!=fields or type(cfg['version']) is not int or cfg['version']!=1 or cfg['mode']!='indefinite':
        raise ValueError('invalid indefinite training configuration')
    for name,low,high in [('epoch_updates',64,8192),('epoch_training_seconds',60,21600),
                          ('epoch_decisions',1,64),('minimum_backoff_seconds',0,3600),
                          ('maximum_backoff_seconds',60,86400),('history_capacity',1,64),
                          ('registry_capacity',2048,32768)]:
        value=cfg[name]
        if type(value) is not int or not low<=value<=high:raise ValueError('invalid continuous '+name)
    if cfg['epoch_updates']%16 or cfg['maximum_backoff_seconds']<cfg['minimum_backoff_seconds']:
        raise ValueError('invalid epoch alignment or backoff')
    learning=spec['continual_learning']
    if learning.get('training_mode')!='cached_prefix' or not learning.get('knowledge_retention'):
        raise ValueError('indefinite service requires persistent native retention')
    r=cfg['lr_recovery']
    if (not isinstance(r,dict) or set(r)!={'accepted_streak','cooldown_updates','growth_factor'}
            or type(r['accepted_streak']) is not int or not 16<=r['accepted_streak']<=1024
            or type(r['cooldown_updates']) is not int or not 1<=r['cooldown_updates']<=1024
            or isinstance(r['growth_factor'],bool) or r['growth_factor']!=2.):
        raise ValueError('invalid continuous guarded LR recovery')
    if 'continuous_training.py' not in spec['script_sha256']:raise ValueError('continuous implementation must be pinned')
    return cfg


def _base(c):
    return dict(update=c['conversation_updates'],seconds=c['elapsed_training_seconds'],
                evaluations=c['learning_search']['evaluations'],
                consolidation=c.get('retention_consolidation_updates',0),
                adoptions=c['morph']['revision'])


def initialize(c,spec,champion_digest):
    validate_configuration(spec)
    if 'continuous_training' in c:
        validate_state(c,spec);return
    if champion_digest!=c['morph']['weights_digest'] or c.get('exploration_pending',False):
        raise ValueError('continuous migration requires the selected champion')
    c['continuous_training']=dict(schema=1,epoch=0,base=_base(c),closed=False,not_before=0.,
                                 zero_gain_epochs=0,history=[],history_hash='0'*64,receipts={})
    c['learning_search'].update(consecutive_rejections=0,stopped=False,
                               trial_start_update=c['conversation_updates'])
    c['exploration_pending']=False
    validate_state(c,spec)


def validate_state(c,spec):
    cfg=validate_configuration(spec);s=c['continuous_training']
    if (s.get('schema')!=1 or type(s.get('epoch')) is not int or s['epoch']<0
            or type(s.get('closed')) is not bool or not isinstance(s.get('base'),dict)
            or not isinstance(s.get('not_before'),(int,float)) or not math.isfinite(s['not_before'])
            or s['not_before']<0 or len(s.get('history',[]))>cfg['history_capacity']):
        raise ValueError('invalid persistent continuous epoch')
    current=_base(c)
    if set(s['base'])!=set(current) or any(s['base'][k]<0 or s['base'][k]>v for k,v in current.items()):
        raise ValueError('continuous lifetime counters regressed')
    return s


def limits(c,spec):
    s=validate_state(c,spec);cfg=spec['continuous_training']
    return dict(update=s['base']['update']+cfg['epoch_updates'],
                seconds=s['base']['seconds']+cfg['epoch_training_seconds'],
                evaluations=s['base']['evaluations']+cfg['epoch_decisions'],
                consolidation=s['base']['consolidation'])


def can_train(c,spec):
    bound=limits(c,spec)
    return (not c['continuous_training']['closed'] and not c['learning_search'].get('stopped',False)
            and c['conversation_updates']<bound['update'] and c['elapsed_training_seconds']<bound['seconds'])


def profile_index(c,spec):
    from continual_learning import profile_index as ordinary
    count=len(spec['continual_learning']['profiles'])
    return (ordinary(c['learning_search']['consecutive_rejections'],count)+c['continuous_training']['epoch'])%count


def decision_configuration(c,spec):
    cfg=dict(spec['continual_learning']);cfg['maximum_evaluations']=limits(c,spec)['evaluations']
    return cfg


def close_epoch(c,spec,champion_digest,now):
    s=validate_state(c,spec)
    if s['closed']:return
    if champion_digest!=c['morph']['weights_digest'] or c.get('exploration_pending',False):
        raise ValueError('cannot close continuous epoch with an unapproved candidate')
    end=_base(c)
    if end['update']==s['base']['update']:
        raise ValueError('an empty epoch cannot renew the service')
    record=dict(epoch=s['epoch'],base=deepcopy(s['base']),end=end,champion=champion_digest,
                attempted_updates=c['retention_transaction']['attempted_updates'],
                committed_updates=c['retention_transaction']['committed_updates'])
    gain=end['adoptions']>s['base']['adoptions']
    s['zero_gain_epochs']=0 if gain else s['zero_gain_epochs']+1
    cfg=spec['continuous_training']
    delay=0 if gain else min(cfg['maximum_backoff_seconds'],cfg['minimum_backoff_seconds']*2**min(s['zero_gain_epochs']-1,16))
    s['not_before']=float(now)+delay;s['closed']=True
    s['history_hash']=digest([s['history_hash'],record]);s['history'].append(record)
    s['history']=s['history'][-cfg['history_capacity']:]
    # Bound diagnostic logs, never the knowledge ledger or TRAIN registry.
    for key in ('reward_history','selection_history','diagnostic_phase_changes','stage_changes'):
        values=c.get(key,[])
        if len(values)>64:
            old=s['receipts'].get(key,dict(removed=0,sha256='0'*64))
            s['receipts'][key]=dict(removed=old['removed']+len(values)-64,
                                    sha256=digest([old['sha256'],values[:-64]]))
            c[key]=values[-64:]


def open_epoch(c,spec,champion_digest,now):
    s=validate_state(c,spec)
    if not s['closed']:return False
    if now<s['not_before']:raise ValueError('continuous epoch cooldown has not elapsed')
    if champion_digest!=c['morph']['weights_digest'] or c.get('exploration_pending',False):
        raise ValueError('new epoch requires the retained champion')
    s.update(epoch=s['epoch']+1,base=_base(c),closed=False,not_before=0.)
    c['learning_search'].update(consecutive_rejections=0,stopped=False,
                               trial_start_update=c['conversation_updates'])
    c['phase']=0
    return True
