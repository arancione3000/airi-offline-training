"""Finite plasticity search with supervised, training-only champion replay.

No evaluation answer is used by the optimizer. Keeping a replay loss small is
only a training constraint; the independent development gates still decide.
"""
import math


def cache_tokenizer(tokenizer, *, capacity=16384):
    """Cache pure encodings, returning fresh lists so callers cannot poison them."""
    from functools import lru_cache
    # ByteTokenizer is immutable and its byte encoding is already linear/cheap.
    if not hasattr(tokenizer,'merges'):return tokenizer
    if getattr(tokenizer,'_continual_cached',False):return tokenizer
    original=tokenizer.encode
    @lru_cache(maxsize=capacity)
    def cached(text,bos,eos):return tuple(original(text,bos=bos,eos=eos))
    def encode(text,*,bos=False,eos=False):return list(cached(str(text),bos,eos))
    tokenizer.encode=encode;tokenizer._continual_cached=True
    return tokenizer


def validate_configuration(spec):
    import hashlib
    import json
    payload={k:v for k,v in spec.items() if k!='fingerprint'}
    expected=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if spec['fingerprint']!=expected:
        raise ValueError('continual experiment fingerprint mismatch')
    value=spec['continual_learning']
    mode=value.get('training_mode','full')
    if mode not in {'full','cached_head'}:raise ValueError('unsupported native training mode')
    if value.get('human_training_scope','legacy_and_reviewed') not in {'legacy_and_reviewed','reviewed_only'}:
        raise ValueError('invalid human training scope')
    if mode=='cached_head':
        cache=value.get('feature_cache_mebibytes')
        if not isinstance(cache,int) or isinstance(cache,bool) or not 16<=cache<=512:
            raise ValueError('invalid bounded native feature cache')
        if value.get('human_training_scope')!='reviewed_only':
            raise ValueError('cached head experiment requires reviewed conversational acquisition')
    if value.get('human_language_policy','legacy_ratio') not in {'legacy_ratio','sqrt_corpus'}:
        raise ValueError('invalid reviewed language sampling')
    examples=value.get('retention_examples_per_language',1)
    if not isinstance(examples,int) or isinstance(examples,bool) or not 1<=examples<=8:
        raise ValueError('invalid bounded retention batch')
    rank=value.get('protected_head_rank',0)
    if not isinstance(rank,int) or isinstance(rank,bool) or not 0<=rank<=48 or (rank and mode!='cached_head'):
        raise ValueError('invalid native head protection rank')
    def integer(name, low, high):
        n=value[name]
        if not isinstance(n,int) or isinstance(n,bool) or not low<=n<=high:
            raise ValueError('invalid finite '+name)
    integer('evaluation_updates',64,1024)
    if value['evaluation_updates']%16:raise ValueError('evaluation must align with update ledger')
    integer('maximum_rejections',1,12);integer('maximum_evaluations',1,64)
    if 'trial_updates' in value:
        integer('trial_updates',64,8192)
        if value['trial_updates'] % value['evaluation_updates']:
            raise ValueError('trial must contain complete evaluation intervals')
        if value['trial_updates'] < value['evaluation_updates']:
            raise ValueError('trial shorter than evaluation interval')
    if not 60<=spec['maximum_training_seconds']<=21600:
        raise ValueError('unbounded continual experiment')
    profiles=value['profiles']
    if not 1<=len(profiles)<=6:raise ValueError('invalid finite profiles')
    for p in profiles:
        for k,low,high in [('learning_rate',1e-7,1e-4),('retention_weight',.5,16.),('champion_kl_weight',.5,16.)]:
            n=p[k]
            if isinstance(n,bool) or not isinstance(n,(int,float)) or not math.isfinite(n) or not low<=n<=high:
                raise ValueError('invalid profile '+k)
    alphas=value['interpolation_alphas']
    if len(alphas)>3 or any(not isinstance(a,(int,float)) or isinstance(a,bool) or not math.isfinite(a) or not 0<a<1 for a in alphas):
        raise ValueError('invalid interpolation probes')
    schedule=value['sampling_schedule']
    allowed={'human-dialogue','curated-dialogue','old-authored','verified-context','composition','native-replay'}
    if len(schedule)!=16 or set(schedule)-allowed or schedule.count('human-dialogue')<4 or 'native-replay' not in schedule:
        raise ValueError('invalid protected sampling schedule')
    slots=value.get('success_transfer_slots',[])
    if (not isinstance(slots,list) or len(slots)>2
            or any(not isinstance(i,int) or isinstance(i,bool) or not 0<=i<16 for i in slots)
            or len(set(slots))!=len(slots)
            or any(schedule[i] in {'human-dialogue','native-replay'} for i in slots)):
        raise ValueError('archived lesson slots overlap protected replay or are invalid')
    return value


def record_decision(search, config, update, accepted, *, force=False):
    """Separate a failed promotion check from a failed learning trajectory.

    Intermediate failures preserve both candidate weights and Adam. Only a
    complete bounded trial counts toward the rejection limit/profile rotation.
    The caller must restore the champion when rollback is returned, including
    at the global evaluation or compute limit. All state is checkpointed.
    """
    start=search.setdefault('trial_start_update',0)
    if not isinstance(update,int) or isinstance(update,bool) or update<start:
        raise ValueError('invalid exploration update ledger')
    search['evaluations']+=1
    finished=force or update-start>=config.get('trial_updates',config['evaluation_updates'])
    if accepted:
        search['consecutive_rejections']=0
        search['accepted_updates']+=1
        search['trial_start_update']=update
    elif finished:
        search['consecutive_rejections']+=1
        search['rejected_trials']=search.get('rejected_trials',0)+1
        search['trial_start_update']=update
    search['stopped']=(search['consecutive_rejections']>=config['maximum_rejections']
                       or search['evaluations']>=config['maximum_evaluations'])
    rollback=not accepted and (finished or search['stopped'])
    return {'rollback':rollback,'trial_complete':finished,
            'trial_updates':update-start,'exploration_pending':not accepted and not rollback}


def try_interpolations(model, champion, *, evaluate, acceptable, alphas=(.5,.25)):
    """Probe smaller parameter steps under the SAME gates; restore on failure.

    This is a bounded search over weights, not a vote between inference models.
    A merged candidate must start a fresh optimizer trajectory at the caller.
    """
    import torch
    state=model.state_dict();base=champion.state_dict();accepted=False;probes=0
    if state.keys()!=base.keys() or any(state[k].shape!=base[k].shape for k in base):
        raise ValueError('interpolation requires the exact same native architecture')
    names=list(state)
    if hasattr(model,'_airi_head_features'):
        if any(not torch.equal(state[k],base[k]) for k in state if k!='lm_head.weight'):
            raise ValueError('head interpolation requires identical frozen backbones')
        names=['lm_head.weight']
    original={k:state[k].detach().clone() for k in names}
    try:
        for alpha in alphas:
            if not isinstance(alpha,(int,float)) or isinstance(alpha,bool) or not math.isfinite(alpha) or not 0<alpha<1:
                raise ValueError('invalid interpolation factor')
            with torch.no_grad():
                for name in names:
                    value=state[name]
                    value.copy_(torch.lerp(base[name],original[name],alpha) if value.is_floating_point() else original[name])
            assessment=evaluate();probes+=1
            if acceptable(assessment):
                accepted=True
                return {'alpha':alpha,'probes':probes,'assessment':assessment}
        return {'alpha':None,'probes':probes,'assessment':None}
    finally:
        if not accepted:
            with torch.no_grad():
                for name in names:state[name].copy_(original[name])


def retention_rows(rows, *, per_family=2):
    """Keep bilingual short skills from being swamped by the large copy corpus."""
    from success_bank import row_key
    groups = {}
    for row in rows:
        family = row['family']
        if not family.startswith(('phrase-', 'extra-', 'foundation-dialogue', 'addition', 'memory')):
            continue
        # Extra-dialogue has many separate intents under a single old family.
        intent=row['expected'] if family in {'extra-dialogue','foundation-dialogue'} else ''
        groups.setdefault((row['language'], family, intent), []).append(row)
    result = []
    for key, values in sorted(groups.items()):
        # Whole prompt histories, not answers, identify examples. These rows have
        # already passed the caller's training/held-out separation audit.
        seen = set()
        for row in sorted(values, key=row_key):
            identity = row_key(row)
            if identity in seen:
                continue
            result.append(row); seen.add(identity)
            if len(seen) == per_family:
                break
    if not result or {r['language'] for r in result} != {'it', 'en'}:
        raise ValueError('training retention memory must cover both languages')
    return result


def verified_memories(champion, rows, encode):
    """Certify greedy training answers through correctly shifted target logits.

    If EVERY gold next token (including EOS) is the argmax with a causal mask,
    greedy generation follows that same answer by induction. Partial token hits
    cannot certify an answer and are excluded from this retention memory.
    """
    import torch
    champion.model.eval();selected=[]
    with torch.no_grad():
        for row in rows:
            ids,labels=encode(row);targets=labels[:,1:];mask=targets!=-100
            predicted=champion.model(ids)['logits'][:,:-1].argmax(-1)
            if bool(mask.any()) and bool((predicted[mask]==targets[mask]).all()):
                selected.append(row)
    if {r['language'] for r in selected}!={'it','en'}:
        raise ValueError('no certified bilingual champion training memory')
    return selected


def profile_index(rejections, count):
    if (not isinstance(rejections, int) or isinstance(rejections, bool) or rejections < 0
            or not isinstance(count, int) or isinstance(count, bool) or count < 1):
        raise ValueError('invalid finite plasticity search')
    return rejections % count


def transfer_replay_options(config, search, memory_by, rng):
    """Archive replay and consolidation must honor the active retention profile."""
    profile=config['profiles'][profile_index(search['consecutive_rejections'],len(config['profiles']))]
    return dict(memories=retention_memories(config,memory_by,rng),
                retention_weight=profile['retention_weight'],champion_kl_weight=profile['champion_kl_weight'])


def retention_memories(config,memory_by,rng):
    if any(not memory_by.get(lang) for lang in ('it','en')):
        raise ValueError('retention requires certified bilingual memories')
    return [memory_by[lang][rng.randrange(len(memory_by[lang]))]
            for lang in ('it','en') for _ in range(config.get('retention_examples_per_language',1))]


def evaluation_due(counters,interval):
    update=counters['conversation_updates'];last=counters.get('last_development_update',0)
    if not 0<=last<=update:raise ValueError('invalid evaluation update ledger')
    return update-last>=interval


def rejection_reasons(before, after):
    """Explain a failed gate without using its examples as training targets."""
    from policy import improved
    from success_bank import retains_correct_answers
    reasons = []
    for name in ('legacy_correct', 'dialogue_correct', 'foundation_correct'):
        if after.get(name, 0) < before.get(name, 0):
            reasons.append(name + '_regressed')
    for name in ('nll', 'human_nll'):
        for key, value in before.get(name, {}).items():
            measured = after.get(name, {}).get(key, float('nan'))
            if not math.isfinite(measured) or measured <= 0 or measured > value * 1.02:
                reasons.append(name + ':' + key + '_regressed_or_invalid')
    if not retains_correct_answers(before, after):
        reasons.append('previously_correct_answer_lost')
    if not improved(before, after) and not reasons:
        reasons.append('no_sufficient_independent_gain_or_other_gate')
    return reasons


def continual_step(model, champion, reference, optimizer, batch, memories, anchor, *,
                   retention_weight=2., champion_kl_weight=2., native_weight=.25):
    """Accumulate losses per example, then perform exactly ONE Adam update.

    The old step had no supervised replay on conversational retention and its
    teacher predated the conversational champion. Here gold memory answers and
    the current champion both protect those skills on every ordinary update.
    The original native reference remains frozen on its own replay anchors.
    """
    import torch
    from generalist_lm.training import causal_training_objective, reference_kl_loss
    from success_transfer import frozen_independent
    frozen_independent(model, champion)
    frozen_independent(model, reference)
    if not memories:
        raise ValueError('supervised retention replay is required')
    weights = (retention_weight, champion_kl_weight, native_weight)
    if any(not math.isfinite(w) or w < 0 for w in weights) or retention_weight <= 0:
        raise ValueError('invalid retention weights')
    model.train(); champion.eval(); reference.eval()
    optimizer.zero_grad(set_to_none=True)
    normalization = 1. + retention_weight + native_weight
    total = 0.; task_ce = 0.; memory_ce = 0.; kl_total = 0.
    supervised = 0; reference_tokens = 0

    def ce_forward(encoded):
        ids, labels = encoded
        logits = model(ids)['logits']
        ce, _ = causal_training_objective(logits, labels, ids,
                                         repetition_unlikelihood_weight=.05,
                                         eos_loss_weight=1.2)
        return logits, ce

    try:
        logits, ce = ce_forward(batch)
        loss = ce / normalization
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('non-finite acquisition loss; no optimizer update')
        loss.backward(); total += float(loss.detach()); task_ce = float(ce.detach())
        supervised += int((batch[1][:, 1:] != -100).sum())
        for memory in memories:
            logits, ce = ce_forward(memory)
            with torch.no_grad(): teacher = champion(memory[0])['logits']
            kl, count = reference_kl_loss(logits, teacher, memory[1])
            loss = (retention_weight * ce + champion_kl_weight * kl) / (normalization * len(memories))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('non-finite retention loss; no optimizer update')
            loss.backward(); total += float(loss.detach())
            memory_ce += float(ce.detach()) / len(memories)
            kl_total += float(kl.detach()); reference_tokens += count
            supervised += int((memory[1][:, 1:] != -100).sum())
        if native_weight:
            logits, ce = ce_forward(anchor)
            with torch.no_grad(): teacher = reference(anchor[0])['logits']
            kl, count = reference_kl_loss(logits, teacher, anchor[1])
            loss = native_weight * (ce + kl) / normalization
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('non-finite native retention loss; no optimizer update')
            loss.backward(); total += float(loss.detach()); reference_tokens += count
            supervised += int((anchor[1][:, 1:] != -100).sum())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    return {'loss': total, 'ce': task_ce, 'retention_ce': memory_ce, 'kl': kl_total,
            'supervised_tokens': supervised, 'reference_tokens': reference_tokens,
            'gradient_norm_before_clipping': float(norm),
            'method': 'gold conversational replay + current champion KL + native replay'}


def acquisition_language(config,pools,rng,fallback):
    """Give a small reviewed language more exposure without repeating it 2/3 times."""
    if config.get('human_language_policy','legacy_ratio')=='legacy_ratio':return fallback
    languages=sorted(pools)
    if not languages or any(not pools[lang] for lang in languages):
        raise ValueError('empty reviewed acquisition language')
    return rng.choices(languages,weights=[math.sqrt(len(pools[lang])) for lang in languages],k=1)[0]
