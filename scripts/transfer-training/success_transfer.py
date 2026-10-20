"""Selective output distillation between AIRI's own native candidates."""
import copy
import math
import torch
from torch.nn import functional as F
from generalist_lm.training import causal_training_objective, reference_kl_loss
from curriculum import exact_correct
from success_bank import MAX_POSITIONS, remember


def frozen_independent(student, reference):
    storage = {p.untyped_storage().data_ptr() for p in student.parameters()}
    if any(p.untyped_storage().data_ptr() in storage for p in reference.parameters()):
        raise ValueError('champion reference must have independent storage')
    if any(p.requires_grad for p in reference.parameters()):
        raise ValueError('champion reference must be frozen')


def harvest(candidate, champion, rows, encode, bank, eligible, source, *, generation_limit=4):
    """Only copy gold-token predictions where the donor is measurably better."""
    frozen_independent(candidate.model, champion.model)
    candidate.model.eval(); champion.model.eval()
    proposals = []
    with torch.no_grad():
        for row in rows:
            ids, labels = encode(row)
            donor = candidate.model(ids)['logits'][0, :-1].float().softmax(-1)
            old = champion.model(ids)['logits'][0, :-1].float().softmax(-1)
            tokens = []
            for shifted in (labels[0, 1:] != -100).nonzero().flatten().tolist():
                target = int(labels[0, shifted + 1])
                probability = float(donor[shifted, target])
                gain = math.log(max(probability, 1e-12)) - math.log(max(float(old[shifted, target]), 1e-12))
                if int(donor[shifted].argmax()) != target or probability < .10 or gain < .05:
                    continue
                tokens.append({'position': shifted + 1, 'target': target, 'gain': gain,
                               'probabilities': donor[shifted].double().tolist()})
            tokens = sorted(tokens, key=lambda p: (-p['gain'], p['position']))[:MAX_POSITIONS]
            if tokens:
                proposals.append({'row': copy.deepcopy(row), 'positions': tokens,
                                  'source': copy.deepcopy(source), 'whole_answer_gain': False,
                                  'scope': 'authored training only'})
        proposals.sort(key=lambda e: -sum(p['gain'] for p in e['positions']))
        for index, entry in enumerate(proposals):
            if index < generation_limit:
                row = entry['row']; maximum = min(160, len(row['expected'].encode()) + 16)
                answer = candidate.chat(row['messages'][:-1], temperature=0, max_new_tokens=maximum)
                entry['donor_answer_correct'] = exact_correct(answer, row['expected'])
                if entry['donor_answer_correct']:
                    previous = champion.chat(row['messages'][:-1], temperature=0, max_new_tokens=maximum)
                    entry['whole_answer_gain'] = not exact_correct(previous, row['expected'])
            remember(bank, entry, eligible)
    bank['harvests'] += 1
    return len(proposals)


def distill_step(model, champion, optimizer, entry, batch, anchor, *, lr_multiplier=5., lr_cap=1e-6,
                 memories=None, retention_weight=.5, champion_kl_weight=1.):
    """Learn selected donor tokens while replaying supervised champion anchors."""
    frozen_independent(model, champion)
    if any(isinstance(w,bool) or not isinstance(w,(int,float)) or not math.isfinite(w) or w<=0
           for w in (retention_weight,champion_kl_weight)):
        raise ValueError('invalid supervised transfer retention weights')
    if memories is not None and not memories:
        raise ValueError('transfer retention memories cannot be empty')
    replay=[anchor] if memories is None else list(memories)
    ids, labels = batch
    positions = [p['position'] for p in entry['positions']]
    for token in entry['positions']:
        position = token['position']
        if position < 1 or position >= labels.shape[1] or int(labels[0, position]) != token['target']:
            raise ValueError('teacher token does not match encoded answer')
    model.train(); champion.eval(); optimizer.zero_grad(set_to_none=True)
    logits = model(ids)['logits'][0, torch.tensor(positions)-1].float()
    teacher = torch.tensor([p['probabilities'] for p in entry['positions']], dtype=logits.dtype)
    teacher = teacher / teacher.sum(-1, keepdim=True)
    if teacher.shape != logits.shape or not bool(torch.isfinite(teacher).all()):
        raise ValueError('teacher shape or probabilities changed')
    target = torch.tensor([p['target'] for p in entry['positions']])
    ce = F.cross_entropy(logits, target)
    kd = F.kl_div(logits.log_softmax(-1), teacher, reduction='batchmean')
    loss = ce + .5 * kd
    count=0;supervised=len(positions);retention_total=0.;anchor_ce_total=0.
    for anchor_ids,anchor_labels in replay:
        anchor_logits = model(anchor_ids)['logits']
        anchor_ce, objective = causal_training_objective(anchor_logits, anchor_labels, anchor_ids,
                                                         repetition_unlikelihood_weight=.05, eos_loss_weight=1.2)
        with torch.no_grad(): protected = champion(anchor_ids)['logits']
        retention, tokens = reference_kl_loss(anchor_logits, protected, anchor_labels)
        loss = loss + (retention_weight * anchor_ce + champion_kl_weight * retention)/len(replay)
        retention_total+=float(retention.detach())/len(replay)
        anchor_ce_total+=float(anchor_ce.detach())/len(replay)
        count+=tokens;supervised+=int((anchor_labels[:,1:]!=-100).sum())
    # Legacy callers retain their exact objective. Continual callers use the
    # supervised acquisition/replay normalization with their active profile.
    if memories is not None:loss=loss/(1.+retention_weight)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('non-finite transfer loss; no optimizer update')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    if not math.isfinite(lr_multiplier) or lr_multiplier <= 0 or not math.isfinite(lr_cap) or lr_cap <= 0:
        raise ValueError('invalid finite transfer learning rate')
    previous_lrs = [g['lr'] for g in optimizer.param_groups]
    try:
        for group in optimizer.param_groups: group['lr'] = min(lr_cap, group['lr'] * lr_multiplier)
        optimizer.step()
    finally:
        for group, lr in zip(optimizer.param_groups, previous_lrs): group['lr'] = lr
    return {'loss': float(loss.detach()), 'ce': float(ce.detach()), 'kl': float(kd.detach()),
            'retention_kl': retention_total, 'retention_ce': anchor_ce_total, 'supervised_tokens': supervised,
            'reference_tokens': count, 'gradient_norm_before_clipping': float(norm), 'objective': objective,
            'retention_weight':retention_weight,'champion_kl_weight':champion_kl_weight,'retention_examples':len(replay),
            'method': 'selected native donor tokens + supervised replay + champion KL'}
