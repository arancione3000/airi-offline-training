"""The predeclared fourth cell: full native parameters with original retention.

This single conditional diagnostic starts at the approved source, reuses arm A's
exact fixed TRAIN/anchor/memory/sentinel trajectory, and changes trainable scope.
The old twelve holdouts are familiar diagnostics. Twenty-four paired fresh
holdouts are committed before training and first forwarded at the endpoint.
No checkpoint promotion, new LR, selection loop, or follow-on arm is authorized.
"""
import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import string
import sys
import time

import controlled_learnability as base

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/transfer-training'))


def confirmation_rows(train, blocked_values):
    rng = random.Random(20261116)
    templates = {
        ('it', 'context-name'): ('Ho scelto il nome {}.', 'Segnato.', 'Scrivi il nome scelto.'),
        ('en', 'context-name'): ('My chosen name is {}.', 'Noted.', 'Write the chosen name.'),
        ('it', 'context-location'): ('La destinazione indicata è {}.', 'Segnato.', 'Scrivi la destinazione indicata.'),
        ('en', 'context-location'): ('The destination is {}.', 'Noted.', 'Write the indicated destination.'),
    }
    used = set(blocked_values)
    def fresh():
        while True:
            value = rng.choice(string.ascii_uppercase) + ''.join(rng.choice(string.ascii_lowercase) for _ in range(2))
            if value.casefold() not in used:
                used.add(value.casefold())
                return value
    rows = []
    for condition, fresh_value, new_template in [('known-values-new-template', False, True),
                                                  ('new-values-known-template', True, False),
                                                  ('new-values-new-template', True, True)]:
        for group in range(4):
            for alternate in range(2):
                source = train[group * 2 + alternate]
                value = fresh() if fresh_value else source['expected']
                if new_template:
                    statement, acknowledgement, question = templates[(source['language'], source['family'])]
                    messages = [dict(role='user', content=statement.format(value)),
                                dict(role='assistant', content=acknowledgement),
                                dict(role='user', content=question),
                                dict(role='assistant', content=value)]
                else:
                    messages = copy.deepcopy(source['messages'])
                    messages[0]['content'] = messages[0]['content'].replace(source['expected'], value)
                    messages[-1]['content'] = value
                rows.append(dict(messages=messages, expected=value, language=source['language'],
                    family=source['family'], pair_id=f'fresh-confirm:{condition}:{group}',
                    counterfactual_index=alternate, condition=condition,
                    provenance='deterministic explicit literal solver; paired fresh confirmation; endpoint only'))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/workspace/airi-lab/learnability-20261115'))
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out = args.output
    if (out / 'D_TRAINING_STARTED.json').exists():
        raise RuntimeError('one conditional D run only; no rerun/tuning')
    prior = json.loads((out / 'REPORT.json').read_text())
    if not prior.get('complete') or not prior.get('source_checkpoint_unchanged') or not prior.get('training_code_unchanged'):
        raise RuntimeError('original three-arm experiment and immutable source checks must finish first')
    for label in 'ABC':
        audit = json.loads((out / f'CACHE_ENDPOINT_{label}.json').read_text())
        if audit.get('passed') is not True:
            raise RuntimeError('independent original endpoint audits must pass first')
    if not any(trial['arm'].startswith('C_') and trial['train']['complete_with_eos'] == 8 for trial in prior['trials']):
        raise RuntimeError('conditional full-native learnability positive control absent')
    original_design = json.loads((out / 'PREREGISTRATION.json').read_text())
    original_rows = json.loads((out / 'ROWS.json').read_text())
    checkpoint = Path('/workspace/airi-lab/rev4/checkpoint/model')
    archive = Path('/workspace/airi-lab/rev4/experiment')
    reference_path = Path('/workspace/airi-lab/reference/generalist-state/bootstrap-data/candidate')
    for path, expected in {**original_design['source_hashes'], **original_design['code_hashes']}.items():
        if base.digest(path) != expected:
            raise RuntimeError('original immutable source changed before D: ' + path)
    import torch
    from torch.nn import functional as F
    from generalist_lm.runtime import GeneralistRuntime
    from generalist_lm.training import SFTExample, encode_sft_example, causal_training_objective
    from generalist_lm.tokenizer import EOS
    from generalist_lm.curriculum import train_rows
    from continual_learning import cache_tokenizer, continual_step
    from curriculum import norm, exact_correct, repetitive
    from success_bank import catalog, row_key, prompt_key
    from retention_memory import validate_memory, make_certifier
    from legacy_run import state_digest
    from prefix_tuning import configure_prefix_tuning
    from retention_guard import make_retention_guard
    from retention_transaction import transactional_update
    from fast_head import install_feature_cache
    from run import validate_training_registry
    torch.set_num_threads(2)
    torch.manual_seed(20261115)
    dataset = json.loads((archive / 'AUTHORED_CURRICULUM.json').read_text())
    def jsonl(name):
        return [json.loads(line) for line in (archive / name).read_text().splitlines()]
    human_dev_all = jsonl('human-development.jsonl')
    forbidden = dataset['development'] + dataset['sealed_transfer'] + human_dev_all + jsonl('human-final.jsonl')
    old_validation = jsonl('development.jsonl') + jsonl('test.jsonl')
    blocked = {norm(m['content']) for row in forbidden + old_validation for m in row['messages']}
    blocked |= {norm(p) for p in json.loads((archive / 'CONVERSATION_TEST.json').read_text())['prompts']}
    memory = json.loads((checkpoint.parent / 'retention_memory.json').read_text())
    registry = json.loads((checkpoint.parent / 'training_registry.json').read_text())
    registered = validate_training_registry(registry,
        [row for row in dataset['train'] if row['family'].startswith('context-')],
        dataset['context_acquisition_values'], forbidden, fingerprint=memory['fingerprint'], blocked=blocked,
        reserved_values={row['expected'] for row in forbidden},
        protected_histories={prompt_key(row) for row in forbidden + old_validation})
    eligible = catalog(dataset['train'] + registered, forbidden)
    old_rows = dataset['train'] + dataset['context_acquisition_values'] + registered + forbidden + old_validation
    blocked_values = {norm(row.get('expected', row['messages'][-1]['content'])) for row in old_rows}
    blocked_values |= {norm(row['expected']) for row in original_rows['train'] + original_rows['heldout']}
    train = original_rows['train']
    fresh = confirmation_rows(train, blocked_values)
    old_prompt_keys = {prompt_key(row) for row in old_rows + train + original_rows['heldout']}
    if len(fresh) != 24 or len({prompt_key(row) for row in fresh}) != 24:
        raise ValueError('confirmation must contain24 unique prompts')
    if any(prompt_key(row) in old_prompt_keys for row in fresh):
        raise ValueError('fresh confirmation prompt collides with old TRAIN/evaluation')
    for index in range(0, 24, 2):
        a, b = fresh[index:index + 2]
        if a['expected'] == b['expected'] or a['messages'][2] != b['messages'][2] or a['messages'][0] == b['messages'][0]:
            raise ValueError('confirmation lacks identical-question counterfactual pair')
    rows_d = dict(train=train, familiar_holdout=original_rows['heldout'], fresh_confirmation=fresh,
        train_policy='exact original8 TRAIN; no additional acquisition examples',
        confirmation_policy='24paired fresh rows,8 percondition; first forwarded only at fixed endpoint',
        fresh_seed=20261116)
    base.save_json(out / 'D_ROWS.json', rows_d)
    runtime = GeneralistRuntime.from_checkpoint(checkpoint)
    cache_tokenizer(runtime.tokenizer)
    for parameter in runtime.model.parameters():
        parameter.requires_grad_(True)
    if any(hasattr(runtime.model, attr) for attr in ('_airi_prefix_features', '_airi_head_features')):
        raise ValueError('full-native D must not install a feature cache')
    validate_memory(memory, eligible, memory['fingerprint'], state_digest(runtime.model))
    tokenizer = runtime.tokenizer
    encoded_cache = {}
    def encode(row):
        key = json.dumps(row['messages'], ensure_ascii=False, sort_keys=True)
        if key not in encoded_cache:
            ids, labels = encode_sft_example(SFTExample.from_dict(row), tokenizer, 1024, require_complete=True)
            width = int((labels != -100).nonzero()[:, 0].max()) + 1
            encoded_cache[key] = ids[:width].unsqueeze(0), labels[:width].unsqueeze(0)
        return encoded_cache[key]
    from context_acquisition import paired_batch
    anchors_by_hash = {}
    for native in train_rows():
        row = dict(messages=native.sft().messages)
        batch = encode(row)
        anchors_by_hash[hashlib.sha256(batch[0].numpy().tobytes()).hexdigest()] = batch
    trajectory = []
    for record in original_design['trajectory']:
        chosen = [row for row in train if row_key(row) in record['train_keys']]
        if [row_key(row) for row in chosen] != record['train_keys']:
            raise ValueError('D TRAIN pair order changed')
        batch = paired_batch([encode(row) for row in chosen])
        if hashlib.sha256(batch[0].numpy().tobytes()).hexdigest() != record['acquisition_ids_sha256']:
            raise ValueError('D input IDs differ from A')
        if hashlib.sha256(batch[1].numpy().tobytes()).hexdigest() != record['acquisition_labels_sha256']:
            raise ValueError('D SFT labels differ from A')
        trajectory.append((batch, [encode(eligible[key]) for key in record['retained_keys']],
            anchors_by_hash[record['anchor_ids_sha256']], [encode(eligible[key]) for key in record['guarded_keys']]))
    if len(trajectory) != 64:
        raise ValueError('requires exact64-unit A trajectory')
    for row in train + fresh + original_rows['heldout']:
        encode(row)
    source_hashes = dict(original_design['source_hashes'])
    code_hashes = dict(original_design['code_hashes'])
    code_hashes[str(Path(__file__).resolve())] = base.digest(__file__)
    code_hashes[str(Path(__file__).with_name('audit_native_chat.py'))] = base.digest(Path(__file__).with_name('audit_native_chat.py'))
    design = dict(scope=__doc__, arm='D_full_native_with_retention', seed=20261115,
        logical_units=64, hard_training_cap_seconds=600, trainable_parameters=sum(p.numel() for p in runtime.model.parameters()),
        source_checkpoint=str(checkpoint), original_design_sha256=base.digest(out / 'PREREGISTRATION.json'),
        original_report_sha256=base.digest(out / 'REPORT.json'), original_rows_sha256=base.digest(out / 'ROWS.json'),
        rows_d_sha256=base.digest(out / 'D_ROWS.json'), objective=original_design['retention'],
        acquisition_objective=original_design['acquisition_objective'], optimizer=original_design['optimizer'],
        trajectory=original_design['trajectory'], source_hashes=source_hashes, code_hashes=code_hashes,
        gradient_audit='Before any actual Adam step: one identical first TRAIN batch in both current scope and full scope; compare acquisition-only gradients to actual continual_step combined gradients via a collecting zero-update optimizer; restore gradients/modes/RNG and assert bitexact unchanged model state. No feedback/tuning.',
        endpoint_policy=['8 TRAIN rawcomplete answers withEOS', 'old12 holdouts familiar diagnostics',
            'fresh24 pairedholdouts endpointonly; frozenbaseline also endpointonly', 'original92+full897 regardless outcome',
            'original195authored development+first16human/language NLL fixed endpoint only if all23originalanswers and897TRAINcertificates survive; otherwise already fails promotion and NLL is explicitly unmeasured',
            'no sealedfinal forwards; no promotion; no further arms/LR tuning'],
        cap_policy='between indivisible logicalunits; update/guard walltime excludes readonlydiagnostics/endpointevaluation')
    base.save_json(out / 'D_PREREGISTRATION.json', design)
    print(json.dumps(dict(phase='D_preflight_ready', parameters=design['trainable_parameters'], fresh_pairs=12,
                         rows=str(out / 'D_ROWS.json'), preregistration=str(out / 'D_PREREGISTRATION.json'))), flush=True)
    if args.prepare_only:
        return
    base.save_json(out / 'D_TRAINING_STARTED.json', dict(preregistration_sha256=base.digest(out / 'D_PREREGISTRATION.json')))
    champion = GeneralistRuntime.from_checkpoint(checkpoint)
    reference = GeneralistRuntime.from_checkpoint(reference_path)
    for teacher in (champion, reference):
        for parameter in teacher.model.parameters():
            parameter.requires_grad_(False)
        teacher.model.eval()
        install_feature_cache(teacher.model, cache_bytes=64 * 1024 * 1024)
    def metrics(active, rows, greedy=False):
        active.model.eval()
        result = []
        total_nll, token_count = 0., 0
        with torch.no_grad():
            for row in rows:
                ids, labels = encode(row)
                targets = labels[:, 1:]; mask = targets != -100
                scores = active.model(ids)['logits'][:, :-1][mask].float()
                gold = targets[mask]
                nll = F.cross_entropy(scores, gold, reduction='none')
                probs = scores.softmax(-1).gather(1, gold[:, None]).squeeze(1)
                selected = scores.gather(1, gold[:, None]).squeeze(1)
                alternatives = scores.clone(); alternatives.scatter_(1, gold[:, None], -float('inf'))
                matches = scores.argmax(-1) == gold
                measured = dict(key=row_key(row), expected=row['expected'], language=row['language'], family=row['family'],
                    condition=row.get('condition'), pair_id=row.get('pair_id'), gold_ids=gold.tolist(),
                    token_count=int(gold.numel()), token_correct=int(matches.sum()), complete_with_eos=bool(matches.all()),
                    raw_ce=float(nll.mean()), gold_token_probabilities=probs.tolist(), first_gold_probability=float(probs[0]),
                    sequence_probability=float((-nll.sum()).exp()), min_margin=float((selected - alternatives.max(-1).values).min()))
                if greedy:
                    measured['native_greedy'] = raw_answer(active, row, 16)
                result.append(measured); total_nll += float(nll.sum()); token_count += int(gold.numel())
        return dict(raw_ce=total_nll / token_count, target_tokens=token_count,
            complete_with_eos=sum(x['complete_with_eos'] for x in result), token_correct=sum(x['token_correct'] for x in result), rows=result)
    def raw_answer(active, row, budget=64):
        prompt = active.tokenizer.serialize_messages(row['messages'][:-1], add_generation_prompt=True)
        ids = active._generate_ids(prompt, max_new_tokens=budget, temperature=0)
        gold = active.tokenizer.encode(row['expected'], eos=True)
        response = active.tokenizer.decode(ids)
        return dict(expected=row['expected'], response=response, generated_ids=ids, gold_ids=gold,
            eos_emitted=bool(ids and ids[-1] == EOS), raw_exact=ids == gold, correct=exact_correct(response, row['expected']))
    def gradient_audit(active):
        parameters = [(name, p) for name, p in active.model.named_parameters() if p.requires_grad]
        saved_modes = [(module, module.training) for module in active.model.modules()]
        saved_grads = [None if p.grad is None else p.grad.detach().clone() for _, p in parameters]
        saved_rng = torch.get_rng_state().clone()
        before_state = state_digest(active.model)
        batch, retained, anchor, _ = trajectory[0]
        try:
            active.model.train(); active.model.zero_grad(set_to_none=True)
            logits = active.model(batch[0])['logits']
            acquisition_loss, _ = causal_training_objective(logits, batch[1], batch[0], eos_loss_weight=1.2,
                                                            repetition_unlikelihood_weight=.05)
            acquisition_loss.backward()
            acquisition_grads = [None if p.grad is None else p.grad.detach().clone() for _, p in parameters]
            acquisition_norm = sum(float(g.double().square().sum()) for g in acquisition_grads if g is not None) ** .5
            class CollectOnly:
                def zero_grad(self, set_to_none=True):
                    active.model.zero_grad(set_to_none=set_to_none)
                def step(self):
                    pass  # Deliberate measurement callback: no weights or optimizer state exist here.
            stats = continual_step(active.model, champion.model, reference.model, CollectOnly(),
                batch, retained, anchor, retention_weight=8., champion_kl_weight=8., native_weight=.25, prompt_kl_weight=8.)
            dot = 0.; combined_sq = 0.
            for old, (_, p) in zip(acquisition_grads, parameters):
                if p.grad is not None:
                    combined_sq += float(p.grad.double().square().sum())
                    if old is not None:
                        dot += float((old.double() * p.grad.double()).sum())
            combined_norm_clipped = combined_sq ** .5
            cosine = dot / (acquisition_norm * combined_norm_clipped) if acquisition_norm * combined_norm_clipped else None
            result = dict(acquisition_loss=float(acquisition_loss.detach()), acquisition_gradient_norm=acquisition_norm,
                combined_gradient_norm_before_clipping=stats['gradient_norm_before_clipping'],
                combined_gradient_norm_after_clipping=combined_norm_clipped, acquisition_combined_gradient_cosine=cosine,
                collection_only_no_optimizer_update=True, combined_loss_statistics=stats)
        finally:
            for (_, p), grad in zip(parameters, saved_grads):
                p.grad = None if grad is None else grad.clone()
            for module, training in saved_modes:
                module.training = training
            torch.set_rng_state(saved_rng)
        if state_digest(active.model) != before_state:
            raise AssertionError('readonly gradient audit changed model weights')
        result.update(model_weights_exactly_unchanged=True, rng_exactly_restored=bool(torch.equal(saved_rng, torch.get_rng_state())),
                      gradients_restored=True, module_modes_restored=True)
        return result
    before_diagnostics_rng = torch.get_rng_state().clone()
    gradient_diagnostics = dict(full_native=gradient_audit(runtime))
    current = GeneralistRuntime.from_checkpoint(checkpoint)
    configure_prefix_tuning(current.model)
    gradient_diagnostics['current_scope'] = gradient_audit(current)
    del current
    gc.collect()
    torch.set_rng_state(before_diagnostics_rng)
    initial = {name: p.detach().clone() for name, p in runtime.model.named_parameters()}
    parameters = list(runtime.model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=3e-5, weight_decay=.01)
    state = dict(schema=1, lr_scale=1., attempted_updates=0, committed_updates=0, rejected_updates=0)
    report = dict(preregistration_sha256=base.digest(out / 'D_PREREGISTRATION.json'), complete=False,
        gradient_diagnostics=gradient_diagnostics, baseline_train=original_rows['baseline_train'],
        production_qualified=False, live_promoted=False, sealed_benchmark_forward=False)
    base.save_json(out / 'D_REPORT.json', report)
    attempts, logical, progress = [], [], []
    training_seconds = 0.
    for step, (batch, retained, anchor, guards) in enumerate(trajectory, 1):
        if training_seconds >= 600:
            break
        started = time.monotonic()
        guard = make_retention_guard(runtime.model, champion.model, guards, max_prompt_kl=.01, max_nll_regression=.02)
        def propose():
            stats = continual_step(runtime.model, champion.model, reference.model, optimizer,
                batch, retained, anchor, retention_weight=8., champion_kl_weight=8., native_weight=.25, prompt_kl_weight=8.)
            attempts.append(dict(unit=step, **stats))
            return stats
        outcome = transactional_update(runtime.model, optimizer, propose, guard, state=state,
            scales=(1., .5, .25, .125), recovery=original_design['retention']['recovery'])
        training_seconds += time.monotonic() - started
        logical.append(dict(unit=step, **outcome))
        if step % 16 == 0:
            measured = metrics(runtime, train)
            progress.append(dict(unit=step, **measured))
            report.update(training_seconds=training_seconds, logical_units=len(logical), transaction=state,
                          progress_train_only=progress, logical_updates=logical, attempt_statistics=attempts)
            base.save_json(out / 'D_REPORT.json', report)
            print(json.dumps(dict(phase='D_training', unit=step, training_seconds=training_seconds,
                train_ce=measured['raw_ce'], train_complete=measured['complete_with_eos'], state=state)), flush=True)
    final_train = metrics(runtime, train, True)
    familiar = metrics(runtime, original_rows['heldout'], True)
    confirmation = metrics(runtime, fresh, True)
    baseline_confirmation = metrics(champion, fresh, True)
    destination = out / 'D_full_native_with_retention-candidate'
    runtime.save_checkpoint(destination, metadata=dict(role='isolated-full-native-protected-learnability-diagnostic',
        production_qualified=False, live_promoted=False, preregistration_sha256=report['preregistration_sha256']))
    exported = GeneralistRuntime.from_checkpoint(destination)
    runtime.model.eval(); exported.model.eval()
    with torch.no_grad():
        logits_exact = all(torch.equal(runtime.model(encode(row)[0])['logits'], exported.model(encode(row)[0])['logits']) for row in train)
        greedy_exact = all(raw_answer(exported, row, 16) == observed['native_greedy'] for row, observed in zip(train, final_train['rows']))
    if not logits_exact or not greedy_exact:
        raise AssertionError('D native export/reload logits or rawIDs/EOS changed')
    delta_sq, max_delta, changed = 0., 0., 0
    for name, p in runtime.model.named_parameters():
        delta = p.detach() - initial[name]
        delta_sq += float(delta.double().square().sum())
        max_delta = max(max_delta, float(delta.abs().max())); changed += int(bool((delta != 0).any()))
    del initial, exported, optimizer, parameters
    gc.collect()
    norms = [x['gradient_norm_before_clipping'] for x in attempts]
    report.update(training_seconds=training_seconds, logical_units=len(logical), transaction=state,
        physical_attempts=len(attempts), exposure='complete64' if len(logical) == 64 else 'hard-wallcap-truncated-prefix',
        train=final_train, familiar_holdout=familiar, fresh_confirmation=confirmation, baseline_fresh_confirmation=baseline_confirmation,
        fresh_confirmation_gained=[index for index, (a, b) in enumerate(zip(baseline_confirmation['rows'], confirmation['rows'])) if not a['native_greedy']['raw_exact'] and b['native_greedy']['raw_exact']],
        fresh_confirmation_lost=[index for index, (a, b) in enumerate(zip(baseline_confirmation['rows'], confirmation['rows'])) if a['native_greedy']['raw_exact'] and not b['native_greedy']['raw_exact']],
        parameter_delta=dict(l2=delta_sq ** .5, max_abs=max_delta, changed_tensors=changed),
        gradient_norm=dict(mean=sum(norms) / len(norms), minimum=min(norms), maximum=max(norms), clipped_fraction=sum(x > 1 for x in norms) / len(norms)),
        physical_supervised_tokens=sum(x['supervised_tokens'] for x in attempts),
        physical_reference_tokens=sum(x['reference_tokens'] for x in attempts),
        candidate=str(destination), native_export_logits_exact=logits_exact, native_export_greedy_ids_and_eos_exact=greedy_exact,
        logical_updates=logical, attempt_statistics=attempts, progress_train_only=progress)
    base.save_json(out / 'D_REPORT.json', report)
    print(json.dumps(dict(phase='D_endpoint', train_complete=final_train['complete_with_eos'],
        familiar_complete=familiar['complete_with_eos'], fresh_complete=confirmation['complete_with_eos'],
        logical_units=len(logical), training_seconds=training_seconds)), flush=True)
    old_samples = prior['baseline']['samples']
    samples = []
    with torch.no_grad():
        for old in old_samples:
            row = dict(messages=old['messages'] + [dict(role='assistant', content=old['expected'])], expected=old['expected'])
            samples.append({**old, **raw_answer(runtime, row)})
    report['original92'] = dict(exact_correct=sum(x['correct'] for x in samples), probes=92,
        repetition=sum(repetitive(x['response']) for x in samples) / 92, samples=samples)
    report['original23_lost'] = [index for index, (a, b) in enumerate(zip(old_samples, samples)) if a['correct'] and not b['correct']]
    report['original92_gained'] = [index for index, (a, b) in enumerate(zip(old_samples, samples)) if not a['correct'] and b['correct']]
    certifier = make_certifier(runtime.model, encode, EOS)
    report['retention897_certificates'] = {key: certifier(entry['row']) for key, entry in memory['entries'].items()}
    report['lost_retention897_keys'] = [key for key, value in report['retention897_certificates'].items() if not value['correct']]
    base.save_json(out / 'D_REPORT.json', report)
    if report['original23_lost'] or report['lost_retention897_keys']:
        report.update(original_nll_gates_evaluated=False,
            nll_skipped_reason='already fails original23 and/or full897 retention; no promotion claim; fixed cost-saving rule',
            measured_retention_gates_pass=False)
        for path, expected in {**source_hashes, **code_hashes}.items():
            if base.digest(path) != expected:
                raise AssertionError('immutable source changed during D: ' + path)
        report.update(source_checkpoint_unchanged=True, training_code_unchanged=True, complete=True)
        base.save_json(out / 'D_REPORT.json', report)
        print(json.dumps(dict(phase='D_complete', original92=report['original92']['exact_correct'],
            original23_lost=len(report['original23_lost']), retention897_lost=len(report['lost_retention897_keys']),
            original_nll_gates_evaluated=False, measured_retention_gates_pass=False)), flush=True)
        return
    # Exact original authored/human development NLL gates, evaluated only after
    # the full fixed D trajectory; source answers never enter the optimizer.
    human_dev = [row for language in ('it', 'en') for row in [r for r in human_dev_all if r['language'] == language][:16]]
    def grouped_nll(active, rows):
        active.model.eval(); result = {}
        with torch.no_grad():
            for language in ('it', 'en'):
                total, count = 0., 0
                for row in rows:
                    if row['language'] != language:
                        continue
                    ids, labels = encode(row)
                    logits = active.model(ids)['logits'][:, :-1]
                    total += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), reduction='sum'))
                    count += int((labels[:, 1:] != -100).sum())
                result[language] = total / count
        return result
    source_nll = dict(nll=grouped_nll(champion, dataset['development']), human_nll=grouped_nll(champion, human_dev))
    saved_champion = torch.load(checkpoint.parent / 'resume.pt', weights_only=True)['counters']['champion']
    for group in ('nll', 'human_nll'):
        for language, measured in source_nll[group].items():
            if abs(measured - saved_champion[group][language]) > 1e-5:
                raise AssertionError('source original NLL gate failed reproduction')
    endpoint_nll = dict(nll=grouped_nll(runtime, dataset['development']), human_nll=grouped_nll(runtime, human_dev))
    nll_failures = [f'{group}:{language}' for group in ('nll', 'human_nll') for language in ('it', 'en')
        if not math.isfinite(endpoint_nll[group][language]) or endpoint_nll[group][language] <= 0 or endpoint_nll[group][language] > source_nll[group][language] * 1.02]
    report.update(original_nll_gates_evaluated=True, original_source_nll=source_nll, original_endpoint_nll=endpoint_nll, nll_regression_failures=nll_failures,
        measured_retention_gates_pass=not report['original23_lost'] and not report['lost_retention897_keys'] and not nll_failures)
    for path, expected in {**source_hashes, **code_hashes}.items():
        if base.digest(path) != expected:
            raise AssertionError('immutable source changed during D: ' + path)
    report.update(source_checkpoint_unchanged=True, training_code_unchanged=True, complete=True)
    base.save_json(out / 'D_REPORT.json', report)
    print(json.dumps(dict(phase='D_complete', original92=report['original92']['exact_correct'],
        original23_lost=len(report['original23_lost']), retention897_lost=len(report['lost_retention897_keys']),
        nll_regression_failures=nll_failures, measured_retention_gates_pass=report['measured_retention_gates_pass'])), flush=True)


if __name__ == '__main__':
    main()
