"""Predeclared tiny native learnability diagnostic; never qualifies/promotes weights.

The combined production retention package, acquisition-only current scope, and
acquisition-only full-native scope share fresh AdamW and an identical fixed TRAIN
trajectory. A finite low-LR overfit test cannot establish an architectural limit.
"""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/transfer-training'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def make_rows():
    """Deterministic labels written before any forward; no model-derived gold."""
    train_values = [('Qyx', 'Zup'), ('Woq', 'Xej'), ('Vux', 'Jaq'), ('Kez', 'Fyv')]
    new_values = [('Tov', 'Piq'), ('Ruz', 'Bex'), ('Daq', 'Giv'), ('Hoz', 'Luq')]
    groups = [('it', 'name'), ('en', 'name'), ('it', 'location'), ('en', 'location')]
    templates = {
        ('it', 'name', 0): ('Nome: {}.', 'Ricevuto.', 'Quale nome?'),
        ('en', 'name', 0): ('Name: {}.', 'Recorded.', 'What name?'),
        ('it', 'location', 0): ('Luogo: {}.', 'Ricevuto.', 'Quale luogo?'),
        ('en', 'location', 0): ('Place: {}.', 'Recorded.', 'Which place?'),
        ('it', 'name', 1): ('Il nominativo indicato è {}.', 'Va bene.', 'Riporta il nominativo.'),
        ('en', 'name', 1): ('The person uses the name {}.', 'Understood.', 'Give the person\'s name.'),
        ('it', 'location', 1): ('Il pacco si trova a {}.', 'Va bene.', 'Dove si trova il pacco?'),
        ('en', 'location', 1): ('The parcel is located at {}.', 'Understood.', 'Where is the parcel located?'),
    }
    def row(index, value, template, condition):
        language, family = groups[index]
        statement, acknowledge, question = templates[(language, family, template)]
        return dict(messages=[dict(role='user', content=statement.format(value)),
                              dict(role='assistant', content=acknowledge),
                              dict(role='user', content=question),
                              dict(role='assistant', content=value)], expected=value,
                    language=language, family='context-' + family, pair_id=f'tiny-{index}',
                    condition=condition, template=template,
                    provenance='deterministic explicit literal TRAIN solver; no student/evaluation labels')
    train = [row(index, value, 0, 'train') for index, values in enumerate(train_values) for value in values]
    heldout = []
    for condition, fresh_value, template in [('known-values-new-template', False, 1),
                                             ('new-values-known-template', True, 0),
                                             ('new-values-new-template', True, 1)]:
        for index in range(4):
            # New-value conditions use different values from each other as well.
            value = new_values[index][int(template == 1)] if fresh_value else train_values[index][0]
            heldout.append(row(index, value, template, condition))
    return train, heldout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/workspace/airi-lab/learnability-20261115'))
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'TRAINING_STARTED.json').exists() and not args.prepare_only:
        raise RuntimeError('single predeclared run already started; do not rerun or tune')
    import torch
    from torch.nn import functional as F
    from generalist_lm.runtime import GeneralistRuntime, checkpoint_model_files
    from generalist_lm.training import SFTExample, encode_sft_example, causal_training_objective
    from generalist_lm.tokenizer import EOS
    from generalist_lm.curriculum import train_rows
    from generalist_lm.phase5_diagnostics import protected_bootstrap_texts
    from continual_learning import cache_tokenizer, continual_step
    from curriculum import norm, exact_correct, repetitive
    from success_bank import catalog, prompt_key, row_key
    from retention_memory import validate_memory, sample_rows, make_certifier
    from run import validate_training_registry
    from legacy_run import state_digest
    from prefix_tuning import configure_prefix_tuning, prefix_summary, native_evaluation
    from retention_guard import make_retention_guard
    from retention_transaction import transactional_update
    from fast_head import install_feature_cache

    seed = 20261115
    steps = 64
    checkpoint = Path('/workspace/airi-lab/rev4/checkpoint/model')
    reference_path = Path('/workspace/airi-lab/reference/generalist-state/bootstrap-data/candidate')
    archive = Path('/workspace/airi-lab/rev4/experiment')
    native_root = Path('/workspace/airi-lab/native-source/computer/generalist_lm')
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    source_paths = list(checkpoint_model_files(checkpoint, verify=True))
    source_paths += [p for p in checkpoint.glob('*.json')]
    source_paths += list(checkpoint_model_files(reference_path, verify=True))
    source_paths += [p for p in reference_path.glob('*.json')]
    source_paths += [checkpoint.parent / name for name in ('resume.pt', 'retention_memory.json', 'training_registry.json')]
    source_paths += [archive / name for name in ('AUTHORED_CURRICULUM.json', 'SPEC.json', 'human-development.jsonl',
                     'human-final.jsonl', 'development.jsonl', 'test.jsonl', 'CONVERSATION_TEST.json')]
    code_paths = list((ROOT / 'scripts/transfer-training').glob('*.py'))
    code_paths += list(native_root.glob('*.py'))
    code_paths += [Path(__file__), ROOT / 'scripts/research-training/ffn_ablation.py']
    source_hashes = {str(p): digest(p) for p in source_paths}
    code_hashes = {str(p): digest(p) for p in code_paths}
    champion = GeneralistRuntime.from_checkpoint(checkpoint)
    cache_tokenizer(champion.tokenizer)
    for parameter in champion.model.parameters():
        parameter.requires_grad_(False)
    champion.model.eval()
    install_feature_cache(champion.model, cache_bytes=64 * 1024 * 1024)
    dataset = json.loads((archive / 'AUTHORED_CURRICULUM.json').read_text())
    def jsonl(name):
        return [json.loads(line) for line in (archive / name).read_text().splitlines()]
    forbidden = dataset['development'] + dataset['sealed_transfer'] + jsonl('human-development.jsonl') + jsonl('human-final.jsonl')
    old_validation = jsonl('development.jsonl') + jsonl('test.jsonl')
    memory = json.loads((checkpoint.parent / 'retention_memory.json').read_text())
    registry = json.loads((checkpoint.parent / 'training_registry.json').read_text())
    blocked = {norm(m['content']) for row in old_validation + forbidden for m in row['messages']}
    blocked |= {norm(p) for p in json.loads((archive / 'CONVERSATION_TEST.json').read_text())['prompts']}
    registered = validate_training_registry(registry,
        [row for row in dataset['train'] if row['family'].startswith('context-')],
        dataset['context_acquisition_values'], forbidden, fingerprint=memory['fingerprint'],
        blocked=blocked, reserved_values={row['expected'] for row in forbidden},
        protected_histories={prompt_key(row) for row in forbidden + old_validation})
    eligible = catalog(dataset['train'] + registered, forbidden)
    validate_memory(memory, eligible, memory['fingerprint'], state_digest(champion.model))
    if len(memory['entries']) != 897:
        raise ValueError('approved source must contain 897 eligible TRAIN certificates')
    encoded_cache = {}
    def encode(row):
        key = json.dumps(row['messages'], ensure_ascii=False, sort_keys=True)
        if key not in encoded_cache:
            ids, labels = encode_sft_example(SFTExample.from_dict(row), champion.tokenizer, 1024, require_complete=True)
            width = int((labels != -100).nonzero()[:, 0].max()) + 1
            encoded_cache[key] = ids[:width].unsqueeze(0), labels[:width].unsqueeze(0)
        return encoded_cache[key]

    train, heldout = make_rows()
    old_rows = dataset['train'] + dataset['context_acquisition_values'] + registered + forbidden + old_validation
    old_prompts = {prompt_key(row) for row in old_rows}
    old_values = {norm(row.get('expected', row['messages'][-1]['content'])) for row in old_rows}
    if any(prompt_key(row) in old_prompts for row in train + heldout):
        raise ValueError('fresh diagnostic prompt collides with existing source')
    fresh_values = {norm(row['expected']) for row in train + heldout}
    if fresh_values & old_values:
        raise ValueError('fresh literal value collides with existing TRAIN/heldout value')
    if {prompt_key(row) for row in train} & {prompt_key(row) for row in heldout}:
        raise ValueError('TRAIN and endpoint-only holdout prompts overlap')
    train_keys = {row_key(row) for row in train}
    if len(train_keys) != 8:
        raise ValueError('requires exactly eight unique TRAIN rows')
    for row in train + heldout:
        encode(row)

    def raw_answer(runtime, row, max_new_tokens=64):
        ids = runtime.tokenizer.serialize_messages(row['messages'][:-1], add_generation_prompt=True)
        generated = runtime._generate_ids(ids, max_new_tokens=max_new_tokens, temperature=0)
        gold = runtime.tokenizer.encode(row['expected'], eos=True)
        response = runtime.tokenizer.decode(generated)
        return dict(expected=row['expected'], response=response, generated_ids=generated, gold_ids=gold,
                    eos_emitted=bool(generated and generated[-1] == EOS), raw_exact=generated == gold,
                    correct=exact_correct(response, row['expected']))

    def metrics(runtime, rows, greedy=False):
        runtime.model.eval()
        observations = []
        nll_sum, tokens = 0., 0
        with torch.no_grad():
            for row in rows:
                ids, labels = encode(row)
                targets = labels[:, 1:]
                mask = targets != -100
                scores = runtime.model(ids)['logits'][:, :-1][mask].float()
                gold = targets[mask]
                token_nll = F.cross_entropy(scores, gold, reduction='none')
                target_probs = scores.softmax(-1).gather(1, gold[:, None]).squeeze(1)
                selected = scores.gather(1, gold[:, None]).squeeze(1)
                alternatives = scores.clone()
                alternatives.scatter_(1, gold[:, None], -float('inf'))
                matches = scores.argmax(-1) == gold
                observation = dict(key=row_key(row), language=row['language'], family=row['family'],
                    condition=row.get('condition'), expected=row['expected'], gold_ids=gold.tolist(),
                    token_count=int(gold.numel()), token_correct=int(matches.sum()),
                    complete_with_eos=bool(matches.all()), raw_ce=float(token_nll.mean()),
                    gold_token_probabilities=target_probs.tolist(),
                    first_gold_probability=float(target_probs[0]), sequence_probability=float((-token_nll.sum()).exp()),
                    min_margin=float((selected - alternatives.max(-1).values).min()))
                if greedy:
                    observation['native_greedy'] = raw_answer(runtime, row, max_new_tokens=16)
                observations.append(observation)
                nll_sum += float(token_nll.sum()); tokens += int(gold.numel())
        return dict(raw_ce=nll_sum / tokens, target_tokens=tokens,
                    token_correct=sum(row['token_correct'] for row in observations),
                    complete_with_eos=sum(row['complete_with_eos'] for row in observations),
                    rows=observations)

    before_train = metrics(champion, train, greedy=True)
    if before_train['complete_with_eos'] or any(row['native_greedy']['correct'] or row['native_greedy']['raw_exact'] for row in before_train['rows']):
        raise ValueError('fixed diagnostic set includes initially correct TRAIN row; preflight rejects before training')
    rows_payload = dict(train=train, heldout=heldout,
        holdout_policy='fixed before training; endpoint only; no optimizer or intermediate forwards',
        baseline_train=before_train)
    save_json(args.output / 'ROWS.json', rows_payload)

    # Short native anchors are TRAIN-only; exact original reference objective.
    anchors = [dict(messages=row.sft().messages) for row in train_rows()]
    anchor_blocked = blocked | protected_bootstrap_texts()
    anchors = [row for row in anchors if encode(row)[0].shape[1] <= 128 and
               not any(norm(m['content']) in anchor_blocked for m in row['messages'])]
    if not anchors:
        raise ValueError('no short unprotected native TRAIN anchors')
    sampled_memory = copy.deepcopy(memory)
    retention_rng = random.Random(seed + 1)
    selector = random.Random(seed)
    trajectory = []
    manifest = []
    from context_acquisition import paired_batch
    for step in range(steps):
        pair_index = step % 4
        chosen = train[pair_index * 2:pair_index * 2 + 2]
        retained = sample_rows(sampled_memory, eligible, retention_rng, per_language=2)
        guarded = sample_rows(sampled_memory, eligible, retention_rng, per_language=4)
        anchor = selector.choice(anchors)
        batch = paired_batch([encode(row) for row in chosen])
        trajectory.append((batch, [encode(row) for row in retained], encode(anchor), [encode(row) for row in guarded]))
        manifest.append(dict(step=step + 1, pair_index=pair_index, train_keys=[row_key(row) for row in chosen],
            acquisition_ids_sha256=hashlib.sha256(batch[0].numpy().tobytes()).hexdigest(),
            acquisition_labels_sha256=hashlib.sha256(batch[1].numpy().tobytes()).hexdigest(),
            retained_keys=[row_key(row) for row in retained], guarded_keys=[row_key(row) for row in guarded],
            anchor_ids_sha256=hashlib.sha256(encode(anchor)[0].numpy().tobytes()).hexdigest()))
    design = dict(scope=__doc__, seed=seed, logical_units=steps,
        arms=[dict(name='A_current_with_retention', cap_seconds=240, scope='final attention/norm/head', trainable_parameters=74304),
              dict(name='B_current_acquisition_only', cap_seconds=60, scope='final attention/norm/head', trainable_parameters=74304),
              dict(name='C_full_native_acquisition_only', cap_seconds=600, scope='all native parameters')],
        optimizer=dict(name='fresh AdamW per arm', learning_rate=3e-5, weight_decay=.01, gradient_clipping=1.),
        acquisition_objective=dict(eos_loss_weight=1.2, repetition_unlikelihood_weight=.05,
            comparison='identical native acquisition objective in all arms; raw shifted CE reported separately'),
        retention=dict(retention_weight=8., champion_kl_weight=8., prompt_kl_weight=8., native_weight=.25,
            max_prompt_kl=.01, max_nll_regression=.02, retry_scales=[1., .5, .25, .125],
            recovery=dict(accepted_streak=64, cooldown_updates=32, growth_factor=2.),
            package='unaltered continual_step + make_retention_guard + transactional_update'),
        rules=['one predeclared run; no LR tuning or checkpoint selection',
            'only TRAIN metrics at units 16/32/48/64', 'heldout matrix first forwarded only at final endpoint',
            'training-only fixed sentinel/anchor sampler; no evaluation labels in optimizer',
            'hard caps checked between units; last indivisible unit may overrun cap',
            'caps measure update/guard wall time, excluding independent endpoint evaluation',
            'full897 endpoint certification if any arm reaches6/8 TRAIN or any newly correct holdout answer vs frozen source',
            'overfit is not generalization; finite fixed-LR failure is not architectural impossibility',
            'no promotion/live jobs/source changes'],
        source_hashes=source_hashes, code_hashes=code_hashes,
        rows_sha256=digest(args.output / 'ROWS.json'), trajectory=manifest,
        source_memory_entries=897, baseline_train=before_train, torch_version=torch.__version__)
    save_json(args.output / 'PREREGISTRATION.json', design)
    print(json.dumps(dict(phase='preflight_ready', rows=str(args.output / 'ROWS.json'),
                         baseline_train_ce=before_train['raw_ce'], baseline_train_complete=0,
                         token_lengths=[encode(row)[0].shape[1] for row in train])), flush=True)
    if args.prepare_only:
        return
    # Independent reviewer can run cache-on/off checks after preparation and
    # before this single run is invoked. Do not change sources after this line.
    audit_path = args.output / 'CACHE_PREFLIGHT.json'
    if not audit_path.exists() or json.loads(audit_path.read_text()).get('passed') is not True:
        raise RuntimeError('independent baseline cache-on/off native preflight required')
    saved_baseline = torch.load(checkpoint.parent / 'resume.pt', weights_only=True)['counters']['champion']

    def original_answers(runtime):
        runtime.model.eval()
        samples = []
        with torch.no_grad():
            for old in saved_baseline['samples']:
                answer = raw_answer(runtime, dict(messages=old['messages'] + [dict(role='assistant', content=old['expected'])],
                                                 expected=old['expected']))
                samples.append({**old, **answer})
        return dict(exact_correct=sum(s['correct'] for s in samples), probes=len(samples),
                    raw_exact=sum(s['raw_exact'] for s in samples),
                    repetition=sum(repetitive(s['response']) for s in samples) / len(samples), samples=samples)

    baseline = original_answers(champion)
    if baseline['exact_correct'] != 23 or len(baseline['samples']) != 92:
        raise ValueError('original23/92 native source reproduction failed')
    for actual, saved in zip(baseline['samples'], saved_baseline['samples']):
        if (actual['response'], actual['correct']) != (saved['response'], saved['correct']):
            raise ValueError('original92 response reproduction differs from archived harness')
    reference = GeneralistRuntime.from_checkpoint(reference_path)
    if digest(reference_path / 'model.pt') != '998530839d9722316b59cb28ff1ef68a7863f517de73fa8fef8ce9677a24b85e':
        raise ValueError('original native teacher changed')
    for parameter in reference.model.parameters():
        parameter.requires_grad_(False)
    reference.model.eval()
    install_feature_cache(reference.model, cache_bytes=64 * 1024 * 1024)
    report = dict(preregistration_sha256=digest(args.output / 'PREREGISTRATION.json'), baseline=baseline,
                  baseline_train=before_train, trials=[], complete=False, production_qualified=False,
                  live_promoted=False, sealed_benchmark_forward=False)
    save_json(args.output / 'TRAINING_STARTED.json', dict(preregistration_sha256=report['preregistration_sha256']))
    save_json(args.output / 'REPORT.json', report)
    for arm in design['arms']:
        arm_name = arm['name']
        torch.manual_seed(seed)
        runtime = GeneralistRuntime.from_checkpoint(checkpoint)
        cache_tokenizer(runtime.tokenizer)
        if arm_name.startswith(('A_', 'B_')):
            configure_prefix_tuning(runtime.model)
            strategy = prefix_summary(runtime.model)
            if strategy['trainable_parameters'] != 74304:
                raise ValueError('current scope changed')
        else:
            for parameter in runtime.model.parameters():
                parameter.requires_grad_(True)
            if any(hasattr(runtime.model, attr) for attr in ('_airi_prefix_features', '_airi_head_features')):
                raise ValueError('full-native arm must not install a feature cache')
            strategy = dict(mode='full_native_no_feature_cache',
                trainable_parameters=sum(p.numel() for p in runtime.model.parameters()),
                trainable_names=[name for name, p in runtime.model.named_parameters() if p.requires_grad])
        parameters = [p for p in runtime.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=3e-5, weight_decay=.01)
        initial = {name: p.detach().clone() for name, p in runtime.model.named_parameters() if p.requires_grad}
        state = dict(schema=1, lr_scale=1., attempted_updates=0, committed_updates=0, rejected_updates=0)
        logical, attempts, progress = [], [], [dict(unit=0, **before_train)]
        training_seconds = 0.
        for step, (batch, retained, anchor, guards) in enumerate(trajectory, 1):
            if training_seconds >= arm['cap_seconds']:
                break
            started = time.monotonic()
            if arm_name.startswith('A_'):
                guard = make_retention_guard(runtime.model, champion.model, guards,
                    max_prompt_kl=.01, max_nll_regression=.02)
                def propose():
                    stats = continual_step(runtime.model, champion.model, reference.model, optimizer,
                        batch, retained, anchor, retention_weight=8., champion_kl_weight=8., native_weight=.25,
                        prompt_kl_weight=8.)
                    attempts.append(dict(unit=step, **stats))
                    return stats
                outcome = transactional_update(runtime.model, optimizer, propose, guard, state=state,
                    scales=(1., .5, .25, .125), recovery=design['retention']['recovery'])
            else:
                runtime.model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = runtime.model(batch[0])['logits']
                loss, stats = causal_training_objective(logits, batch[1], batch[0],
                    eos_loss_weight=1.2, repetition_unlikelihood_weight=.05)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('nonfinite acquisition-only objective')
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                stats.update(loss=float(loss.detach()), gradient_norm_before_clipping=float(gradient_norm),
                             supervised_tokens=int((batch[1][:, 1:] != -100).sum()), reference_tokens=0)
                attempts.append(dict(unit=step, **stats))
                state['attempted_updates'] += 1; state['committed_updates'] += 1
                outcome = dict(accepted=True, attempts=1, scale=1., stats=stats, guard=None)
            training_seconds += time.monotonic() - started
            logical.append(dict(unit=step, **outcome))
            if step % 16 == 0:
                measured = metrics(runtime, train)
                progress.append(dict(unit=step, **measured))
                print(json.dumps(dict(phase='training', arm=arm_name, unit=step, training_seconds=training_seconds,
                                     train_ce=measured['raw_ce'], train_complete=measured['complete_with_eos'], state=state)), flush=True)
        final_train = metrics(runtime, train, greedy=True)
        if progress[-1]['unit'] != len(logical):
            progress.append(dict(unit=len(logical), **final_train))
        final_heldout = metrics(runtime, heldout, greedy=True)
        after = original_answers(runtime)
        delta_squared, max_delta, changed = 0., 0., 0
        for name, parameter in runtime.model.named_parameters():
            if name in initial:
                delta = parameter.detach() - initial[name]
                delta_squared += float(delta.double().square().sum())
                max_delta = max(max_delta, float(delta.abs().max()))
                changed += int(bool((delta != 0).any()))
        del initial
        destination = args.output / (arm_name + '-candidate')
        runtime.save_checkpoint(destination, metadata=dict(production_qualified=False, live_promoted=False,
            role='isolated-tiny-learnability-diagnostic', preregistration_sha256=report['preregistration_sha256']))
        exported = GeneralistRuntime.from_checkpoint(destination)
        runtime.model.eval(); exported.model.eval()
        with torch.no_grad():
            logits_exact = all(torch.equal(runtime.model(encode(row)[0])['logits'], exported.model(encode(row)[0])['logits']) for row in train)
        # Prefix feature computation can differ in floating point by BLAS batch
        # geometry. Exact native-vs-native serialization is the required check.
        if hasattr(runtime.model, '_airi_prefix_features'):
            with native_evaluation(runtime.model), torch.no_grad():
                native_logits_exact = all(torch.equal(runtime.model(encode(row)[0])['logits'], exported.model(encode(row)[0])['logits']) for row in train)
        else:
            native_logits_exact = logits_exact
        if not native_logits_exact:
            raise AssertionError('native export/reload changed exact native logits')
        export_greedy = [raw_answer(exported, row, max_new_tokens=16) for row in train]
        greedy_exact = all(a == b['native_greedy'] for a, b in zip(export_greedy, final_train['rows']))
        if not greedy_exact:
            raise AssertionError('native export/reload changed TRAIN raw greedy IDs/EOS')
        norms = [item['gradient_norm_before_clipping'] for item in attempts]
        trial = dict(arm=arm_name, strategy=strategy, training_seconds=training_seconds,
            logical_units=len(logical), exposure='complete64' if len(logical) == steps else 'hard-wallcap-truncated-prefix',
            cap_seconds=arm['cap_seconds'], transaction=state, physical_attempts=len(attempts),
            physical_supervised_tokens=sum(item['supervised_tokens'] for item in attempts),
            physical_reference_tokens=sum(item['reference_tokens'] for item in attempts),
            gradient_norm=dict(mean=sum(norms) / len(norms), minimum=min(norms), maximum=max(norms),
                               clipped_fraction=sum(value > 1 for value in norms) / len(norms)),
            parameter_delta=dict(l2=delta_squared ** .5, max_abs=max_delta, changed_tensors=changed),
            train=final_train, heldout=final_heldout, original92=after,
            original23_lost=[index for index, (a, b) in enumerate(zip(baseline['samples'], after['samples'])) if a['correct'] and not b['correct']],
            original92_gained=[index for index, (a, b) in enumerate(zip(baseline['samples'], after['samples'])) if not a['correct'] and b['correct']],
            native_export_logits_exact=native_logits_exact, cached_export_logits_exact=logits_exact,
            native_export_greedy_ids_and_eos_exact=greedy_exact, candidate=str(destination),
            progress_train_only=progress, logical_updates=logical, attempt_statistics=attempts)
        report['trials'].append(trial)
        save_json(args.output / 'REPORT.json', report)
        print(json.dumps(dict(phase='endpoint', arm=arm_name, train_ce=final_train['raw_ce'],
            train_complete=final_train['complete_with_eos'], heldout_complete=final_heldout['complete_with_eos'],
            original92=after['exact_correct'], original23_lost=len(trial['original23_lost']),
            logical_units=len(logical), training_seconds=training_seconds)), flush=True)
        del runtime, exported, optimizer, parameters
        gc.collect()
    # Source holdouts are first forwarded here, after every fixed trajectory has
    # finished. They cannot affect optimizer data, length, LR, or checkpoint.
    baseline_heldout = metrics(champion, heldout, greedy=True)
    report['baseline_heldout_endpoint_only'] = baseline_heldout
    for trial in report['trials']:
        trial['holdout_gained'] = [index for index, (base, after) in enumerate(zip(baseline_heldout['rows'], trial['heldout']['rows']))
                                   if not base['native_greedy']['raw_exact'] and after['native_greedy']['raw_exact']]
        trial['holdout_lost'] = [index for index, (base, after) in enumerate(zip(baseline_heldout['rows'], trial['heldout']['rows']))
                                 if base['native_greedy']['raw_exact'] and not after['native_greedy']['raw_exact']]
    qualifies = any(trial['train']['complete_with_eos'] >= 6 or trial['holdout_gained'] for trial in report['trials'])
    report['full897_certification_triggered'] = qualifies
    if qualifies:
        baseline_certifier = make_certifier(champion.model, encode, EOS)
        baseline_certificates = {key: baseline_certifier(entry['row']) for key, entry in memory['entries'].items()}
        if not all(value['correct'] for value in baseline_certificates.values()):
            raise ValueError('source897 ledger failed native reproduction')
        report['baseline897_certificates'] = baseline_certificates
        for trial in report['trials']:
            candidate = GeneralistRuntime.from_checkpoint(trial['candidate'])
            certifier = make_certifier(candidate.model, encode, EOS)
            certificates = {key: certifier(entry['row']) for key, entry in memory['entries'].items()}
            trial['retention897_certificates'] = certificates
            trial['lost_retention897_keys'] = [key for key, value in certificates.items() if not value['correct']]
            save_json(args.output / 'REPORT.json', report)
            del candidate
            gc.collect()
    for path, expected in {**source_hashes, **code_hashes}.items():
        if digest(path) != expected:
            raise AssertionError('immutable source changed during run: ' + path)
    report.update(source_checkpoint_unchanged=True, training_code_unchanged=True, complete=True)
    save_json(args.output / 'REPORT.json', report)
    print(json.dumps(dict(phase='complete', full897_certification_triggered=qualifies)), flush=True)


if __name__ == '__main__':
    main()
