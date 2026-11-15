"""One fixed broad native curriculum trial; isolated diagnostic, no promotion."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import string
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/transfer-training'))
OUT = Path('/workspace/airi-lab/compositional-20261115')
SOURCE = Path('/workspace/airi-lab/rev4/checkpoint/model')
ARCHIVE = Path('/workspace/airi-lab/rev4/experiment')
SEED = 2026111503
CONDITIONS = ('new-values-known-template', 'known-values-new-template', 'new-values-new-template')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def norm(value):
    return ' '.join(str(value).casefold().split())


# All wording is authored before model forwards. Each field has an explicit
# quoted literal and the solver captures exactly the designated source span.
TEMPLATES = {
 'it': [
  ('Nome: "{v}".', 'Quale nome?', r'Nome: "([^"]+)"'),
  ('Il nome scelto è "{v}".', 'Scrivi solo il nome.', r'Il nome scelto è "([^"]+)"'),
  ('Luogo: "{v}".', 'Quale luogo?', r'Luogo: "([^"]+)"'),
  ('Il pacco è a "{v}".', 'Dove si trova il pacco?', r'Il pacco è a "([^"]+)"'),
  ('Codice: "{v}". Scarta "{d}".', 'Quale codice?', r'Codice: "([^"]+)"'),
  ('Ignora "{d}". Il codice valido è "{v}".', 'Scrivi solo il codice valido.', r'Il codice valido è "([^"]+)"'),
  ('Prima: "{d}". Ora: "{v}".', 'Quale valore vale ora?', r'Ora: "([^"]+)"'),
  ('Annulla "{d}". Valore corretto: "{v}".', 'Scrivi il valore corretto.', r'Valore corretto: "([^"]+)"'),
  ('Registro nominativo = "{v}".', 'Riporta il nominativo registrato.', r'Registro nominativo = "([^"]+)"'),
  ('La destinazione è "{v}".', 'Riporta la destinazione.', r'La destinazione è "([^"]+)"'),
  ('Dato utile "{v}"; dato escluso "{d}".', 'Riporta il dato utile.', r'Dato utile "([^"]+)"'),
  ('Rettifica: usa "{v}" invece di "{d}".', 'Quale valore indica la rettifica?', r'Rettifica: usa "([^"]+)"'),
  ('La persona si fa chiamare "{v}".', 'Come si fa chiamare la persona?', r'La persona si fa chiamare "([^"]+)"'),
  ('Ho riposto il foglio presso "{v}".', 'Dove ho riposto il foglio?', r'Ho riposto il foglio presso "([^"]+)"'),
  ('Riferimento attivo "{v}"; non usare "{d}".', 'Scrivi il riferimento attivo.', r'Riferimento attivo "([^"]+)"'),
  ('Sostituisci "{d}" con "{v}".', 'Qual è il valore dopo la sostituzione?', r'Sostituisci "[^"]+" con "([^"]+)"'),
 ],
 'en': [
  ('Name: "{v}".', 'What name?', r'Name: "([^"]+)"'),
  ('The chosen name is "{v}".', 'Write only the name.', r'The chosen name is "([^"]+)"'),
  ('Place: "{v}".', 'Which place?', r'Place: "([^"]+)"'),
  ('The parcel is at "{v}".', 'Where is the parcel?', r'The parcel is at "([^"]+)"'),
  ('Code: "{v}". Discard "{d}".', 'Which code?', r'Code: "([^"]+)"'),
  ('Ignore "{d}". The valid code is "{v}".', 'Write only the valid code.', r'The valid code is "([^"]+)"'),
  ('Before: "{d}". Now: "{v}".', 'Which value applies now?', r'Now: "([^"]+)"'),
  ('Cancel "{d}". Correct value: "{v}".', 'Write the correct value.', r'Correct value: "([^"]+)"'),
  ('Registered personal name = "{v}".', 'Give the registered personal name.', r'Registered personal name = "([^"]+)"'),
  ('The destination is "{v}".', 'Give the destination.', r'The destination is "([^"]+)"'),
  ('Useful detail "{v}"; excluded detail "{d}".', 'Give the useful detail.', r'Useful detail "([^"]+)"'),
  ('Correction: use "{v}" instead of "{d}".', 'Which value does the correction give?', r'Correction: use "([^"]+)"'),
  ('The person goes by "{v}".', 'What does the person go by?', r'The person goes by "([^"]+)"'),
  ('I put the sheet at "{v}".', 'Where did I put the sheet?', r'I put the sheet at "([^"]+)"'),
  ('Active reference "{v}"; do not use "{d}".', 'Write the active reference.', r'Active reference "([^"]+)"'),
  ('Replace "{d}" with "{v}".', 'What is the value after replacement?', r'Replace "[^"]+" with "([^"]+)"'),
 ],
}
FAMILIES = ('name', 'location', 'code', 'correction')
FILLERS = {
 'it': [('Aspetta un momento.', 'Va bene.'), ('Questa è una breve pausa.', 'Ricevuto.')],
 'en': [('Wait a moment.', 'All right.'), ('This is a short pause.', 'Understood.')],
}


def blocked_sources():
    authored = json.loads((ARCHIVE / 'AUTHORED_CURRICULUM.json').read_text())
    rows = sum((authored[key] for key in ('train', 'development', 'sealed_transfer', 'context_acquisition_values')), [])
    for filename in ('human-train.jsonl', 'human-development.jsonl', 'human-final.jsonl', 'development.jsonl', 'test.jsonl'):
        rows.extend(json.loads(line) for line in (ARCHIVE / filename).read_text().splitlines())
    registry = json.loads((SOURCE.parent / 'training_registry.json').read_text())
    # Registry labels are used only as a reservation list, never as fresh gold.
    registry_text = json.dumps(registry, ensure_ascii=False)
    values = {norm(row.get('expected', row['messages'][-1]['content'])) for row in rows}
    histories = {json.dumps(row['messages'][:-1], sort_keys=True, ensure_ascii=False) for row in rows}
    return values, histories, registry_text


def build_rows():
    blocked_values, old_histories, registry_text = blocked_sources()
    used = set(blocked_values)
    rng = random.Random(SEED)
    def fresh(bucket):
        for _ in range(100000):
            if bucket == 0:
                value = ''.join(rng.choices(string.ascii_lowercase, k=rng.choice((2, 3))))
            elif bucket == 1:
                value = rng.choice(string.ascii_uppercase) + ''.join(rng.choices(string.ascii_lowercase, k=rng.choice((3, 4))))
            elif bucket == 2:
                value = rng.choice(string.ascii_uppercase) + ''.join(rng.choices(string.ascii_lowercase + string.digits, k=rng.choice((5, 6, 7))))
            else:
                value = ''.join(rng.choices(string.ascii_lowercase, k=3)) + ' ' + ''.join(rng.choices(string.ascii_lowercase, k=4))
            if norm(value) not in used and ('"expected": "' + value + '"') not in registry_text:
                used.add(norm(value))
                return value
        raise ValueError('fresh literal inventory exhausted')
    def pair(split, condition, template, case, values):
        result = []
        bucket = case % 4
        distractor = fresh(bucket)
        history = case % 3
        before = case % 2 == 1
        family = FAMILIES[template // 2] if template < 8 else FAMILIES[(template - 8) % 4]
        for language in ('it', 'en'):
            statement, question, pattern = TEMPLATES[language][template]
            pair_id = f'{split}:{condition}:{template}:{case}:{language}'
            for alternate, value in enumerate(values):
                source = dict(role='user', content=statement.format(v=value, d=distractor))
                ack = dict(role='assistant', content='Segnato.' if language == 'it' else 'Noted.')
                fillers = [dict(role=role, content=text) for exchange in FILLERS[language][:history]
                           for role, text in zip(('user', 'assistant'), exchange)]
                messages = fillers + [source, ack] if before else [source, ack] + fillers
                source_index = len(fillers) if before else 0
                match = re.search(pattern, messages[source_index]['content'])
                if match is None or match.group(1) != value:
                    raise ValueError('literal solver disagrees with actual source context')
                messages += [dict(role='user', content=question), dict(role='assistant', content=value)]
                result.append(dict(messages=messages, expected=match.group(1), language=language,
                    family='context-' + family, condition=condition, split=split, template=template,
                    pair_id=pair_id, counterfactual_index=alternate, length_bucket=bucket,
                    history_exchanges=history, source_after_filler=before,
                    source_provenance=dict(message_index=source_index, regex=pattern,
                        start=match.start(1), end=match.end(1), literal=match.group(1)),
                    provenance='authored literal capture solver; no model or evaluation labels'))
        return result
    train = []
    train_values = {}
    for template in range(8):
        for case in range(16):
            values = [fresh(case % 4), fresh(case % 4)]
            train_values[(template, case)] = values
            train.extend(pair('train', 'train', template, case, values))
    splits = {'train': train}
    for split, new_start, cases in (('development', 8, 2), ('final', 12, 4)):
        rows = []
        # Number of semantic pairs: (2,2,4) for development; (4,4,8) final.
        for condition in CONDITIONS:
            count = cases * (2 if condition == CONDITIONS[2] else 1)
            for index in range(count):
                family = index % 4
                case = (index + (0 if split == 'development' else 7)) % 16
                template = family * 2 + index % 2 if condition == CONDITIONS[0] else new_start + family
                values = train_values[(family * 2, case)] if condition == CONDITIONS[1] else [fresh(index % 4), fresh(index % 4)]
                rows.extend(pair(split, condition, template, index, values))
        splits[split] = rows
    if [len(splits[x]) for x in ('train', 'development', 'final')] != [512, 32, 64]:
        raise AssertionError('fixed split sizes changed')
    train_literals = {r['expected'] for r in train}
    for split, rows in splits.items():
        groups = defaultdict(list)
        for row in rows:
            groups[row['pair_id']].append(row)
            if json.dumps(row['messages'][:-1], sort_keys=True, ensure_ascii=False) in old_histories:
                raise ValueError('fresh prompt collides with original evaluation or TRAIN')
            p = row['source_provenance']
            if row['messages'][p['message_index']]['content'][p['start']:p['end']] != row['expected']:
                raise AssertionError('literal provenance span mismatch')
            if split != 'train' and row['condition'] != CONDITIONS[1] and row['expected'] in train_literals:
                raise AssertionError('novel literal leaked into TRAIN')
            if split != 'train' and row['condition'] != CONDITIONS[0] and row['template'] < 8:
                raise AssertionError('novel template leaked into TRAIN')
        for members in groups.values():
            a, b = members
            if a['expected'] == b['expected'] or a['messages'][-2] != b['messages'][-2]:
                raise AssertionError('same-question counterfactual pair invalid')
            changed = [i for i, (x, y) in enumerate(zip(a['messages'][:-1], b['messages'][:-1])) if x != y]
            if changed != [a['source_provenance']['message_index']]:
                raise AssertionError('counterfactual changed context beyond its source literal')
    prompt_sets = [{json.dumps(r['messages'][:-1], sort_keys=True, ensure_ascii=False) for r in splits[s]} for s in splits]
    if any(prompt_sets[i] & prompt_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError('split prompts overlap')
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'TRAINING_STARTED.json').exists():
        raise RuntimeError('the single predeclared run already started; no rerun')
    import torch
    from torch.nn import functional as F
    from generalist_lm.runtime import GeneralistRuntime, checkpoint_model_files
    from generalist_lm.training import SFTExample, encode_sft_example, causal_training_objective
    from generalist_lm.tokenizer import EOS, PAD
    from curriculum import exact_correct, repetitive
    from legacy_run import state_digest
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    if args.prepare_only:
        splits = build_rows()
        save(OUT / 'ROWS.json', splits)
    else:
        splits = json.loads((OUT / 'ROWS.json').read_text())
    runtime = GeneralistRuntime.from_checkpoint(SOURCE)
    tokenizer = runtime.tokenizer
    encoded = {}
    def encode(row):
        key = json.dumps(row['messages'], sort_keys=True, ensure_ascii=False)
        if key not in encoded:
            ids, labels = encode_sft_example(SFTExample.from_dict(row), tokenizer, 1024, require_complete=True)
            width = int((labels != -100).nonzero()[:, 0].max()) + 1
            ids, labels = ids[:width].unsqueeze(0), labels[:width].unsqueeze(0)
            prompt = tokenizer.serialize_messages(row['messages'][:-1], add_generation_prompt=True)
            gold = tokenizer.encode(row['expected'], eos=True)
            if ids[0].tolist() != prompt + gold or labels[0].tolist() != [-100] * len(prompt) + gold or gold[-1] != EOS:
                raise AssertionError('native serialization/target mask/EOS preflight failed')
            encoded[key] = ids, labels
        return encoded[key]
    for rows in splits.values():
        for row in rows:
            encode(row)
    def native_batch(rows):
        batches = [encode(row) for row in rows]
        width = max(ids.shape[1] for ids, _ in batches)
        ids = torch.full((len(rows), width), PAD, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        for index, (source_ids, source_labels) in enumerate(batches):
            ids[index, :source_ids.shape[1]] = source_ids[0]
            labels[index, :source_labels.shape[1]] = source_labels[0]
        return ids, labels
    for p in runtime.model.parameters():
        p.requires_grad_(True)
    if sum(p.numel() for p in runtime.model.parameters()) != 50164416:
        raise AssertionError('native parameter count changed')
    if any(hasattr(runtime.model, attr) for attr in ('_airi_prefix_features', '_airi_head_features')):
        raise AssertionError('full native trial cannot use feature caches')
    groups = defaultdict(list)
    for index, row in enumerate(splits['train']):
        groups[row['pair_id']].append(index)
    pairs = list(groups.values())
    trajectory = []
    order = random.Random(SEED + 1)
    # Four fixed epochs, each with 128 batches of two complete pairs.
    for epoch in range(4):
        indices = list(range(len(pairs)))
        order.shuffle(indices)
        for start in range(0, len(indices), 2):
            trajectory.append([i for pair_index in indices[start:start + 2] for i in pairs[pair_index]])
    if len(trajectory) != 512 or Counter(i for batch in trajectory for i in batch) != Counter({i: 4 for i in range(512)}):
        raise AssertionError('fixed trajectory/dose mismatch')
    # Collate every precommitted batch before the start marker; no forwards.
    for indices in trajectory:
        batch_ids, batch_labels = native_batch([splits['train'][i] for i in indices])
        if batch_ids.shape != batch_labels.shape or batch_ids.shape[0] != 4:
            raise AssertionError('four-row native collation preflight failed')
        for index, source_index in enumerate(indices):
            source_ids, source_labels = encode(splits['train'][source_index])
            width = source_ids.shape[1]
            if not torch.equal(batch_ids[index, :width], source_ids[0]) or not torch.equal(batch_labels[index, :width], source_labels[0]):
                raise AssertionError('collation changed native source IDs/labels')
    native_root = Path('/workspace/airi-lab/native-source/computer/generalist_lm')
    source_paths = list(checkpoint_model_files(SOURCE, verify=True)) + list(SOURCE.glob('*.json'))
    source_paths += [SOURCE.parent / name for name in ('resume.pt', 'retention_memory.json', 'training_registry.json')]
    source_paths += [ARCHIVE / name for name in ('AUTHORED_CURRICULUM.json', 'human-train.jsonl', 'human-development.jsonl', 'human-final.jsonl', 'development.jsonl', 'test.jsonl')]
    code_paths = list(native_root.glob('*.py')) + list((ROOT / 'scripts/transfer-training').glob('*.py')) + [Path(__file__)]
    if args.prepare_only:
        design = dict(seed=SEED, source=str(SOURCE), source_state_digest=state_digest(runtime.model),
            source_hashes={str(p): digest(p) for p in source_paths}, code_hashes={str(p): digest(p) for p in code_paths},
            rows_sha256=digest(OUT / 'ROWS.json'), trainable_parameters=50164416,
            train_rows=512, train_unique_values=len({r['expected'] for r in splits['train']}),
            development_rows=32, final_rows=64, trajectory=trajectory,
            max_logical_units=512, batch_size=4, exposure_per_row_if_complete=4,
            training_wallcap_seconds=600, cap_policy='between indivisible updates; excludes readonly endpoint evaluations',
            optimizer=dict(name='fresh AdamW', lr=3e-5, weight_decay=.01, clip_grad_norm=1),
            objective=dict(native_causal_training_objective=True, eos_loss_weight=1.2, repetition_unlikelihood_weight=.05),
            retention_training='none: no replay, no reference model, no KL, no transaction guard',
            endpoint_policy='TRAIN512 + dev32 + final64 once; dev/final and frozen-source comparisons first forwarded after fixed training endpoint; no feedback/tuning',
            expert_threshold=dict(final_new_values_known_template=.80, final_new_values_new_template=.60, metric='native raw exact literal token IDs including EOS'),
            qualification='isolated candidate expert only on threshold; no production acceptance without all old retention and NLL gates; no activation/promotion',
            full897_trigger='expert thresholds pass AND all original23 correct answers preserved',
            native_lengths={split: dict(minimum=min(encode(r)[0].shape[1] for r in rows), maximum=max(encode(r)[0].shape[1] for r in rows)) for split, rows in splits.items()},
            baseline_forward_policy='no fresh TRAIN/development/final model forwards before optimizer',
            torch_version=torch.__version__)
        save(OUT / 'PREREGISTRATION.json', design)
        print(json.dumps(dict(phase='preflight_ready', rows_sha256=design['rows_sha256'], preregistration_sha256=digest(OUT / 'PREREGISTRATION.json'), lengths=design['native_lengths'])), flush=True)
        return
    design = json.loads((OUT / 'PREREGISTRATION.json').read_text())
    if digest(OUT / 'ROWS.json') != design['rows_sha256'] or trajectory != design['trajectory']:
        raise AssertionError('committed dataset/trajectory changed')
    for path, expected in {**design['source_hashes'], **design['code_hashes']}.items():
        if digest(path) != expected:
            raise AssertionError('immutable source/code changed before run: ' + path)
    if state_digest(runtime.model) != design['source_state_digest']:
        raise AssertionError('approved source state digest changed')
    approval = json.loads((OUT / 'APPROVAL.json').read_text())
    if approval['preregistration_sha256'] != digest(OUT / 'PREREGISTRATION.json') or not approval['approved']:
        raise RuntimeError('root approval must name the exact committed design')
    save(OUT / 'TRAINING_STARTED.json', dict(preregistration_sha256=digest(OUT / 'PREREGISTRATION.json'), approval=approval))
    report = dict(complete=False, production_qualified=False, live_promoted=False, neural_metrics_only=True,
        sealed_original_benchmark_forward=False, preregistration_sha256=digest(OUT / 'PREREGISTRATION.json'), attempts=[])
    save(OUT / 'REPORT.json', report)
    params = list(runtime.model.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-5, weight_decay=.01)
    training_seconds = 0.
    for step, batch_indices in enumerate(trajectory, 1):
        if training_seconds >= 600:
            break
        started = time.monotonic()
        batch = native_batch([splits['train'][i] for i in batch_indices])
        runtime.model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = runtime.model(batch[0])['logits']
        loss, stats = causal_training_objective(logits, batch[1], batch[0], eos_loss_weight=1.2, repetition_unlikelihood_weight=.05)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('nonfinite native objective; reject trial')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        training_seconds += time.monotonic() - started
        report['attempts'].append(dict(unit=step, loss=float(loss.detach()), gradient_norm=float(grad_norm),
            supervised_tokens=int((batch[1][:, 1:] != -100).sum()), **stats))
        if step % 32 == 0:
            report.update(logical_units=step, training_seconds=training_seconds)
            save(OUT / 'REPORT.json', report)
            print(json.dumps(dict(phase='training', unit=step, training_seconds=training_seconds, last_loss=float(loss.detach()))), flush=True)
    report.update(logical_units=len(report['attempts']), training_seconds=training_seconds,
        exposure='complete512' if len(report['attempts']) == 512 else 'hard-wallcap-truncated-prefix')
    del optimizer, logits, loss
    runtime.model.eval()
    def answer(active, row, budget=40):
        prompt = active.tokenizer.serialize_messages(row['messages'][:-1], add_generation_prompt=True)
        ids = active._generate_ids(prompt, max_new_tokens=budget, temperature=0)
        gold = active.tokenizer.encode(row['expected'], eos=True)
        response = active.tokenizer.decode(ids)
        return dict(response=response, generated_ids=ids, gold_ids=gold, raw_exact=ids == gold,
            eos_emitted=bool(ids and ids[-1] == EOS), text_correct=exact_correct(response, row['expected']))
    def metrics(active, rows):
        observations = []
        active.model.eval()
        with torch.no_grad():
            for row in rows:
                ids, labels = encode(row)
                targets = labels[:, 1:]; mask = targets != -100
                scores = active.model(ids)['logits'][:, :-1][mask].float()
                gold = targets[mask]
                nll = F.cross_entropy(scores, gold, reduction='sum')
                matches = scores.argmax(-1) == gold
                observations.append({key: row.get(key) for key in ('expected', 'language', 'family', 'condition', 'pair_id', 'counterfactual_index', 'template', 'length_bucket')} |
                    dict(target_tokens=int(gold.numel()), total_nll=float(nll), teacher_forced_complete_with_eos=bool(matches.all()), native_greedy=answer(active, row)))
        def aggregate(selected):
            pair_groups = defaultdict(list)
            for row in selected:
                pair_groups[row['pair_id']].append(row)
            exact = sum(r['native_greedy']['raw_exact'] for r in selected)
            return dict(rows=len(selected), raw_exact_with_eos=exact, raw_exact_rate=exact / len(selected),
                text_exact=sum(r['native_greedy']['text_correct'] for r in selected),
                teacher_forced_complete_with_eos=sum(r['teacher_forced_complete_with_eos'] for r in selected),
                raw_ce=sum(r['total_nll'] for r in selected) / sum(r['target_tokens'] for r in selected),
                pairs=len(pair_groups), both_counterfactuals_exact=sum(len(v) == 2 and all(r['native_greedy']['raw_exact'] for r in v) for v in pair_groups.values()),
                identical_outputs_on_counterfactuals=sum(len(v) == 2 and v[0]['native_greedy']['generated_ids'] == v[1]['native_greedy']['generated_ids'] for v in pair_groups.values()))
        return dict(overall=aggregate(observations), by_condition={condition: aggregate([r for r in observations if r['condition'] == condition]) for condition in sorted({r['condition'] for r in observations})}, rows=observations)
    # The first fresh-data forwards happen here, at the one fixed endpoint.
    report['train'] = metrics(runtime, splits['train'])
    save(OUT / 'REPORT.json', report)
    for split in ('development', 'final'):
        report[split] = metrics(runtime, splits[split])
        save(OUT / 'REPORT.json', report)
    baseline = GeneralistRuntime.from_checkpoint(SOURCE)
    baseline.model.eval()
    report['baseline_endpoint_only'] = {split: metrics(baseline, splits[split]) for split in ('development', 'final')}
    old = torch.load(SOURCE.parent / 'resume.pt', weights_only=True)['counters']['champion']['samples']
    old_samples = []
    with torch.no_grad():
        for saved in old:
            row = dict(messages=saved['messages'] + [dict(role='assistant', content=saved['expected'])], expected=saved['expected'])
            old_samples.append({**saved, **answer(runtime, row, 64)})
    report['original92'] = dict(probes=len(old_samples), text_exact=sum(s['text_correct'] for s in old_samples),
        raw_exact_with_eos=sum(s['raw_exact'] for s in old_samples),
        repetition=sum(repetitive(s['response']) for s in old_samples) / len(old_samples), samples=old_samples)
    report['original23_lost'] = [i for i, (a, b) in enumerate(zip(old, old_samples)) if a['correct'] and not b['text_correct']]
    report['original92_gained'] = [i for i, (a, b) in enumerate(zip(old, old_samples)) if not a['correct'] and b['text_correct']]
    final = report['final']['by_condition']
    expert = final[CONDITIONS[0]]['raw_exact_rate'] >= .8 and final[CONDITIONS[2]]['raw_exact_rate'] >= .6
    report.update(expert_threshold_pass=expert, expert_disposition='isolated-candidate-only' if expert else 'rejected-no-strong-novel-transfer',
        full897_certification_triggered=expert and not report['original23_lost'])
    if report['full897_certification_triggered']:
        from retention_memory import make_certifier
        memory = json.loads((SOURCE.parent / 'retention_memory.json').read_text())
        certifier = make_certifier(runtime.model, encode, EOS)
        report['retention897_certificates'] = {key: certifier(entry['row']) for key, entry in memory['entries'].items()}
        report['lost_retention897_keys'] = [key for key, value in report['retention897_certificates'].items() if not value['correct']]
    else:
        report['retention897_skip_reason'] = 'predeclared potential-qualification trigger failed: expert thresholds and/or original23 retention'
    report.update(original_nll_gates_evaluated=False, production_acceptance_evaluated=False,
        qualification_note='No production acceptance claim. Old NLL and complete original gate package required separately for any acceptance.')
    destination = OUT / 'candidate-model'
    runtime.save_checkpoint(destination, metadata=dict(role='isolated-compositional-native-diagnostic', production_qualified=False,
        live_promoted=False, expert_threshold_pass=expert, preregistration_sha256=report['preregistration_sha256']))
    exported = GeneralistRuntime.from_checkpoint(destination)
    exported.model.eval()
    with torch.no_grad():
        export_rows = [splits['train'][i] for i in (0, 1, 128, 255, 384, 511)]
        report['native_export_logits_exact'] = all(torch.equal(runtime.model(encode(row)[0])['logits'], exported.model(encode(row)[0])['logits']) for row in export_rows)
        report['native_export_raw_ids_eos_exact'] = all(answer(exported, row) == report['train']['rows'][i]['native_greedy'] for i, row in zip((0, 1, 128, 255, 384, 511), export_rows))
    if not report['native_export_logits_exact'] or not report['native_export_raw_ids_eos_exact']:
        raise AssertionError('native export/reload changed logits or raw IDs/EOS')
    report.update(candidate=str(destination), candidate_state_digest=state_digest(exported.model),
        candidate_files_sha256={str(p): digest(p) for p in list(checkpoint_model_files(destination, verify=True)) + list(destination.glob('*.json'))})
    for path, expected in {**design['source_hashes'], **design['code_hashes']}.items():
        if digest(path) != expected:
            raise AssertionError('immutable source/code changed during run: ' + path)
    if digest(OUT / 'ROWS.json') != design['rows_sha256']:
        raise AssertionError('committed dataset changed')
    report.update(source_checkpoint_unchanged=True, training_code_unchanged=True, complete=True)
    save(OUT / 'REPORT.json', report)
    summary = {key: report[key] for key in ('complete', 'logical_units', 'training_seconds', 'exposure', 'expert_threshold_pass', 'expert_disposition', 'original23_lost', 'full897_certification_triggered', 'source_checkpoint_unchanged', 'native_export_logits_exact', 'native_export_raw_ids_eos_exact')}
    summary.update(train=report['train']['overall'], development=report['development']['by_condition'], final=report['final']['by_condition'], original92=report['original92']['text_exact'])
    save(OUT / 'SUMMARY.json', summary)
    print(json.dumps(dict(phase='complete', **summary)), flush=True)


if __name__ == '__main__':
    main()
