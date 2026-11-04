"""Bounded paired checkpoint experiment; no publication or live adoption.

Reuses archived authored development and reviewed OASST2 splits exactly. Training
uses only original TRAIN rows. The ancestral native reference and replay are verified from the pinned source.
The normal runner must still apply its original retention/final gates.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time

SCHEDULE = ('human-dialogue', 'verified-context', 'curated-dialogue', 'composition',
            'human-dialogue', 'verified-context', 'curated-dialogue', 'old-authored',
            'human-dialogue', 'verified-context', 'curated-dialogue', 'composition',
            'human-dialogue', 'verified-context', 'curated-dialogue', 'native-replay')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--native-replay', required=True, type=Path)
    parser.add_argument('--baseline-result', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', choices=('head', 'prefix', 'both'), default='both')
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('--retention-weight', type=float, default=4.)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--spec', type=Path, help='Use this exact continual schedule/profile instead of diagnostic defaults')
    parser.add_argument('--profile-index', type=int, default=0)
    parser.add_argument('--head-protected-rank', type=int, default=0)
    args = parser.parse_args()
    schedule = SCHEDULE
    retention_per_language = 4
    selected_recipe = None
    if args.spec:
        selected_recipe = json.loads(args.spec.read_text())
        config = selected_recipe['continual_learning']
        if 'context_acquisition' in config:
            raise ValueError('legacy learning probe does not support matched context acquisition; use the main runner for this recipe')
        if not 0 <= args.profile_index < len(config['profiles']):
            raise ValueError('invalid selected profile')
        profile = config['profiles'][args.profile_index]
        args.learning_rate = profile['learning_rate']
        args.retention_weight = profile['retention_weight']
        if profile['champion_kl_weight'] != args.retention_weight:
            raise ValueError('paired probe currently requires equal gold/KL retention weights')
        schedule = tuple(config['sampling_schedule'])
        retention_per_language = config.get('retention_examples_per_language',1)
        allowed = set(SCHEDULE)
        if len(schedule)!=16 or set(schedule)-allowed or not 1<=retention_per_language<=8:
            raise ValueError('invalid bounded selected schedule')
    if not 16 <= args.steps <= 4096 or args.steps % 16:
        raise ValueError('complete bounded 16-update schedules required')
    for value in (args.learning_rate, args.retention_weight):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('positive finite training configuration required')
    if args.learning_rate > 1e-4 or args.retention_weight > 16:
        raise ValueError('diagnostic profile outside original finite bounds')
    if not 0 <= args.head_protected_rank <= 48:
        raise ValueError('invalid head protection rank')
    import torch
    from torch.nn import functional as F
    from generalist_lm.runtime import GeneralistRuntime, checkpoint_model_files
    from generalist_lm.training import SFTExample, encode_sft_example
    from generalist_lm.phase5_diagnostics import protected_bootstrap_texts
    from generalist_lm.curriculum import train_rows
    from generalist_lm.bootstrap_data import load_bootstrap_replay
    from continual_learning import cache_tokenizer, continual_step, retention_rows, verified_memories
    from curriculum import norm, exact_correct, repetitive
    from success_bank import catalog, prompt_key, retains_correct_answers
    from human_data import load_corpus
    from policy import improved
    from fast_head import configure_head, install_feature_cache, head_summary, configure_projection, attach_projection
    from prefix_tuning import configure_prefix_tuning, prefix_summary

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_files = checkpoint_model_files(args.checkpoint, verify=True)
    def file_digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    source_hashes = {str(path): file_digest(path) for path in source_files}
    champion = GeneralistRuntime.from_checkpoint(args.checkpoint)
    cache_tokenizer(champion.tokenizer)
    for parameter in champion.model.parameters():
        parameter.requires_grad_(False)
    champion.model.eval()
    install_feature_cache(champion.model, cache_bytes=64*1024*1024)
    reference = GeneralistRuntime.from_checkpoint(args.reference)
    if file_digest(args.reference/'model.pt') != '998530839d9722316b59cb28ff1ef68a7863f517de73fa8fef8ce9677a24b85e':
        raise ValueError('ancestral native reference changed')
    for parameter in reference.model.parameters():
        parameter.requires_grad_(False)
    reference.model.eval()
    install_feature_cache(reference.model, cache_bytes=64*1024*1024)
    baseline_saved = json.loads(args.baseline_result.read_text())['counters']['champion']
    dataset = json.loads((args.archive/'AUTHORED_CURRICULUM.json').read_text())
    authored, heldout, sealed = (dataset[key] for key in ('train', 'development', 'sealed_transfer'))
    spec = json.loads((args.archive/'SPEC.json').read_text())
    old_validation = [json.loads(line) for name in ('development.jsonl', 'test.jsonl')
                      for line in (args.archive/name).read_text().splitlines()]
    legacy_blocked = {norm(m['content']) for row in old_validation for m in row['messages']}
    protected = protected_bootstrap_texts() | legacy_blocked
    old_train = [json.loads(line) for line in (args.archive/'train-1024.jsonl').read_text().splitlines()]
    old_train = [row for row in old_train if not any(norm(m['content']) in protected for m in row['messages'])]
    def fits(row, length):
        return len(champion.tokenizer.serialize_messages(row['messages'][:-1], add_generation_prompt=True)) + len(champion.tokenizer.encode(row['messages'][-1]['content'], eos=True)) <= length
    # The pinned train-1024 source was already audited for complete1024 histories.
    if file_digest(args.archive/'train-1024.jsonl') != '4cd65b21228375b7abaf81da5e32e977dc1182f0a73b942dc81c34a38a698d91':
        raise ValueError('legacy source training corpus changed')
    old_human = [row for row in old_train if 2 <= len(row['messages']) <= 6
                 and 12 <= len(row['messages'][-1]['content'].encode()) <= 480]
    previous_text = {norm(m['content']) for row in old_human for m in row['messages']}
    blocked = legacy_blocked | {norm(p) for p in json.loads((args.archive/'CONVERSATION_TEST.json').read_text())['prompts']}
    all_heldout = {norm(m['content']) for row in heldout for m in row['messages']}
    acquisition_blocked = blocked | all_heldout | {norm(m['content']) for row in authored for m in row['messages']}
    corpus = load_corpus(args.archive, spec['human_corpus']['pins'], acquisition_blocked | previous_text,
                         training_blocked=acquisition_blocked)
    # Match the runner: at most sixteen examples PER language.
    human_dev = [row for lang in ('it', 'en')
                 for row in [r for r in corpus['development'] if r['language'] == lang][:16]]
    human_train = corpus['train']
    eligible = catalog(authored, heldout + sealed + corpus['development'] + corpus['final'])
    if {prompt_key(r) for r in authored} & {prompt_key(r) for r in heldout + sealed}:
        raise ValueError('authored training/heldout prompt overlap')
    dev_texts = {norm(m['content']) for split in ('development', 'final') for row in corpus[split] for m in row['messages']}
    if any(norm(m['content']) in dev_texts for row in human_train for m in row['messages']):
        raise ValueError('reviewed training/heldout text overlap')
    encoded_cache = {}
    def encode(row):
        key = json.dumps(row['messages'], ensure_ascii=False, sort_keys=True)
        if key not in encoded_cache:
            ids, labels = encode_sft_example(SFTExample.from_dict(row), champion.tokenizer, 1024, require_complete=True)
            scored = (labels != -100).nonzero()
            width = int(scored[:, 0].max()) + 1
            encoded_cache[key] = ids[:width].unsqueeze(0), labels[:width].unsqueeze(0)
        return encoded_cache[key]
    validation = [encode(row) for row in heldout]
    human_validation = [encode(row) for row in human_dev]
    memories = verified_memories(champion, retention_rows(list(eligible.values())), encode)
    memory_by = {lang: [encode(row) for row in memories if row['language'] == lang] for lang in ('it', 'en')}
    anchors = [{'messages': r.sft().messages} for r in train_rows()]
    anchors = [encode(row) for row in anchors if fits(row, 128) and not any(norm(m['content']) in protected for m in row['messages'])]
    if not anchors:
        raise ValueError('no native training anchors')
    replay_rows = {'language-it': [], 'language-en': [], 'domains': []}
    replay_source = load_bootstrap_replay(args.native_replay)
    for document in replay_source.documents:
        if document.source.startswith('tatoeba-it-'):
            group, prompt = 'language-it', 'Ripeti questa frase: '+document.text
        elif document.source.startswith('tatoeba-en-'):
            group, prompt = 'language-en', 'Repeat this sentence: '+document.text
        else:
            continue
        row = {'messages': [{'role':'user','content':prompt},{'role':'assistant','content':document.text}]}
        if not any(norm(m['content']) in protected for m in row['messages']) and fits(row,1024):
            replay_rows[group].append(row)
    replay_rows['domains'] = [{'messages':r.sft().messages} for r in train_rows()
                              if r.domain != 'language' and fits({'messages':r.sft().messages},128)
                              and not any(norm(m['content']) in protected for m in r.sft().messages)]
    selector = random.Random(71026)
    replay = {}
    for name, values in replay_rows.items():
        if not values:
            raise ValueError('empty exact native replay pool')
        if len(values)>512:
            values = selector.sample(values,512)
        replay[name] = [encode(row) for row in values]
    native = {lang: [r for r in authored if r['language'] == lang and not r['family'].startswith(('extra-', 'context-', 'foundation-'))] for lang in ('it', 'en')}
    curated = {lang: [r for r in authored if r['language'] == lang and (r['family'].startswith('extra-') or r['family'] == 'foundation-dialogue')] for lang in ('it', 'en')}
    context = {lang: [r for r in authored if r['language'] == lang and r['family'].startswith('context-') and r.get('stage', 0) == 0] for lang in ('it', 'en')}
    composition = {lang: [r for r in authored if r['language'] == lang and r['family'] in {'foundation-copy', 'foundation-recall'}] for lang in ('it', 'en')}
    humans = {lang: [r for r in human_train if r['language'] == lang] for lang in ('it', 'en')}
    print(json.dumps({'phase': 'data ready', 'authored': len(authored), 'development': len(heldout),
                      'human_counts': {k: len(v) for k,v in corpus.items()}, 'human_development': len(human_dev),
                      'certified_memories': len(memories), 'anchors': len(anchors)}), flush=True)

    def evaluate(runtime):
        runtime.model.eval()
        def grouped_nll(rows, batches):
            result = {}
            with torch.no_grad():
                for lang in ('it', 'en'):
                    total, count = 0., 0
                    for row, (ids, labels) in zip(rows, batches):
                        if row['language'] != lang:
                            continue
                        logits = runtime.model(ids)['logits'][:, :-1]
                        total += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), reduction='sum'))
                        count += int((labels[:, 1:] != -100).sum())
                    if count:
                        result[lang] = total/count
            return result
        nll = grouped_nll(heldout, validation)
        human_nll = grouped_nll(human_dev, human_validation)
        samples = []
        with torch.no_grad():
            for old in baseline_saved['samples']:
                response = runtime.chat(old['messages'], temperature=0, max_new_tokens=64)
                samples.append({**old, 'response': response, 'correct': exact_correct(response, old['expected'])})
        contexts = [row for row in samples if row['family'].startswith('context-')]
        rates = {f'{lang}:{task}': sum(s['correct'] for s in contexts if s['language'] == lang and s['family'] == 'context-'+task)/sum(1 for s in contexts if s['language'] == lang and s['family'] == 'context-'+task) for lang in ('it', 'en') for task in ('name', 'location', 'topic', 'correction')}
        pairs = {s['pair_id'] for s in contexts}
        return {'human_nll': human_nll, 'human_development_rows': len(human_dev), 'nll': nll,
                'foundation_correct': sum(s['correct'] for s in samples if s['family'].startswith('foundation-')),
                'foundation_probes': sum(1 for s in samples if s['family'].startswith('foundation-')),
                'context_rates': rates, 'context_pair_rate': sum(all(s['correct'] for s in contexts if s['pair_id'] == pair) for pair in pairs)/len(pairs),
                'legacy_correct': sum(s['correct'] for s in samples if not s['family'].startswith(('extra-', 'context-', 'foundation-'))),
                'dialogue_correct': sum(s['correct'] for s in samples if s['family'].startswith('extra-')),
                'exact_correct': sum(s['correct'] for s in samples), 'probes': len(samples),
                'repetition': sum(repetitive(s['response']) for s in samples)/len(samples), 'samples': samples}
    tick = time.monotonic()
    before = evaluate(champion)
    if before['exact_correct'] != baseline_saved['exact_correct'] or any(abs(before['nll'][k]-v) > 1e-5 for k,v in baseline_saved['nll'].items()) or any(abs(before['human_nll'][k]-v) > 1e-5 for k,v in baseline_saved['human_nll'].items()):
        raise ValueError('archived baseline not reproduced')
    print(json.dumps({'phase': 'baseline reproduced', 'exact_correct': before['exact_correct'], 'nll': before['nll'], 'human_nll': before['human_nll'], 'seconds': time.monotonic()-tick}), flush=True)
    rng = random.Random(args.seed)
    trajectory = []
    for step in range(args.steps):
        lang = 'it' if step%3 != 1 else 'en'
        group = schedule[step%16]
        if group == 'human-dialogue':
            languages = sorted(humans)
            lang = rng.choices(languages, weights=[math.sqrt(len(humans[k])) for k in languages], k=1)[0]
            row = rng.choice(humans[lang]); batch = encode(row)
        elif group == 'native-replay':
            batch = rng.choice(replay[('language-it','language-en','domains')[(step//16)%3]])
        else:
            pool = {'old-authored': native, 'curated-dialogue': curated,
                    'verified-context': context, 'composition': composition}[group][lang]
            if group in ('old-authored', 'composition', 'verified-context'):
                family = rng.choice(sorted({row['family'] for row in pool}))
                pool = [row for row in pool if row['family'] == family]
            row = rng.choice(pool); batch = encode(row)
        retention = [rng.choice(memory_by[language]) for language in ('it', 'en') for _ in range(retention_per_language)]
        trajectory.append((group, batch, retention, rng.choice(anchors)))
    report = {'scope': 'paired isolated native checkpoint diagnostic with original ancestral reference and native replay; not production qualification or sealed-test evidence',
              'checkpoint': str(args.checkpoint), 'source_files': source_hashes, 'seed': args.seed,
              'torch_version': torch.__version__, 'steps': args.steps, 'learning_rate': args.learning_rate,
              'head_protected_rank': args.head_protected_rank,
              'timing_scope': 'observed optimizer time only; other checks may overlap on shared CPU; no throughput comparison claimed',
              'retention_weight': args.retention_weight, 'schedule': schedule,
              'retention_examples_per_language': retention_per_language,
              'selected_spec_fingerprint': selected_recipe.get('fingerprint') if selected_recipe else None,
              'reference_manifest_sha256': file_digest(args.reference/'model.pt'),
              'heldout_training_overlap': False, 'baseline': before, 'trials': [], 'live_promoted': False}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    for mode in (('head', 'prefix') if args.mode == 'both' else (args.mode,)):
        torch.manual_seed(args.seed)
        runtime = GeneralistRuntime.from_checkpoint(args.checkpoint)
        cache_tokenizer(runtime.tokenizer)
        if mode == 'head':
            configure_head(runtime.model)
            if args.head_protected_rank:
                configure_projection(runtime.model, champion.model, memories, encode, rank=args.head_protected_rank)
        else:
            configure_prefix_tuning(runtime.model)
        optimizer = torch.optim.AdamW([p for p in runtime.model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=.01)
        if mode == 'head' and args.head_protected_rank:
            attach_projection(optimizer, runtime.model)
        training_seconds = 0.
        for step, (group, batch, retention, anchor) in enumerate(trajectory, 1):
            tick = time.monotonic()
            stats = continual_step(runtime.model, champion.model, reference.model, optimizer, batch, retention, anchor,
                                   retention_weight=args.retention_weight, champion_kl_weight=args.retention_weight)
            training_seconds += time.monotonic()-tick
            if step%64 == 0:
                print(json.dumps({'phase': 'training', 'mode': mode, 'step': step, 'seconds': training_seconds, 'last_ce': stats['ce']}), flush=True)
        after = evaluate(runtime)
        accepted = improved(before, after) and retains_correct_answers(before, after)
        trial = {'mode': mode, 'strategy': head_summary(runtime.model) if mode == 'head' else prefix_summary(runtime.model),
                 'training_seconds': training_seconds, 'after': after,
                 'development_gate_accepted': accepted,
                 'lost_answers': [i for i,(a,b) in enumerate(zip(before['samples'], after['samples'])) if a['correct'] and not b['correct']],
                 'gained_answers': [i for i,(a,b) in enumerate(zip(before['samples'], after['samples'])) if not a['correct'] and b['correct']]}
        destination = args.output.parent/(args.output.stem+'-'+mode+'-candidate')
        runtime.save_checkpoint(destination, metadata={'production_qualified': False, 'role': 'isolated-learning-diagnostic', 'live_promoted': False})
        exported = GeneralistRuntime.from_checkpoint(destination)
        exported.model.eval(); runtime.model.eval()
        with torch.no_grad():
            check_ids = trajectory[0][1][0]
            if not torch.equal(exported.model(check_ids)['logits'], runtime.model(check_ids)['logits']):
                raise AssertionError('native checkpoint export changed trained logits')
        trial['native_export_logits_exact'] = True
        trial['candidate'] = str(destination)
        del exported
        report['trials'].append(trial)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        print(json.dumps({'phase': 'trial complete', 'mode': mode, 'exact_correct': after['exact_correct'], 'nll': after['nll'], 'human_nll': after['human_nll'], 'development_gate_accepted': accepted, 'lost': trial['lost_answers'], 'gained': trial['gained_answers']}), flush=True)
        del runtime, optimizer
        gc.collect()
    if any(file_digest(path) != expected for path, expected in source_hashes.items()):
        raise AssertionError('source checkpoint changed during isolated diagnostic')
    report['source_checkpoint_unchanged'] = True
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
