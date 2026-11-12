"""Single predeclared 256-unit FFN-output ablation; never promotes or touches live state.

Both arms start at the same native champion, fresh AdamW, fixed identical TRAIN
trajectory and sampled guards. Only native final FFN down trainability differs.
The fixed memory sampler deliberately omits adaptive per-arm drift prioritization;
this is a controlled scope test, not an exact production replay or equal-FLOPs test.
"""
import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'transfer-training'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--native-replay', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.seed = 20261112
    args.steps = 256
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
    resume = torch.load(args.checkpoint.parent/'resume.pt', weights_only=True)
    baseline_saved = resume['counters']['champion']
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
    from composition_data import sampling_pools
    composition = sampling_pools(authored)
    humans = {lang: [r for r in human_train if r['language'] == lang] for lang in ('it', 'en')}
    print(json.dumps({'phase': 'data ready', 'authored': len(authored), 'development': len(heldout),
                      'human_counts': {k: len(v) for k,v in corpus.items()}, 'human_development': len(human_dev),
                      'anchors': len(anchors)}), flush=True)

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
    from context_acquisition import context_skeletons, context_value_pairs, matched_context_pair, paired_batch
    from run import validate_training_registry, register_context_pair
    from retention_memory import validate_memory, sample_rows, make_certifier
    from retention_guard import make_retention_guard
    from retention_transaction import transactional_update
    from generalist_lm.tokenizer import EOS
    memory = json.loads((args.checkpoint.parent/'retention_memory.json').read_text())
    registry = json.loads((args.checkpoint.parent/'training_registry.json').read_text())
    forbidden = heldout + sealed + corpus['development'] + corpus['final']
    context_excluded = blocked | all_heldout | dev_texts
    reserved = {row['expected'] for row in forbidden}
    protected_histories = {prompt_key(row) for row in forbidden + old_validation}
    originals = [row for row in authored if row['family'].startswith('context-')]
    values = dataset['context_acquisition_values']
    validation_options = dict(fingerprint=memory['fingerprint'], blocked=context_excluded,
                              reserved_values=reserved, protected_histories=protected_histories)
    registered = validate_training_registry(registry, originals, values, forbidden, **validation_options)
    eligible.update(catalog(registered, forbidden))
    from legacy_run import state_digest
    validate_memory(memory, eligible, memory['fingerprint'], state_digest(champion.model))
    certify = make_certifier(champion.model, encode, EOS)
    baseline_certificates = {key: certify(entry['row']) for key,entry in memory['entries'].items()}
    if len(baseline_certificates) != 897 or not all(c['correct'] for c in baseline_certificates.values()):
        raise ValueError('complete approved 897-row source retention ledger not reproduced')
    tick = time.monotonic()
    before = evaluate(champion)
    if (before['exact_correct'] != baseline_saved['exact_correct']
            or any(abs(before['nll'][k]-v) > 1e-5 for k,v in baseline_saved['nll'].items())
            or any(abs(before['human_nll'][k]-v) > 1e-5 for k,v in baseline_saved['human_nll'].items())
            or any((a['response'],a['correct']) != (b['response'],b['correct'])
                   for a,b in zip(before['samples'],baseline_saved['samples']))):
        raise ValueError('archived baseline not reproduced')
    print(json.dumps({'phase':'baseline reproduced','score':before['exact_correct'],'certificates':len(baseline_certificates),'seconds':time.monotonic()-tick}),flush=True)
    learning = spec['continual_learning']
    schedule = learning['sampling_schedule']
    recovery = dict(accepted_streak=64, cooldown_updates=32, growth_factor=2.)
    rng = random.Random(args.seed)
    retention_rng = random.Random(args.seed+1)
    sampled_memory = copy.deepcopy(memory)
    skeletons = context_skeletons(originals)
    value_pools = context_value_pairs(values)
    trajectory = []
    manifest = []
    from success_bank import row_key
    for step in range(args.steps):
        lang = 'it' if step%3 != 1 else 'en'
        group = schedule[step%16]
        chosen = []
        if group == 'human-dialogue':
            languages = sorted(humans)
            lang = rng.choices(languages, weights=[math.sqrt(len(humans[k])) for k in languages], k=1)[0]
            chosen = [rng.choice(humans[lang])]; batch = encode(chosen[0])
        elif group == 'native-replay':
            replay_group = ('language-it','language-en','domains')[(step//16)%3]
            batch = rng.choice(replay[replay_group])
        elif group == 'verified-context':
            families = sorted(family for (language,family),groups in skeletons.items()
                              if language == lang and any(rows[0]['stage'] <= resume['counters']['curriculum_stage'] for rows in groups))
            family = rng.choice(families)
            source = rng.choice([rows for rows in skeletons[(lang,family)] if rows[0]['stage'] <= resume['counters']['curriculum_stage']])[0]
            value_pair = rng.choice(value_pools[(lang,family)])
            chosen = matched_context_pair(source,value_pair,blocked=context_excluded,
                         reserved_values=reserved,protected_histories=protected_histories)
            chosen = register_context_pair(registry, source, chosen, values=value_pair)
            eligible.update(catalog(chosen,forbidden))
            batch = paired_batch([encode(row) for row in chosen])
        elif group == 'composition':
            pool = composition[lang]; family = rng.choice(sorted(pool))
            chosen = [rng.choice(pool[family])]; batch = encode(chosen[0])
        else:
            pool = (native if group == 'old-authored' else curated)[lang]
            if group == 'old-authored':
                family = rng.choice(sorted({row['family'] for row in pool}))
                pool = [row for row in pool if row['family'] == family]
            chosen = [rng.choice(pool)]; batch = encode(chosen[0])
        anchor = rng.choice(anchors)
        retained = sample_rows(sampled_memory,eligible,retention_rng,per_language=2)
        guarded = sample_rows(sampled_memory,eligible,retention_rng,per_language=4)
        record = dict(step=step+1,group=group,language=lang,acquisition_keys=[row_key(row) for row in chosen],
                      acquisition_ids_sha256=hashlib.sha256(batch[0].numpy().tobytes()).hexdigest(),
                      anchor_ids_sha256=hashlib.sha256(anchor[0].numpy().tobytes()).hexdigest(),
                      retention_keys=[row_key(row) for row in retained],guard_keys=[row_key(row) for row in guarded])
        manifest.append(record)
        trajectory.append((group,batch,[encode(row) for row in retained],anchor,[encode(row) for row in guarded]))
    validate_training_registry(registry, originals, values, forbidden, **validation_options)
    code_files = list((Path(__file__).resolve().parents[1]/'transfer-training').glob('*.py')) + [Path(__file__).resolve()]
    code_hashes = {str(path):file_digest(path) for path in code_files}
    report = dict(scope=__doc__, seed=args.seed,steps=args.steps,checkpoint=str(args.checkpoint),
                  torch_version=torch.__version__,source_files=source_hashes,code_files=code_hashes,
                  reference_sha256=file_digest(args.reference/'model.pt'),schedule=schedule,
                  objective=dict(learning_rate=3e-5,retention_weight=8.,champion_kl_weight=8.,native_weight=.25,prompt_kl_weight=8.),
                  guards=dict(examples_per_language=4,max_prompt_kl=.01,max_nll_regression=.02,retry_scales=[1.,.5,.25,.125],lr_recovery=recovery),
                  optimizer='fresh AdamW for both arms, weight_decay .01; original Adam not reused because scope changes',
                  sampling_policy='fixed pre-materialized identical TRAIN acquisition, anchors, retention and sentinel IDs; sampler drift observations frozen at champion, unlike production adaptation',
                  generation=dict(temperature=0,max_new_tokens=64,original_samples=92),
                  sealed_forward=False,live_promoted=False,baseline=before,baseline_certificates=baseline_certificates,
                  trajectory=manifest,trials=[])
    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
        temporary.replace(args.output)
    save()
    for arm,train_down in (('A_frozen_ffn',False),('B_train_ffn_output',True)):
        torch.manual_seed(args.seed)
        runtime = GeneralistRuntime.from_checkpoint(args.checkpoint)
        cache_tokenizer(runtime.tokenizer)
        configure_prefix_tuning(runtime.model,train_ffn_output=train_down)
        parameters = [p for p in runtime.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters,lr=3e-5,weight_decay=.01)
        state = dict(schema=1,lr_scale=1.,attempted_updates=0,committed_updates=0,rejected_updates=0)
        attempts, logical = [], []
        started = time.monotonic()
        for step,(group,batch,retained,anchor,guards) in enumerate(trajectory,1):
            guard = make_retention_guard(runtime.model,champion.model,guards,max_prompt_kl=.01,max_nll_regression=.02)
            def propose():
                stats = continual_step(runtime.model,champion.model,reference.model,optimizer,batch,retained,anchor,
                         retention_weight=8.,champion_kl_weight=8.,native_weight=.25,prompt_kl_weight=8.)
                attempts.append(dict(step=step,**stats))
                return stats
            outcome = transactional_update(runtime.model,optimizer,propose,guard,state=state,
                                            scales=(1.,.5,.25,.125),recovery=recovery)
            logical.append(dict(step=step,group=group,**outcome))
            if step%32 == 0:
                print(json.dumps(dict(phase='training',arm=arm,step=step,seconds=time.monotonic()-started,state=state)),flush=True)
        training_seconds = time.monotonic()-started
        after = evaluate(runtime)
        certificates = {key: make_certifier(runtime.model,encode,EOS)(entry['row']) for key,entry in memory['entries'].items()}
        destination = args.output.parent/(args.output.stem+'-'+arm+'-candidate')
        runtime.save_checkpoint(destination,metadata=dict(production_qualified=False,role='isolated-ffn-ablation',live_promoted=False))
        exported = GeneralistRuntime.from_checkpoint(destination)
        runtime.model.eval(); exported.model.eval()
        with torch.no_grad():
            if not torch.equal(exported.model(trajectory[0][1][0])['logits'],runtime.model(trajectory[0][1][0])['logits']):
                raise AssertionError('native export changed trained logits')
        norms = [x['gradient_norm_before_clipping'] for x in attempts]
        trial = dict(arm=arm,train_ffn_output=train_down,strategy=prefix_summary(runtime.model),
                     training_seconds=training_seconds,physical_attempts=len(attempts),transaction=state,
                     physical_supervised_tokens=sum(x['supervised_tokens'] for x in attempts),
                     physical_reference_tokens=sum(x['reference_tokens'] for x in attempts),
                     gradient_norm=dict(mean=sum(norms)/len(norms),minimum=min(norms),maximum=max(norms),clipped_fraction=sum(n>1 for n in norms)/len(norms)),
                     after=after,lost_answers=[i for i,(a,b) in enumerate(zip(before['samples'],after['samples'])) if a['correct'] and not b['correct']],
                     gained_answers=[i for i,(a,b) in enumerate(zip(before['samples'],after['samples'])) if not a['correct'] and b['correct']],
                     context_correct=sum(x['correct'] for x in after['samples'] if x['family'].startswith('context-')),
                     retention_certificates=certificates,lost_retention_keys=[k for k,c in certificates.items() if not c['correct']],
                     native_export_logits_exact=True,candidate=str(destination),logical_updates=logical,attempt_statistics=attempts)
        report['trials'].append(trial); save()
        print(json.dumps(dict(phase='endpoint',arm=arm,score=after['exact_correct'],context=trial['context_correct'],lost=trial['lost_answers'],gained=trial['gained_answers'],retention_lost=len(trial['lost_retention_keys']),seconds=training_seconds)),flush=True)
        del runtime,optimizer,exported,parameters
        gc.collect()
    if any(file_digest(path)!=expected for path,expected in source_hashes.items()):
        raise AssertionError('source checkpoint changed')
    if any(file_digest(path)!=expected for path,expected in code_hashes.items()):
        raise AssertionError('training code changed during experiment')
    report['source_checkpoint_unchanged']=True
    report['training_code_unchanged']=True
    report['complete']=True
    save()


if __name__ == '__main__':
    main()
