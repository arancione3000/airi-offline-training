"""Read-only native SFT/gold/EOS/KV/chat audit on declared diagnostic TRAIN rows.

This does not train, select candidates, consume held-out rows, or promote a model.
Only the requested JSON report is written. Run each endpoint with the same rows.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--rows', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--label', default='native-endpoint')
    args = parser.parse_args()
    import torch
    from generalist_lm.runtime import GeneralistRuntime, checkpoint_model_files
    from generalist_lm.training import SFTExample, encode_sft_example
    from generalist_lm.tokenizer import EOS

    torch.set_num_threads(2)
    source_paths = checkpoint_model_files(args.checkpoint, verify=True)
    source_hashes = {str(path): digest(path) for path in source_paths}
    for name in ('config.json', 'tokenizer.json', 'metadata.json'):
        path = args.checkpoint / name
        if path.exists():
            source_hashes[str(path)] = digest(path)
    loaded = json.loads(args.rows.read_text())
    rows = loaded if isinstance(loaded, list) else loaded.get('train', loaded.get('rows'))
    if not isinstance(rows, list) or not 1 <= len(rows) <= 16:
        raise ValueError('audit requires 1..16 declared TRAIN rows')
    runtime = GeneralistRuntime.from_checkpoint(args.checkpoint)
    runtime.model.eval()
    results = []
    started = time.monotonic()
    with torch.no_grad():
        for index, row in enumerate(rows):
            messages = row['messages']
            example = SFTExample.from_dict(row)
            prompt = runtime.tokenizer.serialize_messages(messages[:-1], add_generation_prompt=True)
            target = runtime.tokenizer.encode(messages[-1]['content'], eos=True)
            if not 1 < len(target) <= 64:
                raise ValueError('audit requires short, nonempty answers with EOS')
            ids, labels = encode_sft_example(example, runtime.tokenizer,
                                             runtime.config.context_length,
                                             require_complete=True)
            length = len(prompt) + len(target)
            ids, labels = ids[:length].unsqueeze(0), labels[:length].unsqueeze(0)
            if ids[0].tolist() != prompt + target:
                raise AssertionError('native prompt / SFT token mismatch')
            if labels[0].tolist() != [-100] * len(prompt) + target:
                raise AssertionError('SFT prompt/target mask mismatch')
            logits = runtime.model(ids)['logits']
            gold_scores = logits[0, len(prompt)-1:length-1].float()
            gold = torch.tensor(target, dtype=torch.long)
            gold_predictions = gold_scores.argmax(-1)
            alternatives = gold_scores.clone()
            alternatives.scatter_(1, gold[:, None], -float('inf'))
            margins = gold_scores.gather(1, gold[:, None]).squeeze(1) - alternatives.max(-1).values
            certificate = bool(torch.equal(gold_predictions, gold))
            # Modifying the final input token must not alter any prior logit.
            future_changed = ids.clone()
            future_changed[0, -1] = (int(ids[0, -1]) + 1) % runtime.config.vocab_size
            future_logits = runtime.model(future_changed)['logits']
            causal_difference = float((logits[:, :-1] - future_logits[:, :-1]).abs().max())
            prompt_tensor = torch.tensor([prompt], dtype=torch.long)
            budget = max(12, len(target) + 2)
            rng_before = torch.get_rng_state().clone()
            cached = runtime.model.generate(prompt_tensor, max_new_tokens=budget,
                                              eos_token_id=EOS, temperature=0.,
                                              use_cache=True)[0, len(prompt):].tolist()
            uncached = runtime.model.generate(prompt_tensor, max_new_tokens=budget,
                                                eos_token_id=EOS, temperature=0.,
                                                use_cache=False)[0, len(prompt):].tolist()
            chat = runtime.chat(messages[:-1], temperature=0., max_new_tokens=budget)
            rng_unchanged = bool(torch.equal(rng_before, torch.get_rng_state()))
            cache_equal = cached == uncached
            chat_equal = chat == runtime.tokenizer.decode(cached)
            chain_equal = cached == target
            findings = []
            if causal_difference != 0.0:
                findings.append('future input changed a causal prefix logit')
            if not cache_equal:
                findings.append('cached and uncached native raw greedy IDs differ')
            if not chat_equal:
                findings.append('native chat differs from raw cached generation decoding')
            if not rng_unchanged:
                findings.append('native greedy consumed torch RNG')
            if certificate and not chain_equal:
                findings.append('gold argmax certificate differs from native greedy chain')
            results.append(dict(index=index, key=row.get('id', row.get('key')),
                prompt_tokens=len(prompt), target_ids=target,
                target_text=messages[-1]['content'],
                shifted_gold_argmax_ids=gold_predictions.tolist(),
                gold_with_eos_certificate=certificate,
                min_gold_margin=float(margins.min()),
                first_gold_prediction_index=len(prompt)-1,
                prompt_mask_exact=True, eos_target_present=target[-1] == EOS,
                future_perturbation_max_logit_difference=causal_difference,
                cached_generated_ids=cached, uncached_generated_ids=uncached,
                cached_uncached_ids_equal=cache_equal,
                native_chat_text=chat, native_chat_equals_raw_decoding=chat_equal,
                native_raw_target_and_eos_exact=chain_equal,
                decoded_text_exact=chat == messages[-1]['content'],
                generated_eos=bool(cached and cached[-1] == EOS),
                greedy_preserves_torch_rng=rng_unchanged,
                max_new_tokens=budget, findings=findings))
    for path, expected in source_hashes.items():
        if digest(path) != expected:
            raise AssertionError('source checkpoint changed')
    report = dict(label=args.label, checkpoint=str(args.checkpoint),
        rows=str(args.rows), rows_sha256=digest(args.rows),
        script_sha256=digest(__file__), torch_version=torch.__version__,
        checkpoint_hashes=source_hashes, source_checkpoint_unchanged=True,
        train_rows_only=True, trained=False, promoted=False,
        elapsed_seconds=time.monotonic()-started, complete=True,
        passed=not any(r['findings'] for r in results), samples=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(args.output)
    print(json.dumps(dict(label=args.label, rows=len(rows), complete=True,
        passed=report['passed'],
        certificates=sum(r['gold_with_eos_certificate'] for r in results),
        raw_target_and_eos_exact=sum(r['native_raw_target_and_eos_exact'] for r in results),
        seconds=report['elapsed_seconds'], report=str(args.output))))


if __name__ == '__main__':
    main()
