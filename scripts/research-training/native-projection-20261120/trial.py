"""Adapt the native output layer and project the update against TRAIN certificates."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from generalist_lm.runtime import GeneralistRuntime
from generalist_lm.training import SFTExample, encode_sft_example, loss_on_examples

SEED = 20261120
RANK = 16
STEPS = 240
BASE_SHA = '57bdb933b82bd8807af5c8b6b545c27ee9114ee4f4c59022434d39b6bf771c79'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def digest(state):
    h = hashlib.sha256()
    for name, value in state.items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sample(rows, seed, size):
    random.Random(seed).shuffle(rows)
    return rows[:size]


def normalized(value):
    return ' '.join(value.casefold().strip().split())


def capture(runtime, rows):
    """Compute once the frozen native features and scores for supervised positions."""
    features, targets = [], []
    for row in rows:
        ids, labels = encode_sft_example(SFTExample.from_dict(row), runtime.tokenizer,
                                         runtime.model.config.context_length)
        length = int((labels != -100).nonzero()[-1, 0]) + 1
        targets.append(labels[1:length][labels[1:length] != -100])
        features.append((ids[:length], labels[1:length] != -100))
    hidden = []
    hook = runtime.model.lm_head.register_forward_pre_hook(lambda _, args: hidden.append(args[0]))
    try:
        with torch.no_grad():
            order = sorted(range(len(rows)), key=lambda i: len(features[i][0]))
            all_hidden, all_scores = [None] * len(rows), [None] * len(rows)
            for start in range(0, len(rows), 8):
                indices = order[start:start + 8]
                batch = nn.utils.rnn.pad_sequence([features[i][0] for i in indices], batch_first=True)
                logits = runtime.model(batch)['logits']
                vectors = hidden.pop()
                for j, i in enumerate(indices):
                    mask = features[i][1]
                    all_hidden[i] = vectors[j, :len(mask)][mask].clone()
                    all_scores[i] = logits[j, :len(mask)][mask].clone()
            cached = list(zip(all_hidden, all_scores, targets))
    finally:
        hook.remove()
    return cached


def stack(rows):
    return tuple(torch.cat(tensors) for tensors in zip(*rows))


def train(pools, anchors, width, vocab):
    rng = random.Random(SEED)
    down = nn.Linear(width, RANK, bias=False)
    up = nn.Linear(RANK, vocab, bias=False)
    nn.init.zeros_(up.weight)
    optimizer = torch.optim.AdamW([*down.parameters(), *up.parameters()], lr=0.003)
    loss_curve = []
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        features, scores, gold = stack([rng.choice(pool) for pool in pools])
        predicted = scores + up(down(features))
        acquire = F.cross_entropy(predicted, gold)
        features, scores, gold = stack(rng.sample(anchors, 16))
        retained = scores + up(down(features))
        replay = F.cross_entropy(retained, gold)
        distill = F.kl_div(retained.log_softmax(-1), scores.softmax(-1), reduction='batchmean')
        loss = acquire + replay + 2 * distill
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*down.parameters(), *up.parameters()], 1., error_if_nonfinite=True)
        optimizer.step()
        if step in (0, 59, 119, 179, STEPS - 1):
            loss_curve.append([step + 1, float(acquire.detach()), float(replay.detach()), float(distill.detach())])
    return (up.weight @ down.weight).detach(), loss_curve


def project(delta, anchors):
    """Scale one linear update to keep positive margins against every alternative."""
    alpha = 1.0
    for start in range(0, len(anchors), 32):
        features, scores, gold = stack(anchors[start:start + 32])
        if not torch.equal(scores.argmax(-1), gold):
            raise ValueError('baseline does not reproduce all TRAIN certificates')
        target = gold[:, None]
        margin = scores.gather(1, target) - scores
        perturbation = F.linear(features, delta)
        adverse = perturbation - perturbation.gather(1, target)
        mask = adverse > 0
        if mask.any():
            alpha = min(alpha, 0.9 * float((margin[mask] / adverse[mask]).min()))
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('invalid projection')
    return alpha


def evaluate(runtime, rows, budget):
    result = []
    for row in rows:
        ids = runtime.tokenizer.serialize_messages(row['messages'][:-1], add_generation_prompt=True)
        generated = runtime._generate_ids(ids, max_new_tokens=budget, temperature=0)
        text = runtime.tokenizer.decode(generated)
        expected = row['expected']
        result.append(dict(family=row.get('family', 'legacy'), expected=expected,
                           answer=text, correct=normalized(text) == normalized(expected)))
    return result


def compare(before, after):
    by_family = defaultdict(lambda: [0, 0, 0])
    for old, new in zip(before, after, strict=True):
        counts = by_family[old['family']]
        counts[0] += old['correct']
        counts[1] += new['correct']
        counts[2] += 1
    return {'total': len(before), 'before': sum(r['correct'] for r in before),
            'after': sum(r['correct'] for r in after), 'by_family': dict(by_family),
            'gains': [i for i, (old, new) in enumerate(zip(before, after)) if new['correct'] and not old['correct']],
            'losses': [i for i, (old, new) in enumerate(zip(before, after)) if old['correct'] and not new['correct']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path, help='parent directory containing model and TRAIN certificates')
    parser.add_argument('data', type=Path, help='rev4 experiment assets')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    dataset = json.loads((args.data / 'AUTHORED_CURRICULUM.json').read_bytes())
    human = read_jsonl(args.data / 'human-train.jsonl')
    authored = dataset['train']
    grouped = {
        'context': [r for r in authored if r['family'].startswith('context-')],
        'calculation': [r for r in authored if r['family'] == 'addition'],
        'copy_and_recall': [r for r in authored if r['family'] in {'foundation-copy', 'foundation-recall'}],
        'dialogue_it': [r for r in authored if r['language'] == 'it'
                        and r['family'] in {'foundation-dialogue', 'extra-dialogue'}] + [r for r in human if r['language'] == 'it'],
        'dialogue_en': [r for r in authored if r['language'] == 'en'
                        and r['family'] in {'foundation-dialogue', 'extra-dialogue'}] + [r for r in human if r['language'] == 'en'],
    }
    selected = {name: sample(rows, SEED + i, 128) for i, (name, rows) in enumerate(grouped.items())}
    memory = json.loads((args.checkpoint / 'retention_memory.json').read_bytes())
    certificates = [entry['row'] for entry in memory['entries'].values()]
    if len(certificates) != 897 or any(not rows for rows in selected.values()):
        raise ValueError('incomplete experiment inputs')
    design = dict(seed=SEED, steps=STEPS, rank=RANK, lr=0.003, source=BASE_SHA,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  split_sha256={name: hashlib.sha256((args.data / name).read_bytes()).hexdigest()
                                for name in ('AUTHORED_CURRICULUM.json', 'human-train.jsonl', 'human-development.jsonl')},
                  sampled_rows={k: len(v) for k, v in selected.items()},
                  gate='2 development gains; zero legacy or development losses; all 897 TRAIN certificates; human NLL <= 2% increase for IT and EN')
    write(args.output / 'PLAN.json', design)
    write(args.output / 'TRAIN.json', selected)
    runtime = GeneralistRuntime.from_checkpoint(args.checkpoint / 'model')
    model = runtime.model.eval()
    if digest(model.state_dict()) != BASE_SHA or memory['base_champion_sha256'] != BASE_SHA:
        raise ValueError('source weights or certificates differ from approved revision')
    print('Capturing frozen TRAIN features', flush=True)
    started = time.monotonic()
    pools = [capture(runtime, rows) for rows in selected.values()]
    anchors = capture(runtime, certificates)
    head = model.lm_head
    delta, losses = train(pools, anchors, head.in_features, head.out_features)
    scale = project(delta, anchors)
    print(json.dumps({'stage': 'trained', 'scale': scale, 'seconds': time.monotonic() - started,
                      'loss_curve': losses}), flush=True)

    resume = torch.load(args.checkpoint / 'resume.pt', weights_only=True)
    legacy = [dict(row, messages=row['messages'] + [{'role': 'assistant', 'content': row['expected']}])
              for row in resume['counters']['champion']['samples']]
    development = dataset['development']
    human_dev = read_jsonl(args.data / 'human-development.jsonl')

    def endpoints():
        result = {'legacy': evaluate(runtime, legacy, 64), 'development': evaluate(runtime, development, 40)}
        result['human_nll'] = {
            lang: loss_on_examples(model, runtime.tokenizer,
                                   [SFTExample.from_dict(row) for row in human_dev if row['language'] == lang])
            for lang in ('it', 'en')}
        return result

    print('Evaluating source', flush=True)
    before = endpoints()
    baseline_weight = head.weight.detach().clone()
    with torch.no_grad():
        head.weight.add_(delta, alpha=scale)
    after_sha = digest(model.state_dict())
    print('Evaluating adapted checkpoint', flush=True)
    after = endpoints()
    cached_after = capture(runtime, certificates)
    kept = sum(torch.equal(scores.argmax(-1), gold) for _, scores, gold in cached_after)
    dev = compare(before['development'], after['development'])
    old = compare(before['legacy'], after['legacy'])
    passed = (after_sha != BASE_SHA and len(dev['gains']) >= 2 and not dev['losses'] and not old['losses']
              and kept == 897 and all(after['human_nll'][k] <= v * 1.02 for k, v in before['human_nll'].items()))
    final = None
    if passed:
        rows = dataset['sealed_transfer']
        final_after = evaluate(runtime, rows, 40)
        with torch.no_grad():
            head.weight.copy_(baseline_weight)
        final = compare(evaluate(runtime, rows, 40), final_after)
        with torch.no_grad():
            head.weight.add_(delta, alpha=scale)
    torch.save({'delta': delta, 'scale': scale, 'parent_sha256': BASE_SHA}, args.output / 'head-update.pt')
    summary = dict(source=BASE_SHA, candidate=after_sha, optimizer_steps=STEPS, scale=scale,
                   retained=kept, legacy=old, development=dev, before_nll=before['human_nll'],
                   after_nll=after['human_nll'], qualified=passed, final=final)
    write(args.output / 'REPORT.json', dict(summary=summary, loss_curve=losses, before=before, after=after))
    write(args.output / 'SUMMARY.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
