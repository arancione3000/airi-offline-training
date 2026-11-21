"""Research-only, zero-initial-output capacity expansion of the native decoder."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

from generalist_lm.runtime import GeneralistRuntime
from generalist_lm.training import SFTExample, encode_sft_example


SEED = 20261121
WIDTH = 64
STEPS = 64
BASE_SHA = '57bdb933b82bd8807af5c8b6b545c27ee9114ee4f4c59022434d39b6bf771c79'
CURRICULUM_SHA = '7636f8610e883877ee149a5676a78a4d96f706b5c6d40a4441224115cb24f56f'
LEDGER_SHA = 'd977083b9ecf127554b5812aac4524f6f8d5be085c1b1d634d151029eaf98609'
RESUME_SHA = 'd47eacdf55da50b98af81b46dd8925a9998b5615239a71f78563cc876d77c984'
CONFIG_SHA = 'e59c79048b447e59710c2f4827966f022b9ad93d4bd19cb111c6c4bb777b889c'
TOKENIZER_SHA = '1099cf0cd718706212492967cb607034ef5fa8321a3e81125a106761b2d247f3'


class ExtraSwiGLU(nn.Module):
    def __init__(self, input_width, expansion_width):
        super().__init__()
        self.up = nn.Linear(input_width, 2 * expansion_width, bias=False)
        self.down = nn.Linear(expansion_width, input_width, bias=False)
        nn.init.normal_(self.up.weight, std=0.12)
        nn.init.zeros_(self.down.weight)

    def forward(self, inputs):
        gate, value = self.up(inputs).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class ExpandedFF(nn.Module):
    def __init__(self, parent, extra):
        super().__init__()
        self.parent = parent
        self.extra = extra

    def forward(self, inputs):
        return self.parent(inputs) + self.extra(inputs)


@dataclass(frozen=True)
class Features:
    residual: torch.Tensor
    normalized: torch.Tensor
    original_ff: torch.Tensor
    scores: torch.Tensor
    targets: torch.Tensor


def pinned(path, expected):
    with path.open('rb') as stream:
        value = stream.read()
    if hashlib.sha256(value).hexdigest() != expected:
        raise ValueError(f'input differs from the approved reference: {path}')
    return value


def tensor_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def encoded(runtime, row):
    ids, labels = encode_sft_example(SFTExample.from_dict(row), runtime.tokenizer,
                                     runtime.model.config.context_length)
    stop = int((labels != -100).nonzero()[-1, 0]) + 1
    return ids[:stop], labels[1:stop]


def capture(runtime, rows):
    """Capture final-block state on only gold next-token positions."""
    model = runtime.model
    block = model.blocks[-1]
    streams = {}
    hooks = [
        block.ff_norm.register_forward_hook(
            lambda _, args, out: streams.update(residual=args[0], normalized=out)),
        block.ff.register_forward_hook(lambda _, args, out: streams.update(original_ff=out)),
    ]
    items = [encoded(runtime, row) for row in rows]
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    recorded = [None] * len(items)
    try:
        with torch.no_grad():
            for start in range(0, len(items), 12):
                batch = order[start:start + 12]
                ids = nn.utils.rnn.pad_sequence([items[i][0] for i in batch], batch_first=True)
                scores = model(ids)['logits']
                for j, i in enumerate(batch):
                    labels = items[i][1]
                    mask = labels != -100
                    vectors = [streams[name][j, :len(labels)][mask].clone()
                               for name in ('residual', 'normalized', 'original_ff')]
                    recorded[i] = Features(*vectors, scores[j, :len(labels)][mask].clone(), labels[mask])
    finally:
        for hook in hooks:
            hook.remove()
    return recorded


def stack(rows):
    return Features(*(torch.cat([getattr(r, field) for r in rows])
                      for field in ('residual', 'normalized', 'original_ff', 'scores', 'targets')))


def expanded_logits(model, branch, data):
    residual = data.residual + (data.original_ff + branch(data.normalized))
    return model.lm_head(model.final_norm(residual))


def train(model, branch, pools, anchors):
    rng = random.Random(SEED)
    optimizer = torch.optim.AdamW(branch.parameters(), lr=0.006)
    steps = []
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        examples = stack([row for pool in pools for row in rng.sample(pool, 4)])
        replay = stack(rng.sample(anchors, 16))
        scores = expanded_logits(model, branch, examples)
        protected = expanded_logits(model, branch, replay)
        acquisition_loss = F.cross_entropy(scores, examples.targets)
        retention_loss = F.kl_div(protected.log_softmax(-1), replay.scores.softmax(-1),
                                  reduction='batchmean')
        (acquisition_loss + 3 * retention_loss).backward()
        nn.utils.clip_grad_norm_(branch.parameters(), 1.0, error_if_nonfinite=True)
        if step in (0, 1, STEPS - 1):
            steps.append({'step': step + 1, 'acquisition_nll': float(acquisition_loss.detach()),
                          'retention_kl': float(retention_loss.detach()),
                          'up_gradient_norm': float(branch.up.weight.grad.norm()),
                          'down_gradient_norm': float(branch.down.weight.grad.norm())})
        optimizer.step()
    if steps[0]['up_gradient_norm'] != 0 or steps[0]['down_gradient_norm'] <= 0 or steps[1]['up_gradient_norm'] <= 0:
        raise AssertionError('gradient does not activate the newly added route as expected')
    return steps


def normalized(text):
    return ' '.join(text.casefold().split())


def greedy(runtime, rows):
    result = []
    for row in rows:
        ids = runtime.tokenizer.serialize_messages(row['messages'], add_generation_prompt=True)
        generated = runtime._generate_ids(ids, max_new_tokens=24, temperature=0)
        text = runtime.tokenizer.decode(generated)
        result.append({'ids': generated, 'correct': normalized(text) == normalized(row['expected'])})
    return result


def checkpoints(runtime, inputs):
    with torch.no_grad():
        return [runtime.model(ids)['logits'].detach().clone() for ids in inputs]


def retention_audit(model, branch, rows):
    kept = 0
    largest_difference = 0.0
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            data = stack(rows[start:start + 32])
            scores = expanded_logits(model, branch, data)
            largest_difference = max(largest_difference, float((scores - data.scores).abs().max()))
            correct = scores.argmax(-1).eq(data.targets)
            kept += sum(bool(t.all()) for t in correct.split([r.targets.numel() for r in rows[start:start + 32]]))
    return {'kept': kept, 'total': len(rows), 'maximum_absolute_logit_change': largest_difference}


def count(model):
    return sum(p.numel() for p in model.parameters())


def write(path, content):
    path.write_text(json.dumps(content, indent=2) + '\n')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('data', type=Path)
    cli.add_argument('output', type=Path)
    opts = cli.parse_args()
    opts.output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)

    data = json.loads(pinned(opts.data / 'AUTHORED_CURRICULUM.json', CURRICULUM_SHA))
    ledger = json.loads(pinned(opts.checkpoint / 'retention_memory.json', LEDGER_SHA))
    from io import BytesIO
    resume = torch.load(BytesIO(pinned(opts.checkpoint / 'resume.pt', RESUME_SHA)), weights_only=True)
    pinned(opts.checkpoint / 'model/config.json', CONFIG_SHA)
    pinned(opts.checkpoint / 'model/tokenizer.json', TOKENIZER_SHA)
    runtime = GeneralistRuntime.from_checkpoint(opts.checkpoint / 'model')
    model = runtime.model.eval()
    if tensor_digest(model) != BASE_SHA or ledger['base_champion_sha256'] != BASE_SHA:
        raise ValueError('native weights do not match revision 4')
    cfg = model.config
    if (cfg.ff_variant, cfg.norm_placement, cfg.recurrent_depth, cfg.dropout) != ('swiglu', 'pre', 1, 0.0):
        raise ValueError('unsupported final block layout')
    train_pools = [[r for r in data['train'] if r['family'].startswith('context-')],
                   [r for r in data['train'] if r['family'] == 'addition']]
    rng = random.Random(SEED)
    selected = [rng.sample(pool, 48) for pool in train_pools]
    protected = [entry['row'] for entry in ledger['entries'].values()]
    if len(protected) != 897:
        raise ValueError('incomplete protected ledger')
    legacy = resume['counters']['champion']['samples']
    development = [dict(row, messages=row['messages'][:-1]) for row in data['development']]
    probes = rng.sample(legacy, 12) + rng.sample(development, 12)
    model.requires_grad_(False)
    parent_ff = model.blocks[-1].ff
    before_parameters = count(model)
    branch = ExtraSwiGLU(cfg.d_model, WIDTH).eval()

    write(opts.output / 'PLAN.json', {'seed': SEED, 'addition_width': WIDTH, 'steps': STEPS,
                                      'source_sha256': BASE_SHA, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                      'train_sha256': CURRICULUM_SHA, 'ledger_sha256': LEDGER_SHA,
                                      'train_rows': [len(pool) for pool in selected], 'greedy_max_new_tokens': 24,
                                      'legacy_sample': 12, 'development_sample': 12})
    print('Frozen activation capture', flush=True)
    anchors = capture(runtime, protected)
    pools = [capture(runtime, rows) for rows in selected]
    if not all(torch.equal(r.scores.argmax(-1), r.targets) for r in anchors):
        raise AssertionError('approved source does not reproduce its certified targets')
    inputs = [encoded(runtime, row)[0].unsqueeze(0) for row in protected[::30]]
    raw_before = checkpoints(runtime, inputs)
    print('Native zero-update check and sampled greedy baseline', flush=True)
    original_generations = greedy(runtime, probes)
    model.blocks[-1].ff = ExpandedFF(parent_ff, branch).eval()
    raw_zero = checkpoints(runtime, inputs)
    initial = {'full_native_forward_probes': len(inputs), 'positions': sum(x.shape[1] for x in inputs),
               'largest_absolute_logit_difference': max(float((a - b).abs().max()) for a, b in zip(raw_before, raw_zero)),
               'greedy_ids_identical': all(a['ids'] == b['ids'] for a, b in zip(original_generations, greedy(runtime, probes)))}
    if initial['largest_absolute_logit_difference'] != 0 or not initial['greedy_ids_identical']:
        raise AssertionError('new capacity must preserve the initial route')
    zero_retention = retention_audit(model, branch, anchors)
    if zero_retention['kept'] != len(anchors):
        raise AssertionError('initial branch disturbed protected targets')
    print('Training only the added SwiGLU weights', flush=True)
    learning = train(model, branch, pools, anchors)
    trained_retention = retention_audit(model, branch, anchors)
    actual_after = checkpoints(runtime, inputs)
    actual_kept = 0
    for row, scores in zip(protected[::30], actual_after, strict=True):
        _, labels = encoded(runtime, row)
        mask = labels != -100
        actual_kept += int(torch.equal(scores[0, :len(mask)][mask].argmax(-1), labels[mask]))
    print('Native sampled greedy candidate', flush=True)
    after_generations = greedy(runtime, probes)
    gains = sum(cur['correct'] and not old['correct'] for old, cur in zip(original_generations, after_generations))
    losses = sum(old['correct'] and not cur['correct'] for old, cur in zip(original_generations, after_generations))
    after_parameters = count(model)
    model.blocks[-1].ff = parent_ff
    if tensor_digest(model) != BASE_SHA:
        raise AssertionError('training changed a pre-existing parameter')
    model.blocks[-1].ff = ExpandedFF(parent_ff, branch).eval()
    candidate = {'parent_sha256': BASE_SHA, 'width': WIDTH, 'state': branch.state_dict()}
    torch.save(candidate, opts.output / 'candidate-branch.pt')
    recovered = torch.load(opts.output / 'candidate-branch.pt', weights_only=True)
    reloaded_branch = ExtraSwiGLU(cfg.d_model, recovered['width']).eval()
    reloaded_branch.load_state_dict(recovered['state'], strict=True)
    model.blocks[-1].ff = ExpandedFF(parent_ff, reloaded_branch).eval()
    export_ok = torch.equal(runtime.model(inputs[0])['logits'], actual_after[0])
    if not export_ok:
        raise AssertionError('round-tripped branch changed full native logits')
    write(opts.output / 'SUMMARY.json', {
        'source_sha256': BASE_SHA, 'parameters_before': before_parameters, 'parameters_after': after_parameters,
        'parameters_added': after_parameters - before_parameters, 'zero_initialization': initial,
        'training': learning, 'retention_at_initialization': zero_retention,
        'retention_after_training': trained_retention, 'full_native_retention_after_training': [actual_kept, len(inputs)],
        'sampled_greedy': {'before': sum(r['correct'] for r in original_generations),
                           'after': sum(r['correct'] for r in after_generations), 'total': len(probes),
                           'gains': gains, 'losses': losses},
        'branch_roundtrip_exact': export_ok, 'promoted': False,
    })


if __name__ == '__main__':
    main()
