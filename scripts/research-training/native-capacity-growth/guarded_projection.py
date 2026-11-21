"""Select a trained residual-branch scale using protected TRAIN rows only."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random

import torch
from torch import nn

from trial import (
    BASE_SHA,
    CONFIG_SHA,
    CURRICULUM_SHA,
    LEDGER_SHA,
    RESUME_SHA,
    SEED,
    TOKENIZER_SHA,
    WIDTH,
    ExpandedFF,
    ExtraSwiGLU,
    GeneralistRuntime,
    capture,
    checkpoints,
    encoded,
    greedy,
    pinned,
    retention_audit,
    tensor_digest,
)

CANDIDATE_SHA = '3aa4b5e3a3eb4b41e3db810c5a16d8f5aa48c2cc352e4742e69048f398748210'
GRID = tuple(2.0 ** -exponent for exponent in range(11)) + (0.0,)
CERTIFICATES = 897


class ScaledBranch(nn.Module):
    def __init__(self, branch):
        super().__init__()
        self.branch = branch
        self.scale = 1.0

    def forward(self, inputs):
        return self.branch(inputs) * self.scale


def diagnostic_prompts(data, resume):
    # Replay the original RNG sequence to select exactly the same 24 rows.
    rng = random.Random(SEED)
    rng.sample([row for row in data['train'] if row['family'].startswith('context-')], 48)
    rng.sample([row for row in data['train'] if row['family'] == 'addition'], 48)
    legacy = resume['counters']['champion']['samples']
    development = [dict(row, messages=row['messages'][:-1]) for row in data['development']]
    return rng.sample(legacy, 12) + rng.sample(development, 12)


def compare_predictions(before, after):
    return {
        'before': sum(row['correct'] for row in before),
        'after': sum(row['correct'] for row in after),
        'total': len(before),
        'gains': sum(new['correct'] and not old['correct'] for old, new in zip(before, after, strict=True)),
        'losses': sum(old['correct'] and not new['correct'] for old, new in zip(before, after, strict=True)),
        'changed_generations': sum(old['ids'] != new['ids'] for old, new in zip(before, after, strict=True)),
    }


def run(checkpoint, data_path, candidate):
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    ledger = json.loads(pinned(checkpoint / 'retention_memory.json', LEDGER_SHA))
    pinned(checkpoint / 'model/config.json', CONFIG_SHA)
    pinned(checkpoint / 'model/tokenizer.json', TOKENIZER_SHA)
    runtime = GeneralistRuntime.from_checkpoint(checkpoint / 'model')
    model = runtime.model.eval().requires_grad_(False)
    if tensor_digest(model) != BASE_SHA or ledger['base_champion_sha256'] != BASE_SHA:
        raise ValueError('native weights or ledger do not match revision 4')
    cfg = model.config
    if (cfg.ff_variant, cfg.norm_placement, cfg.recurrent_depth, cfg.dropout) != ('swiglu', 'pre', 1, 0.0):
        raise ValueError('unsupported final block layout')
    artifact = torch.load(BytesIO(pinned(candidate, CANDIDATE_SHA)), map_location='cpu', weights_only=True)
    if artifact['parent_sha256'] != BASE_SHA or artifact['width'] != WIDTH:
        raise ValueError('candidate dimensions or parent differ from the frozen trial')
    extra = ExtraSwiGLU(cfg.d_model, WIDTH).eval().requires_grad_(False)
    extra.load_state_dict(artifact['state'], strict=True)
    if not all(torch.isfinite(parameter).all() for parameter in extra.parameters()):
        raise ValueError('candidate contains non-finite parameters')
    scaled = ScaledBranch(extra).eval()

    protected = [entry['row'] for entry in ledger['entries'].values()]
    if len(protected) != CERTIFICATES:
        raise ValueError('incomplete protected TRAIN ledger')
    print('Capture all protected TRAIN rows', flush=True)
    anchors = capture(runtime, protected)
    if not all(torch.equal(row.scores.argmax(-1), row.targets) for row in anchors):
        raise AssertionError('approved parent failed a protected target')
    tested = []
    for scale in GRID:
        scaled.scale = scale
        audit = retention_audit(model, scaled, anchors)
        tested.append({'scale': scale, **audit})
        print(f'TRAIN scale {scale:g}: {audit["kept"]}/{audit["total"]}', flush=True)
        if audit['kept'] == CERTIFICATES:
            break
    else:
        raise AssertionError('even the zero residual did not preserve all protected TRAIN rows')
    baseline_audit = tested[-1] if scale == 0 else retention_audit_at_zero(model, scaled, anchors, scale)
    if baseline_audit['kept'] != CERTIFICATES or baseline_audit['maximum_absolute_logit_change'] != 0:
        raise AssertionError('cached replay does not match the neutral baseline')

    # Evaluation inputs are loaded only after the scale is fixed by TRAIN.
    data = json.loads(pinned(data_path / 'AUTHORED_CURRICULUM.json', CURRICULUM_SHA))
    resume = torch.load(BytesIO(pinned(checkpoint / 'resume.pt', RESUME_SHA)), map_location='cpu', weights_only=True)
    rows = diagnostic_prompts(data, resume)
    inputs = [encoded(runtime, row)[0].unsqueeze(0) for row in protected[::30]]
    parent_ff = model.blocks[-1].ff
    print('Run neutral and guarded native diagnostics', flush=True)
    neutral = greedy(runtime, rows)
    neutral_logits = checkpoints(runtime, inputs)
    model.blocks[-1].ff = ExpandedFF(parent_ff, scaled).eval()
    try:
        guarded = greedy(runtime, rows)
        guarded_logits = checkpoints(runtime, inputs)
    finally:
        model.blocks[-1].ff = parent_ff
    if tensor_digest(model) != BASE_SHA:
        raise AssertionError('protected model weights changed')

    return {
        'source_sha256': BASE_SHA,
        'candidate_sha256': CANDIDATE_SHA,
        'trial_source_sha256': hashlib.sha256(Path(__file__).with_name('trial.py').read_bytes()).hexdigest(),
        'projection_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'ledger_sha256': LEDGER_SHA,
        'grid': GRID,
        'selected_scale': scale,
        'tested_until_first_pass': tested,
        'zero_baseline': {'kept': baseline_audit['kept'], 'total': baseline_audit['total'],
                          'maximum_absolute_logit_change': baseline_audit['maximum_absolute_logit_change']},
        'native_forward_probe_sequences': len(inputs),
        'native_forward_maximum_absolute_logit_change': max(float((a - b).abs().max())
                                                            for a, b in zip(neutral_logits, guarded_logits, strict=True)),
        'diagnostic_greedy': compare_predictions(neutral, guarded),
        'independent_qualification_performed': False,
        'promotion_authorized': False,
    }


def retention_audit_at_zero(model, scaled, anchors, chosen):
    scaled.scale = 0.0
    try:
        return retention_audit(model, scaled, anchors)
    finally:
        scaled.scale = chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('data', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a fresh report path')
    result = run(args.checkpoint, args.data, args.candidate)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
