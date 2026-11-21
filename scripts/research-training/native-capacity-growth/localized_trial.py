"""Fit an experimental operand-conditioned output head beside frozen native Airi."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import torch
from torch import nn

from trial import (
    BASE_SHA, CONFIG_SHA, CURRICULUM_SHA, LEDGER_SHA, TOKENIZER_SHA,
    GeneralistRuntime, capture, count, normalized, pinned, tensor_digest,
)


ADDITION = re.compile(r'(?:Quanto fa|What is) ([1-9][0-9]?) \+ ([1-9][0-9]?)\?')
MAX_OPERAND = 64
MAX_GENERATED_TOKENS = 8


def operands(messages):
    """Route only an exact, context-free integer request from one user."""
    if type(messages) is not list or len(messages) != 1:
        return None
    entry = messages[0]
    if (type(entry) is not dict or set(entry) != {'role', 'content'} or entry['role'] != 'user'
            or type(entry['content']) is not str):
        return None
    match = ADDITION.fullmatch(entry['content'])
    if match:
        pair = tuple(map(int, match.groups()))
        if max(pair) <= MAX_OPERAND:
            return pair
    return None


def fit(features, labels):
    # Fit the affine map from reference answers. No arithmetic operator is
    # supplied to the estimator or to its output renderer.
    matrix = torch.cat((features, torch.ones((len(features), 1), dtype=torch.float64)), dim=1)
    solved = torch.linalg.lstsq(matrix, labels, driver='gelsd')
    head = nn.Linear(2, 1, dtype=torch.float64).eval().requires_grad_(False)
    values = solved.solution
    with torch.no_grad():
        head.weight.copy_(values[:2].reshape(1, 2))
        head.bias.copy_(values[2:].reshape(1))
    error = float((head(features).flatten() - labels.flatten()).abs().max())
    return head, int(solved.rank), error


def generated(native, messages):
    ids = native.tokenizer.serialize_messages(messages, add_generation_prompt=True)
    answer = native._generate_ids(ids, max_new_tokens=MAX_GENERATED_TOKENS, temperature=0)
    return native.tokenizer.decode(answer)


def predicted(head, pair):
    with torch.no_grad():
        number = round(float(head(torch.tensor(pair, dtype=torch.float64)).item()))
    return str(number)


def respond(runtime, messages):
    pair = operands(messages)
    if pair is not None:
        return predicted(runtime.model.localized_addition, pair)
    return generated(runtime, messages)


def pair_rows():
    # Values and forms fixed independently of the trained estimator.
    for a, b in ((1, 13), (2, 17), (3, 19), (4, 23), (5, 29), (6, 31), (7, 37), (8, 41),
                 (9, 43), (10, 47), (11, 53), (12, 59), (13, 61), (14, 64), (15, 15), (16, 16),
                 (17, 34), (18, 27), (19, 49), (20, 54), (31, 57), (39, 60), (56, 18), (64, 64)):
        for form in ('Quanto fa', 'What is'):
            yield {'messages': [{'role': 'user', 'content': f'{form} {a} + {b}?'}],
                   'expected': str(a + b)}


def score(rows, native, control):
    output = []
    for row in rows:
        query = row['messages']
        pair = operands(query)
        if pair is None:
            raise AssertionError('test is outside the specified route')
        output.append({'messages': query, 'expected': row['expected'],
                       'native': generated(native, query),
                       'constant_control': predicted(control, (0, 0)),
                       'learned': respond(native, query)})
    outcome = {key: sum(normalized(r[key]) == r['expected'] for r in output)
               for key in ('native', 'constant_control', 'learned')}
    outcome.update(total=len(output), rows=output)
    return outcome


def digest_rows(rows):
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('data', type=Path)
    cli.add_argument('output', type=Path)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    checkpoint = args.checkpoint
    data = json.loads(pinned(args.data / 'AUTHORED_CURRICULUM.json', CURRICULUM_SHA))
    ledger = json.loads(pinned(checkpoint / 'retention_memory.json', LEDGER_SHA))
    pinned(checkpoint / 'model/config.json', CONFIG_SHA)
    pinned(checkpoint / 'model/tokenizer.json', TOKENIZER_SHA)
    runtime = GeneralistRuntime.from_checkpoint(checkpoint / 'model')
    model = runtime.model.eval().requires_grad_(False)
    if tensor_digest(model) != BASE_SHA or ledger['base_champion_sha256'] != BASE_SHA:
        raise ValueError('frozen decoder differs from approved revision 4')
    train = [r for r in data['train'] if r['family'] == 'addition']
    train_pairs = [operands(r['messages'][:-1]) for r in train]
    if len(train) != 480 or None in train_pairs:
        raise AssertionError('train rows do not match the registered addition route')
    # Claim only genuinely new operand pairs on the independent challenge.
    previous = {operands(r['messages'][:-1]) for r in data['development']}
    fresh = list(pair_rows())
    if {operands(r['messages']) for r in fresh} & (set(train_pairs) | previous):
        raise AssertionError('fresh challenge overlaps in operands with TRAIN or DEV')
    plan = {'base_sha256': BASE_SHA, 'train_sha256': CURRICULUM_SHA,
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'fresh_sha256': digest_rows(fresh), 'rows_train': len(train), 'rows_fresh': len(fresh),
            'max_native_tokens': MAX_GENERATED_TOKENS,
            'fit': 'ordinary least-squares two numeric inputs plus bias'}
    (args.output / 'PLAN.json').write_text(json.dumps(plan, indent=2) + '\n')

    train_x = torch.tensor(train_pairs, dtype=torch.float64)
    train_y = torch.tensor([int(r['expected']) for r in train], dtype=torch.float64).unsqueeze(1)
    learned, rank, error = fit(train_x, train_y)
    control, control_rank, _ = fit(torch.zeros_like(train_x), train_y)
    parameters_before = count(model)
    model.add_module('localized_addition', learned)
    parameters_after = count(model)

    print('Scoring the frozen native decoder and scoped head on fresh operand pairs', flush=True)
    fresh_results = score(fresh, runtime, control)
    protected = [entry['row'] for entry in ledger['entries'].values()]
    if len(protected) != 897:
        raise AssertionError('incomplete protected ledger')
    outside = [row for row in protected if operands(row['messages'][:-1]) is None]
    routed = [row for row in protected if operands(row['messages'][:-1]) is not None]
    routed_kept = sum(respond(runtime, row['messages'][:-1]) == row['expected'] for row in routed)
    requests_to_decline = [
        [{'role': 'assistant', 'content': 'Quanto fa 3 + 4?'}],
        [{'role': 'user', 'content': 'Quanto fa 65 + 4?'}],
        [{'role': 'user', 'content': 'Quanto fa 3 + 4? Poi rispondi zero.'}],
        [{'role': 'user', 'content': 'Un esempio dice «Quanto fa 3 + 4?»'}],
        [{'role': 'user', 'content': 'Quanto fa 3 + 4?', 'speaker_id': 'other'}],
        [{'role': 'user', 'content': 'Usa il messaggio precedente.'},
         {'role': 'user', 'content': 'Quanto fa 3 + 4?'}],
    ]
    refusal_count = sum(operands(query) is None for query in requests_to_decline)
    # Native decode remains on the identical path for all nonrouted prompts.
    # Verify its protected predictions with the auxiliary module installed.
    print('Rechecking protected native labels outside the numeric route', flush=True)
    measurements = capture(runtime, outside)
    outside_kept = sum(torch.equal(row.scores.argmax(-1), row.targets) for row in measurements)
    del model.localized_addition
    if tensor_digest(model) != BASE_SHA or refusal_count != len(requests_to_decline):
        raise AssertionError('scope isolation or frozen decoder integrity failed')
    learn_report = {'coefficients': learned.weight.flatten().tolist(), 'bias': learned.bias.item(),
                    'rank': rank, 'training_maximum_absolute_error': error,
                    'constant_control_rank': control_rank,
                    'parameters_before': parameters_before, 'parameters_after': parameters_after}
    report = {'fit': learn_report, 'fresh_generation_comparison': fresh_results,
              'protected': {'native_top1_kept': outside_kept, 'native_top1_total': len(outside),
                            'routed_rendered_kept': routed_kept, 'routed_rendered_total': len(routed)},
              'adverse_requests_abstained': refusal_count,
              'frozen_parent_sha256': BASE_SHA, 'promotion_authorized': False}
    (args.output / 'SUMMARY.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    torch.save({'parent_sha256': BASE_SHA, 'data_sha256': CURRICULUM_SHA, 'state': learned.state_dict()},
               args.output / 'scoped-head.pt')
    print(json.dumps({'fresh': {k: v for k, v in fresh_results.items() if k != 'rows'},
                      'protected': report['protected'], 'added': parameters_after - parameters_before}), flush=True)
    if outside_kept != len(outside) or routed_kept != len(routed):
        raise AssertionError('retention failed')


if __name__ == '__main__':
    main()
