"""Native byte-token full-answer/EOS certificates and persistent TRAIN retention."""
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/transfer-training'))
import torch
from generalist_lm.model import GeneralistLMConfig
from generalist_lm.runtime import GeneralistRuntime
from generalist_lm.tokenizer import EOS
from generalist_lm.training import SFTExample, encode_sft_example
from prefix_tuning import configure_prefix_tuning
from retention_memory import (empty_memory, make_certifier, migrate_memory, observe,
                              refresh, sample_rows, validate_memory)
from success_bank import catalog, row_key


def run_checks():
    torch.set_num_threads(2)
    torch.manual_seed(881)
    runtime = GeneralistRuntime.fresh(GeneralistLMConfig(
        d_model=32, n_layers=2, n_heads=2, d_ff=64, context_length=64,
        position_encoding='rope', attention_type='gqa', n_kv_heads=1, norm_type='rmsnorm'))
    model = runtime.model
    configure_prefix_tuning(model, cache_bytes=65536, cache_entries=16)
    families = ('context-name', 'foundation-copy', 'foundation-recall')
    rows = [dict(messages=[dict(role='user', content=language + str(index)),
                           dict(role='assistant', content='ok')], expected='ok',
                 language=language, family=family,
                 provenance='deterministic exact TRAIN solver')
            for language in ('it', 'en') for index, family in enumerate(families)]
    eligible = catalog(rows, [])
    def encode(row):
        ids, labels = encode_sft_example(SFTExample(messages=row['messages']), runtime.tokenizer,
                                         model.config.context_length, require_complete=True)
        end = int((labels != -100).nonzero()[-1]) + 1
        return ids[:end].unsqueeze(0), labels[:end].unsqueeze(0)
    certify = make_certifier(model, encode, EOS)
    adam = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.025)
    for step in range(160):
        row = rows[step % len(rows)]
        ids, labels = encode(row)
        adam.zero_grad(set_to_none=True)
        model(ids, labels)['loss'].backward(); adam.step()
        if step > 20 and step % len(rows) == 0 and all(certify(r)['correct'] for r in rows):
            break
    assert all(certify(row)['correct'] for row in rows), 'tiny native fixture did not learn its complete answers'
    model.train(); model.blocks[0].eval()
    modes = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()
    certificate = certify(rows[0])
    assert certificate['token_count'] == len(rows[0]['expected'].encode()) + 1
    assert certificate['complete'] and certificate['includes_eos'] and certificate['correct']
    assert modes == [module.training for module in model.modules()]
    assert torch.equal(rng, torch.get_rng_state())
    ids, labels = encode(rows[0])
    prompt = ids[:, :int((labels[0] != -100).nonzero()[0])]
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=3, eos_token_id=EOS, use_cache=False)
    assert generated[0, prompt.shape[1]:].tolist() == runtime.tokenizer.encode('ok', eos=True)

    # Check the answer tail and final EOS independently. A prefix-only verifier
    # would accept these corruptions; the full native certificate must reject.
    original_forward = model.forward
    for position in (-3, -2):
        def corrupted(input_ids, *args, **kwargs):
            result = original_forward(input_ids, *args, **kwargs)
            logits = result['logits'].clone()
            logits[:, position, :] = -100
            logits[:, position, 11] = 100
            return {**result, 'logits': logits}
        model.forward = corrupted
        wrong = certify(rows[0])
        assert not wrong['correct'] and wrong['min_margin'] < 0
    model.forward = original_forward

    # Incomplete/padded encodings and bad gold labels cannot create a permanent
    # certificate. Production encode requires complete prompt+answer fit.
    invalid_encoders = (
        lambda row: (ids[:, :-1], labels[:, :-1]),
        lambda row: (torch.cat((ids, torch.zeros((1, 1), dtype=torch.long)), dim=1),
                     torch.cat((labels, torch.full((1, 1), -100, dtype=torch.long)), dim=1)),
        lambda row: (ids, labels.clone().index_fill(1, torch.tensor([labels.shape[1] - 2]), 9)),
    )
    for bad_encode in invalid_encoders:
        try: make_certifier(model, bad_encode, EOS)(rows[0])
        except ValueError: pass
        else: raise AssertionError('incomplete/corrupt answer certificate accepted')
    overflow = deepcopy(rows[0]); overflow['messages'][0]['content'] = 'x' * 80
    try: certify(overflow)
    except ValueError as error: assert 'complete SFT example exceeds context' in str(error)
    else: raise AssertionError('truncated prompt certified as complete')

    champion = 'a' * 64
    memory = refresh(champion, eligible, certify, empty_memory('native-one'), probe_limit=16)
    assert set(memory['entries']) == set(eligible)
    restored = json.loads(json.dumps(memory, allow_nan=False))
    validate_memory(restored, eligible, 'native-one', champion)
    restored = migrate_memory(restored, eligible, 'native-one', 'native-two', champion)
    for key in eligible: assert key in restored['entries']
    baseline = deepcopy(restored)
    # One wrong EOS invalidates the proposed champion atomically; all old
    # answers remain permanently registered, including context/copy/recall.
    def wrong_eos(input_ids, *args, **kwargs):
        result = original_forward(input_ids, *args, **kwargs)
        logits = result['logits'].clone(); logits[:, -2, EOS] = -1000
        return {**result, 'logits': logits}
    model.forward = wrong_eos
    try: refresh('b' * 64, eligible, certify, restored, probe_limit=0)
    except ValueError as error: assert 'lost a previously certified' in str(error)
    else: raise AssertionError('forgotten native answer was silently dropped')
    assert restored == baseline
    observed = observe(restored, eligible, certify)
    assert all(not entry['observation']['correct'] for entry in observed['entries'].values())
    assert set(observed['entries']) == set(restored['entries'])
    model.forward = original_forward
    draws = sample_rows(observed, eligible, random.Random(8), per_language=3, uniform_fraction=.25)
    assert {row_key(row) for row in draws} == set(eligible)
    print(json.dumps({'retention_memory_native': 'PASS', 'full_answer_and_eos_certified': True,
                      'native_greedy_answer_matches_certificate': True,
                      'truncation_and_wrong_tail_rejected': True,
                      'lost_old_answer_rejects_champion_atomically': True,
                      'json_fingerprint_migration_all_families_retained': True}))


if __name__ == '__main__': run_checks()
