"""Real PREFIX smoke: learn TRAIN answers, preserve memory, and resume exactly.

This fixture uses only authored TRAIN rows. Any attempt to inspect held-out
results through the evaluation callback fails the check immediately.
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/transfer-training'))
import torch
from generalist_lm.model import GeneralistLMConfig
from generalist_lm.runtime import GeneralistRuntime
from generalist_lm.tokenizer import EOS
from generalist_lm.training import SFTExample, encode_sft_example
from prefix_tuning import configure_prefix_tuning
from retention_memory import make_certifier
from success_bank import catalog


def state_digest(model):
    """Bind names, shape, dtype, and every native tensor without pickle RNG."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)],
                                 separators=(',', ':')).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_files(path):
    return {str(file.relative_to(path)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(path.rglob('*')) if file.is_file()}


def run_checks():
    torch.set_num_threads(2)
    torch.manual_seed(881)
    runtime = GeneralistRuntime.fresh(GeneralistLMConfig(
        d_model=32, n_layers=2, n_heads=2, d_ff=64, context_length=64,
        position_encoding='rope', attention_type='gqa', n_kv_heads=1,
        norm_type='rmsnorm'))
    configure_prefix_tuning(runtime.model, cache_bytes=16 * 1024 * 1024,
                            cache_entries=32)
    known_rows = []
    for language in ('it', 'en'):
        families = ('context-name', 'foundation-copy', 'foundation-recall',
                    'phrase-' + language)
        for index, family in enumerate(families):
            known_rows.append(dict(
                messages=[dict(role='user', content=language + str(index)),
                          dict(role='assistant', content='ok')],
                expected='ok', language=language, family=family,
                provenance='deterministic exact TRAIN smoke fixture'))
    task = dict(messages=[dict(role='user', content='en new task'),
                          dict(role='assistant', content='no')],
                expected='no', language='en', family='extra-dialogue',
                provenance='authored reviewed TRAIN smoke fixture')
    eligible = catalog(known_rows + [task], [])

    def encode(row):
        ids, labels = encode_sft_example(
            SFTExample.from_dict(row), runtime.tokenizer,
            runtime.model.config.context_length, require_complete=True)
        width = int((labels != -100).nonzero()[-1]) + 1
        return ids[:width].unsqueeze(0), labels[:width].unsqueeze(0)

    certify = make_certifier(runtime.model, encode, EOS)
    pretrain = torch.optim.AdamW(
        [parameter for parameter in runtime.model.parameters() if parameter.requires_grad],
        lr=.025)
    for step in range(240):
        ids, labels = encode(known_rows[step % len(known_rows)])
        pretrain.zero_grad(set_to_none=True)
        runtime.model(ids, labels)['loss'].backward()
        pretrain.step()
        if step > 24 and step % len(known_rows) == 0:
            measurements = [certify(row) for row in known_rows]
            if all(result['correct'] and result['min_margin'] > 1.
                   for result in measurements):
                break
    assert all(certify(row)['correct'] for row in known_rows), \
        'tiny native fixture did not learn all eight complete TRAIN answers'
    initial_task = certify(task)
    assert not initial_task['correct'], 'new TRAIN task was already solved by fixture pretraining'
    frozen_prefix = {name: parameter.detach().clone()
                     for name, parameter in runtime.model.named_parameters()
                     if not parameter.requires_grad}
    for parameter in runtime.model.parameters():
        parameter.grad = None
    anchors = [encode(known_rows[0])]
    retention = dict(version=1, capacity=1024, refresh_probe_limit=128,
                     guard_examples_per_language=2, max_prompt_kl=.01,
                     max_nll_regression=.02, prompt_kl_weight=1.,
                     retry_scales=[1., .5, .25, .125], consolidation_updates=32,
                     source_policy='eligible_train_only')
    learning = dict(
        profiles=[dict(learning_rate=1e-4, retention_weight=2., champion_kl_weight=2.)],
        retention_examples_per_language=2, feature_cache_mebibytes=16,
        knowledge_retention=retention)
    evaluation_calls = []

    def forbidden_evaluate(*args, **kwargs):
        evaluation_calls.append(True)
        raise AssertionError('retention preflight read held-out development data')

    from retention_smoke import real_retention_smoke
    with tempfile.TemporaryDirectory(prefix='airi-native-retention-smoke-') as temporary:
        checkpoint = Path(temporary) / 'initial'
        runtime.save_checkpoint(checkpoint / 'model')
        initial_files = checkpoint_files(checkpoint)
        reference = GeneralistRuntime.from_checkpoint(checkpoint / 'model')
        for parameter in reference.model.parameters():
            parameter.requires_grad_(False)
        reference_digest = state_digest(reference.model)
        assert reference.model is not runtime.model
        assert reference_digest == state_digest(runtime.model)
        assert not any(a.data_ptr() == b.data_ptr()
                       for a, b in zip(runtime.model.parameters(), reference.model.parameters()))
        report = real_retention_smoke(
            runtime, checkpoint, reference, eligible, [], encode, anchors,
            forbidden_evaluate, state_digest, retention, learning,
            'tiny-native-retention-smoke')
        assert isinstance(report, dict), 'smoke must return its reviewable integrity report'
        assert report['units'] == 32
        assert report['committed_updates'] > 0
        assert report['final_training_task_nll'] < report['initial_training_task_nll']
        assert set(report['retention_languages']) == {'it', 'en'}
        assert report['training_retention_rows'] >= len(known_rows)
        assert report['champion_unchanged'] is True
        assert report['resume_exact'] is True
        assert report['native_export_exact'] is True
        assert report['live_promoted'] is False
        assert certify(task)['mean_nll'] < initial_task['mean_nll'], \
            'actual native student did not improve the new TRAIN answer'
        assert all(certify(row)['correct'] for row in known_rows), \
            'actual native student lost an initially correct TRAIN answer'
        assert all(torch.equal(value, dict(runtime.model.named_parameters())[name])
                   for name, value in frozen_prefix.items()), 'native frozen prefix changed'
        assert not evaluation_calls
        assert state_digest(reference.model) == reference_digest
        assert all(parameter.grad is None for parameter in reference.model.parameters())
        assert checkpoint_files(checkpoint) == initial_files, 'preflight rewrote its input champion'
        json.dumps(report, allow_nan=False)
    print(json.dumps(dict(
        retention_smoke_native='PASS', units=report['units'],
        committed_updates=report['committed_updates'],
        initial_training_task_nll=report['initial_training_task_nll'],
        final_training_task_nll=report['final_training_task_nll'],
        training_retention_rows=report['training_retention_rows'],
        heldout_callback_never_called=True, frozen_reference_unchanged=True,
        input_checkpoint_unchanged=True, resume_exact=True,
        native_export_exact=True, live_promoted=False)), flush=True)


if __name__ == '__main__':
    run_checks()
