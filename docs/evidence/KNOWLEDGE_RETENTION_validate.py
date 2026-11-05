"""Verify archived retention evidence without loading Torch or any model."""
import ast
import gzip
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def metrics(value):
    samples = value['samples']
    assert len(samples) == value['probes']
    assert sum(row['correct'] for row in samples) == value['exact_correct']
    context = [row for row in samples if row['family'].startswith('context-')]
    return dict(exact_correct=value['exact_correct'], probes=len(samples),
                context_correct=sum(row['correct'] for row in context), context_probes=len(context),
                native_nll=value['nll'], human_nll=value['human_nll'], repetition=value['repetition'])


def derive(raw):
    assert raw['complete'] is True and raw['live_promoted'] is False
    assert all(raw[name] is True for name in ('source_checkpoint_unchanged',
               'frozen_champion_unchanged', 'frozen_ancestral_reference_unchanged'))
    assert raw['code_hashes_at_start'] == raw['code_hashes_at_end']
    initial = raw['ledger_baseline']
    assert initial['protected_legacy_rows'] + initial['coverage_added_before_training'] == initial['protected_bank_rows']
    output = dict(schema_version=1, date='2026-11-05',
                  scope='Fixed native retention and persistence diagnostic; no candidate adoption or production promotion',
                  seed=raw['seed'], steps_per_arm=raw['steps_per_arm'], segment_units=raw['segment_units'],
                  approved_model_tensor_sha256=raw['source_teacher_tensor_sha256'],
                  original_reference_tensor_sha256=raw['original_reference_tensor_sha256'],
                  source_checkpoint_unchanged=raw['source_checkpoint_unchanged'],
                  frozen_teachers_unchanged=True, baseline=metrics(raw['baseline']),
                  initial_training_coverage=initial,
                  common_acquisition_anchor_sha256=raw['common_acquisition_anchor_sha256'],
                  common_acquisition_supervised_tokens=raw['acquisition_supervised_tokens'],
                  predeclared_profile=raw['predeclared_profile'],
                  equal_optimizer_units=True, equal_tokens_or_compute=False,
                  live_promoted=False, candidate_adoptions=0, trials=[])
    before = raw['baseline']['samples']
    for trial in raw['trials']:
        after = trial['after']['samples']
        for old, new in zip(before, after):
            assert all(old[key] == new[key] for key in ('messages', 'expected', 'family', 'language'))
        lost = [i for i, (old, new) in enumerate(zip(before, after)) if old['correct'] and not new['correct']]
        gained = [i for i, (old, new) in enumerate(zip(before, after)) if new['correct'] and not old['correct']]
        assert lost == trial['lost_answers'] and gained == trial['gained_answers']
        assert trial['original_development_gate_accepted'] is False
        certificates = trial['endpoint_full_bank_certificates']
        assert len(certificates) == initial['protected_bank_rows']
        assert all(v['correct'] and v['complete'] and v['includes_eos'] and v['token_count'] > 0
                   for v in certificates.values())
        assert not trial['endpoint_full_bank_lost'] and not trial['endpoint_old_memory_lost']
        assert trial['native_export_logits_exact'] is True
        segments = []
        for segment in trial['segments']:
            assert segment['old_lost'] == segment['bank_lost'] == []
            assert segment['newly_correct_vs_baseline'] == []
            segments.append({key: segment[key] for key in ('unit', 'correct_measured_train_rows',
                             'old_correct', 'bank_correct', 'ledger_entries_ever_correct',
                             'ledger_entries_currently_correct', 'newly_correct_vs_baseline')})
        assert [s['unit'] for s in segments] == [64, 128, 192]
        accounting = trial['accounting']
        validation = trial['resume_validation_extra_physical_accounting']
        interruption = trial.get('interruption_extra_physical_accounting')
        extra_attempts = interruption['extra_discarded_physical_optimizer_attempts'] if interruption else 0
        physical_attempts = accounting['physical_attempts'] + validation['physical_attempts'] + extra_attempts
        physical_tokens = {}
        for kind in ('supervised_tokens', 'reference_tokens', 'prompt_reference_tokens'):
            discarded = interruption['extra_discarded_physical_' + kind] if interruption else 0
            physical_tokens[kind] = accounting['physical_' + kind] + validation['physical_' + kind] + discarded
        if interruption:
            assert physical_attempts == trial['physical_attempts_including_resume_validation_and_discarded']
            assert physical_tokens == trial['physical_tokens_including_resume_validation_and_discarded']
        entry = dict(arm=trial['arm'], after=metrics(trial['after']), lost_heldout_indices=lost,
                     gained_heldout_indices=gained, original_development_gate_accepted=False,
                     complete_train_certificates_retained=len(certificates),
                     complete_train_certificates_lost=0, segments=segments,
                     logical_trajectory_accounting=accounting,
                     extra_resume_validation_accounting=validation,
                     total_physical_attempts_including_validation_and_discarded=physical_attempts,
                     total_physical_token_exposures_including_validation_and_discarded=physical_tokens,
                     interruption=interruption, resume=trial['resume'], native_export_logits_exact=True)
        audit = trial.get('endpoint_exhaustive_train_bank_fullprompt_audit')
        if audit:
            values = list(audit['records'].values())
            assert len(values) == audit['rows'] == len(certificates)
            assert all(v['teacher_complete_answer_correct'] and v['candidate_complete_answer_correct'] for v in values)
            assert max(v['full_sequence_kl'] for v in values) == audit['max_full_sequence_kl']
            assert max(v['answer_nll_increase'] for v in values) == audit['max_answer_nll_increase']
            assert sum(v['real_next_token_positions'] for v in values) == audit['real_next_token_positions']
            assert sum(v['full_sequence_argmax_changed'] for v in values) == audit['full_sequence_argmax_changed']
            assert sum(v['full_sequence_kl'] > .01 for v in values) == audit['rows_above_sampled_kl_threshold']
            assert sum(v['answer_nll_increase'] > .02 for v in values) == audit['rows_above_sampled_answer_nll_threshold']
            entry['exhaustive_train_bank_fullprompt_audit'] = {k: v for k, v in audit.items() if k != 'records'}
        if entry['resume']:
            checks = entry['resume']
            assert checks['verification_pending'] is False
            assert all(checks[name] is True for name in ('trainable_weights_exact', 'optimizer_exact',
                       'bank_json_exact', 'transaction_json_exact', 'memory_rng_exact', 'guard_rng_exact',
                       'torch_rng_exact', 'update_statistics_exact'))
        output['trials'].append(entry)
    assert len(output['trials']) == 2
    return output


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'KNOWLEDGE_RETENTION_PROVENANCE_20261105.json').read_text())
    blobs = {}
    for item in manifest['artifacts']:
        packed = (root / item['archive']).read_bytes()
        raw = gzip.decompress(packed)
        assert len(packed) == item['archive_bytes'] and sha(packed) == item['archive_sha256']
        assert len(raw) == item['source_bytes'] and sha(raw) == item['source_sha256']
        name = Path(item['source_path']).name
        if name.endswith('.json'):
            json.loads(raw)
        elif name.endswith('.py'):
            ast.parse(raw)
        blobs[name] = raw
    raw = json.loads(blobs['retention-system-trial.json'])
    summary = (root / manifest['summary']['path']).read_bytes()
    assert sha(summary) == manifest['summary']['sha256']
    assert json.loads(summary) == derive(raw)
    initial_bank = json.loads(blobs['retention-system-trial-initial-bank.json'])
    endpoint_bank = next(t['endpoint_bank_json'] for t in raw['trials'] if t['arm'].startswith('new-'))
    assert set(initial_bank['entries']) == set(endpoint_bank['entries'])
    for path, expected in raw['code_hashes_at_start'].items():
        assert sha(blobs[Path(path).name]) == expected
    assert sha(blobs['retention-system-trial.py']) == raw['harness_sha256']
    assert sha(blobs['retention-system-trial-fullkl.py']) == raw['read_only_fullprompt_audit_script_sha256']
    print('PASS: %d exact archives, source hashes, derived metrics, complete TRAIN certificates, resume and physical accounting' % len(blobs))
    remote_path = root / 'KNOWLEDGE_RETENTION_REMOTE_20261105.json'
    if remote_path.exists():
        remote = json.loads(remote_path.read_text())
        reports = []
        for item in remote['archived_logs']:
            packed = (root / item['archive']).read_bytes()
            data = gzip.decompress(packed)
            assert len(packed) == item['archive_bytes'] and sha(packed) == item['archive_sha256']
            assert len(data) == item['source_bytes'] and sha(data) == item['source_sha256']
            if item['archive'].startswith('KNOWLEDGE_RETENTION_REMOTE_job-'):
                for line in data.decode('utf-8-sig').splitlines():
                    try:
                        record = json.loads(line[line.index('{'):])
                    except (ValueError, json.JSONDecodeError):
                        continue
                    if isinstance(record, dict) and record.get('phase') == 'real native persistent retention preflight PASS':
                        reports.append(record)
        assert reports and reports[-1] == remote['real_checkpoint_preflight']
        proof = reports[-1]
        assert proof['full_bank_lost'] == proof['development_evaluations'] == 0
        assert proof['training_retention_rows'] == 518
        assert proof['units'] == proof['physical_attempts'] == proof['committed_updates'] == 32
        assert all(proof[key] is True for key in ('resume_exact', 'native_export_exact',
                                                 'champion_unchanged', 'reference_unchanged'))
        assert proof['live_promoted'] is False and remote['heldout_improvement_evaluated'] is False
        assert all(run['status'] == 'completed' and run['conclusion'] == 'success'
                   and run['head_sha'] == remote['frozen_runner_commit'] for run in remote['runs'])
        print('PASS: hashed remote CI logs and exact full-checkpoint retention preflight report')
        checkpoint_material = {}
        for checkpoint_key in ('first_checkpoint_proof', 'first_candidate_checkpoint_proof'):
            checkpoint = remote.get(checkpoint_key)
            if not checkpoint:
                continue
            material = {}
            for item in checkpoint['files']:
                packed = (root / item['archive']).read_bytes()
                data = gzip.decompress(packed)
                assert len(packed) == item['archive_bytes'] and sha(packed) == item['archive_sha256']
                assert len(data) == item['source_bytes'] and sha(data) == item['source_sha256']
                material[item['kind']] = (item, data, json.loads(data))
            index = material['index'][2]
            manifest = material['manifest'][2]
            state = material['run_state'][2]
            memory = material['ledger'][2]
            registry = material['registry'][2]
            assert all(value['fingerprint'] == remote['pilot_fingerprint']
                       for value in (index, manifest, state, memory, registry))
            for kind in ('run_state', 'ledger', 'registry'):
                item, data, _ = material[kind]
                assert manifest['files'][item['name']] == sha(data)
            assert index['run_state']['sha256'] == sha(material['run_state'][1])
            assert checkpoint['optimizer_archive_descriptor'] in index['archives']
            assert len(memory['entries']) == checkpoint['verified']['retention_rows'] == 518
            assert all(entry['certificate']['correct'] and entry['certificate']['includes_eos']
                       for entry in memory['entries'].values())
            assert state['revision'] == 2 and state['search']['accepted_updates'] == 0
            checkpoint_material[checkpoint_key] = material
        if checkpoint_material:
            print('PASS: durable ledger/registry and hash-committed continuation states')
        if 'first_candidate_checkpoint_proof' in checkpoint_material:
            initial = checkpoint_material['first_checkpoint_proof']
            candidate = checkpoint_material['first_candidate_checkpoint_proof']
            assert set(initial['ledger'][2]['entries']) == set(candidate['ledger'][2]['entries'])
            assert all(entry['observation']['correct'] for entry in candidate['ledger'][2]['entries'].values())
            assert len(candidate['registry'][2]['entries']) == 252
            extra = {}
            for key in ('training_assessments', 'continuation_progress', 'first_training_job_log'):
                item = remote[key]
                packed = (root / item['archive']).read_bytes()
                data = gzip.decompress(packed)
                assert len(packed) == item['archive_bytes'] and sha(packed) == item['archive_sha256']
                assert len(data) == item['source_bytes'] and sha(data) == item['source_sha256']
                if key != 'first_training_job_log': extra[key] = json.loads(data)
            assessments = extra['training_assessments']
            assert assessments['source']['sha256'] == candidate['manifest'][2]['files']['resume.pt']
            before, after = assessments['champion'], assessments['last_candidate_assessment']
            assert metrics(before)['exact_correct'] == 23 and metrics(after)['exact_correct'] == 22
            assert sum(old['correct'] and not new['correct'] for old, new in zip(before['samples'], after['samples'])) == 1
            assert not any(new['correct'] and not old['correct'] for old, new in zip(before['samples'], after['samples']))
            progress = extra['continuation_progress']
            continued = remote['automatic_continuation']
            assert progress['fingerprint'] == remote['pilot_fingerprint']
            assert progress['checkpoint_release'] == continued['source_release']
            assert progress['completed_steps'] == continued['last_observed_step'] == 450044
            assert progress['conversation_updates'] == continued['last_observed_logical_units'] == 768
            assert progress['elapsed_training_seconds'] == continued['elapsed_training_seconds']
            transaction = progress['retention_transaction']
            assert transaction['attempted_updates'] == 774
            assert transaction['committed_updates'] == 741 and transaction['rejected_updates'] == 33
            assert progress['learning_search']['history'] == candidate['run_state'][2]['search']['history']
            assert progress['learning_search']['trial_start_update'] == 0
            print('PASS: unchanged certificate membership, rejected 22/92 candidate and actual automatic training resume')


if __name__ == '__main__':
    main()
