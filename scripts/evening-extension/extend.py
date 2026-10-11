"""One authorized four-hour sequel, after the frozen dialogue chain finishes."""
import base64
import hashlib
import json
import math
import os
import re
import subprocess
from urllib.parse import quote

SOURCE_BRANCH = 'dialogue-cycle-20261011-pm'
SOURCE_SHA = 'cda908aa80818d4ece8f230e13cde07bf6553b47'
SOURCE_FP = 'eda81e8c7453e00b63e937e89da74d91f2c9693a880b7464b764122e8bf06635'
SOURCE_SECONDS = 14400
EXTRA_SECONDS = 14400
BRANCH = 'dialogue-cycle-20261011-evening'
REQUEST = 'thomas-test-then-extra4h-20261011-evening'
WORKFLOW = 'airi-transfer-background.yml'
PREFIX = 'scripts/transfer-training/'
ACTIVE = {'in_progress', 'queued', 'requested', 'waiting', 'pending'}

def sha(data):
    return hashlib.sha256(data).hexdigest()

class Client:
    def __init__(self, repository):
        if repository != 'arancione3000/airi-offline-training':
            raise ValueError('unexpected repository')
        self.base = 'repos/' + repository + '/'

    def api(self, path, payload=None):
        args = ['gh', 'api', self.base + path]
        if payload is not None:
            args += ['--method', 'PUT' if path.endswith('/disable') else 'POST', '--input', '-']
        result = subprocess.run(args, input=None if payload is None else json.dumps(payload),
                                capture_output=True, text=True, check=True, timeout=60)
        return json.loads(result.stdout) if result.stdout.strip() else None

    def pages(self, path, key=None):
        page = 1
        while True:
            sep = '&' if '?' in path else '?'
            value = self.api(path + sep + f'per_page=100&page={page}')
            values = value[key] if key else value
            yield from values
            if len(values) < 100:
                break
            page += 1

    def content(self, path, ref):
        value = self.api('contents/' + path + '?ref=' + quote(ref, safe=''))
        return base64.b64decode(value['content'], validate=False)

    def asset(self, item):
        data = subprocess.check_output(['gh', 'api', self.base + 'releases/assets/' + str(item['id']),
                                       '-H', 'Accept: application/octet-stream'], timeout=60)
        if len(data) != item['size']:
            raise ValueError('release asset size mismatch')
        digest = item.get('digest')
        if digest and digest != 'sha256:' + sha(data):
            raise ValueError('release asset digest mismatch')
        return json.loads(data)

    def branch(self):
        try:
            return self.api('git/ref/heads/' + BRANCH)
        except subprocess.CalledProcessError as error:
            if '(HTTP 404)' in error.stderr:
                return None
            raise

def validate_terminal(result, index, release, source_run):
    counters = result['counters']
    elapsed = counters['elapsed_training_seconds']
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or not math.isfinite(elapsed):
        raise ValueError('invalid completed compute')
    expected_tag = rf'airi-offline-[0-9]+-{source_run["id"]}-{source_run["run_attempt"]}-[0-9]+'
    if (release.get('draft') or release.get('prerelease') is not True or release.get('target_commitish') != SOURCE_SHA
            or not re.fullmatch(expected_tag, release['tag_name'])
            or source_run['head_sha'] != SOURCE_SHA or source_run['head_branch'] != SOURCE_BRANCH
            or source_run['status'] != 'completed' or source_run['conclusion'] != 'success'
            or result['next_action'] is not None or result['maximum_training_seconds'] != SOURCE_SECONDS
            or elapsed < SOURCE_SECONDS or result['production_qualified'] is not False
            or result['live_promoted'] is not False or index['fingerprint'] != SOURCE_FP
            or index['completed_steps'] != counters['completed_steps']
            or index.get('production_qualified') is not False or index.get('live_promoted') is not False):
        raise ValueError('terminal checkpoint provenance or completion mismatch')
    step = index['completed_steps']
    if not isinstance(step, int) or isinstance(step, bool) or not 129599 < step < 1000000:
        raise ValueError('invalid terminal step')
    if len(index['archives']) != 2:
        raise ValueError('complete model and optimizer required')
    names = {f'AIRI_TRAINING_{step:05d}_{part}.zip' for part in ('MODEL', 'OPTIMIZER')}
    for item in index['archives']:
        name = item.get('name') or item.get('file_name')
        if name not in names or not re.fullmatch('[a-f0-9]{64}', item['sha256']) or item['bytes'] <= 0:
            raise ValueError('invalid checkpoint archive index')
        names.remove(name)
    return step

def build_files(source, terminal, index, release, source_run):
    step = validate_terminal(terminal, index, release, source_run)
    spec = json.loads(source[PREFIX + 'SPEC.json'])
    if spec['fingerprint'] != SOURCE_FP or spec['maximum_training_seconds'] != SOURCE_SECONDS:
        raise ValueError('source experiment changed')
    for name, digest in spec['script_sha256'].items():
        if sha(source[PREFIX + name]) != digest:
            raise ValueError('source script hash mismatch: ' + name)
    run = source[PREFIX + 'run.py'].decode()
    files = {}
    spec.update(start_step=step, pilot_step=step, maximum_training_seconds=EXTRA_SECONDS,
                migration_source_release=release['tag_name'], migration_source_fingerprint=SOURCE_FP,
                recipes=[{'learning_rate': 1e-6, 'kl_weight': 1.0} for _ in range(3)])
    spec['schedule'] = 'Four-hour offline sequel; same data and sampling; predeclared lower LR 1e-6 and KL 1.0 after development stagnation; preserve native weights, Adam and RNG; separate fresh compute ledger; previous champion retained externally; repeated final probes excluded from selection.'
    spec['extension'] = {'request': REQUEST, 'source_sha': SOURCE_SHA, 'source_branch': SOURCE_BRANCH,
                         'source_run': source_run['id'], 'seed_release': release['tag_name'],
                         'seed_step': step, 'index_sha256': sha(json.dumps(index, sort_keys=True).encode()),
                         'additional_training_seconds': EXTRA_SECONDS}
    spec.pop('fingerprint')
    spec['fingerprint'] = sha(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode())
    files[PREFIX + 'SPEC.json'] = json.dumps(spec, indent=2) + '\n'
    # Preserve budget/ledger checks for the explicitly authorized successor seed.
    path = 'tests/transfer-training/test_diagnostics.py'
    tests = source[path].decode()
    replacements = {
        "917dee2511c1d30be8b2be9d784850d16b0daab55d503614f6b0ce6b3f87b92e": SOURCE_FP,
        "b766a3d7ab8c5e0db1a9058d7c5b789f09568418": SOURCE_SHA,
    }
    for old, new in replacements.items():
        if tests.count(old) != 1:
            raise ValueError('unexpected original regression fixture')
        tests = tests.replace(old, new)
    files[path] = tests
    return files

def runs(client, branch):
    return [r for r in client.pages('actions/workflows/' + WORKFLOW + '/runs?branch=' + branch, 'workflow_runs')
            if r.get('display_title') != 'AIRI transfer preflight']

def completed_release(client, last):
    candidates = [r for r in client.pages('releases')
                  if re.fullmatch(rf'airi-offline-[0-9]+-{last["id"]}-{last["run_attempt"]}-[0-9]+', r['tag_name'])
                  and any(a['name'] == 'FINAL_RESULT.json' for a in r['assets'])]
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError('ambiguous terminal release')
    return candidates[0]

def tick(client):
    # An existing immutable successor is an idempotency record, never overwritten.
    branch = client.branch()
    if branch:
        ref = branch['object']['sha']
        spec = json.loads(client.content(PREFIX + 'SPEC.json', ref))
        ext = spec['extension']
        if ext['request'] != REQUEST or ext['source_sha'] != SOURCE_SHA or spec['maximum_training_seconds'] != EXTRA_SECONDS:
            raise ValueError('successor branch identity mismatch')
        existing = runs(client, BRANCH)
        if existing:
            last = max(existing, key=lambda r: r['id'])
            if last['head_sha'] != ref:
                raise ValueError('successor source changed')
            if last['status'] == 'completed' and last['conclusion'] == 'success':
                release = completed_release(client, last)
                if release:
                    result = client.asset(next(a for a in release['assets'] if a['name'] == 'FINAL_RESULT.json'))
                    elapsed = result['counters']['elapsed_training_seconds']
                    if (result['next_action'] is not None or result['maximum_training_seconds'] != EXTRA_SECONDS
                            or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < EXTRA_SECONDS
                            or result['production_qualified'] is not False or result['live_promoted'] is not False):
                        raise ValueError('successor completion mismatch')
                    client.api('actions/workflows/airi-evening-extension.yml/disable', {})
                    return {'action': 'four_hours_completed_controller_disabled', 'run': last['id'], 'extra_seconds': EXTRA_SECONDS}
            if last['status'] == 'completed' and last['conclusion'] in {'failure', 'timed_out'}:
                # Use the existing fail-closed transient classifier; never retry integrity/quality failures.
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'autonomous-control'))
                from watchdog import transient
                jobs = list(client.pages('actions/runs/' + str(last['id']) + '/jobs', 'jobs'))
                logs = '\n'.join(subprocess.check_output(['gh', 'api', client.base + 'actions/jobs/' + str(j['id']) + '/logs'], text=True, timeout=60)
                                 for j in jobs if j.get('conclusion') in {'failure', 'timed_out'})
                busy = any('/airi-' in r.get('path', '') and 'background.yml' in r.get('path', '')
                           for status in sorted(ACTIVE)
                           for r in client.pages('actions/runs?status=' + status, 'workflow_runs'))
                if busy:
                    return {'action': 'waiting_for_training_slot', 'extra_seconds': EXTRA_SECONDS}
                if last['run_attempt'] < 3 and transient(logs):
                    client.api('actions/runs/' + str(last['id']) + '/rerun-failed-jobs', {})
                    return {'action': 'retry_verified_transient_failure', 'run': last['id']}
                client.api('actions/workflows/airi-evening-extension.yml/disable', {})
                return {'action': 'stopped_after_error_checkpoint_retained', 'run': last['id']}
            if last['status'] == 'completed' and last['conclusion'] == 'cancelled':
                client.api('actions/workflows/airi-evening-extension.yml/disable', {})
                return {'action': 'intentional_cancellation_respected', 'run': last['id']}
            return {'action': 'already_started', 'run': last['id'], 'extra_seconds': EXTRA_SECONDS}
        seed = spec['migration_source_release']
    else:
        original = runs(client, SOURCE_BRANCH)
        if not original:
            raise ValueError('source chain not found')
        last = max(original, key=lambda r: r['id'])
        if last['head_sha'] != SOURCE_SHA:
            raise ValueError('frozen source ref changed')
        if last['status'] in ACTIVE:
            return {'action': 'waiting_for_current_cycle', 'run': last['id'], 'extra_seconds': EXTRA_SECONDS}
        if last['status'] != 'completed' or last['conclusion'] != 'success':
            return {'action': 'source_stopped_without_success', 'run': last['id']}
        release = completed_release(client, last)
        if release is None:
            return {'action': 'waiting_for_terminal_release', 'run': last['id']}
        results = [a for a in release['assets'] if a['name'] == 'FINAL_RESULT.json']
        indexes = [a for a in release['assets'] if re.fullmatch(r'AIRI_TRAINING_CHECKPOINT_[0-9]+\.json', a['name'])]
        if len(results) != 1 or len(indexes) != 1:
            raise ValueError('one final result and committed index required')
        terminal, index = client.asset(results[0]), client.asset(indexes[0])
        validate_terminal(terminal, index, release, last)
        spec = json.loads(client.content(PREFIX + 'SPEC.json', SOURCE_SHA))
        paths = [PREFIX + 'SPEC.json', 'tests/transfer-training/test_diagnostics.py'] + [PREFIX + n for n in spec['script_sha256']]
        source = {p: client.content(p, SOURCE_SHA) for p in paths}
        files = build_files(source, terminal, index, release, last)
        parent = client.api('git/commits/' + SOURCE_SHA)
        tree = client.api('git/trees', {'base_tree': parent['tree']['sha'], 'tree': [
            {'path': p, 'mode': '100644', 'type': 'blob', 'content': content} for p, content in files.items()]})
        commit = client.api('git/commits', {'message': 'feat: frozen four-hour sequel from verified terminal checkpoint',
                                         'tree': tree['sha'], 'parents': [SOURCE_SHA]})
        client.api('git/refs', {'ref': 'refs/heads/' + BRANCH, 'sha': commit['sha']})
        ref, seed = commit['sha'], release['tag_name']
    # Shared training concurrency is retained; never displace an active run.
    for status in ('in_progress', 'queued', 'waiting', 'requested', 'pending'):
        if any('/airi-' in r.get('path', '') and 'background.yml' in r.get('path', '')
               for r in client.pages('actions/runs?status=' + status, 'workflow_runs')):
            return {'action': 'waiting_for_training_slot', 'extra_seconds': EXTRA_SECONDS}
    client.api('actions/workflows/' + WORKFLOW + '/dispatches', {
        'ref': BRANCH, 'inputs': {'checkpoint_release': seed, 'continuation': '0'}})
    return {'action': 'dispatched', 'branch': BRANCH, 'sha': ref, 'seed': seed, 'extra_seconds': EXTRA_SECONDS}

if __name__ == '__main__':
    client = Client(os.environ['GITHUB_REPOSITORY'])
    probe = client.api('actions/runs/38166921422')
    if probe['head_sha'] != 'f833e7e874d83d8458bfd18ff299b41bbabb4bbf' or probe['path'] != '.github/workflows/airi-conversation-probe.yml':
        raise ValueError('read-only probe provenance changed')
    if probe['status'] in ACTIVE:
        print(json.dumps({'action': 'waiting_for_authorized_read_only_probe'}))
    elif probe['status'] != 'completed' or probe['conclusion'] != 'success':
        raise ValueError('read-only conversation probe did not complete')
    else:
        print(json.dumps(tick(client), sort_keys=True))
