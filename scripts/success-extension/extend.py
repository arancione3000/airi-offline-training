"""One bounded selective-transfer pilot after verified fifty-hour completion."""
import base64
import hashlib
import json
import math
import os
import re
import subprocess
from urllib.parse import quote

SOURCE_BRANCH = 'dialogue-success-transfer-20261015'
SOURCE_SHA = 'cf6e52928b7bd92d535928c7a10abb1beea66b8e'
SOURCE_FP = '7bb6a78ce1859ffdf8fe7a0db5b374cfddb5abace8a349a67be1538004f9856e'
SOURCE_SECONDS = 180000
EXTRA_SECONDS = 3600
BRANCH = 'dialogue-human-extra50h-20261013'
REQUEST = 'thomas-selective-native-success-transfer-20261015'
WORKFLOW = 'airi-success-background.yml'
SOURCE_WORKFLOW = 'airi-human-background.yml'
TEMPLATE_SHA = 'f6e627f3202828a3295e99a4177687aaac4549e6'
TEMPLATE_BRANCH = 'success-transfer-template-20261015'
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

def build_files(source, terminal, index, release, source_run, template=None):
    from pathlib import Path
    step = validate_terminal(terminal,index,release,source_run)
    source_spec=json.loads(source[PREFIX+'SPEC.json'])
    if source_spec['fingerprint'] != SOURCE_FP or source_spec['maximum_training_seconds'] != SOURCE_SECONDS:
        raise ValueError('source experiment changed')
    for name,digest in source_spec['script_sha256'].items():
        if sha(source[PREFIX+name]) != digest:
            raise ValueError('source script changed: '+name)
    if template is None:
        root=Path(os.environ['AIRI_SUCCESS_TEMPLATE'])
        template={PREFIX+'SPEC.json':(root/PREFIX/'SPEC.json').read_bytes()}
        spec=json.loads(template[PREFIX+'SPEC.json'])
        for name in spec['script_sha256']:
            template[PREFIX+name]=(root/PREFIX/name).read_bytes()
    spec=json.loads(template[PREFIX+'SPEC.json'])
    if (spec['success_transfer']['request'] != REQUEST
            or spec['success_transfer']['source_sha'] != SOURCE_SHA
            or spec['maximum_training_seconds'] != EXTRA_SECONDS):
        raise ValueError('wrong selective transfer template')
    for name,digest in spec['script_sha256'].items():
        if sha(template[PREFIX+name]) != digest:
            raise ValueError('template script changed: '+name)
    spec.update(start_step=step,pilot_step=step,maximum_training_seconds=EXTRA_SECONDS,
                migration_source_release=release['tag_name'],migration_source_fingerprint=SOURCE_FP)
    spec['extension']={'request':REQUEST,'source_sha':SOURCE_SHA,'source_branch':SOURCE_BRANCH,
                       'source_run':source_run['id'],'seed_release':release['tag_name'],
                       'seed_step':step,'index_sha256':sha(json.dumps(index,sort_keys=True).encode()),
                       'additional_training_seconds':EXTRA_SECONDS,'template_sha':TEMPLATE_SHA}
    spec.pop('fingerprint')
    spec['fingerprint']=sha(json.dumps(spec,sort_keys=True,separators=(',',':')).encode())
    return {PREFIX+'SPEC.json':json.dumps(spec,indent=2)+'\n'}

def runs(client, branch):
    workflow=SOURCE_WORKFLOW if branch==SOURCE_BRANCH else WORKFLOW
    return [r for r in client.pages('actions/workflows/' + workflow + '/runs?branch=' + branch, 'workflow_runs')
            if r.get('display_title') not in {'AIRI transfer preflight','AIRI success transfer preflight'}]

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
    # A real frozen-template preflight is required before any training dispatch.
    checks=[r for r in client.pages('actions/workflows/'+WORKFLOW+'/runs?branch='+TEMPLATE_BRANCH,'workflow_runs')
            if r.get('display_title')=='AIRI success transfer preflight' and r['head_sha']==TEMPLATE_SHA]
    if not checks or max(checks,key=lambda r:r['id']).get('conclusion') != 'success':
        return {'action':'waiting_for_real_native_preflight','extra_seconds':EXTRA_SECONDS}
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
                    client.api('actions/workflows/airi-success-extension.yml/disable', {})
                    return {'action': 'selective_transfer_pilot_completed_controller_disabled', 'run': last['id'], 'extra_seconds': EXTRA_SECONDS}
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
                client.api('actions/workflows/airi-success-extension.yml/disable', {})
                return {'action': 'stopped_after_error_checkpoint_retained', 'run': last['id']}
            if last['status'] == 'completed' and last['conclusion'] == 'cancelled':
                client.api('actions/workflows/airi-success-extension.yml/disable', {})
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
        parent = client.api('git/commits/' + TEMPLATE_SHA)
        tree = client.api('git/trees', {'base_tree': parent['tree']['sha'], 'tree': [
            {'path': p, 'mode': '100644', 'type': 'blob', 'content': content} for p, content in files.items()]})
        commit = client.api('git/commits', {'message': 'feat: frozen selective transfer pilot from verified fifty-hour terminal',
                                         'tree': tree['sha'], 'parents': [TEMPLATE_SHA]})
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
    print(json.dumps(tick(Client(os.environ['GITHUB_REPOSITORY'])),sort_keys=True))
