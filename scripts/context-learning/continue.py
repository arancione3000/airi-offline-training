"""Continue only a committed checkpoint on the same frozen finite experiment."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def validate_action(action):
    if (not isinstance(action, dict) or set(action) != {'release'}
            or not isinstance(action['release'], str)
            or not re.fullmatch(r'airi-offline-[A-Za-z0-9._-]+', action['release'])):
        raise ValueError('invalid checkpoint continuation action')


def continuation_payload(action, *, spec, continuation, branch, runner_sha,
                         branch_sha, index=None, state_bytes=None):
    if action is None:
        return None
    if (not isinstance(continuation, int) or isinstance(continuation, bool)
            or not 0 <= continuation < spec['job_limit']):
        raise ValueError('invalid finite continuation counter')
    validate_action(action)
    if branch != spec['frozen_runner_branch'] or branch_sha != runner_sha:
        raise ValueError('frozen runner branch changed; continuation refused')
    if not isinstance(index, dict) or index.get('fingerprint') != spec['fingerprint']:
        raise ValueError('continuation checkpoint fingerprint mismatch')
    descriptor = index.get('run_state', {})
    if (not isinstance(state_bytes, bytes) or descriptor.get('name') != 'AIRI_RUN_STATE.json'
            or descriptor.get('bytes') != len(state_bytes)
            or descriptor.get('sha256') != hashlib.sha256(state_bytes).hexdigest()):
        raise ValueError('continuation state is not hash-committed')
    state = json.loads(state_bytes)
    if (state.get('fingerprint') != spec['fingerprint']
            or state.get('job_index') != continuation
            or state.get('completed_steps') != index.get('completed_steps')
            or not spec['start_step'] <= state['completed_steps'] <= spec['max_steps']
            or state.get('continue') is not True
            or state.get('production_qualified') is not False
            or state.get('live_promoted') is not False):
        raise ValueError('continuation state does not match this offline experiment')
    if continuation + 1 >= spec['job_limit']:
        return None
    return {'ref': branch, 'inputs': {'checkpoint_release': action['release'],
                                      'continuation': str(continuation + 1),
                                      'preflight_only': 'false'}}


def main():
    spec = json.loads(Path(os.environ['AIRI_EXPERIMENT_SPEC']).read_text())
    action = json.loads(os.environ['NEXT_ACTION'])
    if action is None:
        print('Finite experiment stopped; selected checkpoint retained.')
        return
    validate_action(action)
    repo = os.environ['GITHUB_REPOSITORY']
    branch = os.environ['GITHUB_REF_NAME']
    if os.environ['GITHUB_REF_TYPE'] != 'branch':
        raise ValueError('continuation requires the frozen branch')
    ref = json.loads(subprocess.check_output(
        ['gh', 'api', 'repos/' + repo + '/git/ref/heads/' + branch], text=True))
    # Publication writes the model, optimizer and state before committing this
    # index. The next runner verifies both archive digests before loading.
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(['gh', 'release', 'download', action['release'], '--repo', repo,
                        '--pattern', 'AIRI_TRAINING_CHECKPOINT_*.json',
                        '--pattern', 'AIRI_RUN_STATE.json', '--dir', directory], check=True)
        paths = list(Path(directory).glob('AIRI_TRAINING_CHECKPOINT_*.json'))
        if len(paths) != 1:
            raise ValueError('one committed continuation index required')
        payload = continuation_payload(
            action, spec=spec, continuation=int(os.environ['CONTINUATION']),
            branch=branch, runner_sha=os.environ['GITHUB_SHA'],
            branch_sha=ref['object']['sha'], index=json.loads(paths[0].read_text()),
            state_bytes=(Path(directory) / 'AIRI_RUN_STATE.json').read_bytes())
    if payload is None:
        print('Finite job limit reached; last verified checkpoint retained.')
        return
    subprocess.run(['gh', 'api', '--method', 'POST',
                    'repos/' + repo + '/actions/workflows/airi-context-background.yml/dispatches',
                    '--input', '-'], input=json.dumps(payload), text=True, check=True)


if __name__ == '__main__':
    main()
