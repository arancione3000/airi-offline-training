"""Atomic stable reference to one offline AIRI champion, never to an untested candidate."""
import base64
import json
import os
import re
import subprocess

BRANCH = 'airi-model-current'
PATH = 'models/AIRI_CURRENT.json'
REPOSITORY = 'arancione3000/airi-offline-training'


def validate(record):
    if (record['schema'] != 1 or record['model_id'] != 'airi-native'
            or record['scope'] != 'verified offline champion'
            or record['production_qualified'] is not False or record['live_promoted'] is not False
            or not isinstance(record['revision'], int) or isinstance(record['revision'], bool)
            or record['revision'] < 0 or not isinstance(record['completed_steps'], int)
            or record['completed_steps'] < 0
            or not re.fullmatch(r'airi-offline-[A-Za-z0-9._-]+',record['checkpoint_release'])):
        raise ValueError('invalid stable AIRI model record')
    for name in ('weights_digest','fingerprint','model_file_sha256'):
        if not re.fullmatch('[a-f0-9]{64}',record[name]):
            raise ValueError('invalid stable AIRI digest')
    parent = record['parent_weights_digest']
    if parent is not None and not re.fullmatch('[a-f0-9]{64}',parent):
        raise ValueError('invalid parent model digest')
    files=record['model_files']
    if not files or files.get('model/model.pt')!=record['model_file_sha256']:
        raise ValueError('stable model file manifest missing')
    for name,digest in files.items():
        if (not name.startswith('model/') or '..' in name.split('/') or '\\' in name
                or not re.fullmatch('[a-f0-9]{64}',digest)):
            raise ValueError('invalid stable model file manifest')
    return record


def transition(previous, current):
    """Equal weights are idempotent; an update must extend the verified lineage."""
    validate(current)
    if previous is None:
        if current['revision'] != 0:
            raise ValueError('initialize the stable model before acquiring skills')
        return 'initialize'
    validate(previous)
    if current['revision'] == previous['revision'] and current['weights_digest'] == previous['weights_digest']:
        return 'unchanged'
    if (current['revision'] != previous['revision'] + 1
            or current['parent_weights_digest'] != previous['weights_digest']):
        raise ValueError('stale, skipped or unrelated model update')
    return 'advance'


class Client:
    def __init__(self, repository):
        if repository != REPOSITORY:
            raise ValueError('unexpected model repository')
        self.base = 'repos/' + repository + '/'

    def api(self, path, payload=None, *, method=None, missing=False):
        args = ['gh','api',self.base+path]
        if payload is not None:
            args += ['--method',method or 'POST','--input','-']
        result = subprocess.run(args,input=None if payload is None else json.dumps(payload),
                                capture_output=True,text=True,timeout=60)
        if result.returncode:
            if missing and '(HTTP 404)' in result.stderr:
                return None
            raise subprocess.CalledProcessError(result.returncode,args,result.stdout,result.stderr)
        return json.loads(result.stdout) if result.stdout.strip() else None


def publish_current(client, record, source_sha):
    """A contents API commit updates the pointer with a SHA compare-and-swap."""
    validate(record)
    branch = client.api('git/ref/heads/'+BRANCH,missing=True)
    if branch is None:
        client.api('git/refs',{'ref':'refs/heads/'+BRANCH,'sha':source_sha})
    previous_file = client.api('contents/'+PATH+'?ref='+BRANCH,missing=True)
    previous = (json.loads(base64.b64decode(previous_file['content'])) if previous_file else None)
    action = transition(previous,record)
    if action == 'unchanged':
        return {'action':action,'model_id':record['model_id'],'revision':record['revision']}
    payload = {'message':f"Update verified AIRI model to revision {record['revision']}",
               'branch':BRANCH,'content':base64.b64encode((json.dumps(record,indent=2)+'\n').encode()).decode()}
    if previous_file:
        payload['sha'] = previous_file['sha']
    client.api('contents/'+PATH,payload,method='PUT')
    return {'action':action,'model_id':record['model_id'],'revision':record['revision']}


def read_current(client):
    item=client.api('contents/'+PATH+'?ref='+BRANCH)
    return validate(json.loads(base64.b64decode(item['content'])))


if __name__=='__main__':
    print(json.dumps(read_current(Client(os.environ.get('GITHUB_REPOSITORY',REPOSITORY))),indent=2))
