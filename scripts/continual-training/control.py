"""Resume ONE authorized, finite learning experiment from verified releases."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/autonomous-control'))
from watchdog import transient

BRANCH='dialogue-selective-replay-20261018'
WORKFLOW='airi-continual-background.yml'
CONTROLLER='airi-continual-controller.yml'
TITLE='AIRI continual learning trial'
SPEC=json.loads((Path(__file__).parent/'SPEC.json').read_text())


class Client:
    def __init__(self,repository):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repository):raise ValueError('invalid repository')
        self.base='repos/'+repository
    def get(self,path):return json.loads(subprocess.check_output(['gh','api',self.base+'/'+path],text=True))
    def post(self,path,payload):
        subprocess.run(['gh','api','--method','POST',self.base+'/'+path,'--input','-'],
                       input=json.dumps(payload),text=True,check=True)
    def disable(self):subprocess.run(['gh','api','--method','PUT',self.base+'/actions/workflows/'+CONTROLLER+'/disable'],check=True)
    def asset(self,item):
        raw=subprocess.check_output(['gh','api',self.base+'/releases/assets/'+str(item['id']),'-H','Accept: application/octet-stream'])
        if len(raw)!=item['size'] or len(raw)>1024*1024:raise ValueError('invalid controller asset size')
        if item.get('digest') and item['digest']!='sha256:'+hashlib.sha256(raw).hexdigest():raise ValueError('asset digest mismatch')
        return raw
    def logs(self,run):
        jobs=self.get('actions/runs/'+str(run['id'])+'/jobs?per_page=100')['jobs']
        return '\n'.join(subprocess.check_output(['gh','api',self.base+'/actions/jobs/'+str(j['id'])+'/logs'],text=True)
                         for j in jobs if j['conclusion']=='failure')


def verified_state(client,release):
    assets={a['name']:a for a in release['assets']}
    indexes=[n for n in assets if re.fullmatch(r'AIRI_TRAINING_CHECKPOINT_[0-9]+\.json',n)]
    if len(indexes)!=1:return None
    index=json.loads(client.asset(assets[indexes[0]]))
    if index.get('fingerprint')!=SPEC['fingerprint']:raise ValueError('continuation fingerprint mismatch')
    if index.get('production_qualified') is not False or index.get('live_promoted') is not False:
        raise ValueError('unqualified continuation state')
    archives=index['archives'];step=index['completed_steps']
    expected={f'AIRI_TRAINING_{step:05d}_{part}.zip' for part in ('MODEL','OPTIMIZER')}
    if len(archives)!=2 or {a['name'] for a in archives}!=expected:raise ValueError('invalid checkpoint archives')
    for a in archives:
        remote=assets.get(a['name'])
        if (not remote or not isinstance(a['bytes'],int) or a['bytes']<=0 or remote['size']!=a['bytes']
                or not re.fullmatch('[a-f0-9]{64}',a['sha256'])
                or remote.get('digest')!='sha256:'+a['sha256']):
            raise ValueError('checkpoint archive digest mismatch')
    descriptor=index.get('run_state');remote=assets.get('AIRI_RUN_STATE.json')
    if not descriptor or not remote:raise ValueError('checkpoint lacks committed continuation state')
    raw=client.asset(remote)
    if descriptor!={'name':'AIRI_RUN_STATE.json','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}:
        raise ValueError('continuation state digest mismatch')
    state=json.loads(raw)
    if (state['fingerprint']!=SPEC['fingerprint'] or state['completed_steps']!=step
            or state['maximum_training_seconds']!=SPEC['maximum_training_seconds']
            or state['production_qualified'] is not False or state['live_promoted'] is not False
            or not isinstance(state['continue'],bool)):
        raise ValueError('invalid committed continuation identity')
    seconds=state['elapsed_training_seconds'];job=state['job_index']
    if (isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<0
            or not isinstance(job,int) or isinstance(job,bool) or not 0<=job<32
            or not SPEC['start_step']<=step<=SPEC['max_steps']):
        raise ValueError('invalid continuation budget')
    stopped=state['search']['stopped']
    if not isinstance(stopped,bool) or state['continue']!=(not stopped and seconds<SPEC['maximum_training_seconds']):
        raise ValueError('continuation stop/budget mismatch')
    return state


def tick(client):
    runs=client.get('actions/workflows/'+WORKFLOW+'/runs?branch='+BRANCH+'&per_page=10')['workflow_runs']
    runs=[r for r in runs if r['head_branch']==BRANCH and r['display_title']==TITLE]
    if not runs:return {'action':'awaiting_initial_start'}
    if any(r['status']!='completed' for r in runs):return {'action':'active_job_retained'}
    run=runs[0]
    if run['conclusion']=='cancelled':client.disable();return {'action':'cancellation_respected'}
    head=client.get('git/ref/heads/'+BRANCH)['object']['sha']
    if head!=run['head_sha']:client.disable();return {'action':'changed_branch_stopped'}
    failed=0
    for r in runs:
        if r['conclusion']=='success':break
        failed+=1
    if run['conclusion']!='success':
        if failed>=3 or not transient(client.logs(run)):
            client.disable();return {'action':'nontransient_or_retry_limit_stopped','run':run['id']}
    releases=client.get('releases?per_page=100')
    choices=[r for r in releases if r['target_commitish']==head and r.get('draft') is False
             and r.get('prerelease') is True and re.fullmatch(r'airi-offline-[0-9]+-[0-9]+-[0-9]+-[0-9]+',r['tag_name'])]
    choices.sort(key=lambda r:tuple(int(n) for n in r['tag_name'].split('-')[2:]),reverse=True)
    selected=None
    for release in choices:
        state=verified_state(client,release)
        if state is not None:selected=(release,state);break
    if selected is None:
        if run['conclusion']=='success':raise ValueError('successful training job has no verified checkpoint')
        # Failed before a first checkpoint: bounded replay of the pinned seed.
        release=SPEC['migration_source_release'];continuation=failed
    else:
        release,state=selected
        if not state['continue']:
            client.disable();return {'action':'finite_experiment_stopped','revision':state['revision'],
                                     'elapsed_training_seconds':state['elapsed_training_seconds']}
        continuation=state['job_index']+1
        release=release['tag_name']
    if continuation>=32:client.disable();return {'action':'finite_job_limit_stopped'}
    # Repository-wide lock: never start another native training workflow in parallel.
    active=client.get('actions/runs?per_page=100')['workflow_runs']
    if any(r['status']!='completed' and r.get('path','').endswith('-background.yml') for r in active):
        return {'action':'waiting_for_other_training'}
    # A final fresh lease protects cancellation/branch changes during reads.
    if client.get('git/ref/heads/'+BRANCH)['object']['sha']!=head:raise ValueError('runner ref changed')
    client.post('actions/workflows/'+WORKFLOW+'/dispatches',{'ref':BRANCH,'inputs':{
        'checkpoint_release':release,'continuation':str(continuation)}})
    return {'action':'resumed_verified_checkpoint','release':release,'continuation':continuation}


if __name__=='__main__':print(json.dumps(tick(Client(os.environ['GITHUB_REPOSITORY']))),flush=True)
