"""GitHub-only transient recovery; no assistant, model API, or new training recipe."""
import argparse
import json
import re
import subprocess
import time

BRANCH = 'overnight-training-v1'
COMMIT = 'f4bb6c14a9be26adaa45c50b5db2638867a8df44'
WORKFLOW = 'airi-overnight-background.yml'
EXPERIMENTS={WORKFLOW:(BRANCH,COMMIT,None),
    'airi-transfer-background.yml':('dialogue-cycle-20261010','d00bf9770884295c8f17877c97ab711cda743401',None),
    'airi-focused-background.yml':('focused-learning-v1','bd98259560582bb64258a1a8c7a6c0c0f0c90dfd',None),
    'airi-noon-background.yml':('morning-dialogue-v1','6cc1ea287133b671ab73b5504350fd33ebb483fd',1791547200.0),
    'airi-evolving-background.yml':('evolving-context-v1','6c6c72dea5b7512cb45ec174ee69510674b7bb1f',None),
    'airi-stable-background.yml':('stable-dialogue-v1','f436b73d71e9eee06595f58b96945672f36a6527',None),
    'airi-afternoon-background.yml':('afternoon-dialogue-v1','0a502c0242528f475837167501c2056eaaeecdc2',1791459600.0)}

TRAINING_PATHS = {
    '.github/workflows/airi-transfer-background.yml',
    '.github/workflows/airi-focused-background.yml',
    '.github/workflows/airi-noon-background.yml',
    '.github/workflows/airi-evolving-background.yml',
    '.github/workflows/airi-stable-background.yml',
    '.github/workflows/airi-afternoon-background.yml',
    '.github/workflows/airi-overnight-background.yml',
    '.github/workflows/airi-conversation-background.yml',
    '.github/workflows/airi-offline-background.yml',
    '.github/workflows/airi-recovery-background.yml',
    '.github/workflows/airi-expanded-background.yml',
}
ACTIVE = {'in_progress','queued','waiting','requested','pending'}

def workflow_path(run):
    return run.get('path','').split('@',1)[0]

def transient(log):
    log = log.casefold()
    if any(marker in log for marker in [
        'fingerprint mismatch','digest mismatch','provenance mismatch','unsafe archive',
        'reference changed','trial or curriculum mismatch','runner ref changed',
        'continuation limit reached','finite job limit reached',
        'all recipes failed','three validation rounds without',
    ]):
        return False
    return bool(re.search(r'(?:http|status code|returned error|gh:)[^\n]{0,160}\b(?:429|500|502|503|504)\b',log)
                or any(marker in log for marker in [
                    'connection reset by peer','temporary failure in name resolution',
                    'tls handshake timeout','connection timed out','remote end closed connection',
                    'the runner has lost communication','failed to connect to',
                ]))

class Client:
    def __init__(self, repository):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository):
            raise ValueError('invalid repository')
        self.base='repos/'+repository
    def get(self,path):
        return json.loads(subprocess.check_output(['gh','api',self.base+'/'+path],text=True))
    def logs(self,job_id):
        return subprocess.check_output(['gh','api',self.base+f'/actions/jobs/{int(job_id)}/logs'],text=True)
    def rerun(self,run_id):
        subprocess.run(['gh','api','--method','POST',self.base+f'/actions/runs/{int(run_id)}/rerun-failed-jobs'],check=True)
    def pages(self,path,key):
        page=1
        while True:
            separator='&' if '?' in path else '?'
            values=self.get(path+separator+f'per_page=100&page={page}')[key]
            yield from values
            if len(values)<100:break
            page+=1

def recover(client,run_id=None,dry_run=False):
    if run_id:
        run=client.get(f'actions/runs/{int(run_id)}')
    else:
        runs=[]
        for workflow,(branch,commit,cutoff) in EXPERIMENTS.items():
            runs+=client.get(f'actions/workflows/{workflow}/runs?branch={branch}&per_page=1')['workflow_runs']
        if not runs:return {'action':'none','reason':'no configured run'}
        run=max(runs,key=lambda r:r['id'])
    workflow=workflow_path(run).rsplit('/',1)[-1]
    experiment=EXPERIMENTS.get(workflow)
    if experiment is None:return {'action':'none','reason':'outside configured frozen experiment'}
    branch,commit,cutoff=experiment
    if run.get('head_branch')!=branch or run.get('head_sha')!=commit:
        return {'action':'none','reason':'outside configured frozen experiment'}
    if cutoff is not None and time.time()>=cutoff:
        return {'action':'none','reason':'afternoon training deadline passed; do not restart'}
    if run['status']!='completed' or run.get('conclusion') not in {'failure','timed_out'}:
        return {'action':'none','reason':'running, successful, quality stop, or intentionally cancelled'}
    if run.get('run_attempt',1)>=3:
        return {'action':'none','reason':'two automatic recovery attempts exhausted; checkpoint retained'}
    # Never restart an old failed ancestor after a newer continuation has started.
    latest=client.get(f'actions/workflows/{workflow}/runs?branch={branch}&per_page=1')['workflow_runs']
    if not latest or latest[0]['id']!=run['id']:
        return {'action':'none','reason':'superseded by a newer run'}
    for status in sorted(ACTIVE):
        if any(workflow_path(r) in TRAINING_PATHS for r in client.pages(f'actions/runs?status={status}','workflow_runs')):
            return {'action':'none','reason':'another training job is active'}
    if client.get('git/ref/heads/'+branch)['object']['sha']!=commit:
        return {'action':'none','reason':'frozen experiment ref changed'}
    failed=[j for j in client.pages(f'actions/runs/{run["id"]}/jobs','jobs') if j.get('conclusion')=='failure']
    if not failed:
        return {'action':'none','reason':'no completed failed job logs to diagnose'}
    logs='\n'.join(client.logs(j['id']) for j in failed)
    if not transient(logs):
        return {'action':'none','reason':'no verified transient network/platform error; checkpoint retained'}
    if not dry_run:client.rerun(run['id'])
    return {'action':'would retry' if dry_run else 'retry submitted','run_id':run['id'],
            'next_attempt':run.get('run_attempt',1)+1,
            'reason':'verified transient error; same frozen inputs and checkpoint checks'}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repository',required=True)
    parser.add_argument('--run-id',type=int)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    print(json.dumps(recover(Client(args.repository),args.run_id,args.dry_run)))

if __name__=='__main__':main()
