"""Run a new semantic experiment only after trusted data/code/bindings change.

No source self-modification or self-generated truth. Only qualified model data
may be staged; the workflow tests that tree before a fast-forward publication.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
RUNTIME=ROOT/'scripts/assistant-runtime'
sys.path.insert(0,str(RUNTIME))
from exact_aliases import read_aliases
from skill_bank import SkillBank, RegistryError, canonical, digest
from flexible_skills import LearnedQueryRouter

HERE=Path(__file__).resolve().parent
STATE=RUNTIME/'queries/research-state.json'
MODEL=RUNTIME/'queries/approved-router.json'
RETENTION=RUNTIME/'queries/retention.json.gz'
ALIASES=RUNTIME/'queries/approved-aliases.json'


def exact_transfer():
    path=HERE.parent/'exact-transfer-20261120/cycle.py'
    spec=importlib.util.spec_from_file_location('exact_transfer_cycle',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint():
    paths=list(RUNTIME.glob('*.py'))+[HERE/'run.py',HERE/'cycle.py',HERE/'ENCODER_MANIFEST.json',HERE/'requirements-encoder.txt']
    paths.append(HERE.parent/'exact-transfer-20261120/cycle.py')
    paths+=list((HERE/'evidence/precommitted').glob('*.gz'))
    source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    bank=SkillBank(RUNTIME/'skills/approved-bank.json')._read()[0]
    return digest({'sources':source,'programs':[(entry['id'],entry['lesson_id']) for entry in bank['entries']]})


def needed():
    if not STATE.exists():return True
    return json.loads(STATE.read_text()).get('input_sha256')!=fingerprint()


def run(output):
    if not needed():return {'action':'unchanged','reason':'same trusted inputs and code; no repeated training'}
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    corpus=output/'corpus';corpus.mkdir()
    for name in ('FAMILIES','TRAIN','QUALIFICATION','FINAL','DIAGNOSTIC','MANIFEST'):
        compressed=(HERE/'evidence/precommitted'/(name+'.json.gz')).read_bytes()
        raw=gzip.decompress(compressed)
        if len(raw)>2*1024*1024:raise ValueError('trusted corpus exceeds bound')
        (corpus/(name+'.json')).write_bytes(raw)
    bank=RUNTIME/'skills/approved-bank.json'
    subprocess.run([sys.executable,str(HERE/'run.py'),'--backend','encoder','--bank',str(bank),
                    '--corpus',str(corpus),'--output',str(output/'trial')],check=True)
    result=json.loads((output/'trial/RESULT.json').read_text())
    eligible=result.get('eligible_for_activation') is True
    state={'input_sha256':fingerprint(),'action':'rejected','result':result,
           'scope':'operational replay after publication, not a new blind confirmation',
           'model_staged':False,'neural_weights_changed':False}
    if eligible:
        import query_router_encoder as backend
        candidate=output/'trial/candidate.json';new=backend.read_model(candidate)
        source=SkillBank(bank)
        routed=LearnedQueryRouter(source,candidate,frozen=True,backend=backend)
        transfer=exact_transfer()
        old=transfer.read_semantic_retention(RETENTION)
        qualification=json.loads((output/'trial/QUALIFICATION_REPORT.json').read_text())['observations']
        final=json.loads((output/'trial/FINAL_REPORT.json').read_text())['observations']
        checks=[*old]
        for row in qualification+final:
            checks.append({'messages':row['messages'],'expected':row['expected'],
                           'allow_abstention':not row['correct']})
        try:
            state['composition_audit']=transfer.audit_candidate(
                routed.snapshot,read_aliases(ALIASES),routed,checks,seed=20261120)
        except RegistryError as error:
            eligible=False
            state.update(reason=str(error),composition_audit={'passed':False})
        if eligible:
            retained={digest(row['messages']):row for row in old}
            for row in qualification+final:
                if row['correct']:
                    retained[digest(row['messages'])]={'messages':row['messages'],'expected':row['expected']}
            MODEL.parent.mkdir(parents=True,exist_ok=True)
            backend.write_model(MODEL,new)
            RETENTION.write_bytes(gzip.compress(canonical(list(retained.values())),mtime=0))
            state.update(action='candidate staged',model_staged=True,model_sha256=new['sha256'],
                         retention_cases=len(retained),retention_used_for_training=False)
    STATE.parent.mkdir(parents=True,exist_ok=True)
    STATE.write_bytes(canonical(state)+b'\n')
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--needed',action='store_true')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.needed:print(str(needed()).lower());return
    if a.output is None:p.error('--output is required')
    print(json.dumps(run(a.output)),flush=True)


if __name__=='__main__':main()
