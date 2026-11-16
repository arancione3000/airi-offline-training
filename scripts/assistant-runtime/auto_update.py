"""Gated automatic procedural updates from a finite, exact-reference curriculum.

No paid API or native-weight training. Candidates live in a temporary bank until
new-value audits and historical routing pass. Only then is one snapshot committed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import time

from auto_curriculum import CATALOG, VERSION, build
from context_memory import resolve
from learn import private_bank
from skill_bank import MAX_SKILLS, RegistryError, SkillBank, canonical, digest, _route
from skill_validation import StableGate


def _write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(canonical(value)+b'\n')


def _source_hashes():
    root=Path(__file__).resolve().parent
    paths=[root/name for name in ('auto_update.py','auto_curriculum.py','skill_programs.py',
           'skill_bank.py','context_memory.py','skill_validation.py')]
    paths.append(root/'skills/stable_contracts.json.gz')
    return {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def cycle(bank, *, seed, limit=2, report_dir=None, progress=None):
    if type(limit) is not int or not 1<=limit<=4:raise ValueError('cycle limit must be1..4')
    original,_=bank._read();before=digest(original)
    acquired={entry['lesson_id'] for entry in original['entries']}
    pending=[family for family in CATALOG if family['id'] not in acquired]
    selected=pending[:min(limit,MAX_SKILLS-len(original['entries']))]
    selected_ids={family['id'] for family in selected}
    bundles={family['id']:build(family['id'],seed) for family in CATALOG
             if family['id'] in acquired or family['id'] in selected_ids}
    sources=_source_hashes()
    prereg={'schema':1,'curriculum_version':VERSION,'seed':seed,'source_hashes':sources,
            'starting_bank_sha256':before,'selected':[f['id'] for f in selected],
            'bundle_sha256':{key:digest(value) for key,value in bundles.items()},
            'rule':'TRAIN selects program; qualification and heldout only accept/reject; no retries on a failed seed'}
    if report_dir is not None:
        report_dir=Path(report_dir);report_dir.mkdir(parents=True,exist_ok=False)
        _write_json(report_dir/'PREREGISTRATION.json',prereg)
        _write_json(report_dir/'INPUTS.json',bundles)
    with tempfile.TemporaryDirectory(prefix='airi-auto-update-') as directory:
        staged=SkillBank(Path(directory)/'candidate.json');staged._write(original)
        growth=[]
        for family in selected:
            gate=StableGate()
            transition=staged.learn(bundles[family['id']]['lesson'],regression_gate=gate)
            growth.append({'lesson_id':family['id'],'transition':transition,'stable_gate':gate.last_report})
            if transition['action']!='activated':raise RegistryError('automatic candidate admission failed: '+family['id'])
        proposal,_=staged._read()
        observations=[];by_family={}
        for family_id,bundle in bundles.items():
            values=[]
            for row in bundle['heldout']:
                result=_route(proposal['entries'],row['messages'])
                answer=None if result is None else result['answer']
                values.append({'messages':row['messages'],'expected':row['answer'],
                               'prediction':result,'correct':answer==row['answer']})
            by_family[family_id]={'correct':sum(r['correct'] for r in values),'total':len(values)}
            observations.extend({'lesson_id':family_id,**r} for r in values)
        gate=StableGate()
        stored_contracts=0
        for entry in proposal['entries']:
            for row in entry['train']+entry['qualification']:
                result=_route(proposal['entries'],row['messages'])
                if (None if result is None else result['answer'])!=row['answer']:
                    raise RegistryError('previous skill contract failed; live bank unchanged')
                stored_contracts+=1
        # Route all experts in their real priority order; StableGate retains memory priority.
        if not gate(lambda messages:_route(proposal['entries'],messages)):
            raise RegistryError('historical routing audit failed')
        passed=all(row['correct'] for row in observations)
        report={'schema':1,'checked_at_utc':datetime.now(timezone.utc).isoformat(),
                'seed':seed,'preregistration_sha256':digest(prereg),'before_sha256':before,
                'after_sha256':digest(proposal),'before_revision':original['revision'],
                'after_revision':proposal['revision'],'activated':len(selected),'growth':growth,
                'audit':{'correct':sum(r['correct'] for r in observations),'total':len(observations),
                         'by_family':by_family,'observations':observations,'passed':passed},
                'stable_gate':gate.last_report,'remaining_families':len(pending)-len(selected),
                'stored_expert_contracts_passed':stored_contracts,
                'catalog_size':len(CATALOG),'capacity':MAX_SKILLS,'neural_weights_changed':False,
                'scope':'finite authored procedural templates; independent exact labels, not general chat or neural improvement',
                'status':'candidate','published':False}
        if report_dir is not None:_write_json(report_dir/'REPORT.json',report)
        if not passed:raise RegistryError('new-value audit failed; live bank unchanged')
        if _source_hashes()!=sources:raise RegistryError('source changed during cycle')
        if selected:
            with bank._lock():
                current,_=bank._read()
                if digest(current)!=before:raise RegistryError('concurrent bank update; candidate not published')
                publication=bank._write(proposal)
            report.update(status='updated',published=True,publication=publication)
        else:report['status']='capacity' if pending else 'audited'
        if report_dir is not None:_write_json(report_dir/'REPORT.json',report)
        if progress is not None and selected:
            summary={key:value for key,value in report.items() if key not in ('audit','growth')}
            summary['audit']={key:value for key,value in report['audit'].items() if key!='observations'}
            summary['report_sha256']=digest(report)
            _write_json(progress,summary)
        return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,help='Explicit bank; default uses the private chat bank')
    p.add_argument('--limit',type=int,default=2)
    p.add_argument('--seed',type=int)
    p.add_argument('--report-dir',type=Path)
    p.add_argument('--progress',type=Path)
    p.add_argument('--watch',action='store_true')
    p.add_argument('--interval',type=int,default=3600)
    p.add_argument('--cycles',type=int,default=0,help='0 continues until stopped')
    a=p.parse_args()
    if not 1<=a.limit<=4 or not 60<=a.interval<=86400 or a.cycles<0:p.error('invalid cycle bounds')
    count=0
    while True:
        seed=a.seed if a.seed is not None else time.time_ns()%(2**64)
        directory=a.report_dir
        if a.watch and directory is not None:directory=directory/f'cycle-{time.time_ns()}'
        try:
            result=cycle(private_bank(a.bank),seed=seed,limit=a.limit,report_dir=directory,progress=a.progress)
            print(json.dumps({key:result[key] for key in ('status','activated','after_revision','remaining_families')},ensure_ascii=False),flush=True)
        except (ValueError,OSError) as error:
            print(json.dumps({'status':'failed','error':str(error)[:240],'live_update':'only committed qualified snapshots apply'}),flush=True)
            if not a.watch:raise SystemExit(1)
        count+=1
        if not a.watch or (a.cycles and count>=a.cycles):return
        time.sleep(a.interval)


if __name__=='__main__':main()
