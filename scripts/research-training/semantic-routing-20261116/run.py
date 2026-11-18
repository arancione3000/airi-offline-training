"""Single TRAIN-calibrated lexical router; independent qualification then FINAL."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
RUNTIME=ROOT/'scripts/assistant-runtime'
sys.path.insert(0,str(RUNTIME))
from learn import selected_bank
from skill_bank import SkillBank
import query_router
from query_learning import bindings,evaluate


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bank',type=Path)
    p.add_argument('--backend',choices=['centroid','linear','factorized','encoder'],default='centroid')
    a=p.parse_args()
    backend=query_router
    if a.backend=='linear':
        import query_router_linear
        backend=query_router_linear
    elif a.backend=='factorized':
        import query_router_factorized
        backend=query_router_factorized
    elif a.backend=='encoder':
        import query_router_encoder
        backend=query_router_encoder
    if a.output.exists():raise ValueError('choose a fresh output; no replacement of an observed run')
    a.output.mkdir(parents=True)
    bank=selected_bank(a.bank)
    families=json.loads((a.corpus/'FAMILIES.json').read_text())['families']
    raw=json.loads((a.corpus/'TRAIN.json').read_text())['rows']
    labels=bindings(bank,families);known={r['lesson_id'] for r in labels}
    train=[{key:row[key] for key in ('id','query','lesson_id','source_lesson_id')}
           for row in raw if row['source_lesson_id'] in known]
    source={str(path):sha(path) for path in RUNTIME.glob('*.py')}
    inputs={str(path):sha(path) for path in a.corpus.glob('*.json')}
    prereg={'source_hashes':source,'input_hashes':inputs,'bank':bank.status(),
            'rule':'TRAIN-only calibration; qualification and final require >=50% correct positives, no wrong answers and no retention regressions; final data never selects a model'}
    write(a.output/'PREREGISTRATION.json',prereg)
    (a.output/'sources').mkdir()
    for path in RUNTIME.glob('*.py'):(a.output/'sources'/(path.name+'.gz')).write_bytes(gzip.compress(path.read_bytes(),mtime=0))
    start=time.perf_counter()
    if a.backend in ('factorized','encoder'):
        metadata=[{key:f[key] for key in ('id','operation','object_phrase')} for f in families if f['id'] in known]
        model=backend.train(train,labels,family_metadata=metadata)
    else:model=backend.train(train,labels)
    backend.write_model(a.output/'candidate.json',model)
    write(a.output/'FREEZE.json',{'model_sha256':model['sha256'],'calibration':model['calibration'],
                                'thresholds':model['thresholds'],'train_seconds':time.perf_counter()-start})
    print(json.dumps({'phase':'frozen','calibration':model['calibration'],'thresholds':model['thresholds']}),flush=True)
    qualification=json.loads((a.corpus/'QUALIFICATION.json').read_text())['rows']
    qual=evaluate(bank,a.output/'candidate.json',qualification,families,value_offset=10000,backend=backend)
    write(a.output/'QUALIFICATION_REPORT.json',qual)
    print(json.dumps({'phase':'qualification',**{k:v for k,v in qual.items() if k not in ('observations','historical_gate')}}),flush=True)
    if not qual['passed']:
        write(a.output/'RESULT.json',{'activated':False,'reason':'qualification failed','final_observed':False})
        return
    final=json.loads((a.corpus/'FINAL.json').read_text())['rows']
    result=evaluate(bank,a.output/'candidate.json',final,families,value_offset=20000,backend=backend)
    write(a.output/'FINAL_REPORT.json',result)
    diagnostic=json.loads((a.corpus/'DIAGNOSTIC.json').read_text())['rows']
    diag=evaluate(bank,a.output/'candidate.json',diagnostic,families,value_offset=30000,backend=backend)
    write(a.output/'DIAGNOSTIC_REPORT.json',diag)
    assert all(sha(Path(path))==value for path,value in source.items())
    assert all(sha(Path(path))==value for path,value in inputs.items())
    write(a.output/'RESULT.json',{'eligible_for_activation':result['passed'],'model_sha256':model['sha256'],
                                 'qualification_passed':True,'final_observed':True,'scope':'bounded lexical paraphrase routing; no modern Transformer comparison or neural weight gain'})
    print(json.dumps({'phase':'final',**{k:v for k,v in result.items() if k not in ('observations','historical_gate')}}),flush=True)


if __name__=='__main__':main()
