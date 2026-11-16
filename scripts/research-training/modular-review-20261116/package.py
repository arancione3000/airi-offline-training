"""Archive one completed review without model weights or private environment data."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('evidence'))
    a=p.parse_args()
    report=json.loads((a.run/'REPORT.json').read_text())
    prereg=json.loads((a.run/'PREREGISTRATION.json').read_text())
    assert report['complete'] and report['native_weights_revalidated']
    if a.output.exists():raise ValueError('refusing to rewrite evidence')
    a.output.mkdir(parents=True)
    artifacts=[]
    for path in sorted(a.run.rglob('*')):
        if not path.is_file():continue
        if path.suffix not in ('.json','.py','.gz'):raise ValueError('unexpected evidence type')
        raw=path.read_bytes();stored=gzip.compress(raw,mtime=0)
        relative=str(path.relative_to(a.run))+'.gz';target=a.output/relative
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(stored)
        assert gzip.decompress(target.read_bytes())==raw
        artifacts.append({'path':relative,'raw_bytes':len(raw),'stored_bytes':len(stored),
            'raw_sha256':hashlib.sha256(raw).hexdigest(),'stored_sha256':hashlib.sha256(stored).hexdigest()})
    summary={k:v for k,v in report.items() if k!='observations'}
    summary.update(model_revision=prereg['model_revision'],weights_digest=prereg['weights_digest'],
        preregistration_time_utc=prereg['created_at_utc'],seed=prereg['seed'])
    (a.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    manifest={'schema':1,'scope':report['scope'],'artifacts':artifacts}
    (a.output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'artifacts':len(artifacts),'compressed_bytes':sum(x['stored_bytes'] for x in artifacts),
        'manifest_sha256':hashlib.sha256((a.output/'MANIFEST.json').read_bytes()).hexdigest(),
        'summary':summary}))


if __name__=='__main__':main()
