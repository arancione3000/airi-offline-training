"""Publish immutable experiment snapshots, committing the index last."""
import json,subprocess,sys,zipfile
from pathlib import Path
from session import atomic_json,sha256_file

ROOT=Path(__file__).resolve().parents[1]

def upload(paths):
    request={'uploads':[{'local_path':str(Path(p).resolve()),'purpose':'create_library_file',
                         'library_artifact_type':'report'} for p in paths]}
    proc=subprocess.run([sys.executable,str(ROOT/'recovery-helpers/library_upload.py')],
                        input=json.dumps(request),capture_output=True,text=True)
    if proc.returncode:raise RuntimeError('checkpoint upload failed; training halted; inspect local transport result')
    result=json.loads(proc.stdout)
    items=result['results']
    if len(items)!=len(paths):raise RuntimeError('checkpoint upload count mismatch')
    for item,path in zip(items,paths):
        if item.get('status')!='succeeded' or not item.get('local_metadata_applied'):
            raise RuntimeError('checkpoint upload incomplete; training halted')
        if Path(item['local_path']).resolve()!=Path(path).resolve():
            raise RuntimeError('checkpoint upload identity mismatch')
    return items

def publish(checkpoint,output,*,extra_files=()):
    checkpoint=Path(checkpoint);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    step=json.loads((checkpoint/'checkpoint.json').read_text())['completed_steps']
    archives=[]
    for part in ['MODEL','OPTIMIZER']:
        dest=output/f'AIRI_TRAINING_{step:05d}_{part}.zip'
        with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for p in sorted(checkpoint.rglob('*')):
                if p.is_file() and ((p.relative_to(checkpoint).parts[0]=='model')==(part=='MODEL')):
                    z.write(p,str(Path('checkpoint')/p.relative_to(checkpoint)))
            if part=='OPTIMIZER':
                for p in extra_files:z.write(p,str(Path('experiment')/Path(p).name))
        archives.append(dest)
    items=upload(archives)
    index=output/f'AIRI_TRAINING_CHECKPOINT_{step:05d}.json'
    atomic_json(index,{'completed_steps':step,'fingerprint':json.loads((checkpoint/'checkpoint.json').read_text())['fingerprint'],
                       'production_qualified':False,'live_promoted':False,
                       'archives':[{'sha256':sha256_file(p),'bytes':p.stat().st_size,**item}
                                   for p,item in zip(archives,items)]})
    # This small file is the commit record; partial uploads are never durable checkpoints.
    committed=upload([index])[0]
    atomic_json(output/'LATEST_DURABLE.json',{'completed_steps':step,'index':committed})
    return committed
