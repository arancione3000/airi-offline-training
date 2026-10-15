"""Load the stable AIRI identity; checkpoint names are resolved automatically."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from model_channel import Client,REPOSITORY,read_current,validate


def validate_selected_model(checkpoint, record):
    import hashlib
    validate(record); checkpoint=Path(checkpoint)
    manifest=(json.loads((checkpoint/'checkpoint.json').read_text())
              if (checkpoint/'checkpoint.json').exists() else
              {'fingerprint':record['fingerprint'],'completed_steps':record['completed_steps'],
               'files':record['model_files']})
    if manifest['fingerprint']!=record['fingerprint'] or manifest['completed_steps']!=record['completed_steps']:
        raise ValueError('selected checkpoint identity mismatch')
    model_files={n:d for n,d in manifest['files'].items() if n.startswith('model/')}
    if model_files!=record['model_files']:
        raise ValueError('stable pointer and native model manifest differ')
    if 'model/model.pt' not in model_files:
        raise ValueError('selected checkpoint has no native model')
    for name,digest in model_files.items():
        path=checkpoint/name
        if not path.resolve().is_relative_to(checkpoint.resolve()):
            raise ValueError('unsafe selected model path')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:
                raise ValueError('selected model integrity failed')
    if model_files['model/model.pt']!=record['model_file_sha256']:
        raise ValueError('stable model pointer does not match the checkpoint weights')
    return checkpoint/'model'


def load_current(root, record=None):
    from generalist_lm.runtime import GeneralistRuntime
    from session import sha256_file
    from transport import safe_extract
    record=record or read_current(Client(os.environ.get('GITHUB_REPOSITORY',REPOSITORY)))
    validate(record)
    root=Path(root)/record['model_file_sha256'];root.mkdir(parents=True,exist_ok=True)
    checkpoint=root/'checkpoint'
    if not (checkpoint/'checkpoint.json').exists():
        download=root/'download';download.mkdir(exist_ok=True)
        step=record['completed_steps'];tag=record['checkpoint_release']
        index_name=f'AIRI_TRAINING_CHECKPOINT_{step:05d}.json'
        subprocess.run(['gh','release','download',tag,'--repo',REPOSITORY,'--pattern',index_name,
                        '--dir',str(download),'--clobber'],check=True,timeout=120)
        index=json.loads((download/index_name).read_text())
        if (index['fingerprint']!=record['fingerprint'] or index['completed_steps']!=step
                or index['production_qualified'] is not False or index['live_promoted'] is not False):
            raise ValueError('stable checkpoint index provenance mismatch')
        name=f'AIRI_TRAINING_{step:05d}_MODEL.zip'
        items=[a for a in index['archives'] if a['name']==name]
        if len(items)!=1:
            raise ValueError('one immutable native model archive required')
        item=items[0]
        subprocess.run(['gh','release','download',tag,'--repo',REPOSITORY,'--pattern',name,
                        '--dir',str(download),'--clobber'],check=True,timeout=180)
        archive=download/name
        if archive.stat().st_size!=item['bytes'] or sha256_file(archive)!=item['sha256']:
            raise ValueError('stable model archive integrity failed')
        safe_extract(archive,root)
        archive.unlink()
    model=validate_selected_model(checkpoint,record)
    if not (checkpoint/'checkpoint.json').exists():
        from session import atomic_json
        atomic_json(checkpoint/'checkpoint.json',{'fingerprint':record['fingerprint'],
                    'completed_steps':record['completed_steps'],'files':record['model_files']})
    return GeneralistRuntime.from_checkpoint(model),record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,default=Path('airi-current-cache'))
    args=parser.parse_args()
    runtime,record=load_current(args.cache)
    print(f"AIRI, revisione {record['revision']} — modello sperimentale")
    history=[]
    while True:
        try:message=input('Tu: ')
        except (EOFError,KeyboardInterrupt):break
        if message.strip() in {'/exit','/quit'}:break
        current=read_current(Client(os.environ.get('GITHUB_REPOSITORY',REPOSITORY)))
        if current['revision']>record['revision']:
            runtime,record=load_current(args.cache,current)
        history.append({'role':'user','content':message})
        answer=runtime.chat(history,temperature=0,max_new_tokens=192)
        history.append({'role':'assistant','content':answer})
        print('AIRI:',answer)


if __name__=='__main__':main()
