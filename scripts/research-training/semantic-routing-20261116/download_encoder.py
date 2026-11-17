"""Download the pinned optional research encoder; verify every exact file hash."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    manifest_path=Path(__file__).with_name('ENCODER_MANIFEST.json')
    manifest=json.loads(manifest_path.read_text())
    a.output.mkdir(parents=True,exist_ok=True)
    for name,descriptor in manifest['files'].items():
        path=a.output/name
        if path.is_symlink():raise ValueError('symlink encoder destination refused')
        if path.exists() and path.stat().st_size==descriptor['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==descriptor['sha256']:
            continue
        path.parent.mkdir(parents=True,exist_ok=True)
        url=f"https://huggingface.co/{manifest['model_id']}/resolve/{manifest['revision']}/{name}"
        fd,temporary=tempfile.mkstemp(dir=path.parent)
        try:
            count=0;digest=hashlib.sha256()
            with urllib.request.urlopen(url,timeout=60) as source,os.fdopen(fd,'wb') as output:
                while True:
                    chunk=source.read(1024*1024)
                    if not chunk:break
                    count+=len(chunk)
                    if count>descriptor['bytes']:raise ValueError('encoder exceeds declared bytes')
                    digest.update(chunk);output.write(chunk)
                output.flush();os.fsync(output.fileno())
            if count!=descriptor['bytes'] or digest.hexdigest()!=descriptor['sha256']:raise ValueError('encoder hash mismatch')
            os.replace(temporary,path)
        finally:
            if os.path.exists(temporary):os.unlink(temporary)
        print(json.dumps({'verified':name,'bytes':count}),flush=True)
    (a.output/'MANIFEST.json').write_bytes(manifest_path.read_bytes())


if __name__=='__main__':main()
