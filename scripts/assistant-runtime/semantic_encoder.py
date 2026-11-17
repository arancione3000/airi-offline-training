"""Optional frozen local semantic encoder. Never downloads or calls a model API."""
import hashlib
import json
import os
from pathlib import Path
import stat

MAX_FILE_BYTES=512*1024*1024
_RUNTIME=None


def _sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _signature(path):
    info=path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size>MAX_FILE_BYTES:raise ValueError('invalid local encoder file')
    return (str(path),info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)


def manifest():
    root=Path(os.environ.get('AIRI_QUERY_ENCODER_CACHE','')).expanduser()
    if not os.environ.get('AIRI_QUERY_ENCODER_CACHE'):raise ValueError('local encoder cache is not configured')
    path=root/'MANIFEST.json';_signature(path)
    if path.stat().st_size>128*1024:raise ValueError('encoder manifest exceeds bound')
    raw=path.read_bytes()
    try:value=json.loads(raw)
    except (RecursionError,TypeError) as error:raise ValueError('invalid encoder manifest nesting') from error
    if type(value) is not dict:raise ValueError('invalid encoder manifest schema')
    if value.get('schema')!=1 or not isinstance(value.get('model_id'),str) or not isinstance(value.get('revision'),str):raise ValueError('invalid encoder manifest')
    if type(value.get('max_tokens')) is not int or not 16<=value['max_tokens']<=512:raise ValueError('invalid encoder token bound')
    files=value.get('files')
    if type(files) is not dict or not 2<=len(files)<=16:raise ValueError('invalid encoder file inventory')
    paths={}
    for key in (value.get('model_file'),value.get('tokenizer_file')):
        if not isinstance(key,str) or Path(key).is_absolute() or '..' in Path(key).parts or key not in files:raise ValueError('unsafe encoder path')
        path=root/key
        if any(parent.is_symlink() for parent in [path,*path.parents] if parent!=root.parent):raise ValueError('encoder paths cannot be symlinks')
        paths[key]=path
        descriptor=files[key]
        if not isinstance(descriptor,dict) or type(descriptor.get('bytes')) is not int or not isinstance(descriptor.get('sha256'),str):raise ValueError('invalid encoder digest descriptor')
        if path.stat().st_size!=descriptor['bytes']:raise ValueError('encoder size mismatch')
    signature=(_signature(root/'MANIFEST.json'),)+tuple(_signature(paths[key]) for key in sorted(paths))
    return value,hashlib.sha256(raw).hexdigest(),paths,signature


def _runtime():
    global _RUNTIME
    value,digest,paths,signature=manifest()
    if _RUNTIME is not None and _RUNTIME[0]==signature:return _RUNTIME[1:]
    # Optional packages are imported only when this explicitly selected backend
    # runs, keeping the ordinary assistant and CI standard-library compatible.
    try:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as error:raise ValueError('optional local encoder dependencies are unavailable') from error
    for key,path in paths.items():
        if _sha(path)!=value['files'][key]['sha256']:raise ValueError('encoder digest mismatch')
    tokenizer=Tokenizer.from_file(str(paths[value['tokenizer_file']]))
    tokenizer.no_padding();tokenizer.no_truncation()
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(paths[value['model_file']]),sess_options=options,providers=['CPUExecutionProvider'])
    if not {entry.name for entry in session.get_inputs()} <= {'input_ids','attention_mask','token_type_ids'}:raise ValueError('unexpected encoder graph inputs')
    _RUNTIME=(signature,value,digest,tokenizer,session,np)
    return _RUNTIME[1:]


def encode(texts):
    if type(texts) is not list or not 1<=len(texts)<=768 or any(type(t) is not str or len(t)>512 for t in texts):raise ValueError('invalid bounded encoder texts')
    value,digest,tokenizer,session,np=_runtime();batches=[]
    for start in range(0,len(texts),32):
        items=tokenizer.encode_batch(texts[start:start+32])
        if any(not item.ids or len(item.ids)>value['max_tokens'] for item in items):raise ValueError('encoder token bound exceeded; no silent truncation')
        width=max(len(item.ids) for item in items)
        ids=np.zeros((len(items),width),dtype=np.int64);mask=np.zeros_like(ids);types=np.zeros_like(ids)
        for row,item in enumerate(items):
            ids[row,:len(item.ids)]=item.ids;mask[row,:len(item.ids)]=1;types[row,:len(item.ids)]=item.type_ids
        data={'input_ids':ids,'attention_mask':mask,'token_type_ids':types}
        outputs=session.run(None,{entry.name:data[entry.name] for entry in session.get_inputs()})
        states=np.asarray(outputs[0],dtype=np.float64)
        if states.ndim!=3 or states.shape[:2]!=ids.shape:raise ValueError('unexpected encoder output shape')
        pooled=(states*mask[:,:,None]).sum(axis=1)/mask.sum(axis=1,keepdims=True)
        norms=np.linalg.norm(pooled,axis=1,keepdims=True)
        if not np.isfinite(pooled).all() or (norms<=0).any():raise ValueError('invalid encoder features')
        batches.append(pooled/norms)
    return np.concatenate(batches,axis=0),digest
