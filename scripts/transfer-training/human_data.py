"""Verified human conversations, tree-disjoint splits, no model-generated labels."""
import gzip
import hashlib
import json
from pathlib import Path
from curriculum import norm

REVISION='179dd21fc55192153d94adb0e0ce8f69e222bf75'
SOURCE_SHA256='a9f240c4c77aa1378364f70d37e753c07ba284e247b019d700e1947a0e5da751'

def quality_message(row):
    if (row.get('deleted') or row.get('synthetic') or row.get('model_name') or
        row.get('review_result') is not True or row.get('lang') not in ('it','en')):return False
    text=row.get('text','').strip()
    if not 2<=len(text)<=900 or '\x00' in text:return False
    labels=row.get('labels') or {}
    for name in ('spam','lang_mismatch','pii','not_appropriate','fails_task'):
        label=labels.get(name) or {}
        if label.get('value',0)>=.5:return False
    if row.get('role')=='assistant':
        label=labels.get('quality') or {}
        if row.get('rank')!=0 or label.get('count',0)<1 or label.get('value',0)<.6:return False
        if not 12<=len(text.encode('utf-8'))<=480:return False
    return row.get('role') in ('prompter','assistant')

def context_size(messages):
    # ByteTokenizer's exact BOS + role/text/SEP + generation role + target/EOS.
    return 1+sum(len(m['content'].encode('utf-8'))+2 for m in messages[:-1])+1+len(messages[-1]['content'].encode('utf-8'))+1

def prepare(raw):
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA256:raise ValueError('human source digest mismatch')
    source=[json.loads(line) for line in gzip.decompress(raw).splitlines()]
    rows={r['message_id']:r for r in source if quality_message(r)}
    candidates=[]
    for identity,row in sorted(rows.items()):
        if row['role']!='assistant':continue
        chain=[];cursor=row;seen=set()
        while cursor is not None:
            mid=cursor['message_id']
            if mid in seen or len(chain)>=6:chain=[];break
            seen.add(mid);chain.append(cursor)
            parent=cursor.get('parent_id')
            if parent is None:break
            cursor=rows.get(parent)
            if cursor is None:chain=[];break
        chain.reverse()
        if len(chain)<2 or any(x['role']!=('prompter' if i%2==0 else 'assistant') or x['lang']!=row['lang'] for i,x in enumerate(chain)):continue
        messages=[{'role':'user' if x['role']=='prompter' else 'assistant','content':x['text'].strip()} for x in chain]
        if context_size(messages)>1024:continue
        candidates.append({'messages':messages,'expected':messages[-1]['content'],'language':row['lang'],'family':'human-oasst2','provenance':{'dataset':'OpenAssistant/oasst2','revision':REVISION,'license':'Apache-2.0','message_id':identity,'tree_id':chain[0]['message_id'],'synthetic':False}})
    # Join trees with duplicated normalized user text before assigning splits.
    parent={r['provenance']['tree_id']:r['provenance']['tree_id'] for r in candidates}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    prompts={}
    for r in candidates:
        tree=r['provenance']['tree_id']
        for m in r['messages']:
            if m['role']!='user':continue
            key=norm(m['content'])
            if key in prompts:
                a,b=find(tree),find(prompts[key]);parent[max(a,b)]=min(a,b)
            else:prompts[key]=tree
    splits={'train':[],'development':[],'final':[]};seen=set();group_splits={}
    for lang in ('it','en'):
        groups=sorted({find(r['provenance']['tree_id']) for r in candidates if r['language']==lang},key=lambda g:hashlib.sha256(('airi-oasst2-v1:'+g).encode()).hexdigest())
        count=max(2,len(groups)//20)
        for i,g in enumerate(groups):group_splits[g]='development' if i<count else 'final' if i<2*count else 'train'
    for r in candidates:
        key=json.dumps(r['messages'],sort_keys=True,ensure_ascii=False)
        if key in seen:continue
        seen.add(key);group=find(r['provenance']['tree_id'])
        split=group_splits[group]
        splits[split].append(r)
    # Exact validation text in any role is never fed into training.
    excluded={norm(m['content']) for split in ('development','final') for r in splits[split] for m in r['messages']}
    splits['train']=[r for r in splits['train'] if not any(norm(m['content']) in excluded for m in r['messages'])]
    return splits

def load_corpus(directory,pins,blocked=(),*,training_blocked=None):
    directory=Path(directory);out={};blocked=set(blocked)
    for split in ('train','development','final'):
        raw=(directory/('human-'+split+'.jsonl')).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=pins[split]:raise ValueError('human corpus digest mismatch: '+split)
        out[split]=[json.loads(s) for s in raw.splitlines()]
    # Exclude previous human training/validation prompts at runtime.
    for split in out:
        excluded=blocked if split!='train' or training_blocked is None else set(training_blocked)
        out[split]=[r for r in out[split] if not any(norm(m['content']) in excluded for m in r['messages'])]
    # Even a separately filtered training replacement cannot reuse a remaining
    # development/final message. This never relaxes held-out separation.
    if training_blocked is not None:
        heldout={norm(m['content']) for split in ('development','final') for r in out[split] for m in r['messages']}
        out['train']=[r for r in out['train'] if not any(norm(m['content']) in heldout for m in r['messages'])]
    if not out['train'] or not out['development']:raise ValueError('empty human corpus after overlap removal')
    return out
