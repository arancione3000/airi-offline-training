"""Read-only inventory of archived TRAIN coverage; no model or evaluation forward."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def describe(rows):
    serialized=[json.dumps(r['messages'],sort_keys=True,ensure_ascii=False) for r in rows]
    return dict(rows=len(rows),unique_message_sequences=len(set(serialized)),
                languages=dict(Counter(r.get('language','unspecified') for r in rows)),
                families=dict(Counter(r.get('family','unspecified') for r in rows)),
                multiturn=sum(len(r['messages'])>2 for r in rows),
                assistant_utf8_bytes=sum(len(r['messages'][-1]['content'].encode()) for r in rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--previous-log',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sources={}
    def read(path):
        raw=path.read_bytes();sources[str(path)]=hashlib.sha256(raw).hexdigest();return raw
    authored=json.loads(read(a.archive/'AUTHORED_CURRICULUM.json'))['train']
    human=[json.loads(line) for line in read(a.archive/'human-train.jsonl').splitlines()]
    old=[json.loads(line) for line in read(a.archive/'train-1024.jsonl').splitlines()]
    meta=json.loads(read(a.metadata))
    prior=[json.loads(line) for line in read(a.previous_log).splitlines()]
    observations=[r for r in prior if r.get('phase')=='data ready']
    if len(observations)!=1:raise ValueError('expected one archived runtime-pool observation')
    observation=observations[0]
    report=dict(scope='available archived TRAIN corpus, not a reconstruction of unique lifetime pretraining tokens',
        sources=sources,authored_train=describe(authored),raw_reviewed_human_train=describe(human),
        legacy_train_archive=describe(old),previous_runtime_pool_observation=observation,
        model_parameters=meta['parameters'],trainable_parameters=meta['trainable_parameters'],
        trainable_fraction=meta['trainable_parameters']/meta['parameters'],
        metadata_attempted_supervised_tokens=meta['attempted_supervised_tokens'],
        unique_lifetime_pretraining_tokens=None,
        limitations=['Byte counts are not token counts; overlapping pools must not be summed as independent data.',
          'The previous runtime observation is attributed to the archived Nov12 run, not recomputed by this inventory.',
          '18 Italian human conversations describe one acquisition pool, not every Italian example seen by the model.',
          'Attempted supervised tokens include repetition; they do not certify pretraining adequacy or new skills.'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),authored=report['authored_train']['rows'],
        raw_human=report['raw_reviewed_human_train']['languages'],runtime=observation.get('human_counts'))))


if __name__=='__main__':main()
