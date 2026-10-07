"""Native language expansion, gated pilot then finite autonomous training."""
import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from generalist_lm.runtime import GeneralistRuntime
from generalist_lm.research_cycle import _research_eligible
from generalist_lm.phase5_diagnostics import degeneration_gate
from session import atomic_json, sha256_file, train_step, save_session, restore_session
from persist import publish
from corpus import prepare as prepare_corpus, verify as verify_corpus
from policy import START, PILOT, TARGET, DeckSampler, group_for, pilot_decision
from legacy_run import prepare, common_metrics, state_digest, EXPECTED_PARENT

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'recovered/evidence/dialogue-anchors'
PARENT = ROOT/'context-trial/generalist-state/bootstrap-data/candidate'
OUT = ROOT/'long-training/results'
SPEC = json.loads((Path(__file__).parent/'SPEC.json').read_text())
FINGERPRINT = SPEC['fingerprint']


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=2e-6, weight_decay=.01)


def encode_text(row, tokenizer):
    ids = tokenizer.encode(row['text'], bos=True, eos=True)
    if not 8<=len(ids)<=512:
        raise ValueError('raw text exceeds authorized complete example budget')
    ids = torch.tensor([ids], dtype=torch.long)
    labels = ids.clone()
    labels[:,0] = -100
    return ids, labels


def corpus_nll(runtime, rows):
    runtime.model.eval()
    result = {}
    with torch.no_grad():
        for lang in ['it','en']:
            loss, tokens = 0., 0
            for row in rows:
                if row['language']!=lang:
                    continue
                ids, labels = encode_text(row, runtime.tokenizer)
                logits = runtime.model(ids)['logits'][:,:-1,:]
                loss += float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]), labels[:,1:].reshape(-1), reduction='sum'))
                tokens += labels.shape[1]-1
            if tokens==0:
                raise ValueError('empty heldout language')
            result[lang] = loss/tokens
    return result


def evaluate(runtime, heldout):
    return {**common_metrics(runtime), 'corpus': corpus_nll(runtime, heldout)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=Path, required=True)
    parser.add_argument('--max-steps', type=int, default=TARGET)
    args = parser.parse_args()
    if args.max_steps!=TARGET:
        raise ValueError('finite target mismatch')
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if sha256_file(PARENT/'model.pt')!=EXPECTED_PARENT:
        raise ValueError('native parent changed')
    reference = GeneralistRuntime.from_checkpoint(PARENT)
    teacher_digest = state_digest(reference.model)
    for p in reference.model.parameters():
        p.requires_grad_(False)
    dialogue, replay, anchors, deck, old_rows = prepare(reference.tokenizer)
    runtime, opt, payload = restore_session(args.resume, optimizer, fingerprint=FINGERPRINT)
    counters = payload['counters']
    rng, anchor_rng = random.Random(), random.Random()
    rng.setstate(payload['sampler_rng'])
    anchor_rng.setstate(payload['anchor_rng'])
    if not START<=counters['completed_steps']<=TARGET:
        raise ValueError('expanded recipe checkpoint range mismatch')
    if not (DATA/'CORPUS_MANIFEST.json').exists():
        if counters['completed_steps']!=START or 'expanded_corpus_sha256' in counters:
            raise ValueError('pinned corpus lost; refuse to download different data')
        prepare_corpus(DATA, runtime.tokenizer, DATA/'data', PARENT.parent)
    manifest = verify_corpus(DATA, runtime.tokenizer.digest)
    corpus_sha = sha256_file(DATA/'CORPUS_MANIFEST.json')
    if counters.get('expanded_corpus_sha256', corpus_sha)!=corpus_sha:
        raise ValueError('checkpoint/corpus identity mismatch')
    counters['expanded_corpus_sha256'] = corpus_sha
    train = [json.loads(x) for x in (DATA/'CORPUS_TRAIN.jsonl').read_text().splitlines()]
    heldout = [json.loads(x) for x in (DATA/'CORPUS_HELDOUT.jsonl').read_text().splitlines()]
    by_language = {lang:[i for i,r in enumerate(old_rows) if r['provenance']['language']==lang] for lang in ['it','en']}
    texts = {lang:[r for r in train if r['language']==lang] for lang in ['it','en']}
    if any(len(v)!=20000 for v in texts.values()) or len(heldout)!=256:
        raise ValueError('corpus size mismatch')
    if {r['sha256'] for r in train}&{r['sha256'] for r in heldout}:
        raise ValueError('corpus split leakage')
    baseline_file = DATA/'CORPUS_BASELINE.json'
    if not baseline_file.exists():
        if counters['completed_steps']!=START:
            raise ValueError('seed assessment missing')
        atomic_json(baseline_file, evaluate(runtime, heldout))
    baseline_sha = sha256_file(baseline_file)
    if counters.get('expanded_baseline_sha256', baseline_sha)!=baseline_sha:
        raise ValueError('seed assessment identity mismatch')
    counters['expanded_baseline_sha256'] = baseline_sha
    baseline = json.loads(baseline_file.read_text())
    counters.setdefault('expanded_elapsed_seconds',0.)
    counters.setdefault('expanded_updates',0)
    counters.setdefault('expanded_group_updates',{})
    counters.setdefault('expanded_cursors',{'it':0,'en':0})
    if counters['expanded_updates']!=counters['completed_steps']-START:
        raise ValueError('real update counter mismatch')
    if counters['completed_steps']>PILOT and not counters.get('expanded_pilot',{}).get('passed'):
        raise ValueError('long training cannot bypass the pilot gate')
    extras = [Path(__file__).parent/name for name in SPEC['script_sha256']]
    extras += [Path(__file__).parent/'SPEC.json']
    extras += list((DATA/'data').glob('*.jsonl'))+[DATA/'CONVERSATION_TEST.json']
    extras += [DATA/name for name in ['CORPUS_TRAIN.jsonl','CORPUS_HELDOUT.jsonl','CORPUS_MANIFEST.json','CORPUS_BASELINE.json']]
    atomic_json(OUT/'SPEC.json', {**SPEC, 'corpus_manifest_sha256':corpus_sha, 'corpus':manifest})
    def checkpoint(durable=False):
        path = OUT/f'checkpoint-{counters["completed_steps"]:05d}'
        if not path.exists():
            save_session(runtime,opt,rng,anchor_rng,counters,path,fingerprint=FINGERPRINT)
        if durable:
            publish(path,OUT/'archives',extra_files=extras)
        for old in sorted(OUT.glob('checkpoint-*'))[:-2]:
            shutil.rmtree(old)
        return path
    if counters['completed_steps']==START:
        checkpoint(durable=True)
    sampler = DeckSampler()
    while counters['completed_steps']<TARGET:
        if counters.get('expanded_pilot',{}).get('passed') is False:
            break
        step = counters['completed_steps']
        group = group_for(step)
        lang = group.rsplit('-',1)[1]
        if group.startswith('corpus-'):
            cursor = counters['expanded_cursors'][lang]
            index = sampler.index(lang,cursor,len(texts[lang]))
            batch = encode_text(texts[lang][index], runtime.tokenizer)
        else:
            batch = dialogue[rng.choice(by_language[lang])]
        anchor = anchors[anchor_rng.randrange(len(anchors))]
        tick = time.monotonic()
        stats = train_step(runtime.model,reference.model,opt,*batch,*anchor)
        elapsed = time.monotonic()-tick
        counters['elapsed_training_seconds']+=elapsed
        counters['expanded_elapsed_seconds']+=elapsed
        counters['completed_steps']+=1
        counters['expanded_updates']+=1
        counters['supervised_tokens']+=stats['supervised_tokens']
        counters['reference_tokens']+=stats['reference_tokens']
        counters['tokens_by_group'][group]=counters['tokens_by_group'].get(group,0)+stats['supervised_tokens']
        counters['expanded_group_updates'][group]=counters['expanded_group_updates'].get(group,0)+1
        if group.startswith('corpus-'):
            counters['expanded_cursors'][lang]+=1
        step = counters['completed_steps']
        if step==PILOT and 'expanded_pilot' not in counters:
            after = evaluate(runtime,heldout)
            retention = _research_eligible(baseline['domains'],after['domains'],minimum_loss_gain=.01,max_domain_regression=.03)
            degeneration = degeneration_gate(baseline['language'],after['language'],max_repetition_regression=0.,max_entropy_collapse_fraction=.70)
            counters['expanded_pilot'] = pilot_decision(baseline,after,retention,degeneration)
            atomic_json(OUT/'PILOT_RESULT.json', {'completed_steps':step, 'before':baseline,'after':after,**counters['expanded_pilot']})
            checkpoint(durable=True)
            from persist import upload
            upload([OUT/'PILOT_RESULT.json'])
        if step%16==0 or step in [PILOT,TARGET]:
            progress = {**counters,'status':'training' if counters.get('expanded_pilot',{}).get('passed') is not False else 'blocked: expanded corpus pilot failed',
                        'target_steps':TARGET,'last_update':stats,'live_promoted':False,
                        'eta_training_seconds':counters['expanded_elapsed_seconds']/counters['expanded_updates']*(TARGET-step)}
            atomic_json(OUT/'PROGRESS.json',progress)
            print(json.dumps(progress),flush=True)
            if step!=PILOT:
                checkpoint(durable=(step%1024==0))
    checkpoint(durable=True)
    after = evaluate(runtime,heldout)
    before = common_metrics(reference)
    retention = _research_eligible(before['domains'],after['domains'],minimum_loss_gain=.01,max_domain_regression=.03)
    degeneration = degeneration_gate(before['language'],after['language'],max_repetition_regression=.04,max_entropy_collapse_fraction=.70)
    dev_ok = all(after['development'][lang]['nll_per_byte']<=before['development'][lang]['nll_per_byte'] for lang in ['it','en'])
    lang_ok = after['language']['language_nll']<=before['language']['language_nll']
    prompts = json.loads((DATA/'CONVERSATION_TEST.json').read_text())['prompts']
    generations = [{'prompt':p,'response':runtime.chat([{'role':'user','content':p}],max_new_tokens=48)} for p in prompts]
    atomic_json(OUT/'NEW_GENERATIONS.json',generations)
    if state_digest(reference.model)!=teacher_digest or any(p.grad is not None for p in reference.model.parameters()):
        raise ValueError('frozen native reference changed')
    result = {'counters':counters,'status':'training completed' if counters['completed_steps']==TARGET else 'blocked: pilot failed; longer cycle not started',
              'before':before,'after':after,'seed_baseline':baseline,'corpus':manifest,
              'retention_gate':list(retention),'degeneration_gate':list(degeneration),
              'development_nonregression':dev_ok,'language_nonregression':lang_ok,
              'metric_gates_passed':bool(retention[0] and degeneration[0] and dev_ok and lang_ok),
              'conversation_assessment':'pending; no conversational success claimed',
              'production_qualified':False,'live_promoted':False,'reference_unchanged':True}
    atomic_json(OUT/'FINAL_RESULT.json',result)
    from persist import upload
    upload([OUT/'FINAL_RESULT.json',OUT/'NEW_GENERATIONS.json',OUT/'SPEC.json',DATA/'CORPUS_MANIFEST.json'])
    atomic_json(OUT/'PROGRESS.json',{**counters,'status':result['status'],'live_promoted':False})


if __name__=='__main__':
    main()
