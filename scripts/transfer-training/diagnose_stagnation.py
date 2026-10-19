"""Offline bounded probe; never publishes a model or feeds evaluation targets to training."""
import sys,json,copy,random,time,argparse,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
p=argparse.ArgumentParser(description='Diagnostic stress test, not production evaluation; champion anchors replace original native anchors.')
p.add_argument('--checkpoint',required=True,type=Path)
p.add_argument('--baseline-result',required=True,type=Path)
p.add_argument('--output',required=True,type=Path)
p.add_argument('--steps',type=int,default=256)
p.add_argument('--learning-rate',type=float,default=5e-7)
args=p.parse_args()
if not 64<=args.steps<=512 or args.steps%64:raise ValueError('bounded complete diagnostic intervals required')
if not math.isfinite(args.learning_rate) or not 0<args.learning_rate<=1e-4:
 raise ValueError('invalid diagnostic learning rate')
import torch
from generalist_lm.runtime import GeneralistRuntime
from generalist_lm.training import SFTExample,encode_sft_example
from continual_learning import continual_step,cache_tokenizer,retention_rows,verified_memories
from curriculum import build,norm,exact_correct
from extra_dialogues import build_extra
from success_bank import catalog
root=Path(__file__).resolve().parents[2]
torch.set_num_threads(2);torch.manual_seed(42);rng=random.Random(42)
r=GeneralistRuntime.from_checkpoint(args.checkpoint);cache_tokenizer(r.tokenizer)
champ=copy.deepcopy(r.model).eval()
for p in champ.parameters():p.requires_grad_(False)
def encode(row):
 ids,labels=encode_sft_example(SFTExample.from_dict(row),r.tokenizer,1024,require_complete=True)
 width=int((labels!=-100).nonzero()[-1])+1
 return ids[:width].unsqueeze(0),labels[:width].unsqueeze(0)
old=json.loads(args.baseline_result.read_text())['counters']['champion']
probes=old['samples'];blocked={norm(m['content']) for s in probes for m in s['messages']}|{norm(s['expected']) for s in probes}
human=[json.loads(s) for s in (root/'data/oasst2/human-train.jsonl').read_text().splitlines()]
human=[x for x in human if not any(norm(m['content']) in blocked for m in x['messages'])]
train,held=build();extra,dev=build_extra(old_validation=held)
rows=retention_rows(list(catalog(train+extra,held+dev).values()))
from types import SimpleNamespace
mem=verified_memories(SimpleNamespace(model=champ),rows,encode)
mems={l:[encode(x) for x in mem if x['language']==l] for l in ['it','en']}
# Independent human development references are never trained.
validation=[json.loads(s) for s in (root/'data/oasst2/human-development.jsonl').read_text().splitlines()][:16]
v=[encode(x) for x in validation]
def evaluate(step):
 r.model.eval()
 with torch.no_grad():
  loss=count=0
  for ids,labels in v:
   logits=r.model(ids)['logits'][:,:-1];loss+=float(torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),reduction='sum'));count+=int((labels[:,1:]!=-100).sum())
  answers=[]
  for s in probes:
   if not s['correct']:continue
   response=r.chat(s['messages'],temperature=0,max_new_tokens=64)
   answers.append({'expected':s['expected'],'before':s['response'],'response':response,'correct':exact_correct(response,s['expected'])})
 result={'step':step,'human_nll':loss/count,'retained':sum(x['correct'] for x in answers),'original_correct':len(answers),'lost':[x for x in answers if not x['correct']]}
 print(json.dumps(result),flush=True);return result
results=[evaluate(0)];opt=torch.optim.AdamW(r.model.parameters(),lr=args.learning_rate,weight_decay=.01)
for step in range(1,args.steps+1):
 row=rng.choice(human);memories=[rng.choice(mems[l]) for l in ['it','en']]
 # Isolate continuous ordinary learning: champion anchors, no archive or rollback.
 stats=continual_step(r.model,champ,champ,opt,encode(row),memories,memories[0],retention_weight=16,champion_kl_weight=16)
 if step%64==0:results.append(evaluate(step))
args.output.write_text(json.dumps({'scope':'isolated human training, champion native anchor, no archive; diagnostic only','results':results},indent=2))
