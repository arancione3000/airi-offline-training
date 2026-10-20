"""Real native transformer: salvage skill B from a donor that forgot skill A."""
import copy
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/transfer-training'))
import torch
from generalist_lm.model import CausalTransformerLM,GeneralistLMConfig
from generalist_lm.runtime import GeneralistRuntime
from generalist_lm.training import SFTExample,encode_sft_example
from session import train_step,save_session,restore_champion
from success_bank import catalog,empty_bank,validate_bank,row_key,summary
from success_transfer import harvest,distill_step
from curriculum import exact_correct

torch.set_num_threads(2);torch.manual_seed(40)
config=GeneralistLMConfig(d_model=32,n_heads=2,n_layers=1,d_ff=64,context_length=128)
base=GeneralistRuntime(CausalTransformerLM(config),config)
rows=[dict(messages=[dict(role='user',content=prompt),dict(role='assistant',content=answer)],
           expected=answer,language='en',family=family,provenance='deterministic exact solver')
      for prompt,answer,family in [('Write only: a','a','copy-a'),('Write only: b','b','copy-b')]]
eligible=catalog(rows,[])
def encode(row):
    ids,labels=encode_sft_example(SFTExample.from_dict(row),base.tokenizer,128,require_complete=True)
    width=int((labels!=-100).nonzero()[-1])+1
    return ids[:width].unsqueeze(0),labels[:width].unsqueeze(0)
def correct(runtime,row):
    return exact_correct(runtime.chat(row['messages'][:-1],temperature=0,max_new_tokens=6),row['expected'])
def freeze(runtime):
    runtime.model.eval()
    for p in runtime.model.parameters():p.requires_grad_(False)
    return runtime
def digest(model):
    return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()

def clone(runtime):
    result=GeneralistRuntime(CausalTransformerLM(config),config)
    result.model.load_state_dict(runtime.model.state_dict())
    return result

initial=freeze(clone(base));optimizer=torch.optim.AdamW(base.model.parameters(),lr=.003)
for _ in range(100):train_step(base.model,initial.model,optimizer,*encode(rows[0]),*encode(rows[0]),kl_weight=0.)
assert correct(base,rows[0]) and not correct(base,rows[1]),'base must have A but not B'
champion=freeze(clone(base));champion_digest=digest(champion.model)
donor=clone(base);optimizer=torch.optim.AdamW(donor.model.parameters(),lr=.003)
for _ in range(100):train_step(donor.model,champion.model,optimizer,*encode(rows[1]),*encode(rows[1]),kl_weight=0.)
assert correct(donor,rows[1]) and not correct(donor,rows[0]),'donor must improve B but lose A'
bank=empty_bank('native-demo')
harvest(donor,champion,rows,encode,bank,eligible,
        dict(step=200,fingerprint='native-demo',model_sha256=digest(donor.model)),generation_limit=2)
validate_bank(bank,eligible,'native-demo',config.vocab_size)
assert bank['entries'][row_key(rows[1])]['whole_answer_gain']
assert all(p['target'] in range(config.vocab_size) for e in bank['entries'].values() for p in e['positions'])
student=clone(base);optimizer=torch.optim.AdamW(student.model.parameters(),lr=.001)
lesson=bank['entries'][row_key(rows[1])]
for _ in range(160):
    stats=distill_step(student.model,champion.model,optimizer,lesson,encode(rows[1]),encode(rows[0]),lr_multiplier=1.,lr_cap=.001)
    bank['transfer_updates']+=1
assert correct(student,rows[0]) and correct(student,rows[1]),'selective student must retain A and acquire B'
assert digest(champion.model)==champion_digest and all(p.grad is None for p in champion.model.parameters())

with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'champion';rng=random.Random(4);ar=random.Random(5)
    save_session(student,optimizer,rng,ar,{'completed_steps':360},path,fingerprint='native-demo',success_bank=bank)
    restored,oo,_=restore_champion(path,lambda m:torch.optim.AdamW(m.parameters(),lr=.001),rng,ar,fingerprint='native-demo')
    saved=json.loads((path/'success_bank.json').read_text())
    validate_bank(saved,eligible,'native-demo',config.vocab_size)
    assert saved==bank and correct(restored,rows[0]) and correct(restored,rows[1])
    assert 'success_bank.json' in json.loads((path/'checkpoint.json').read_text())['files']
    (path/'success_bank.json').write_text('{}')
    try:restore_champion(path,lambda m:torch.optim.AdamW(m.parameters(),lr=.001),rng,ar,fingerprint='native-demo')
    except ValueError:pass
    else:raise AssertionError('corrupted lesson archive bypassed checkpoint integrity')

# Catch causal off-by-one errors before an optimizer can update any parameter.
bad=copy.deepcopy(lesson);bad['positions'][0]['position']=0
before=digest(student.model)
try:distill_step(student.model,champion.model,optimizer,bad,encode(rows[1]),encode(rows[0]))
except ValueError:pass
else:raise AssertionError('prompt/invalid teacher position accepted')
assert digest(student.model)==before
print(json.dumps({'native_complementary_skill_demo':'PASS','base':['A'],'discarded_donor':['B'],
                  'consolidated_student':['A','B'],'bank':summary(bank),
                  'scope':'two trained toy tasks in the native transformer; not proof of general chat'}))

# Confirm the real weighted archive objective uses BOTH retention histories,
# including their shifted target masks, in one atomic optimizer step.
from torch.nn import functional as F
from generalist_lm.training import causal_training_objective,reference_kl_loss
weighted=clone(student);weighted_opt=torch.optim.AdamW(weighted.model.parameters(),lr=0.)
batch=encode(rows[1]);memories=[encode(r) for r in rows]
with torch.no_grad():
    positions=[p['position'] for p in lesson['positions']]
    logits=weighted.model(batch[0])['logits'][0,torch.tensor(positions)-1].float()
    teacher=torch.tensor([p['probabilities'] for p in lesson['positions']],dtype=logits.dtype)
    teacher=teacher/teacher.sum(-1,keepdim=True)
    expected=F.cross_entropy(logits,torch.tensor([p['target'] for p in lesson['positions']]))
    expected+=.5*F.kl_div(logits.log_softmax(-1),teacher,reduction='batchmean')
    for ids,labels in memories:
        logits=weighted.model(ids)['logits'];gold,_=causal_training_objective(logits,labels,ids,
            repetition_unlikelihood_weight=.05,eos_loss_weight=1.2)
        kl,_=reference_kl_loss(logits,champion.model(ids)['logits'],labels)
        expected+=(16*gold+16*kl)/2
    expected/=17
stats=distill_step(weighted.model,champion.model,weighted_opt,lesson,batch,memories[0],
                   memories=memories,retention_weight=16,champion_kl_weight=16,lr_multiplier=1.,lr_cap=.001)
assert abs(stats['loss']-float(expected))<1e-5,(stats['loss'],float(expected))
assert stats['retention_examples']==2 and stats['retention_weight']==16
assert stats['supervised_tokens']==len(positions)+sum(int((labels[:,1:]!=-100).sum()) for _,labels in memories)
assert all(int(v['step'])==1 for v in weighted_opt.state.values())
before=digest(weighted.model);moments=copy.deepcopy(weighted_opt.state_dict())
for bad_weight in [float('nan'),-1,0,True]:
    try:distill_step(weighted.model,champion.model,weighted_opt,lesson,batch,memories[0],retention_weight=bad_weight)
    except ValueError:pass
    else:raise AssertionError('invalid transfer retention accepted')
    assert digest(weighted.model)==before
for key,values in moments['state'].items():
    for name,value in values.items():assert torch.equal(value,weighted_opt.state_dict()['state'][key][name])
print('Weighted bilingual archive objective matches causal reference; one Adam update; invalid weights preserve state.')

# Exercise the exact automatic per-skill transaction used by the real runner.
from morph_model import new_state,adopt,consolidate
morph=clone(base);morph_opt=torch.optim.AdamW(morph.model.parameters(),lr=.002)
state=new_state(digest(morph.model));spent=[]
initial_model=copy.deepcopy(morph.model.state_dict());initial_opt=copy.deepcopy(morph_opt.state_dict())
def learn(entry):
    distill_step(morph.model,champion.model,morph_opt,entry,encode(entry['row']),encode(rows[0]),lr_multiplier=1.,lr_cap=.002)
    spent.append(1)
def restore_initial():
    morph.model.load_state_dict(initial_model);morph_opt.load_state_dict(initial_opt)
def assess():return {'A':correct(morph,rows[0]),'B':correct(morph,rows[1])}
identity=id(morph.model)
result=consolidate(state,bank,{'A':True,'B':False},train=learn,evaluate=assess,restore=restore_initial,
                   can_train=lambda:True,can_adopt=lambda before,after:after['A'] and after['B'],
                   step=lambda:len(spent),max_updates=64)
assert result['action']=='skill_accepted',(result,assess())
adopt(state,digest(morph.model),step=len(spent),reason='native verified skill B')
assert id(morph.model)==identity and state['model_id']=='airi-native' and state['revision']==1
accepted_model=copy.deepcopy(morph.model.state_dict());accepted_opt=copy.deepcopy(morph_opt.state_dict())
def restore_accepted():
    morph.model.load_state_dict(accepted_model);morph_opt.load_state_dict(accepted_opt)
# An update with no new verified capability is rejected; A+B remain intact.
before_digest=digest(morph.model)
result=consolidate(state,bank,assess(),train=learn,evaluate=assess,restore=restore_accepted,
                   can_train=lambda:True,can_adopt=lambda before,after:any(not before[k] and after[k] for k in before),
                   step=lambda:len(spent),max_updates=4)
assert result['action']=='skill_rejected' and digest(morph.model)==before_digest and assess()=={'A':True,'B':True}
actual=morph_opt.state_dict()
for key,values in accepted_opt['state'].items():
    for name,value in values.items():
        assert torch.equal(actual['state'][key][name],value) if torch.is_tensor(value) else actual['state'][key][name]==value
assert len(spent)==68 and state['attempted_updates']==68
print(json.dumps({'native_single_model_transaction':'PASS','identity':state['model_id'],'revision':state['revision'],
                  'verified_capabilities':['A','B'],'rejected_attempt_restores_weights_and_adam':True,
                  'all_attempted_updates_counted':len(spent)}))
