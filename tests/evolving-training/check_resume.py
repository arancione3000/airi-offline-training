"""Run after the native runtime installation: real model/Adam recovery equivalence."""
import random,tempfile,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/evolving-training'))
import torch
from generalist_lm.runtime import GeneralistRuntime
from generalist_lm.model import CausalTransformerLM,GeneralistLMConfig
from session import train_step,save_session,restore_session

torch.set_num_threads(2);torch.manual_seed(12)
c=GeneralistLMConfig(d_model=32,n_heads=2,n_layers=1,d_ff=64,context_length=32)
r=GeneralistRuntime(CausalTransformerLM(c),c);ref=GeneralistRuntime(CausalTransformerLM(c),c)
for p in ref.model.parameters():p.requires_grad_(False)
f=lambda model:torch.optim.AdamW(model.parameters(),lr=3e-6)
o=f(r.model);ids=torch.tensor([[2,12,13,14,15,1]]);labels=ids.clone();labels[:,:2]=-100
train_step(r.model,ref.model,o,ids,labels,ids,labels)
with tempfile.TemporaryDirectory() as tmp:
 path=Path(tmp)/'champion';rng=random.Random(42);ar=random.Random(43)
 save_session(r,o,rng,ar,dict(completed_steps=27444),path,fingerprint='recovery-check')
 restored,oo,payload=restore_session(path,f,fingerprint='recovery-check')
 train_step(r.model,ref.model,o,ids,labels,ids,labels)
 train_step(restored.model,ref.model,oo,ids,labels,ids,labels)
 assert all(torch.equal(a,b) for a,b in zip(r.model.parameters(),restored.model.parameters()))
 assert payload['sampler_rng']==rng.getstate()
 print('Native champion + Adam recovery: next update bitwise identical; sampler RNG retained.')

from context_curriculum import build_context
from generalist_lm.training import SFTExample,encode_sft_example
from generalist_lm.tokenizer import ByteTokenizer
from curriculum import build
from extra_dialogues import build_extra
old,heldout=build();extra,more=build_extra((),heldout);context,tests=build_context()
tok=ByteTokenizer()
for row in context+tests+old+extra+heldout+more:
 ids,labels=encode_sft_example(SFTExample.from_dict(row),tok,1024,require_complete=True)
 prompt=tok.serialize_messages(row['messages'][:-1],add_generation_prompt=True)
 target=tok.encode(row['messages'][-1]['content'],eos=True)
 assert ids[:len(prompt)].tolist()==prompt
 assert labels[:len(prompt)].tolist()==[-100]*len(prompt)
 assert labels[len(prompt):len(prompt)+len(target)].tolist()==target
print('All curriculum histories and answers fit full 1024 context; chat/SFT prompt and masked targets agree.')
