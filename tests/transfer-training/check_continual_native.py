"""Native optimizer atomicity and interpolation failures, with real tensors."""
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/transfer-training'))
import torch
from generalist_lm.model import CausalTransformerLM,GeneralistLMConfig
from generalist_lm.runtime import GeneralistRuntime
from generalist_lm.training import SFTExample,encode_sft_example
from continual_learning import continual_step,try_interpolations,cache_tokenizer,verified_memories

torch.set_num_threads(2);torch.manual_seed(42)
config=GeneralistLMConfig(d_model=32,n_heads=2,n_layers=1,d_ff=64,context_length=128)
student=GeneralistRuntime(CausalTransformerLM(config),config)
raw=student.tokenizer.encode('ciao',bos=True,eos=True)
cache_tokenizer(student.tokenizer)
assert student.tokenizer.encode('ciao',bos=True,eos=True)==raw
from generalist_lm.bpe_tokenizer import BPETokenizer
from generalist_lm.tokenizer import BYTE_OFFSET
bpe=BPETokenizer(((BYTE_OFFSET+99,BYTE_OFFSET+105),))
bpe_raw=bpe.encode('ciao ciao',bos=True,eos=True);cache_tokenizer(bpe)
assert bpe.encode('ciao ciao',bos=True,eos=True)==bpe_raw
bpe.encode('ciao ciao',bos=True,eos=True).clear()
assert bpe.encode('ciao ciao',bos=True,eos=True)==bpe_raw
student.tokenizer.encode('ciao',bos=True,eos=True).clear()
assert student.tokenizer.encode('ciao',bos=True,eos=True)==raw
def frozen():
    result=GeneralistRuntime(CausalTransformerLM(config),config)
    result.model.load_state_dict(student.model.state_dict())
    for p in result.model.parameters():p.requires_grad_(False)
    return result
champion=frozen();reference=frozen()
def encode(prompt,answer):
    row=SFTExample([dict(role='user',content=prompt),dict(role='assistant',content=answer)])
    return tuple(v.unsqueeze(0) for v in encode_sft_example(row,student.tokenizer,128,require_complete=True))
a=encode('Write A','a');b=encode('Write B','b')
optimizer=torch.optim.AdamW(student.model.parameters(),lr=.002)
protected=copy.deepcopy(champion.model.state_dict())
stats=continual_step(student.model,champion.model,reference.model,optimizer,b,[a],a)
assert stats['supervised_tokens']>0 and all(int(v['step'])==1 for v in optimizer.state.values())
assert all(torch.equal(protected[k],v) for k,v in champion.model.state_dict().items())
saved=copy.deepcopy(student.model.state_dict());moments=copy.deepcopy(optimizer.state_dict())
result=try_interpolations(student.model,champion.model,evaluate=lambda:None,acceptable=lambda a:False)
assert result['alpha'] is None and result['probes']==2
assert all(torch.equal(saved[k],v) for k,v in student.model.state_dict().items())
def explode():raise RuntimeError('failed independent evaluation')
try:try_interpolations(student.model,champion.model,evaluate=explode,acceptable=lambda a:True)
except RuntimeError:pass
else:raise AssertionError('evaluation failure swallowed')
assert all(torch.equal(saved[k],v) for k,v in student.model.state_dict().items())
result=try_interpolations(student.model,champion.model,evaluate=lambda:None,acceptable=lambda a:True,alphas=(.5,))
assert result['alpha']==.5
assert all(torch.allclose(v,torch.lerp(protected[k],saved[k],.5)) for k,v in student.model.state_dict().items())
# A non-finite reference can fail after task.backward(), but before optimizer.step().
saved=copy.deepcopy(student.model.state_dict())
next(reference.model.parameters()).data.fill_(float('nan'))
try:continual_step(student.model,champion.model,reference.model,optimizer,b,[a],a)
except RuntimeError:pass
else:raise AssertionError('non-finite retention update accepted')
assert all(torch.equal(saved[k],v) for k,v in student.model.state_dict().items())
for key,values in moments['state'].items():
    for name,value in values.items():
        assert torch.equal(optimizer.state_dict()['state'][key][name],value)
print(json.dumps({'native_continual_atomicity':'PASS','single_optimizer_update':True,
                  'failed_merge_restores_weights':True,'evaluation_exception_restores_weights':True,
                  'nonfinite_retention_preserves_weights_and_adam':True}))

# A correct prefix alone must never certify a whole answer for retention.
from types import SimpleNamespace
class CausalMemory(torch.nn.Module):
    def forward(self,ids):
        logits=torch.zeros((1,3,264));logits[0,0,12]=10
        logits[0,1,13 if int(ids[0,0])!=9 else 14]=10
        return {'logits':logits}
rows=[dict(language='it',key=1),dict(language='en',key=2),dict(language='en',key=9)]
def encoded_memory(row):return torch.tensor([[row['key'],12,13]]),torch.tensor([[-100,12,13]])
selected=verified_memories(SimpleNamespace(model=CausalMemory()),rows,encoded_memory)
assert [r['key'] for r in selected]==[1,2]
print('Full-answer training-memory certificate rejects partial teacher-forced hits.')
