import base64
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/transfer-training'))
from morph_model import new_state,inherit_state,adopt,consolidate,model_record,MODEL_ID
from model_channel import publish_current,transition,validate,BRANCH,PATH
from chat_current import validate_selected_model


def bank():
    return {'entries':{'b':{'row':{'messages':[],'expected':'b','language':'en','family':'skill-b'},
                          'positions':[{'gain':.3}],'source':{'model_sha256':'b'*64},'whole_answer_gain':True}}}


def record(state=None):
    return model_record(state or new_state('a'*64),release='airi-offline-1-2-1-1',fingerprint='f'*64,
                        step=1,model_files={'model/model.pt':'a'*64,'model/config.json':'c'*64})


class FakeChannel:
    def __init__(self):self.branch=False;self.file=None;self.calls=[]
    def api(self,path,payload=None,**options):
        self.calls.append((path,copy.deepcopy(payload),options))
        if path.startswith('git/ref/'):return {} if self.branch else None
        if path=='git/refs':self.branch=True;return {}
        if path.startswith('contents/') and payload is None:return copy.deepcopy(self.file)
        if path=='contents/'+PATH:
            expected=self.file['sha'] if self.file else None
            if payload.get('sha')!=expected:raise ValueError('concurrent index modification')
            self.file={'sha':'file-'+str(len(self.calls)),'content':payload['content']};return {}
        raise AssertionError(path)


class MorphTests(unittest.TestCase):
    def test_new_training_cycle_inherits_identity_revision_and_skills(self):
        state=new_state('a'*64);adopt(state,'b'*64,step=20,reason='verified skill')
        state['skills']['en:copy']={'attempts':3};state['attempted_updates']=40
        inherited=inherit_state(state,'b'*64)
        self.assertEqual(inherited,state);self.assertIsNot(inherited,state)
        self.assertEqual(inherited['revision'],1);self.assertEqual(inherited['attempted_updates'],40)
        self.assertRaises(ValueError,inherit_state,state,'c'*64)

    def test_failed_skill_restores_weights_and_adam_but_not_spent_budget(self):
        model={'A':True,'B':False};optimizer={'moment':5};spent=[];state=new_state('a'*64)
        old_model=copy.deepcopy(model);old_optimizer=copy.deepcopy(optimizer)
        def train(entry):model.update(A=False,B=True);optimizer['moment']+=1;spent.append(1)
        def restore():model.clear();model.update(old_model);optimizer.clear();optimizer.update(old_optimizer)
        result=consolidate(state,bank(),dict(model),train=train,evaluate=lambda:dict(model),restore=restore,
                           can_train=lambda:True,can_adopt=lambda a,b:b['A'] and b['B'],step=lambda:len(spent),max_updates=3)
        self.assertEqual(result['action'],'skill_rejected')
        self.assertEqual(model,old_model);self.assertEqual(optimizer,old_optimizer)
        self.assertEqual(len(spent),3);self.assertEqual(state['attempted_updates'],3)
        self.assertEqual(state['revision'],0)

    def test_success_updates_same_model_identity_and_later_failure_keeps_it(self):
        state=new_state('a'*64);model={'A':True,'B':False}
        result=consolidate(state,bank(),dict(model),train=lambda e:model.update(B=True),evaluate=lambda:dict(model),
                           restore=lambda:self.fail('successful update was discarded'),can_train=lambda:True,
                           can_adopt=lambda a,b:b['A'] and b['B'],step=lambda:1,max_updates=1)
        self.assertEqual(result['action'],'skill_accepted')
        adopt(state,'b'*64,step=1,reason='verified skill B')
        self.assertEqual(state['model_id'],MODEL_ID);self.assertEqual(state['revision'],1)
        self.assertEqual(state['parent_weights_digest'],'a'*64)
        preserved=dict(model)
        result=consolidate(state,bank(),preserved,train=lambda e:model.update(A=False),evaluate=lambda:dict(model),
                           restore=lambda:(model.clear(),model.update(preserved)),can_train=lambda:True,
                           can_adopt=lambda a,b:b['A'] and b['B'],step=lambda:2,max_updates=1)
        self.assertEqual(result['action'],'skill_rejected');self.assertEqual(model,{'A':True,'B':True})

    def test_zero_budget_cannot_train_or_evaluate(self):
        forbidden=lambda *a:self.fail('budget exceeded')
        result=consolidate(new_state('a'*64),bank(),{},train=forbidden,evaluate=forbidden,restore=forbidden,
                           can_train=lambda:False,can_adopt=forbidden,step=lambda:0)
        self.assertEqual(result['action'],'budget_exhausted');self.assertEqual(result['updates'],0)

    def test_exception_also_restores_transaction_and_is_not_silenced(self):
        restored=[]
        def train(e):raise RuntimeError('nonfinite native gradient')
        with self.assertRaises(RuntimeError):
            consolidate(new_state('a'*64),bank(),{},train=train,evaluate=lambda:{},restore=lambda:restored.append(1),
                        can_train=lambda:True,can_adopt=lambda a,b:True,step=lambda:0)
        self.assertEqual(restored,[1])

    def test_rejected_identical_skill_is_backed_off(self):
        state=new_state('a'*64)
        args=dict(train=lambda e:None,evaluate=lambda:{},restore=lambda:None,can_train=lambda:True,
                  can_adopt=lambda a,b:False,step=lambda:1,max_updates=1)
        self.assertEqual(consolidate(state,bank(),{},**args)['action'],'skill_rejected')
        self.assertEqual(consolidate(state,bank(),{},**args)['action'],'no_ready_skill')

    def test_channel_uses_one_identity_and_compare_and_swap(self):
        c=FakeChannel();first=record()
        self.assertEqual(publish_current(c,first,'d'*40)['action'],'initialize')
        writes=len([x for x in c.calls if x[1] and x[0].startswith('contents/')])
        self.assertEqual(publish_current(c,first,'d'*40)['action'],'unchanged')
        self.assertEqual(len([x for x in c.calls if x[1] and x[0].startswith('contents/')]),writes)
        state=new_state('a'*64);adopt(state,'b'*64,step=2,reason='verified')
        second=record(state)
        self.assertEqual(publish_current(c,second,'d'*40)['action'],'advance')
        last=c.calls[-1]
        self.assertEqual(last[2]['method'],'PUT');self.assertIn('sha',last[1])
        self.assertEqual(json.loads(base64.b64decode(c.file['content']))['model_id'],'airi-native')
        self.assertRaises(ValueError,publish_current,c,first,'d'*40)

    def test_channel_cannot_skip_lineage_or_publish_live_or_wrong_model(self):
        before=record();after=copy.deepcopy(before)
        after.update(revision=1,weights_digest='b'*64,parent_weights_digest='c'*64)
        self.assertRaises(ValueError,transition,before,after)
        after['parent_weights_digest']='a'*64;after['revision']=2
        self.assertRaises(ValueError,transition,before,after)
        for name,value in [('live_promoted',True),('model_id','other'),('model_file_sha256','wrong')]:
            bad=copy.deepcopy(before);bad[name]=value;self.assertRaises(ValueError,validate,bad)

    def test_stable_loader_verifies_configuration_and_weights_without_optimizer_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'model').mkdir();(p/'model/model.pt').write_bytes(b'native')
            (p/'model/config.json').write_text('{}')
            files={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (p/'model').iterdir()}
            current=model_record(new_state('a'*64),release='airi-offline-1-2-1-1',fingerprint='f'*64,step=1,model_files=files)
            self.assertEqual(validate_selected_model(p,current),p/'model')
            (p/'model/config.json').write_text('{"changed":true}')
            self.assertRaises(ValueError,validate_selected_model,p,current)

if __name__=='__main__':unittest.main()
