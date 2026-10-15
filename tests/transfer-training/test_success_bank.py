import copy
import json
import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/transfer-training'))
from success_bank import (catalog,row_key,empty_bank,remember,validate_bank,summary,
                          retains_correct_answers,probe_rows,CAPACITY)

def row(prompt='copy a',answer='a',language='en',family='foundation-copy'):
    return dict(messages=[dict(role='user',content=prompt),dict(role='assistant',content=answer)],
                expected=answer,language=language,family=family,provenance='deterministic exact solver')

def donation(r,step=1,gain=.3):
    return {'row':r,'positions':[{'position':2,'target':1,'gain':gain,'probabilities':[.1,.8,.1]}],
            'source':dict(step=step,fingerprint='test',model_sha256='a'*64),
            'whole_answer_gain':False,'scope':'authored training only'}

class SuccessBankTests(unittest.TestCase):
    def test_transfer_slots_preserve_human_and_native_replay_for_every_focus(self):
        import run
        for focus in run.SAMPLING_SCHEDULES:
            selected=[i for i in range(48) if run.success_slot(i,run.sampling_group(i,focus),True)]
            self.assertLessEqual(len(selected),6)
            self.assertTrue(selected)
            self.assertTrue(all(run.sampling_group(i,focus) not in {'human-dialogue','native-replay'} for i in selected))
        self.assertFalse(run.success_slot(2,'old-authored',False))

    def test_heldout_prompt_cannot_become_training_lesson_even_with_changed_answer(self):
        test=row(); forged=row(answer='wrong')
        eligible=catalog([row('safe'),forged],[test])
        self.assertNotIn(row_key(forged),eligible)
        self.assertRaises(ValueError,remember,empty_bank('test'),donation(test),eligible)

    def test_human_and_model_labels_cannot_become_verified_lessons(self):
        human=row('human');human['provenance']={'source':'OpenAssistant'}
        model=row('guess');model['provenance']='student says correct'
        result=catalog([row(),human,model],[])
        self.assertEqual(len(result),1)

    def test_lesson_survives_json_restart_and_keeps_multiple_donor_lineages(self):
        a,b=row(),row('copy b','b');eligible=catalog([a,b],[]);bank=empty_bank('test')
        remember(bank,donation(a,7),eligible);remember(bank,donation(b,11),eligible)
        restored=json.loads(json.dumps(bank))
        validate_bank(restored,eligible,'test',3)
        self.assertEqual({e['source']['step'] for e in restored['entries'].values()},{7,11})
        self.assertEqual(summary(restored)['verified_whole_answer_gains'],0)

    def test_less_informative_duplicate_cannot_overwrite_better_donor(self):
        r=row();eligible=catalog([r],[]);bank=empty_bank('test')
        remember(bank,donation(r,1,.8),eligible);remember(bank,donation(r,2,.1),eligible)
        self.assertEqual(bank['entries'][row_key(r)]['source']['step'],1)

    def test_corrupted_wrong_and_nonfinite_distributions_are_refused(self):
        r=row();eligible=catalog([r],[]);bank=empty_bank('test');remember(bank,donation(r),eligible)
        for probs in ([.1,float('nan'),.9],[.9,.05,.05],[.1,.7,.1],[-.1,1.,.1]):
            bad=copy.deepcopy(bank);bad['entries'][row_key(r)]['positions'][0]['probabilities']=probs
            self.assertRaises(ValueError,validate_bank,bad,eligible,'test',3)
        self.assertRaises(ValueError,validate_bank,bank,eligible,'other',3)

    def test_finite_capacity_and_both_languages_survive(self):
        rows=[row(str(i),language='it' if i%2 else 'en') for i in range(100)]
        eligible=catalog(rows,[]);bank=empty_bank('test')
        for i,r in enumerate(rows):remember(bank,donation(r,i,.1+i/100),eligible)
        self.assertEqual(len(bank['entries']),CAPACITY)
        self.assertEqual({e['row']['language'] for e in bank['entries'].values()},{'it','en'})

    def test_rotating_probes_visit_different_training_rows(self):
        eligible=catalog([row(str(i),family='task'+str(i%3)) for i in range(60)],[])
        first=probe_rows(eligible,0,6);later=probe_rows(eligible,3,6)
        self.assertNotEqual(first,later)
        self.assertTrue(all(row_key(r) in eligible for r in first+later))

    def test_equal_aggregate_cannot_hide_lost_old_answer(self):
        a=dict(messages=[{'role':'user','content':'A'}],expected='a',family='x',language='en',correct=True)
        b={**a,'messages':[{'role':'user','content':'B'}],'expected':'b','correct':False}
        before={'samples':[a,b]};after={'samples':[{**a,'correct':False},{**b,'correct':True}]}
        self.assertFalse(retains_correct_answers(before,after))
        self.assertTrue(retains_correct_answers(before,{'samples':[a,{**b,'correct':True}]}))

if __name__=='__main__':unittest.main()
