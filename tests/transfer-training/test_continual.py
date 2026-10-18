import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/transfer-training'))
from continual_learning import retention_rows,profile_index,validate_configuration,rejection_reasons
from success_bank import catalog,prompt_key
from curriculum import build
from extra_dialogues import build_extra
from unittest.mock import patch
import run as runner


class ContinualTests(unittest.TestCase):
    def test_archived_lessons_receive_slots_without_replacing_human_or_native_replay(self):
        root=Path(__file__).resolve().parents[2]
        spec=json.loads((root/'scripts/continual-training/SPEC.json').read_text())
        cfg=spec['continual_learning'];schedule=cfg['sampling_schedule']
        with patch.object(runner,'LEARNING',cfg):
            actual=[i for i,g in enumerate(schedule) if runner.success_slot(i,g,True)]
            self.assertEqual(actual,cfg['success_transfer_slots'])
            self.assertEqual(len(actual),2)
            for i,g in enumerate(schedule):
                if g in {'human-dialogue','native-replay'}:
                    self.assertFalse(runner.success_slot(i,g,True))
                self.assertFalse(runner.success_slot(i,g,False))
        bad=copy.deepcopy(spec);bad['continual_learning']['success_transfer_slots']=[0,15]
        bad.pop('fingerprint');bad['fingerprint']=hashlib.sha256(json.dumps(bad,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.assertRaises(ValueError,validate_configuration,bad)

    def test_reviewed_dialogue_can_donate_but_evaluation_and_human_references_cannot(self):
        authored,heldout=build();extra,development=build_extra(old_validation=heldout)
        foreign=copy.deepcopy(extra[0]);foreign['provenance']={'dataset':'unverified human'}
        foreign['messages'][0]['content']='foreign prompt'
        rows=catalog(authored+extra+[foreign],heldout+development)
        self.assertTrue(any(r['family']=='extra-dialogue' for r in rows.values()))
        self.assertNotIn(prompt_key(foreign),{prompt_key(r) for r in rows.values()})
        forbidden=extra[0]
        rows=catalog(authored+extra,heldout+development+[forbidden])
        self.assertNotIn(prompt_key(forbidden),{prompt_key(r) for r in rows.values()})

    def test_memory_preserves_bilingual_families_without_copy_corpus_dominating(self):
        authored,heldout=build();extra,development=build_extra(old_validation=heldout)
        rows=list(catalog(authored+extra,heldout+development).values())
        selected=retention_rows(rows)
        self.assertTrue(all(r in rows for r in selected))
        self.assertEqual({r['language'] for r in selected},{'it','en'})
        self.assertTrue(any(r['family']=='extra-dialogue' for r in selected))
        self.assertTrue(any(r['family'].startswith('phrase-') for r in selected))
        self.assertRaises(ValueError,retention_rows,[r for r in rows if r['language']=='it'])

    def test_profile_exploration_has_a_floor_and_revisits_different_recipes(self):
        self.assertEqual([profile_index(i,3) for i in range(6)],[0,1,2,0,1,2])
        for bad in (-1,True,1.5):self.assertRaises(ValueError,profile_index,bad,3)

    def test_configuration_rejects_unbounded_search_and_tampering(self):
        root=Path(__file__).resolve().parents[2]
        spec=json.loads((root/'scripts/continual-training/SPEC.json').read_text())
        self.assertTrue(validate_configuration(spec))
        bad=copy.deepcopy(spec);bad['maximum_training_seconds']+=1
        self.assertRaises(ValueError,validate_configuration,bad)
        bad=copy.deepcopy(spec);bad['continual_learning']['maximum_rejections']=1000
        bad.pop('fingerprint');bad['fingerprint']=hashlib.sha256(json.dumps(bad,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.assertRaises(ValueError,validate_configuration,bad)

    def test_losing_old_answer_is_reported_even_when_aggregate_is_equal(self):
        def sample(name):return dict(messages=[dict(role='user',content=name)],expected=name,family='phrase',language='it',correct=True)
        m=dict(context_rates={'it:name':0.},context_pair_rate=0.,legacy_correct=1,dialogue_correct=0,
               foundation_correct=0,foundation_probes=16,repetition=0.,nll={'it':1.},samples=[sample('a')])
        changed=copy.deepcopy(m);changed['samples']=[sample('b')]
        self.assertIn('previously_correct_answer_lost',rejection_reasons(m,changed))


if __name__=='__main__':unittest.main()
