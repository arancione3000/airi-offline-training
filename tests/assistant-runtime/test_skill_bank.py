import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/assistant-runtime'))
from skill_bank import SkillBank,RegistryError
from assistant import AiriAssistant


def lesson(label='demo',prefix='Packet: ',query='What is the packet value?'):
    def example(value):return {'messages':[{'role':'user','content':prefix+value+'!'},
                                          {'role':'user','content':query}],'answer':value}
    return {'trusted':True,'lesson_id':label,
        'examples':[example(v) for v in ('aBc','Zulu','xYz','Mønt')],
        'qualification':[example(v) for v in ('Beta','Oméga','Nova','Delta','Gámma','Rune','Vertex','Sierra')]
            +[{'messages':[{'role':'assistant','content':prefix+v+'!'},
                           {'role':'user','content':query}],'answer':None} for v in ('One','Two','Three','Four')]}


class BankTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'bank.json';self.bank=SkillBank(self.path)

    def test_learning_adds_persistent_expert_and_replay_is_idempotent(self):
        self.assertEqual(self.bank.status()['active_skills'],0)
        result=self.bank.learn(lesson());self.assertEqual(result['action'],'activated')
        self.assertEqual(SkillBank(self.path).predict(lesson()['qualification'][0]['messages'])['answer'],'Beta')
        before=self.path.read_bytes();self.assertEqual(self.bank.learn(lesson())['action'],'unchanged')
        self.assertEqual(before,self.path.read_bytes());self.assertEqual(len(list(self.bank.snapshots.glob('*.json'))),1)

    def test_rejected_qualification_never_mutates_existing_knowledge(self):
        self.bank.learn(lesson());before=self.path.read_bytes()
        bad=lesson('other','Other: ','What is the other value?');bad['qualification'][0]['answer']='incorrect'
        self.assertEqual(self.bank.learn(bad)['action'],'rejected');self.assertEqual(before,self.path.read_bytes())

    def test_bad_labels_and_untrusted_lessons_cannot_write(self):
        bad=lesson();bad['examples'][0]['answer']='inconsistent'
        self.assertEqual(self.bank.learn(bad)['action'],'rejected');self.assertFalse(self.path.exists())
        bad['trusted']=False
        with self.assertRaises(RegistryError):self.bank.learn(bad)

    def test_snapshot_recovers_corruption_and_interrupted_main_publication(self):
        self.bank.learn(lesson());self.path.write_text('{corrupt')
        self.assertEqual(self.bank.status()['revision'],1);self.assertTrue(self.bank.status()['recovered_from_snapshot'])
        original=self.bank._atomic
        def interrupted(path,raw):
            if path==self.path:raise OSError('simulated interruption after snapshot commit')
            original(path,raw)
        with patch.object(self.bank,'_atomic',side_effect=interrupted):
            result=self.bank.learn(lesson('other','Other: ','What is the other value?'))
            self.assertEqual(result['action'],'activated')
            self.assertTrue(result['snapshot_committed']);self.assertFalse(result['main_copy_updated'])
        self.assertEqual(SkillBank(self.path).status()['revision'],2)
        self.assertEqual(SkillBank(self.path).predict(lesson()['qualification'][0]['messages'])['answer'],'Beta')

    def test_complete_registry_damage_falls_back_without_touching_base(self):
        self.path.write_text('{bad')
        a=AiriAssistant(lambda messages:'stable',skill_bank=self.bank)
        result=a.respond([{'role':'user','content':'unknown'}])
        self.assertEqual(result['answer'],'stable');self.assertEqual(result['source'],'native_model')
        self.assertIn('component_error',result)
        self.assertEqual(a.respond([{'role':'user','content':'Mi chiamo Ada. Come mi chiamo?'}])['answer'],'Ada')

    def test_deeply_nested_registry_corruption_recovers_or_uses_native(self):
        damaged=b'['*10000+b'0'+b']'*10000
        self.path.write_bytes(damaged)
        assistant=AiriAssistant(lambda _: 'stable',skill_bank=self.bank)
        result=assistant.respond([{'role':'user','content':'unknown'}])
        self.assertEqual(result['answer'],'stable')
        self.assertIn('component_error',result)
        self.path.unlink();self.bank.learn(lesson());self.path.write_bytes(damaged)
        row=lesson()['qualification'][0]
        self.assertEqual(self.bank.predict(row['messages'])['answer'],row['answer'])
        self.assertTrue(self.bank.status()['recovered_from_snapshot'])

    def test_seed_revision_zero_does_not_consume_expert_capacity(self):
        import skill_bank
        with patch.object(skill_bank,'MAX_SKILLS',2):
            self.bank._write(skill_bank._empty())
            self.bank.learn(lesson())
            self.bank.learn(lesson('other','Other: ','What is the other value?'))
            self.assertEqual(self.bank.status()['revision'],2)
            self.assertEqual(len(list(self.bank.snapshots.glob('*.json'))),3)

    def test_cas_conflict_does_not_replace_a_committed_revision(self):
        self.bank.learn(lesson(),expected_revision=0);before=self.path.read_bytes()
        with self.assertRaises(RegistryError):self.bank.learn(lesson('other','Other: ','What is the other value?'),expected_revision=0)
        self.assertEqual(before,self.path.read_bytes())

    def test_overlap_and_failed_system_gate_are_rejected(self):
        bad=lesson();bad['qualification'][0]=copy.deepcopy(bad['examples'][0])
        with self.assertRaises(RegistryError):self.bank.learn(bad)
        self.assertEqual(self.bank.learn(lesson(),regression_gate=lambda route:False)['action'],'rejected')
        self.assertFalse(self.path.exists())

    def test_private_bank_and_snapshots_have_owner_only_permissions(self):
        self.bank.learn(lesson())
        self.assertEqual(self.path.stat().st_mode&0o777,0o600)
        self.assertEqual(self.bank.snapshots.stat().st_mode&0o777,0o700)
        self.assertEqual(next(self.bank.snapshots.glob('*.json')).stat().st_mode&0o777,0o600)

    def test_new_operation_cannot_shadow_an_older_qualified_skill(self):
        self.bank.learn(lesson());before=self.path.read_bytes()
        other=lesson('reverse')
        for row in other['examples']+other['qualification']:
            if row['answer'] is not None:row['answer']=row['answer'][::-1]
        self.assertEqual(self.bank.learn(other)['action'],'rejected')
        self.assertEqual(before,self.path.read_bytes())

    def test_newer_snapshot_is_reported_when_valid_main_copy_is_stale(self):
        self.bank.learn(lesson());stale=self.path.read_bytes()
        self.bank.learn(lesson('other','Other: ','What is the other value?'))
        self.path.write_bytes(stale)
        self.assertEqual(self.bank.status()['revision'],2)
        self.assertTrue(self.bank.status()['recovered_from_snapshot'])

    def test_same_assistant_uses_new_skill_without_restart(self):
        assistant=AiriAssistant(lambda _: 'old native',skill_bank=self.bank)
        messages=lesson()['qualification'][0]['messages']
        self.assertEqual(assistant.respond(messages)['answer'],'old native')
        self.bank.learn(lesson())
        self.assertEqual(assistant.respond(messages)['answer'],'Beta')
        self.assertEqual(assistant.respond(messages)['source'],'learned_skill')


if __name__=='__main__':unittest.main()
