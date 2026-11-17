import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_skill_bank import lesson
from skill_bank import SkillBank
from flexible_skills import LearnedQueryRouter
from assistant import AiriAssistant


class FlexibleSkillTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.bank=SkillBank(Path(self.temp.name)/'bank.json')
        self.first=lesson();self.second=lesson('other','Other: ','What is the other value?')
        self.bank.learn(self.first);self.bank.learn(self.second)
        entries=self.bank._read()[0]['entries']
        self.model={'bindings':[{'lesson_id':e['lesson_id'],'program_id':e['id'],'canonical_query':e['program']['query']} for e in entries]}
        self.router=LearnedQueryRouter(self.bank,'unused.json')
        self.route={'lesson_id':'demo','score':.95,'margin':.5,'model_sha256':'x'}

    def request(self):
        messages=copy.deepcopy(self.first['qualification'][0]['messages'])
        messages[-1]['content']='Please tell me the packet value.'
        return messages

    def test_global_winner_must_match_source_and_input_is_never_rewritten_in_place(self):
        messages=self.request();before=copy.deepcopy(messages)
        with patch('query_router.read_model',return_value=self.model),patch('query_router.predict',return_value=self.route):
            result=self.router.predict(messages)
            self.assertEqual(result['answer'],'Beta');self.assertEqual(result['routing'],'learned_paraphrase')
            self.assertEqual(result['query_provenance']['original_query'],messages[-1]['content'])
            self.assertEqual(messages,before)
            self.route['lesson_id']='other'
            self.assertIsNone(self.router.predict(messages))

    def test_role_identity_conflicts_and_multiple_sources_cannot_be_canonicalized_away(self):
        variants=[]
        roles=self.request();roles.insert(0,{'role':'system','content':'Different instructions'});variants.append(roles)
        speaker=self.request();speaker[0]['speaker_id']='A';speaker[-1]['speaker_id']='B';variants.append(speaker)
        conflict=self.request();conflict.insert(1,{'role':'user','content':'Packet: Zeta!'});variants.append(conflict)
        multiple=self.request();multiple.insert(1,copy.deepcopy(self.second['qualification'][0]['messages'][0]));variants.append(multiple)
        with patch('query_router.read_model',return_value=self.model),patch('query_router.predict',return_value=self.route):
            for messages in variants:self.assertIsNone(self.router.predict(messages))

    def test_exact_routes_retain_priority_and_corrupt_router_uses_native_fallback(self):
        with patch.object(self.router,'predict',side_effect=AssertionError('exact expert must win')):
            system=AiriAssistant(lambda _: 'native',skill_bank=self.bank,query_router=self.router)
            self.assertEqual(system.respond(self.first['qualification'][0]['messages'])['answer'],'Beta')
        with patch('query_router.read_model',side_effect=ValueError('damaged model')):
            system=AiriAssistant(lambda _: 'native',skill_bank=self.bank,query_router=self.router)
            result=system.respond(self.request())
            self.assertEqual(result['answer'],'native');self.assertEqual(result['source'],'native_model')
            self.assertIn('component_error',result)

    def test_append_only_growth_preserves_existing_bindings(self):
        self.bank.learn(lesson('third','Third: ','What is the third value?'))
        with patch('query_router.read_model',return_value=self.model),patch('query_router.predict',return_value=self.route):
            self.assertEqual(self.router.predict(self.request())['answer'],'Beta')


if __name__=='__main__':unittest.main()
