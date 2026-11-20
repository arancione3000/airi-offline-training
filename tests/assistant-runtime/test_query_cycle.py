import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

from test_skill_bank import lesson
from skill_bank import SkillBank

PATH=Path(__file__).resolve().parents[2]/'scripts/research-training/semantic-routing-20261116/cycle.py'
SPEC=importlib.util.spec_from_file_location('semantic_research_cycle',PATH)
cycle=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(cycle)
ORIGINAL_FINGERPRINT=cycle.fingerprint


class QueryCycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.here=self.root/'research'
        corpus=self.here/'evidence/precommitted';corpus.mkdir(parents=True)
        for name in ('FAMILIES','TRAIN','QUALIFICATION','FINAL','DIAGNOSTIC','MANIFEST'):
            (corpus/(name+'.json.gz')).write_bytes(gzip.compress(b'{}',mtime=0))
        self.runtime=self.root/'runtime';self.runtime.mkdir()
        self.state=self.runtime/'state.json';self.model=self.runtime/'approved.json';self.retention=self.runtime/'retention.gz'
        self.aliases=self.runtime/'aliases.json'
        self.model.write_text('previous approved model')
        self.retention.write_bytes(gzip.compress(b'[]',mtime=0))
        settings={'HERE':self.here,'RUNTIME':self.runtime,'STATE':self.state,'MODEL':self.model,
                  'RETENTION':self.retention,'ALIASES':self.aliases}
        self.patcher=patch.multiple(cycle,**settings);self.patcher.start();self.addCleanup(self.patcher.stop)
        self.fingerprint=patch.object(cycle,'fingerprint',return_value='new-inputs');self.fingerprint.start();self.addCleanup(self.fingerprint.stop)

    def fake_trial(self,eligible,*,qualification=(),final=()):
        def execute(command,**_):
            trial=Path(command[-1]);trial.mkdir(parents=True)
            (trial/'RESULT.json').write_text(json.dumps({'eligible_for_activation':eligible}))
            (trial/'candidate.json').write_text('{}')
            for split, observations in (('QUALIFICATION',qualification),('FINAL',final)):
                (trial/(split+'_REPORT.json')).write_text(json.dumps({'observations':observations}))
        return execute

    def test_identical_inputs_do_not_train_or_rewrite_verdict(self):
        self.state.write_text(json.dumps({'input_sha256':'new-inputs','action':'rejected'}));before=self.state.read_bytes()
        with patch.object(cycle.subprocess,'run',side_effect=AssertionError('unchanged data must not retrain')):
            result=cycle.run(self.root/'out')
        self.assertEqual(result['action'],'unchanged');self.assertEqual(before,self.state.read_bytes())

    def test_new_program_bindings_invalidate_a_previous_verdict(self):
        for name in ('run.py','cycle.py','ENCODER_MANIFEST.json','requirements-encoder.txt'):
            (self.here/name).write_text('fixed trusted source')
        transfer_cycle=self.root/'exact-transfer-20261120/cycle.py'
        transfer_cycle.parent.mkdir()
        transfer_cycle.write_text('fixed transfer audit')
        (self.runtime/'example.py').write_text('fixed runtime')
        bank=SkillBank(self.runtime/'skills/approved-bank.json');bank.learn(lesson())
        with patch.object(cycle,'ROOT',self.root),patch.object(cycle,'fingerprint',ORIGINAL_FINGERPRINT):
            self.state.write_text(json.dumps({'input_sha256':cycle.fingerprint()}))
            self.assertFalse(cycle.needed())
            bank.learn(lesson('other','Other: ','What is the other value?'))
            self.assertTrue(cycle.needed())

    def test_rejected_candidate_keeps_previous_model_and_knowledge(self):
        before=self.model.read_bytes();retained=self.retention.read_bytes()
        with patch.object(cycle.subprocess,'run',side_effect=self.fake_trial(False)):
            result=cycle.run(self.root/'out')
        self.assertEqual(result['action'],'rejected');self.assertFalse(result['model_staged'])
        self.assertEqual(self.model.read_bytes(),before);self.assertEqual(self.retention.read_bytes(),retained)

    def test_new_candidate_cannot_lose_a_previously_qualified_paraphrase(self):
        bank=SkillBank(self.runtime/'skills/approved-bank.json');bank.learn(lesson())
        messages=lesson()['qualification'][0]['messages'];messages[-1]['content']='Please report the packet.'
        self.retention.write_bytes(gzip.compress(json.dumps([{'messages':messages,'expected':'Beta'}]).encode(),mtime=0))
        previous=self.model.read_bytes()
        routed=Mock(snapshot=bank._read()[0],predict=lambda value: {'answer':'wrong'} if value==messages else None)
        with patch.object(cycle,'HERE',PATH.parent):
            module=cycle.exact_transfer()
        with patch.object(cycle.subprocess,'run',side_effect=self.fake_trial(True)),patch('query_router_encoder.read_model',return_value={'sha256':'candidate'}),patch.object(cycle,'LearnedQueryRouter',return_value=routed),patch.object(cycle,'exact_transfer',return_value=module),patch('query_router_encoder.write_model',side_effect=AssertionError('regressing candidate must not publish')):
            result=cycle.run(self.root/'out')
        self.assertEqual(result['action'],'rejected');self.assertIn('regress',result['reason'])
        self.assertEqual(result['composition_audit'],{'passed':False})
        self.assertFalse(result['result']['eligible_for_activation'])
        self.assertEqual(self.model.read_bytes(),previous)

    def test_composed_gate_persists_final_success_only_as_retention(self):
        bank=SkillBank(self.runtime/'skills/approved-bank.json');bank.learn(lesson())
        messages=lesson()['qualification'][0]['messages']
        messages[-1]['content']='Please report this packet.'
        observed={'messages':messages,'expected':'Beta','correct':True}
        routed=Mock(snapshot=bank._read()[0],predict=lambda value: {'answer':'Beta'} if value==messages else None)
        with patch.object(cycle,'HERE',PATH.parent):
            module=cycle.exact_transfer()
        model={'sha256':'candidate'}
        with patch.object(cycle.subprocess,'run',side_effect=self.fake_trial(True,final=[observed])), \
                patch('query_router_encoder.read_model',return_value=model), \
                patch.object(cycle,'LearnedQueryRouter',return_value=routed), \
                patch.object(cycle,'exact_transfer',return_value=module), \
                patch('query_router_encoder.write_model') as write:
            result=cycle.run(self.root/'out')
        self.assertTrue(result['model_staged'])
        self.assertFalse(result['retention_used_for_training'])
        self.assertEqual(result['composition_audit']['semantic_checks'],1)
        self.assertEqual(json.loads(gzip.decompress(self.retention.read_bytes())),
                         [{'messages':messages,'expected':'Beta'}])
        write.assert_called_once_with(self.model,model)


if __name__=='__main__':unittest.main()
