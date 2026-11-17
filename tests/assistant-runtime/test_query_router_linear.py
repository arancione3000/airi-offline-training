import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/assistant-runtime'))
import query_router_linear as router
from test_query_router import training


class LinearRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows,cls.bindings=training();cls.model=router.train(cls.rows,cls.bindings)

    def test_supervised_weights_distinguish_close_operations_globally(self):
        self.assertTrue(self.model['calibration']['precision_gate_passed'])
        self.assertEqual(self.model['calibration']['false_accepts'],0)
        self.assertEqual(self.model['calibration']['wrong_classes'],0)
        self.assertEqual(router.predict(self.model,'Mostrami la etichetta in minuscolo.')['lesson_id'],'lower')
        self.assertEqual(router.predict(self.model,'Mostrami la etichetta in maiuscolo.')['lesson_id'],'upper')
        self.assertEqual(router.predict(self.model,'Il codice del pacco, vorrei.')['lesson_id'],'parcel')

    def test_guards_and_unknown_questions_abstain(self):
        for query in ('Non mostrarmi il codice del pacco.','"Mostrami il codice del pacco."',
                      'Mostrami il secondo meno il primo.','qzxwvkj xyzabcd','x'*513):
            self.assertIsNone(router.predict(self.model,query),query)

    def test_persisted_weights_load_without_training_and_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'linear.json';router.write_model(path,self.model)
            with patch.object(router,'_fit',side_effect=AssertionError('inference must not train')):
                restored=router.read_model(path)
                self.assertEqual(router.predict(restored,'Mostrami il codice del pacco.')['lesson_id'],'parcel')
            self.assertEqual(path.stat().st_mode & 0o777,0o600)
            raw=copy.deepcopy(self.model);feature=next(iter(raw['weights'][0]));raw['weights'][0][feature]=float('inf')
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):router.read_model(path)
            raw=copy.deepcopy(self.model);raw['classes'][0]='unknown'
            with self.assertRaises(ValueError):router.predict(raw,'Mostrami il codice del pacco.')
            path.write_text('['*10000+'0'+']'*10000)
            with self.assertRaises(ValueError):router.read_model(path)


if __name__=='__main__':unittest.main()
