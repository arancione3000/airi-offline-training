import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/assistant-runtime'))
import query_router_factorized as router
from test_query_router import training


def metadata():
    return [{'id':'parcel','operation':'copy','object_phrase':'il codice del pacco'},
            {'id':'upper','operation':'uppercase','object_phrase':'la etichetta'},
            {'id':'lower','operation':'lowercase','object_phrase':'la etichetta'}]


class FactorizedRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows,cls.bindings=training()
        cls.model=router.train(cls.rows,cls.bindings,family_metadata=metadata())

    def test_global_family_and_independent_operation_consensus(self):
        self.assertEqual(router.predict(self.model,'Mostrami la etichetta in minuscolo.')['lesson_id'],'lower')
        self.assertEqual(router.predict(self.model,'Mostrami la etichetta in maiuscolo.')['lesson_id'],'upper')
        self.assertEqual(router.predict(self.model,'Mostrami il codice del pacco.')['lesson_id'],'parcel')
        # A known operation applied to another known object must not inherit
        # the object's ordinary copy program merely because the noun matches.
        self.assertIsNone(router.predict(self.model,'Mostrami il codice del pacco in minuscolo.'))
        self.assertIsNone(router.predict(self.model,'Mostrami il codice del pacco in maiuscolo.'))
        self.assertEqual(self.model['calibration']['false_accepts'],0)
        self.assertEqual(self.model['calibration']['wrong_classes'],0)

    def test_operation_evidence_is_learned_without_object_identifiers(self):
        cues=self.model['operation_cues']
        self.assertEqual(cues['w:minuscolo'],'lowercase')
        self.assertEqual(cues['w:maiuscolo'],'uppercase')
        self.assertNotIn('w:pacco',cues)
        self.assertNotIn('w:etichetta',cues)
        self.assertIsNone(router.predict(self.model,'Non mostrarmi il codice del pacco.'))
        self.assertIsNone(router.predict(self.model,'qzxwvkj xyzabcd'))

    def test_metadata_identity_and_serialization_remain_bound_and_no_inference_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'factorized.json';router.write_model(path,self.model)
            with patch.object(router,'_fit',side_effect=AssertionError('no inference training')):
                restored=router.read_model(path)
                self.assertEqual(router.predict(restored,'Mostrami il codice del pacco.')['lesson_id'],'parcel')
            self.assertEqual(path.stat().st_mode & 0o777,0o600)
            corrupt=copy.deepcopy(self.model);corrupt['operations']['parcel']='multiply'
            with self.assertRaises(ValueError):router.predict(corrupt,'Mostrami il codice del pacco.')
        bad=metadata();bad[0]['operation']='execute'
        with self.assertRaises(ValueError):router.train(self.rows,self.bindings,family_metadata=bad)


if __name__=='__main__':unittest.main()
