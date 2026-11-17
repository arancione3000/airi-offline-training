import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/assistant-runtime'))
import query_router_encoder as router
from test_query_router import training

HAS_NUMPY=importlib.util.find_spec('numpy') is not None


@unittest.skipUnless(HAS_NUMPY,'optional numpy is not installed')
class SemanticRouterTests(unittest.TestCase):
    @staticmethod
    def features(texts):
        import numpy as np
        rows=[]
        for text in texts:
            if text.startswith('Non ') or ('pacco' not in text and 'etichetta' not in text):rows.append([0.,0.,1.])
            elif 'pacco' in text:rows.append([1.,0.,0.])
            elif 'minuscol' in text:rows.append([0.,1.,.05])
            else:rows.append([0.,1.,-.05])
        return np.array(rows,dtype=np.float64),'a'*64

    def setUp(self):
        self.rows,self.bindings=training()
        with patch.object(router.semantic_encoder,'encode',side_effect=self.features):
            self.model=router.train(self.rows,self.bindings)

    def test_fixed_encoder_features_and_persisted_ridge_heads_are_separate(self):
        self.assertEqual(self.model['schema'],4)
        self.assertEqual(self.model['encoder_manifest_sha256'],'a'*64)
        self.assertEqual(self.model['calibration']['false_accepts'],0)
        self.assertEqual(self.model['calibration']['wrong_classes'],0)
        with patch.object(router.semantic_encoder,'encode',side_effect=self.features):
            self.assertEqual(router.predict(self.model,'Mostrami il codice del pacco.')['lesson_id'],'parcel')
            self.assertIsNone(router.predict(self.model,'Non mostrarmi il codice del pacco.'))

    def test_compilation_freezes_weights_and_encoder_identity_remains_checked(self):
        compiled=router.compile_model(self.model)
        self.assertFalse(compiled.weights.flags.writeable)
        with patch.object(router,'_validate',side_effect=AssertionError('validation should happen once')),patch.object(router.semantic_encoder,'encode',side_effect=self.features):
            self.assertEqual(compiled.predict('Mostrami il codice del pacco.')['lesson_id'],'parcel')
        def changed(texts):
            values,_=self.features(texts);return values,'b'*64
        with patch.object(router.semantic_encoder,'encode',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'identity'):compiled.predict('Mostrami il codice del pacco.')

    def test_shape_nonfinite_and_nested_corruption_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'semantic.json';router.write_model(path,self.model)
            self.assertEqual(router.read_model(path),self.model)
            self.assertEqual(path.stat().st_mode & 0o777,0o600)
            model=copy.deepcopy(self.model);model['weights'][0][0]=float('nan')
            path.write_text(json.dumps(model))
            with self.assertRaises(ValueError):router.read_model(path)
            path.write_text('['*10000+'0'+']'*10000)
            with self.assertRaises(ValueError):router.read_model(path)
