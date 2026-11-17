import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/assistant-runtime'))
import semantic_encoder


class EncoderBoundaryTests(unittest.TestCase):
    def test_missing_cache_and_unsafe_manifest_fail_before_optional_runtime(self):
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaisesRegex(ValueError,'not configured'):semantic_encoder.manifest()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest={'schema':1,'model_id':'test','revision':'pinned','max_tokens':128,
                      'model_file':'../outside.onnx','tokenizer_file':'tokenizer.json',
                      'files':{'../outside.onnx':{'sha256':'a'*64,'bytes':1},'tokenizer.json':{'sha256':'b'*64,'bytes':1}}}
            (root/'MANIFEST.json').write_text(json.dumps(manifest))
            with patch.dict('os.environ',{'AIRI_QUERY_ENCODER_CACHE':directory}):
                with self.assertRaisesRegex(ValueError,'unsafe'):semantic_encoder.manifest()
                (root/'MANIFEST.json').write_text('['*10000+'0'+']'*10000)
                with self.assertRaises(ValueError):semantic_encoder.manifest()

    def test_inputs_are_bounded_before_loading_optional_assets(self):
        for values in ([],['x'*513],[None],['x']*769):
            with self.assertRaises(ValueError):semantic_encoder.encode(values)
