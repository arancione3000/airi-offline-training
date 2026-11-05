import hashlib,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/transfer-training'))
from continual_learning import validate_configuration
from continuous_training import validate_configuration as validate_service
class RecipeTests(unittest.TestCase):
 def test_current_service_matches_every_pinned_implementation(self):
  spec=json.loads((ROOT/'scripts/continuous-training/SPEC.json').read_text())
  validate_configuration(spec);validate_service(spec)
  for name,pin in spec['script_sha256'].items():self.assertEqual(hashlib.sha256((ROOT/'scripts/transfer-training'/name).read_bytes()).hexdigest(),pin,name)
  self.assertIsNone(spec['job_limit']);self.assertEqual(spec['migration_source_step'],450364)
  self.assertEqual(spec['start_step'],449276)
if __name__=='__main__':unittest.main()
