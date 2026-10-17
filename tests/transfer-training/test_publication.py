import sys, subprocess, unittest, tempfile, json, hashlib, os
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/transfer-training'))
import transport

class PublicationTests(unittest.TestCase):
 def test_continuation_state_is_hash_committed_after_all_checkpoint_assets(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);checkpoint=root/'checkpoint';(checkpoint/'model').mkdir(parents=True)
   (checkpoint/'model/model.pt').write_bytes(b'weights');(checkpoint/'optimizer.pt').write_bytes(b'adam')
   state=b'{"continue":true}';(checkpoint/'AIRI_RUN_STATE.json').write_bytes(state)
   (checkpoint/'checkpoint.json').write_text(json.dumps({'fingerprint':transport.FINGERPRINT,'completed_steps':400000,
       'files':{str(p.relative_to(checkpoint)):transport.digest(p) for p in checkpoint.rglob('*') if p.is_file()}}))
   uploads=[];indexes=[]
   def fake_gh(*args):
    if args[:2]==('release','upload'):
     paths=[Path(v) for v in args[3:-2]];uploads.extend(p.name for p in paths)
     indexes.extend(json.loads(p.read_text()) for p in paths if p.name.startswith('AIRI_TRAINING_CHECKPOINT_'))
    return ''
   with patch.object(transport,'gh',side_effect=fake_gh),patch.dict(os.environ,{'GITHUB_REPOSITORY':'owner/repo','GITHUB_SHA':'a'*40,'GITHUB_RUN_ID':'10','GITHUB_RUN_ATTEMPT':'1'}):
    transport.ReleasePublisher(root/'releases').publish(checkpoint)
   self.assertEqual(uploads[-2:],['AIRI_RUN_STATE.json','AIRI_TRAINING_CHECKPOINT_400000.json'])
   self.assertEqual(indexes[0]['run_state'],{'name':'AIRI_RUN_STATE.json','bytes':len(state),'sha256':hashlib.sha256(state).hexdigest()})
 def test_transient_edit_is_retried_without_reupload_or_new_release(self):
  error=subprocess.CalledProcessError(1,['gh','release','edit'],stderr='release not found')
  with patch.object(transport,'gh',side_effect=[error,error,'published']) as call:
   waits=[];self.assertEqual(transport.finalize_release('airi-offline-test','owner/repo',pause=waits.append),'published')
   self.assertEqual(waits,[1,2]);self.assertEqual(call.call_count,3)
   self.assertTrue(all(c.args==('release','edit','airi-offline-test','--repo','owner/repo','--draft=false','--latest=false') for c in call.call_args_list))
 def test_exhausted_publication_stays_failed(self):
  error=subprocess.CalledProcessError(1,['gh'])
  with patch.object(transport,'gh',side_effect=error) as call:
   with self.assertRaises(subprocess.CalledProcessError):transport.finalize_release('airi-offline-test','owner/repo',pause=lambda _:None)
   self.assertEqual(call.call_count,4)
