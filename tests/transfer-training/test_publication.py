import sys, subprocess, unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/transfer-training'))
import transport

class PublicationTests(unittest.TestCase):
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
