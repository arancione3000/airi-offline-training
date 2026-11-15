import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/assistant-runtime'))
import chat


class ChatTests(unittest.TestCase):
    def test_real_batch_cli_answers_from_memory_without_model_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'messages.json'
            path.write_text(json.dumps([{'role':'user','content':'Mi chiamo Áda e vivo a Torino. Come mi chiamo?'}]))
            run=subprocess.run([sys.executable,str(ROOT/'scripts/assistant-runtime/chat.py'),
                                '--messages-file',str(path)],capture_output=True,text=True,check=True,timeout=10)
            value=json.loads(run.stdout)
            self.assertEqual(value['answer'],'Áda')
            self.assertEqual(value['source'],'bounded_context_memory')
            self.assertEqual(run.stderr,'')

    def test_reset_drops_history_and_does_not_reuse_a_previous_name(self):
        seen=[]
        def native(_self,messages):
            seen.append(list(messages));return 'fallback'
        prompts=iter(['My name is Ada. What is my name?','/reset','What is my name?','/exit'])
        with patch.object(sys,'argv',['chat.py']),patch('builtins.input',side_effect=lambda _:next(prompts)),\
             patch.object(chat.VerifiedNative,'__call__',native),contextlib.redirect_stdout(io.StringIO()):
            chat.main()
        self.assertEqual(len(seen),1)
        self.assertEqual(seen[0],[{'role':'user','content':'What is my name?'}])

    def test_local_checkpoint_requires_hash_bound_approved_record(self):
        with self.assertRaises(ValueError):
            chat.VerifiedNative(cache=Path('unused'),checkpoint=Path('unverified'))


if __name__=='__main__':unittest.main()
