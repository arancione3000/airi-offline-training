import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import unittest

from test_skill_bank import lesson

ROOT=Path(__file__).resolve().parents[2]


class LearnCLITests(unittest.TestCase):
    def test_default_chat_uses_bundle_and_private_learning_survives_a_new_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);env={**os.environ,'HOME':directory}
            messages=root/'messages.json'
            messages.write_text(json.dumps([
                {'role':'user','content':'Il segnale acquisito è Abé42.'},
                {'role':'user','content':'Come restituisci il segnale?'}]))
            chat=[sys.executable,str(ROOT/'scripts/assistant-runtime/chat.py')]
            result=subprocess.run(chat+['--messages-file',str(messages)],env=env,capture_output=True,text=True,check=True,timeout=10)
            answer=json.loads(result.stdout)
            self.assertEqual(answer['source'],'learned_skill')
            self.assertEqual(answer['answer'],'24ébA')
            training=root/'lesson.json';training.write_text(json.dumps(lesson()))
            learned=subprocess.run(chat+['--learn',str(training)],env=env,capture_output=True,text=True,check=True,timeout=10)
            self.assertEqual(json.loads(learned.stdout)['registry']['revision'],7)
            messages.write_text(json.dumps(lesson()['qualification'][0]['messages']))
            restarted=subprocess.run(chat+['--messages-file',str(messages)],env=env,capture_output=True,text=True,check=True,timeout=10)
            self.assertEqual(json.loads(restarted.stdout)['answer'],lesson()['qualification'][0]['answer'])
            self.assertEqual(json.loads(restarted.stdout)['skill_revision'],7)

    def test_watcher_learns_new_file_without_intervention_and_restarts_idempotently(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);queue=root/'queue';queue.mkdir();bank=root/'bank.json'
            (queue/'first.json').write_text(json.dumps(lesson()))
            cmd=[sys.executable,str(ROOT/'scripts/assistant-runtime/learn.py'),'--watch',str(queue),
                 '--bank',str(bank),'--interval','1','--cycles','3']
            child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                self.assertTrue(select.select([child.stdout],[],[],10)[0],'watcher failed to produce its first result')
                first=json.loads(child.stdout.readline());self.assertEqual(first['registry']['revision'],1)
                (queue/'second.json').write_text(json.dumps(lesson('other','Other: ','What is the other value?')))
                stdout,stderr=child.communicate(timeout=10)
                self.assertEqual(child.returncode,0,stderr)
                rows=[json.loads(line) for line in stdout.splitlines()]
                self.assertTrue(any(r.get('registry',{}).get('revision')==2 for r in rows))
                before=bank.read_bytes()
                replay=subprocess.run(cmd[:-1]+['1'],capture_output=True,text=True,check=True,timeout=10)
                self.assertEqual(before,bank.read_bytes())
                self.assertTrue(all(e['action']=='unchanged' for r in map(json.loads,replay.stdout.splitlines()) for e in r['results']))
            finally:
                if child.poll() is None:child.kill();child.wait(timeout=5)


if __name__=='__main__':unittest.main()
