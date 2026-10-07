import importlib.util
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/conversation-training'))
import curriculum

class CurriculumTests(unittest.TestCase):
    def test_split_and_arithmetic(self):
        train,heldout=curriculum.build()
        self.assertGreater(len(train),600)
        self.assertGreater(len(heldout),100)
        self.assertEqual({r['language'] for r in heldout},{'it','en'})
        for row in train+heldout:
            if row['family'].startswith('addition'):
                import re
                values=[int(x) for x in re.findall(r'\d+',row['messages'][0]['content'])]
                self.assertEqual(int(row['expected']),sum(values))
        seen={r['messages'][0]['content'] for r in train}
        self.assertFalse(seen&{r['messages'][0]['content'] for r in heldout})
    def test_protected_text_excluded_in_all_roles(self):
        train,heldout=curriculum.build({'mi chiamo luca.','my name is airi.'})
        for row in train+heldout:
            self.assertFalse({'mi chiamo luca.','my name is airi.'}&{curriculum.norm(m['content']) for m in row['messages']})
    def test_no_false_exact_success(self):
        self.assertTrue(curriculum.exact_correct(' 7. ','7'))
        self.assertFalse(curriculum.exact_correct('7 giorni ma non so, una pistola','7'))
        self.assertFalse(curriculum.exact_correct('17','7'))
    def test_search_requires_both_language_and_responses(self):
        before={'nll':{'it':2,'en':2},'exact_correct':0,'repetition':0}
        after={'nll':{'it':1.8,'en':1.8},'exact_correct':3,'repetition':0}
        self.assertTrue(curriculum.pilot_eligible(before,after))
        after['exact_correct']=0
        self.assertFalse(curriculum.pilot_eligible(before,after))
        after['exact_correct']=3;after['nll']['it']=2.1
        self.assertFalse(curriculum.pilot_eligible(before,after))
    def test_no_eligible_winner(self):
        self.assertIsNone(curriculum.choose_winner([{'eligible':False}]))
    def test_scripts_and_spec_pinned(self):
        import hashlib,json
        here=ROOT/'scripts/conversation-training'
        spec=json.loads((here/'SPEC.json').read_text())
        for name,digest in spec['script_sha256'].items():
            self.assertEqual(hashlib.sha256((here/name).read_bytes()).hexdigest(),digest,name)
        pinned=spec.pop('fingerprint')
        self.assertEqual(hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),pinned)
        self.assertEqual(spec['max_steps'],spec['start_step']+spec['pilot_updates']+spec['learning_updates'])

if __name__=='__main__':unittest.main()
