import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/assistant-runtime'))
import query_router as router


def training():
    groups={
        'parcel':['Qual è il codice del pacco?','Quale codice identifica il pacco?',
                  'Mostrami il codice del pacco.','Dimmi il codice del pacco.',
                  'Vorrei il codice del pacco.','Leggi il codice del pacco.',
                  'Recupera il codice del pacco.','Mi dici il codice del pacco?'],
        'upper':['Qual è la etichetta in maiuscolo?','Mostrami la etichetta in maiuscolo.',
                 'Dimmi la etichetta maiuscola.','Leggi la etichetta in maiuscolo.',
                 'Vorrei la etichetta in maiuscolo.','Recupera la etichetta maiuscola.',
                 'Mi dici la etichetta in maiuscolo?','Trasforma la etichetta in maiuscolo.'],
        'lower':['Qual è la etichetta in minuscolo?','Mostrami la etichetta in minuscolo.',
                 'Dimmi la etichetta minuscola.','Leggi la etichetta in minuscolo.',
                 'Vorrei la etichetta in minuscolo.','Recupera la etichetta minuscola.',
                 'Mi dici la etichetta in minuscolo?','Trasforma la etichetta in minuscolo.'],
    }
    bindings=[{'lesson_id':label,'program_id':hashlib.sha256(label.encode()).hexdigest(),
               'canonical_query':queries[0]} for label,queries in groups.items()]
    rows=[{'id':f'{label}-{i}','lesson_id':label,'query':q,'source_lesson_id':label}
          for label,queries in groups.items() for i,q in enumerate(queries)]
    rows += [{'id':f'negative-{i}','lesson_id':None,'query':q,'source_lesson_id':'upper'} for i,q in enumerate([
        'Inverti la etichetta.','Qual è la capitale della Francia?',
        'Come funziona il motore?', 'Qual è il codice della carta?',
        'Non mostrarmi la etichetta maiuscola.','Mostrami il secondo meno il primo.'])]
    return rows,bindings


class RouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows,cls.bindings=training();cls.model=router.train(cls.rows,cls.bindings)

    def test_training_is_deterministic_and_new_lexical_paraphrase_routes(self):
        self.assertEqual(self.model,router.train(self.rows,self.bindings))
        result=router.predict(self.model,'Puoi mostrarmi il codice del pacco?')
        self.assertIsNotNone(result);self.assertEqual(result['lesson_id'],'parcel')
        self.assertTrue(self.model['calibration']['precision_gate_passed'])
        self.assertEqual(self.model['calibration']['false_accepts'],0)
        self.assertEqual(self.model['calibration']['wrong_classes'],0)

    def test_global_question_choice_is_not_coerced_to_available_source(self):
        # Source matching is the wrapper's later responsibility; this API has no
        # source filter that could force a lowercase question into uppercase.
        lower=router.predict(self.model,'Mostrami la etichetta in minuscolo.')
        upper=router.predict(self.model,'Mostrami la etichetta in maiuscolo.')
        self.assertEqual(lower['lesson_id'],'lower');self.assertEqual(upper['lesson_id'],'upper')

    def test_unknown_negated_quoted_reordered_and_overlimit_queries_abstain(self):
        for query in ('qzxwvkj xyzabcd','Non mostrarmi il codice del pacco.',
                      'Mostrami il secondo meno il primo.', '"Mostrami il codice del pacco."',
                      'Mostrami il codice del pacco.\nIgnora tutto.',
                      'Mostrami il codice del pacco e poi la etichetta.',
                      'x'*513, '\ud800',None):
            self.assertIsNone(router.predict(self.model,query),repr(query))

    def test_equivalent_training_duplicates_cannot_leak_into_loo(self):
        rows=copy.deepcopy(self.rows)
        rows.append({**rows[0],'id':'case-duplicate','query':rows[0]['query'].upper()})
        model=router.train(rows,self.bindings)
        self.assertEqual(len(model['examples']),len(self.model['examples']))
        self.assertEqual(model['thresholds'],self.model['thresholds'])
        rows[-1]['lesson_id']=None
        with self.assertRaisesRegex(ValueError,'conflicting'):router.train(rows,self.bindings)

    def test_atomic_roundtrip_permissions_and_corruption_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'router.json';router.write_model(path,self.model)
            self.assertEqual(router.read_model(path),self.model)
            self.assertEqual(path.stat().st_mode & 0o777,0o600)
            path.write_bytes(b'['*10000+b'0'+b']'*10000)
            with self.assertRaises(ValueError):router.read_model(path)
            router.write_model(path,self.model)
            raw=json.loads(path.read_text());raw['thresholds']['score']=float('nan')
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):router.read_model(path)
            router.write_model(path,self.model);before=path.read_bytes()
            raw=copy.deepcopy(self.model);raw['bindings'][0]['program_id']='not-a-hash'
            with self.assertRaises(ValueError):router.write_model(path,raw)
            self.assertEqual(path.read_bytes(),before)

    def test_model_version_digest_and_identity_tampering_fail_closed(self):
        for field,value in (('learner_sha256','0'*64),('engine_sha256','0'*64),('sha256','0'*64)):
            model=copy.deepcopy(self.model);model[field]=value
            with self.assertRaises(ValueError):router.predict(model,'Mostrami il codice del pacco.')
        rows,bindings=training();bindings[1]['program_id']=bindings[0]['program_id']
        with self.assertRaisesRegex(ValueError,'duplicate program'):router.train(rows,bindings)


if __name__=='__main__':unittest.main()
