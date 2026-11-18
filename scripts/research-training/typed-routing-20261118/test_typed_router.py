import unittest

from typed_router import Prototypes, accepts, entity_frame, records


class TypedInputTests(unittest.TestCase):
    def setUp(self):
        self.entities = {'saldo': 'il saldo', 'scorta': 'la scorta'}

    def test_masks_only_one_unambiguous_complete_span(self):
        self.assertEqual(entity_frame('Mostrami il saldo.', self.entities),
                         ('saldo', 'mostrami il dato'))
        for query in ('Mostrami il saldone', 'Mostrami il saldo e la scorta',
                      'Confronta il saldo con il saldo', 'Non mostrare il saldo'):
            self.assertIsNone(entity_frame(query, self.entities))

    def test_negative_evidence_is_specific_to_source_operation(self):
        rows = [
            dict(query='Somma il saldo', source_lesson_id='saldo', lesson_id='saldo'),
            dict(query='Somma la scorta', source_lesson_id='scorta', lesson_id=None),
        ]
        distinct = records(rows * 2, self.entities, {'saldo': 'add', 'scorta': 'subtract'})
        self.assertEqual(distinct, [('somma il dato', 'add', True),
                                    ('somma il dato', 'subtract', False)])


class FixedSimilarities:
    """A deterministic stand-in for a matrix of stored encoder prototypes."""
    def __init__(self, values):
        self.values = values

    def __matmul__(self, _):
        return self.values


class OperationGateTests(unittest.TestCase):
    def setUp(self):
        cases = [('dimmi il dato', 'copy', True), ('inverti il dato', 'reverse', True),
                 ('inverti il dato', 'copy', False), ('mostrami il dato', 'copy', True)]
        self.prototypes = Prototypes(cases, FixedSimilarities([.81, .93, .79]))

    def test_governing_verb_cannot_borrow_nested_speech_act(self):
        self.assertIsNone(self.prototypes.rank('smetti di mostrarmi il dato', None, 'copy'))

    def test_wrong_operation_competes_even_when_object_is_correct(self):
        rank = self.prototypes.rank('inverti il dato', None, 'copy')
        self.assertLess(rank[1], 0)
        self.assertFalse(accepts(rank, [.55, .01]))

    def test_whole_frame_is_excluded_across_conditional_labels(self):
        rank = self.prototypes.rank('inverti il dato', None, 'reverse', exclude='inverti il dato')
        self.assertIsNone(rank)

    def test_agreeing_supported_operation_is_accepted(self):
        rank = self.prototypes.rank('inverti il dato', None, 'reverse')
        self.assertTrue(accepts(rank, [.55, .01]))


if __name__ == '__main__':
    unittest.main()
