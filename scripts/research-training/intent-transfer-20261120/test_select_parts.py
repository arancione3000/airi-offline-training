import unittest

from select_parts import predict, select


FAMILIES = [{'id': 'a', 'operation': 'copy'}, {'id': 'b', 'operation': 'add'}]


def record(query_id, expected, prediction, source):
    if expected is None:
        outcome = 'negative_abstain' if prediction is None else 'negative_activation'
    elif expected == prediction:
        outcome = 'positive_correct'
    else:
        outcome = 'positive_abstain' if prediction is None else 'positive_wrong'
    return {'id': query_id, 'lesson_id': expected, 'source_lesson_id': source,
            'outcomes': [outcome, outcome], 'predictions': [prediction, prediction]}


class PolicyTests(unittest.TestCase):
    def test_competing_donors_are_selected_by_executed_operation(self):
        labels = ['a', 'a', 'b', 'b', None]
        sources = ['a', 'a', 'b', 'b', 'b']
        def examples(predicted):
            return [record(str(i), gold, output, source)
                    for i, (gold, output, source) in enumerate(zip(labels, predicted, sources, strict=True))]
        observations = {
            'approved': examples([None] * 5),
            'v6': examples(['a', 'a', 'b', 'b', 'a']),
            'parser': examples(['a', 'a', None, None, None]),
        }
        policy = select(observations, FAMILIES)
        self.assertEqual(policy['winners'], {'copy': 'parser', 'add': 'v6'})
        self.assertEqual(policy['global_winner'], 'parser')
        callbacks = {'v6': lambda q, f: 'b', 'parser': lambda q, f: 'a'}
        self.assertIsNone(predict('conflicting', FAMILIES, policy, callbacks))
        callbacks['v6'] = lambda q, f: 'a'
        self.assertEqual(predict('unique', FAMILIES, policy, callbacks), 'a')

    def test_repeated_wording_and_half_complete_pair_cannot_qualify(self):
        rows = [record('1', 'a', 'a', 'a'), record('2', 'a', 'a', 'a'), record('3', 'a', 'a', 'a')]
        rows[0]['normalized_query'] = rows[1]['normalized_query'] = 'same text'
        rows[2]['outcomes'][1] = 'positive_abstain'
        rows[2]['predictions'][1] = None
        baseline = [{**row, 'predictions': [None, None], 'outcomes': ['positive_abstain'] * 2} for row in rows]
        policy = select({'approved': baseline, 'parser': rows}, FAMILIES)
        self.assertEqual(policy['development']['parser']['copy']['correct_wordings'], 1)
        self.assertEqual(policy['winners']['copy'], 'approved')


if __name__ == '__main__':
    unittest.main()
