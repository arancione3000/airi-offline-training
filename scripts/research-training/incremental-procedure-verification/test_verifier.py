import unittest

from verifier import Verifier


INPUTS = ((-3, 2), (1, -2), (0, 1), (4, 3))


def node(op, *deps):
    return {'op': op, 'deps': list(deps)}


def recipe():
    return {
        'total': node('add', 'a', 'b'),
        'doubled': node('add', 'total', 'total'),
        'magnitude': node('absolute', 'doubled'),
        'unrelated': node('negate', 'a'),
    }


def reference():
    return {
        'total': (-1, -1, 1, 7),
        'doubled': (-2, -2, 2, 14),
        'magnitude': (2, 2, 2, 14),
        'unrelated': (3, -1, 0, -4),
    }


class VerifierTests(unittest.TestCase):
    def test_equivalent_refactor_does_not_invalidate_users_of_the_result(self):
        graph = recipe()
        expected = reference()
        verifiers = {mode: Verifier(INPUTS, mode=mode)
                     for mode in ('semantic', 'structural', 'full')}
        for verifier in verifiers.values():
            initial = verifier.update(graph, expected)
            self.assertEqual(initial['status'], 'accepted')
            self.assertEqual(initial['evaluated_cells'], len(graph) * len(INPUTS))
            self.assertEqual(initial['reused_nodes'], 0)

        refactored = {**graph, 'total': node('add', 'b', 'a')}
        revisions = {mode: verifier.update(refactored, expected)
                     for mode, verifier in verifiers.items()}
        for mode, verifier in verifiers.items():
            self.assertEqual(revisions[mode]['status'], 'accepted')
            self.assertEqual(dict(verifier.tables), expected)
        self.assertLess(revisions['semantic']['evaluated_nodes'], revisions['structural']['evaluated_nodes'])
        self.assertLess(revisions['structural']['evaluated_nodes'], revisions['full']['evaluated_nodes'])
        self.assertEqual(revisions['semantic']['reused_nodes'] + revisions['semantic']['evaluated_nodes'], len(graph))

    def test_changed_reference_is_checked_against_identical_cached_graph(self):
        verifier = Verifier(INPUTS)
        graph, expected = recipe(), reference()
        self.assertEqual(verifier.update(graph, expected)['status'], 'accepted')
        before = verifier.version

        revised_reference = {**expected, 'magnitude': (2, 2, 3, 14)}
        rejected = verifier.update(graph, revised_reference)

        self.assertEqual(rejected['status'], 'rejected')
        self.assertEqual(rejected['evaluated_nodes'], 0)
        self.assertGreater(rejected['compared_reference_cells'], 0)
        self.assertEqual(rejected['mismatch']['node'], 'magnitude')
        self.assertEqual(rejected['mismatch']['index'], 2)
        self.assertEqual(tuple(rejected['mismatch']['input']), INPUTS[2])
        self.assertEqual(rejected['mismatch']['expected'], 3)
        self.assertEqual(rejected['mismatch']['actual'], 2)
        self.assertEqual(verifier.version, before)
        self.assertEqual(dict(verifier.tables), expected)

    def test_changed_parent_outputs_force_unchanged_downstream_code_to_run(self):
        verifier = Verifier(INPUTS)
        graph, expected = recipe(), reference()
        self.assertEqual(verifier.update(graph, expected)['status'], 'accepted')
        before = verifier.version

        changed = {**graph, 'total': node('subtract', 'a', 'b')}
        new_parent = {**expected, 'total': (-5, 3, -1, 1)}
        rejected = verifier.update(changed, new_parent)
        self.assertEqual(rejected['status'], 'rejected')
        self.assertEqual(rejected['mismatch']['node'], 'doubled')
        self.assertEqual(rejected['mismatch']['actual'], -10)
        self.assertEqual(verifier.version, before)
        self.assertEqual(dict(verifier.tables), expected)

        all_revised = {**new_parent, 'doubled': (-10, 6, -2, 2), 'magnitude': (10, 6, 2, 2)}
        accepted = verifier.update(changed, all_revised)
        self.assertEqual(accepted['status'], 'accepted')
        self.assertEqual(accepted['evaluated_nodes'], 3)
        self.assertEqual(accepted['reused_nodes'], 1)
        self.assertEqual(dict(verifier.tables), all_revised)

    def test_rejected_revision_preserves_previous_tables_and_recovers(self):
        verifier = Verifier(INPUTS)
        graph, expected = recipe(), reference()
        self.assertEqual(verifier.update(graph, expected)['status'], 'accepted')
        before = verifier.version
        staged = {
            'total': node('add', 'b', 'a'),
            'doubled': node('subtract', 'total', 'total'),
            'magnitude': graph['magnitude'],
            'unrelated': node('negate', 'b'),
        }
        candidate_reference = {**expected, 'unrelated': (-2, 2, -1, -3)}
        rejected = verifier.update(staged, candidate_reference)

        self.assertEqual(rejected['status'], 'rejected')
        self.assertEqual(rejected['mismatch']['node'], 'doubled')
        self.assertEqual(rejected['mismatch']['actual'], 0)
        self.assertEqual(rejected['version'], before)
        self.assertEqual(verifier.version, before)
        self.assertEqual(dict(verifier.tables), expected)

        repaired = {**staged, 'doubled': graph['doubled']}
        accepted = verifier.update(repaired, candidate_reference)
        self.assertEqual(accepted['status'], 'accepted')
        self.assertEqual(accepted['version'], before + 1)
        self.assertEqual(dict(verifier.tables), candidate_reference)
        with self.assertRaises(TypeError):
            verifier.tables['doubled'] = ()

    def test_graph_order_and_deletion(self):
        verifier = Verifier(INPUTS)
        graph, expected = recipe(), reference()
        reversed_graph = dict(reversed(tuple(graph.items())))
        self.assertEqual(verifier.update(reversed_graph, expected)['status'], 'accepted')
        before = verifier.version

        del reversed_graph['unrelated']
        del expected['unrelated']
        result = verifier.update(reversed_graph, expected)

        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(result['version'], before + 1)
        self.assertEqual(result['evaluated_nodes'], 0)
        self.assertEqual(result['reused_nodes'], len(expected))
        self.assertNotIn('unrelated', verifier.tables)

    def test_rejects_cycles_unknown_dependencies_and_wrong_arity(self):
        verifier = Verifier(INPUTS)
        graph, expected = recipe(), reference()
        self.assertEqual(verifier.update(graph, expected)['status'], 'accepted')
        before = verifier.version
        invalid = (
            {**graph, 'total': node('add', 'magnitude', 'b')},
            {**graph, 'total': node('add', 'absent', 'b')},
            {**graph, 'total': node('add', 'a')},
            {**graph, 'total': node('unrecognized', 'a', 'b')},
        )
        for staged in invalid:
            with self.subTest(total=staged['total']):
                with self.assertRaises(ValueError):
                    verifier.update(staged, expected)
                self.assertEqual(verifier.version, before)
                self.assertEqual(dict(verifier.tables), expected)

    def test_division_by_zero_inside_a_candidate_rejects_transaction(self):
        verifier = Verifier(INPUTS)
        graph, expected = recipe(), reference()
        self.assertEqual(verifier.update(graph, expected)['status'], 'accepted')
        before = verifier.version
        staged = {
            **graph,
            'zero': node('subtract', 'a', 'a'),
            'quotient': node('floor_quotient', 'a', 'zero'),
        }
        rejected = verifier.update(staged, {**expected, 'zero': (0,) * 4, 'quotient': (0,) * 4})
        self.assertEqual(rejected['status'], 'rejected')
        self.assertEqual(rejected['mismatch']['node'], 'quotient')
        self.assertEqual(tuple(rejected['mismatch']['input']), INPUTS[0])
        self.assertIsNone(rejected['mismatch']['actual'])
        self.assertEqual(verifier.version, before)
        self.assertEqual(dict(verifier.tables), expected)

    def test_conventions_agree_only_on_their_actual_finite_results(self):
        graph = {
            'floor': node('floor_quotient', 'a', 'b'),
            'truncation': node('trunc_quotient', 'a', 'b'),
            'euclidean': node('euclidean_quotient', 'a', 'b'),
        }
        expected = {
            'floor': (-3, 2, -3),
            'truncation': (-2, 2, -2),
            'euclidean': (-3, 3, -2),
        }
        verifier = Verifier(((-7, 3), (-7, -3), (7, -3)))
        self.assertEqual(verifier.update(graph, expected)['status'], 'accepted')
        self.assertEqual(dict(verifier.tables), expected)


if __name__ == '__main__':
    unittest.main()
