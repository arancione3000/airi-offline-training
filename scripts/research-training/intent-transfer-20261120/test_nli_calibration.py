import unittest

from nli import expand_train_cases


class CalibrationTests(unittest.TestCase):
    def test_generic_wording_can_support_multiple_operations(self):
        trusted = {('request @', 'copy', True), ('request @', 'uppercase', True)}
        self.assertEqual(expand_train_cases(trusted), trusted | {
            ('request @', 'lowercase', False), ('request @', 'reverse', False)})

    def test_contradictory_observed_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'inconsistent trusted TRAIN'):
            expand_train_cases({('request @', 'add', True), ('request @', 'add', False)})
