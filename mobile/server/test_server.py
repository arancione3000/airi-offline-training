from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from server import Model, conversation


class ApiTests(unittest.TestCase):
    def test_only_alternating_dialogue_with_a_final_user(self):
        self.assertEqual(conversation(b'{"messages":[{"role":"user","content":"Ciao"}]}')[0]['content'], 'Ciao')
        for raw in (b'{"messages":[{"role":"system","content":"Ciao"}]}',
                    b'{"messages":[{"role":"user","content":"Ciao"},{"role":"assistant","content":"Ciao"}]}',
                    b'{"messages":[{"role":"user","content":" "}]}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                conversation(raw)

    def test_model_refresh_keeps_approved_runtime_on_failure_or_stale_revision(self):
        with TemporaryDirectory() as directory:
            model = Model(Path(directory))
            original, replacement = object(), object()
            model.current = (original, {'revision': 4})
            with patch('server.validate'), patch('server.Client'), patch('server.read_current') as pointer, \
                    patch('server.load_current') as loader:
                pointer.return_value = {'revision': 3}
                model.refresh()
                loader.assert_not_called()
                pointer.return_value = {'revision': 5}
                loader.side_effect = ValueError('invalid model archive')
                with self.assertRaises(ValueError):
                    model.refresh()
                self.assertEqual(model.current, (original, {'revision': 4}))
                loader.side_effect = None
                loader.return_value = (replacement, {'revision': 5})
                model.refresh()
                self.assertEqual(model.current, (replacement, {'revision': 5}))


if __name__ == '__main__':
    unittest.main()
