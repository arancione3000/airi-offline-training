from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cycle import INPUTS, cycle


class SnapshotTests(unittest.TestCase):
    def reference(self, directory, name, tasks):
        path = directory / name
        raw = json.dumps({'schema': 1, 'tasks': {task: [f(a, b) for a, b in INPUTS]
                                                for task, f in tasks.items()}}).encode()
        path.write_bytes(raw)
        return path, sha256(raw).hexdigest()

    def test_restart_requalification_and_new_reference_guard(self):
        class Learner:
            def __init__(self, graph, _tables):
                self.graph = graph

            def learn(self, oracle, qualify, *, max_depth):
                recipe = {'result': {'op': 'add', 'deps': ['a', 'b']}}
                return {'accepted': qualify(recipe, 'result', {})}

        with TemporaryDirectory() as folder, patch.dict('sys.modules', {
                'synthesis_growth': SimpleNamespace(GrowingLearner=Learner)}):
            root = Path(folder)
            registry = root / 'state.json'
            path, pin = self.reference(root, 'source.json', {'sum': lambda a, b: a+b})
            self.assertEqual(cycle(registry, path, pin)['status'], 'accepted')
            snapshot = registry.read_bytes()
            self.assertEqual(cycle(registry, path, pin)['status'], 'audited')
            self.assertEqual(registry.read_bytes(), snapshot)
            changed_path, changed_pin = self.reference(root, 'changed.json', {'sum': lambda a, b: a-b})
            with self.assertRaisesRegex(ValueError, 'reference changed'):
                cycle(registry, changed_path, changed_pin)
            self.assertEqual(registry.read_bytes(), snapshot)
            altered = json.loads(snapshot)
            altered['payload']['entries'][0]['status'] = 'rejected'
            registry.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                cycle(registry, path, pin)

    def test_rejected_task_retries_only_after_a_verified_graph_change(self):
        class Learner:
            def __init__(self, graph, _tables):
                self.graph = graph

            def learn(self, oracle, qualify, *, max_depth):
                if oracle(0) == (INPUTS[0][0] + INPUTS[0][1]) ** 2:
                    dependency = next((n for n in self.graph if n.startswith('growth_')), None)
                    if dependency is None:
                        return {'accepted': False}
                    op, args = 'multiply', [dependency, dependency]
                else:
                    op, args = 'add', ['a', 'b']
                return {'accepted': qualify({'result': {'op': op, 'deps': args}}, 'result', {})}

        with TemporaryDirectory() as folder, patch.dict('sys.modules', {
                'synthesis_growth': SimpleNamespace(GrowingLearner=Learner)}):
            root = Path(folder)
            state = root / 'state.json'
            path, pin = self.reference(root, 'first.json', {'a_square': lambda a,b: (a+b)**2})
            self.assertEqual(cycle(state, path, pin)['status'], 'rejected')
            self.assertEqual(cycle(state, path, pin)['status'], 'audited')
            path, pin = self.reference(root, 'appended.json', {
                'a_square': lambda a,b: (a+b)**2, 'b_sum': lambda a,b: a+b})
            self.assertEqual(cycle(state, path, pin)['task'], 'b_sum')
            recovered = cycle(state, path, pin)
            self.assertEqual((recovered['task'], recovered['status'], recovered['approved']), ('a_square', 'accepted', 2))
            self.assertEqual(cycle(state, path, pin)['pending'], 0)


if __name__ == '__main__':
    unittest.main()
