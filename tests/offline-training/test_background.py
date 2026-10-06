import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / 'scripts/offline-training'
spec = importlib.util.spec_from_file_location('background_runner', SOURCE / 'background.py')
bg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bg)


class BackgroundTests(unittest.TestCase):
    def test_frozen_recipe_and_sources_unchanged(self):
        for name, expected in bg.SPEC['script_sha256'].items():
            self.assertEqual(bg.digest(SOURCE / name), expected)
        original = {k: v for k, v in bg.SPEC.items() if k not in ['fingerprint', 'parameter_count']}
        self.assertEqual(bg.hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest(), bg.FINGERPRINT)
        self.assertEqual(bg.TARGET, 13107)

    def test_safe_archive_and_traversal_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for member in ['../escaped', '/absolute', 'checkpoint/../../escaped', 'bad\\path']:
                z = root / 'unsafe.zip'
                with zipfile.ZipFile(z, 'w') as archive:
                    archive.writestr(member, 'bad')
                with self.assertRaises(ValueError):
                    bg.safe_extract(z, root / 'restore')
            z = root / 'safe.zip'
            with zipfile.ZipFile(z, 'w') as archive:
                archive.writestr('checkpoint/file', 'okay')
            bg.safe_extract(z, root / 'restore')
            self.assertEqual((root / 'restore/checkpoint/file').read_text(), 'okay')

    def test_checkpoint_corruption_and_provenance_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'file').write_text('weights')
            payload = {'fingerprint': bg.FINGERPRINT, 'completed_steps': 1280,
                       'files': {'file': bg.digest(root / 'file')}}
            (root / 'checkpoint.json').write_text(json.dumps(payload))
            self.assertEqual(bg.verify_checkpoint(root)['completed_steps'], 1280)
            (root / 'file').write_text('corruption')
            with self.assertRaises(ValueError):
                bg.verify_checkpoint(root)
            payload['fingerprint'] = 'other'
            (root / 'checkpoint.json').write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                bg.verify_checkpoint(root)

    def test_stop_only_after_save_and_remote_commit(self):
        order = []
        def save(*args, **kwargs):
            order.append('save'); return {'completed_steps': 1408}
        def publish(path):
            order.append('publish')
        bounded = bg.bounded_save(save, publish, started=0, seconds=60, clock=lambda: 61)
        with self.assertRaises(bg.ChunkComplete):
            bounded(None, None, None, None, None, Path('checkpoint'))
        self.assertEqual(order, ['save', 'publish'])
        def failed(path):
            raise RuntimeError('upload failed')
        bounded = bg.bounded_save(save, failed, started=0, seconds=60, clock=lambda: 61)
        with self.assertRaisesRegex(RuntimeError, 'upload failed'):
            bounded(None, None, None, None, None, Path('checkpoint'))

    def test_final_checkpoint_continues_to_evaluation(self):
        bounded = bg.bounded_save(lambda *a, **k: {'completed_steps': bg.TARGET},
                                 lambda p: self.fail('final interrupted'),
                                 started=0, seconds=60, clock=lambda: 10000)
        self.assertEqual(bounded(None, None, None, None, None, Path('final'))['completed_steps'], bg.TARGET)

    def test_split_seed_reconstruction_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            content = b'original archive bytes'
            parts = [content[:7], content[7:]]
            items = []
            for i, data in enumerate(parts):
                name = f'seed.zip.part{i:02d}'
                (root / name).write_bytes(data)
                items.append({'name': name, 'bytes': len(data), 'sha256': bg.digest(root / name)})
            entry = {'name': 'seed.zip', 'bytes': len(content),
                     'sha256': bg.hashlib.sha256(content).hexdigest(), 'parts': items}
            self.assertEqual(bg.assemble_archive(root, entry).read_bytes(), content)
            (root / 'seed.zip').unlink()
            (root / items[1]['name']).write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'part integrity'):
                bg.assemble_archive(root, entry)
            self.assertFalse((root / 'seed.zip.partial').exists())

    def test_live_release_names_rejected(self):
        for tag in ['generalist-state', 'v1', 'airi-offline-../../live', 'airi-offline-x;echo bad']:
            with self.assertRaises(ValueError): bg.check_tag(tag)
        self.assertEqual(bg.check_tag('airi-offline-seed-01280'), 'airi-offline-seed-01280')

    def test_workflow_continues_only_success_and_is_finite(self):
        workflow = (SOURCE.parents[1] / '.github/workflows/airi-offline-background.yml').read_text()
        self.assertNotIn('schedule:', workflow)
        self.assertNotIn('always()', workflow)
        self.assertIn("if: steps.train.outputs.completed == 'false'", workflow)
        self.assertIn('next_count >= 64', workflow)
        self.assertIn("CHUNK_SECONDS: ${{ inputs.continuation == '0' && '60' || '14400' }}", workflow)
        self.assertIn('--seconds "$CHUNK_SECONDS"', workflow)
        self.assertIn('cancel-in-progress: false', workflow)
        self.assertIn('path: background-work/context-trial\n', workflow)
        self.assertIn("python -m pip install 'numpy>=2,<3'", workflow)
        self.assertIn('test -s background-work/context-trial/generalist-state/bootstrap-data/candidate/model.pt', workflow)
        self.assertNotIn('generalist-bootstrap.yml/dispatches', workflow)

if __name__ == '__main__': unittest.main()
