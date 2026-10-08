"""Finite GitHub runner for the frozen AIRI experiment. Never promotes a model."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / 'SPEC.json').read_text())
FINGERPRINT = SPEC['fingerprint']
TARGET = SPEC['max_steps']
LEGACY_FINGERPRINT = '9cd021f40fb592f09c2533e92ecb0b126d7af440253380d23d989069aee3d2f2'
SEED_TAG = 'airi-offline-17203-37613251656-1-1'
SOURCE_FINGERPRINTS = {}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def gh(*args):
    return subprocess.check_output(['gh', *map(str, args)], text=True).strip()


def best_effort_upload(upload, *, attempts=4, pause=time.sleep):
    """Retry non-critical telemetry without aborting model training."""
    if attempts < 1:
        raise ValueError('at least one telemetry upload attempt is required')
    for attempt in range(attempts):
        try:
            upload()
            return True
        except subprocess.CalledProcessError as error:
            if attempt + 1 < attempts:
                pause(2 ** attempt)
            else:
                print(json.dumps({'warning': 'progress upload failed; training continues',
                                  'attempts': attempts, 'returncode': error.returncode}), flush=True)
    return False


def check_tag(tag):
    if not re.fullmatch(r'airi-offline-[A-Za-z0-9._-]+', tag):
        raise ValueError('only dedicated offline experiment release tags are allowed')
    return tag


def safe_extract(archive, root):
    root = root.resolve()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            p = root / info.filename
            if not p.resolve().is_relative_to(root) or '\\' in info.filename:
                raise ValueError('unsafe archive member')
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('archive symlink refused')
        z.extractall(root)


def verify_checkpoint(path, expected_fingerprint=FINGERPRINT):
    manifest = json.loads((path / 'checkpoint.json').read_text())
    if manifest['fingerprint'] != expected_fingerprint:
        raise ValueError('checkpoint fingerprint mismatch')
    if not 17203 <= manifest['completed_steps'] <= TARGET:
        raise ValueError('checkpoint outside the authorized experiment')
    if expected_fingerprint == LEGACY_FINGERPRINT and manifest['completed_steps'] != 17203:
        raise ValueError('only the completed first-pass checkpoint can migrate')
    if not manifest['files']:
        raise ValueError('empty checkpoint')
    for name, expected in manifest['files'].items():
        p = path / name
        if not p.resolve().is_relative_to(path.resolve()) or digest(p) != expected:
            raise ValueError('checkpoint integrity failed: ' + name)
    return manifest



def assemble_archive(seed, entry):
    """Reconstruct a split seed upload, verifying every part and the original ZIP."""
    archive = seed / entry['name']
    if archive.exists():
        return archive
    parts = entry.get('parts', [])
    if not 1 <= len(parts) <= 64 or sum(x['bytes'] for x in parts) != entry['bytes']:
        raise ValueError('invalid archive parts')
    partial = archive.with_suffix(archive.suffix + '.partial')
    try:
        with partial.open('wb') as out:
            for i, item in enumerate(parts):
                if item['name'] != entry['name'] + f'.part{i:02d}':
                    raise ValueError('invalid part identity or order')
                path = seed / item['name']
                if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
                    raise ValueError('part integrity failed')
                with path.open('rb') as source:
                    shutil.copyfileobj(source, out)
        if partial.stat().st_size != entry['bytes'] or digest(partial) != entry['sha256']:
            raise ValueError('reconstructed archive digest mismatch')
        partial.replace(archive)
        return archive
    finally:
        partial.unlink(missing_ok=True)


def load_release(tag, root):
    check_tag(tag)
    seed = root / 'download'
    seed.mkdir(parents=True)
    gh('release', 'download', tag, '--repo', os.environ['GITHUB_REPOSITORY'],
       '--pattern', 'AIRI_TRAINING_*', '--dir', seed)
    indexes = list(seed.glob('AIRI_TRAINING_CHECKPOINT_*.json'))
    if len(indexes) != 1:
        raise ValueError('one committed checkpoint index required')
    index = json.loads(indexes[0].read_text())
    expected = SOURCE_FINGERPRINTS.get(tag, FINGERPRINT)
    if index['fingerprint'] != expected or len(index['archives']) != 2:
        raise ValueError('release provenance mismatch')
    step = index['completed_steps']
    names = {f'AIRI_TRAINING_{step:05d}_{part}.zip' for part in ['MODEL', 'OPTIMIZER']}
    found = set()
    for entry in index['archives']:
        name = entry.get('name') or entry.get('file_name') or Path(entry.get('local_path', '')).name
        if name not in names or name in found:
            raise ValueError('unexpected archive identity')
        archive = assemble_archive(seed, {**entry, 'name': name})
        if archive.stat().st_size != entry['bytes'] or digest(archive) != entry['sha256']:
            raise ValueError('release archive digest mismatch')
        safe_extract(archive, root / 'restored')
        found.add(name)
    checkpoint = root / 'restored/checkpoint'
    manifest = verify_checkpoint(checkpoint, expected)
    if manifest['completed_steps'] != step:
        raise ValueError('index/checkpoint step mismatch')
    return checkpoint


class ChunkComplete(BaseException):
    """Successful bounded stop after a remote checkpoint commit."""


def bounded_save(save, publish, *, started, seconds, target=TARGET, clock=time.monotonic):
    def wrapped(*args, **kwargs):
        result = save(*args, **kwargs)
        path = Path(args[5])
        step = result['completed_steps']
        # Never interrupt the final checkpoint: the frozen runner evaluates it.
        if step not in [target, SPEC['pilot_step']] and clock() - started >= seconds:
            publish(path)
            raise ChunkComplete()
        return result
    return wrapped


def progress_writer(write, report):
    reported = False
    def wrapped(path, payload):
        nonlocal reported
        write(path, payload)
        if (Path(path).name == 'PROGRESS.json' and payload.get('status') == 'training'
                and (not reported or payload['completed_steps'] % 128 == 0)):
            report(payload)
            reported = True
    return wrapped


class ReleasePublisher:
    def __init__(self, output):
        self.output = output
        self.next_release = None
        self.sequence = 0

    def release_tag(self, step):
        return check_tag(f'airi-offline-{step:05d}-{os.environ["GITHUB_RUN_ID"]}-'
                         f'{os.environ["GITHUB_RUN_ATTEMPT"]}-{self.sequence+1}')

    def publish(self, checkpoint, _unused=None, *, extra_files=()):
        manifest = verify_checkpoint(checkpoint)
        step = manifest['completed_steps']
        self.sequence += 1
        tag = check_tag(f'airi-offline-{step:05d}-{os.environ["GITHUB_RUN_ID"]}-'
                        f'{os.environ["GITHUB_RUN_ATTEMPT"]}-{self.sequence}')
        dest = self.output / tag
        dest.mkdir(parents=True)
        archives = []
        for part in ['MODEL', 'OPTIMIZER']:
            archive = dest / f'AIRI_TRAINING_{step:05d}_{part}.zip'
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
                for p in sorted(checkpoint.rglob('*')):
                    if p.is_file() and ((p.relative_to(checkpoint).parts[0] == 'model') == (part == 'MODEL')):
                        z.write(p, str(Path('checkpoint') / p.relative_to(checkpoint)))
                if part == 'OPTIMIZER':
                    for p in extra_files:
                        z.write(p, str(Path('experiment') / Path(p).name))
            archives.append({'name': archive.name, 'sha256': digest(archive), 'bytes': archive.stat().st_size})
        index = dest / f'AIRI_TRAINING_CHECKPOINT_{step:05d}.json'
        index.write_text(json.dumps({'completed_steps': step, 'fingerprint': FINGERPRINT,
                                    'archives': archives, 'production_qualified': False,
                                    'live_promoted': False}, indent=2))
        repo = os.environ['GITHUB_REPOSITORY']
        gh('release', 'create', tag, '--repo', repo, '--target', os.environ['GITHUB_SHA'],
           '--title', f'AIRI offline checkpoint {step}', '--notes',
           'Offline experiment only. No live promotion. SHA256 index committed after assets.', '--draft', '--prerelease')
        gh('release', 'upload', tag, *[dest / x['name'] for x in archives], '--repo', repo)
        gh('release', 'upload', tag, index, '--repo', repo)
        gh('release', 'edit', tag, '--repo', repo, '--draft=false', '--latest=false')
        self.next_release = tag
        # Bounded local disk use; previously committed releases remain available.
        shutil.rmtree(dest)
        return {'release': tag, 'completed_steps': step}

    def progress(self, payload, source_tag):
        check_tag(source_tag)
        path = self.output / 'status' / f'AIRI_PROGRESS_{os.environ["GITHUB_RUN_ID"]}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**payload, 'fingerprint': FINGERPRINT,
                                   'production_qualified': False, 'live_promoted': False,
                                   'checkpoint_release': self.next_release or source_tag}, indent=2))
        return best_effort_upload(
            lambda: gh('release', 'upload', source_tag, path, '--repo',
                       os.environ['GITHUB_REPOSITORY'], '--clobber'))

    def upload(self, paths):
        if not self.next_release:
            raise RuntimeError('no committed checkpoint for final reports')
        gh('release', 'upload', self.next_release, *paths, '--repo', os.environ['GITHUB_REPOSITORY'])
        return [{'status': 'succeeded'} for _ in paths]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=14400)
    args = parser.parse_args()
    if not 60 <= args.seconds <= 14400:
        raise ValueError('chunk time budget outside 1 minute–4 hours')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = load_release(args.release, root)
    if json.loads((checkpoint / 'checkpoint.json').read_text())['completed_steps'] == TARGET:
        # A final checkpoint may have been saved immediately before an interrupted evaluation.
        print('Final checkpoint: re-running only the frozen evaluation.')
    scripts = root / 'long-training'
    scripts.mkdir(exist_ok=True)
    for name, expected in SPEC['script_sha256'].items():
        if digest(HERE / name) != expected:
            raise ValueError('frozen experiment source mismatch: ' + name)
        shutil.copy2(HERE / name, scripts / name)
    shutil.copy2(HERE / 'SPEC.json', scripts / 'SPEC.json')
    data = root / 'recovered/evidence/dialogue-anchors'
    (data / 'data').mkdir(parents=True)
    for name in ['train-1024.jsonl', 'development.jsonl', 'test.jsonl']:
        shutil.copy2(root / 'restored/experiment' / name, data / 'data' / name)
    shutil.copy2(root / 'restored/experiment/CONVERSATION_TEST.json', data / 'CONVERSATION_TEST.json')
    for name in ['CORPUS_TRAIN.jsonl', 'CORPUS_HELDOUT.jsonl', 'CORPUS_MANIFEST.json', 'CORPUS_BASELINE.json']:
        source = root / 'restored/experiment' / name
        if source.exists():
            shutil.copy2(source, data / name)
        elif args.release != SEED_TAG:
            raise ValueError('committed corpus/baseline missing')
    sys.path.insert(0, str(scripts))
    run = importlib.import_module('run')
    # Only the exact completed first-pass release may cross the explicit recovery recipe boundary.
    if args.release == SEED_TAG:
        restore = run.restore_session
        def migrate_seed(path, factory, *, fingerprint):
            if fingerprint != FINGERPRINT:
                raise ValueError('FP32 recipe fingerprint mismatch')
            runtime, opt, payload = restore(path, factory, fingerprint=LEGACY_FINGERPRINT)
            if payload['counters']['completed_steps'] != 17203:
                raise ValueError('only the verified completed recovery can migrate')
            for group in opt.param_groups:
                group['lr'] = SPEC['learning_rate']
            print(json.dumps({'phase': 'explicit recovery migration', 'from_fingerprint': LEGACY_FINGERPRINT,
                              'to_fingerprint': FINGERPRINT, 'completed_steps': payload['counters']['completed_steps'],
                              'learning_rate': SPEC['learning_rate']}), flush=True)
            return runtime, opt, payload
        run.restore_session = migrate_seed
    print(json.dumps({'phase': 'starting', 'precision': SPEC['precision'],
                      'fingerprint': FINGERPRINT, 'source_release': args.release}), flush=True)
    persist = importlib.import_module('persist')
    transport = ReleasePublisher(root / 'releases')
    run.publish = persist.publish = transport.publish
    persist.upload = transport.upload
    run.atomic_json = progress_writer(run.atomic_json, lambda payload: transport.progress(payload, args.release))
    extras = [scripts / name for name in SPEC['script_sha256']] + [HERE / 'SPEC.json']
    extras += list((data / 'data').glob('*.jsonl')) + [data / 'CONVERSATION_TEST.json']
    extras += [data / name for name in ['CORPUS_TRAIN.jsonl', 'CORPUS_HELDOUT.jsonl', 'CORPUS_MANIFEST.json', 'CORPUS_BASELINE.json']]
    run.save_session = bounded_save(run.save_session,
                                    lambda path: transport.publish(path, extra_files=extras),
                                    started=time.monotonic(), seconds=args.seconds)
    sys.argv = ['run.py', '--resume', str(checkpoint), '--max-steps', str(TARGET)]
    completed = False
    try:
        run.main()
        completed = True
    except ChunkComplete:
        pass
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        output.write(f'checkpoint_release={transport.next_release}\n')
        output.write(f'completed={str(completed).lower()}\n')


if __name__ == '__main__':
    main()
