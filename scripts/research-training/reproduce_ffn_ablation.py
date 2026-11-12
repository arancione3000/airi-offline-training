"""Reproduce the rejected FFN experiment in an isolated temporary source tree."""
import gzip
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE/'ffn-output-20261112'
TRANSFER = HERE.parent/'transfer-training'
EXPECTED_REPORT = '96143e2d15c13e3f28b0417f780274ad85a2146c5e953a96c05dceeab339425d'
EXPECTED_PREFIX = '520548fde4f31044956f891e7a32eaa4965c745562fe6d344f91637b7f987fe5'
EXPECTED_HARNESS = '32909f805d8e0a615a48b4e12bf2894f5434d3d7d1796188131068616b48a638'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if sys.argv[1:] == ['--verify-report']:
        data = gzip.decompress((ARCHIVE/'FFN_ABLATION_20261112.json.gz').read_bytes())
        if hashlib.sha256(data).hexdigest() != EXPECTED_REPORT:
            raise ValueError('archived raw experiment report changed')
        report = json.loads(data)
        if not report.get('complete') or report.get('live_promoted'):
            raise ValueError('unexpected experiment completion/adoption state')
        print(json.dumps(dict(raw_sha256=EXPECTED_REPORT,complete=True,
              scores=[t['after']['exact_correct'] for t in report['trials']])))
        return
    if digest(ARCHIVE/'prefix_tuning.py') != EXPECTED_PREFIX:
        raise ValueError('archived experimental implementation changed')
    if digest(HERE/'ffn_ablation.py') != EXPECTED_HARNESS:
        raise ValueError('original experiment harness changed')
    check = sys.argv[1:2] == ['--check-plasticity']
    arguments = sys.argv[2:] if check else sys.argv[1:]
    # The original harness remains byte-identical. Its relative import and
    # source hashing now target an isolated copy, never the live recipe.
    with tempfile.TemporaryDirectory(prefix='airi-ffn-research-') as directory:
        stage = Path(directory)
        transfer = stage/'scripts/transfer-training'
        transfer.mkdir(parents=True)
        for source in TRANSFER.glob('*.py'):
            shutil.copy2(source,transfer/source.name)
        shutil.copy2(TRANSFER/'SPEC.json',transfer/'SPEC.json')
        shutil.copy2(ARCHIVE/'prefix_tuning.py',transfer/'prefix_tuning.py')
        relative = ('tests/transfer-training/check_prefix_plasticity.py' if check
                    else 'scripts/research-training/ffn_ablation.py')
        target = stage/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ARCHIVE/'check_prefix_plasticity.py' if check else HERE/'ffn_ablation.py',target)
        raw = gzip.decompress((ARCHIVE/'FFN_ABLATION_20261112.json.gz').read_bytes())
        if hashlib.sha256(raw).hexdigest() != EXPECTED_REPORT:
            raise ValueError('archived raw experiment report changed')
        for original, expected in json.loads(raw)['code_files'].items():
            source = HERE/'ffn_ablation.py' if Path(original).name == 'ffn_ablation.py' else transfer/Path(original).name
            if digest(source) != expected:
                raise ValueError('reproduction source differs from the original run: '+str(source))
        sys.argv = [str(target),*arguments]
        runpy.run_path(str(target),run_name='__main__')


if __name__ == '__main__':
    main()
