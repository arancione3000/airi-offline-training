"""Verify the archived candidate evidence and record its admission decision."""
import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CANDIDATE_SHA256 = '3aa4b5e3a3eb4b41e3db810c5a16d8f5aa48c2cc352e4742e69048f398748210'


def select(summary):
    reasons = []
    init = summary['zero_initialization']
    retention = summary['retention_after_training']
    greedy = summary['sampled_greedy']
    if (summary['parameters_added'] <= 0 or init['largest_absolute_logit_difference'] != 0
            or not init['greedy_ids_identical'] or not summary['branch_roundtrip_exact']):
        reasons.append('initial expansion or checkpoint roundtrip failed')
    if retention['total'] != 897 or retention['kept'] != retention['total']:
        reasons.append('trained branch did not retain all 897 protected TRAIN certificates')
    if greedy['losses'] != 0:
        reasons.append('at least one sampled previously correct answer regressed')
    if greedy['gains'] < 1:
        reasons.append('trained branch acquired no correct answer in sampled generation')
    return {'passed_screen': not reasons, 'independent_qualification_performed': False,
            'promotion_authorized': False, 'rejection_reasons': reasons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, help='optional original local tensor artifact')
    args = parser.parse_args()
    summary = json.loads((HERE / 'SUMMARY.json').read_bytes())
    plan = json.loads((HERE / 'PLAN.json').read_bytes())
    if hashlib.sha256((HERE / 'trial.py').read_bytes()).hexdigest() != plan['script_sha256']:
        parser.error('trial source differs from the evidence plan')
    if summary['source_sha256'] != plan['source_sha256']:
        parser.error('source checkpoint differs from the evidence plan')
    if args.candidate:
        with args.candidate.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != CANDIDATE_SHA256:
            parser.error('artifact differs from the original recorded candidate')
    print(json.dumps(select(summary), indent=2))


if __name__ == '__main__':
    main()
