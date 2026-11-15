"""Read completed artifacts only; no model import, inference, or training."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil

out = Path('/workspace/airi-lab/compositional-20261115')
repo = Path(__file__).resolve().parent
r = json.loads((out / 'REPORT.json').read_text())
design = json.loads((out / 'PREREGISTRATION.json').read_text())
assert r['complete'] and r['source_checkpoint_unchanged'] and r['training_code_unchanged']
exposure = Counter(i for batch in design['trajectory'][:r['logical_units']] for i in batch)
histogram = dict(Counter(exposure.get(i, 0) for i in range(512)))
assert histogram == {4: 436, 3: 76}
assert all(r['baseline_endpoint_only'][s]['overall']['raw_exact_with_eos'] == 0 for s in ('development', 'final'))
analysis = dict(native_neural_only=True, actual_exposures=sum(exposure.values()),
    actual_per_row_exposure_histogram=histogram, full_planned_exposures=2048,
    original92_baseline=23, original92_endpoint=r['original92']['text_exact'],
    original23_retained=23-len(r['original23_lost']), original23_lost=len(r['original23_lost']),
    original92_gained=len(r['original92_gained']), full897_evaluated=r['full897_certification_triggered'],
    original_nll_evaluated=False,
    baseline_fresh={s: r['baseline_endpoint_only'][s]['by_condition'] for s in ('development', 'final')},
    teacher_forced_native_greedy_disagreements={s: sum(x['teacher_forced_complete_with_eos'] != x['native_greedy']['raw_exact'] for x in r[s]['rows']) for s in ('train', 'development', 'final')},
    eos_emitted={s: sum(x['native_greedy']['eos_emitted'] for x in r[s]['rows']) for s in ('train', 'development', 'final')},
    run_marker_to_complete_report_wall_seconds=(out / 'REPORT.json').stat().st_mtime - (out / 'TRAINING_STARTED.json').stat().st_mtime,
    posthoc_analysis_only_no_new_model_forwards=True)
(out / 'ANALYSIS.json').write_text(json.dumps(analysis, indent=2))
text = '''# Native compositional result, 2026-11-15

The isolated candidate is rejected. It failed both predeclared novel-transfer
thresholds and lost 21 of the approved source's 23 correct old probe answers.
No production weights, runtime, activation, or deployment changed. These are
native neural metrics, separate from any explicit software memory results.

| Endpoint | Native exact literal + EOS | Both answers in counterfactual pair exact |
|---|---:|---:|
| TRAIN | 391/512 (76.37%) | 162/256 |
| Final: unseen values, familiar wording | 4/16 (25%) | 2/8 |
| Final: familiar values, new wording | 3/16 (18.75%) | 1/8 |
| Final: unseen values and new wording | 0/32 | 0/16 |

Development was 13/32: 4/8 unseen values, 6/8 new wording, and 3/16 both-new.
The frozen approved source was 0/32 development and 0/64 final, first measured
at this same endpoint. All TRAIN/development/final whole-gold teacher-forced
certificates agreed with native greedy raw-ID/EOS correctness. Final
counterfactual outputs were never identical within a pair, but only 3/32 pairs
were wholly correct; changing output with context is insufficient evidence of
reliable copying.

Training stopped at the committed wall cap after 493 of 512 planned updates,
600.543 seconds including the last indivisible update. It completed 1,972 row
exposures: 436 rows received four exposures and 76 received three. No learning
rate, data, dose, checkpoint selection, or follow-on trial was changed using
these results. The raw TRAIN CE was 0.2133. TRAIN was not fully learned, so this
bounded result does not establish an architectural or asymptotic limit.

Old native probes fell from 23/92 to 3/92. Two original successes survived and
one previously wrong probe became correct. Full 897-row certification was
skipped under the fixed potential-qualification trigger because both expert
transfer thresholds failed; the old exact-answer gate also already failed.
Old NLL gates were not evaluated, and there is no production acceptance claim.

The immutable approved tensor digest remains
`57bdb933b82bd8807af5c8b6b545c27ee9114ee4f4c59022434d39b6bf771c79`.
All pinned source/checkpoint/code hashes remained unchanged. The isolated
candidate export reload reproduced bit-exact native logits and raw greedy
IDs/EOS on six fixed TRAIN examples. Full raw answers, gold/generated IDs,
source provenance, fixed trajectory, and hashes are in the evidence artifacts.

See REPRODUCTION.md for exact design, costs, the transparent zero-update
collation repair, and the scheduled-bucket annotation limitation. Primary
condition metrics do not depend on that annotation.
'''
for dest in (out / 'RESULTS.md', repo / 'RESULTS.md'):
    dest.write_text(text)
evidence = repo / 'evidence'
evidence.mkdir(exist_ok=True)
files = ('PREREGISTRATION.json', 'ROWS.json', 'APPROVAL.json', 'PREUPDATE_REPAIR.json',
         'REPORT.json', 'SUMMARY.json', 'DATA_AUDIT.json', 'ANALYSIS.json', 'TRAINING_STARTED.json', 'RUN.log')
manifest = dict(artifacts={}, candidate_state_digest=r['candidate_state_digest'],
    candidate_files_sha256=r['candidate_files_sha256'],
    immutable_source_checkpoint=r['source_checkpoint_unchanged'], training_code_unchanged=r['training_code_unchanged'],
    production_qualified=False, live_promoted=False)
for name in files:
    source = out / name
    target = evidence / (name + '.gz')
    with target.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
            zipped.write(source.read_bytes())
    manifest['artifacts'][name] = dict(sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        gzip_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), bytes=source.stat().st_size)
    assert gzip.decompress(target.read_bytes()) == source.read_bytes()
archive = evidence / 'aborted-preupdate-start'
archive.mkdir(exist_ok=True)
manifest['aborted_zero_update_start'] = {}
for source in sorted((out / 'aborted-preupdate-start').iterdir()):
    if not source.is_file():
        continue
    target = archive / (source.name + '.gz')
    with target.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
            zipped.write(source.read_bytes())
    manifest['aborted_zero_update_start'][source.name] = dict(
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        gzip_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
for name in ('SUMMARY.json', 'DATA_AUDIT.json'):
    shutil.copy2(out / name, evidence / name)
manifest['postprocessing_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(evidence / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(analysis, indent=2))
print('Packaged', len(files), 'artifacts; isolated candidate:', r['candidate'])
