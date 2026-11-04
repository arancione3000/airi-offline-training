"""Validate this evidence package using only the Python standard library.

This verifies archived measurements; it does not run or train a model.
"""

import ast
import gzip
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
REPORT = HERE.parent / "CONTEXT_DIVERSITY_20261104.md"
manifest = json.loads((HERE / "CONTEXT_DIVERSITY_PROVENANCE_20261104.json").read_text())
summary_path = HERE / manifest["summary"]["path"]
assert hashlib.sha256(summary_path.read_bytes()).hexdigest() == manifest["summary"]["sha256"]
summary = json.loads(summary_path.read_text())
checks_meta = manifest["observed_checks_record"]
checks_raw = (HERE / checks_meta["path"]).read_bytes()
assert hashlib.sha256(checks_raw).hexdigest() == checks_meta["sha256"]
checks = json.loads(checks_raw)
assert [(r["tests"], r["skipped"], r["exit_code"]) for r in checks["checks"]] == [(71, 1, 0), (10, 0, 0)]
sources = {}

for entry in manifest["artifacts"]:
    archive = HERE / entry["archive"]
    packed = archive.read_bytes()
    assert len(packed) == entry["archive_bytes"], archive
    assert hashlib.sha256(packed).hexdigest() == entry["archive_sha256"], archive
    raw = gzip.decompress(packed)
    assert len(raw) == entry["source_bytes"], archive
    assert hashlib.sha256(raw).hexdigest() == entry["source_sha256"], archive
    source_name = Path(entry["source_path"]).name
    assert source_name not in sources
    if source_name.endswith(".json"):
        sources[source_name] = json.loads(raw)
    elif source_name.endswith(".py"):
        ast.parse(raw)
        sources[source_name] = raw.decode()
    else:
        assert source_name.endswith(".log")
        sources[source_name] = raw.decode()

assert len(sources) == 21
assert all(not Path(e["source_path"]).name.endswith(".pt") for e in manifest["artifacts"])


def stripped_metrics(metrics):
    return {k: v for k, v in metrics.items() if k != "samples"}


def generation_delta(before, after):
    old, new = before["samples"], after["samples"]
    assert len(old) == len(new) == 92
    assert sum(row["correct"] for row in new) == after["exact_correct"]
    for left, right in zip(old, new):
        for key in ("messages", "expected", "family", "language", "pair_id"):
            assert left[key] == right[key]
    context = [row for row in new if row["family"].startswith("context-")]
    assert len(context) == 32
    return {
        "lost_answers": [i for i, (x, y) in enumerate(zip(old, new)) if x["correct"] and not y["correct"]],
        "gained_answers": [i for i, (x, y) in enumerate(zip(old, new)) if not x["correct"] and y["correct"]],
        "context_correct": sum(row["correct"] for row in context),
        "context_probes": len(context),
    }


paired = sources["context-values-paired-512.json"]
interpolation = sources["diverse-interpolation-evidence.json"]
only = sources["context-only-512.json"]
assert paired["baseline"] == interpolation["baseline"] == only["baseline"]
assert paired["baseline"]["exact_correct"] == 23
assert paired["steps"] == only["steps"] == 512
assert len(paired["trials"]) == 2
assert paired["acquisition_pool"]["rows"] == 2048
assert paired["acquisition_pool"]["unique_values"] == 1868
assert paired["acquisition_pool"] == summary["paired512"]["acquisition_pool"]
assert interpolation["training_exposure"] == summary["paired512"]["exposure"]
assert [interpolation["training_exposure"][arm]["examples"] for arm in paired["treatments"]] == [256, 256]
assert [interpolation["training_exposure"][arm]["target_tokens_including_eos"] for arm in paired["treatments"]] == [1083, 1960]

for original, compact in zip(paired["trials"], summary["paired512"]["trials"]):
    delta = generation_delta(paired["baseline"], original["after"])
    assert delta == compact["recomputed_generation_delta"]
    assert delta["lost_answers"] == original["lost_answers"] == [0, 64]
    assert delta["gained_answers"] == original["gained_answers"] == []
    assert delta["context_correct"] == 0
    assert stripped_metrics(original["after"]) == compact["metrics"]
    assert original["after"]["exact_correct"] == 21
    assert original["development_gate_accepted"] is False
    assert original["native_export_logits_exact"] is True

assert [row["alpha"] for row in interpolation["interpolation_results"]] == [0.5, 0.25]
for original, compact in zip(interpolation["interpolation_results"], summary["interpolation"]["results"]):
    assert generation_delta(interpolation["baseline"], original["metrics"]) == compact["recomputed_generation_delta"]
    assert original["lost_answers"] == [0, 64]
    assert original["gained_answers"] == []
    assert original["metrics"]["exact_correct"] == 21
    assert original["development_gate_accepted"] is False

assert summary["interpolation"]["first_token_summary"] == interpolation["first_token_summary"]
for label, diagnostic in interpolation["first_token_diagnostics"].items():
    rows = diagnostic["rows"]
    compact = interpolation["first_token_summary"][label]
    assert len(rows) == compact["count"] == 32
    assert sum(row["top_equals_target"] for row in rows) == compact["first_token_top1"] == 0
    assert abs(sum(row["target_probability"] for row in rows) / 32 - compact["target_probability_mean"]) < 1e-12
    assert abs(sum(row["target_rank"] for row in rows) / 32 - compact["target_rank_mean"]) < 1e-12

original = only["trials"][0]
compact = summary["context_only512"]["trials"][0]
assert generation_delta(only["baseline"], original["after"]) == compact["recomputed_generation_delta"]
assert original["after"]["exact_correct"] == 18
assert original["lost_answers"] == [0, 4, 49, 60, 64]
assert original["gained_answers"] == []
assert original["development_gate_accepted"] is False
assert original["native_export_logits_exact"] is True
assert original["independent_frozen_teachers_unchanged"] is True
assert only["predeclared_optional_continuation"]["proceed_beyond_512"] is False
assert only["sealed_final_forward_passes"] == 0
exposure = only["treatments"]["diverse-context-only"]
assert (exposure["examples"], exposure["unique_values"], exposure["target_tokens_including_eos"]) == (1024, 737, 7489)

for name in ("context-values-paired-512.json", "context-only-512.json"):
    assert sources[name]["source_checkpoint_unchanged"] is True
    assert sources[name]["live_promoted"] is False
    assert sources[name]["heldout_training_overlap"] is False
assert interpolation["source_checkpoint_unchanged"] is True
assert interpolation["candidate_checkpoint_unchanged"] is True
assert interpolation["no_adoption"] is True
assert interpolation["sealed_final_evaluated"] is False
static = sources["diverse-static-audit.json"]
assert static["all_source_checkpoint_sha256_match"] is True
assert static["pool_reproduction"]["sha256_match"] is True
assert static["pool_reproduction"]["all_audit_values_match"] is True
assert summary["implementation"]["audit"] == sources["diverse-review-data-audit.json"]

for label in ("original", "revision2"):
    data = sources[f"context-representation-{label}.json"]
    assert data["rows"] == 64 and len(data["names"]) == 8
    assert data["all_parameter_gradients_absent"] is True
    assert data["source_checkpoint_unchanged"] is True
    final_readouts = [r for r in data["readouts"] if r["position"] == "question_tail" and r["layer"] in ("block10", "block11")]
    assert len(final_readouts) == 4 and all(r["accuracy"] == 1 for r in final_readouts)

novel = sources["novel-value-representation.json"]
assert (novel["selected_rows"], novel["selected_unique_values"]) == (512, 485)
assert novel["value_overlap_fit_test"] == []
assert novel["model_training"] is False
assert novel["source_checkpoint_unchanged"] is True
assert novel["all_parameter_gradients_absent"] is True
for language in ("it", "en"):
    for split in ("fit", "test"):
        assert len(novel["split"][language][split]["class_counts"]) == 26
    source_rows = [r for r in novel["readouts"] if r["language"] == language and r["position"] == "source_first_character_token"]
    assert len(source_rows) == 6 and all(r["accuracy"] == 1 for r in source_rows)
tails = [r for r in novel["readouts"] if r["position"] == "question_tail" and r["layer"] == "block11"]
assert [(r["language"], r["correct"], r["test_rows"]) for r in tails] == [("it", 30, 61), ("en", 42, 64)]

assert "Ran 70 tests" in sources["diverse-review-pure.log"]
assert "OK (skipped=1)" in sources["diverse-review-pure.log"]
assert "Ran 9 tests" in sources["diverse-review-native.log"]
assert "Ran 6 tests" in sources["diverse-review-prefix-integration.log"]
assert all(value is False for key, value in summary["final_decision"].items() if key in ("capability_gain_demonstrated", "model_adopted", "new_live_cycle_activated", "opt_in_curriculum_activated"))

for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", REPORT.read_text()):
    if not target.startswith(("https://", "http://", "#")):
        assert (REPORT.parent / target.split("#")[0]).exists(), target
assert (HERE / "CONTEXT_DIVERSITY_VALIDATE_20261104.py").exists()
print("PASS: 21 exact archives, observed-check provenance, JSON/scripts, SHA256, generation deltas, TRAIN readouts, test logs and report links; no model execution.")
