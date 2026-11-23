"""Replay the frozen format changes and compare a maintained Python parser."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata

from cases import INITIAL

HERE = Path(__file__).resolve().parent
CORPUS_SHA256 = "9bd048c1db127b8939468c2544f1d8e7665f579ff9a227a4e46175d513c9311b"
FIELD = re.compile(r"(?P<label>[^;\n:=]+)\s*[:=]\s*(?P<value>[0-9]{1,7})")


def fields(report):
    parts = re.split("[;\n]", report.strip())
    if len(parts) != 2:
        return {}
    parsed = {}
    for part in parts:
        match = FIELD.fullmatch(part.strip())
        if not match:
            return {}
        label = unicodedata.normalize("NFC", match["label"].strip()).casefold()
        value = int(match["value"])
        if not label or label in parsed or value > 1_000_000:
            return {}
        parsed[label] = value
    return parsed


class ConfiguredParser:
    """A Python parser whose operator declares each approved label mapping."""

    def __init__(self, mappings=None):
        self.mappings = {frozenset(INITIAL): INITIAL[0]} if mappings is None else mappings

    def assess(self, reports):
        values = set()
        for report in reports:
            data = fields(report)
            shipped = self.mappings.get(frozenset(data))
            if shipped is None:
                return {"status": "review"}
            cancelled, = data.keys() - {shipped}
            values.add((data[shipped], data[cancelled]))
        if len(values) != 1:
            return {"status": "review"}
        sent, returned = values.pop()
        return {"status": "answered", "answer": sent - returned}

    def configure(self, report, shipped_label, checks):
        data = fields(report)
        if shipped_label not in data:
            raise RuntimeError("The proposed script mapping does not match the report")
        candidate = ConfiguredParser({**self.mappings, frozenset(data): shipped_label})
        for row in checks:
            answer = candidate.assess(row["reports"]).get("answer")
            if answer != row["expected"]:
                raise RuntimeError(f"Maintained script disagrees with a check: {row['id']}")
        return candidate


def evaluate(assess, cases):
    rows, metrics = [], Counter()
    for item in cases:
        result = assess(item["reports"])
        answer = result.get("answer") if result["status"] == "answered" else None
        expected = item["expected"]
        if expected is None:
            metrics["correct_nonanswers" if answer is None else "unsafe_answers"] += 1
        elif answer is None:
            metrics["missed_answers"] += 1
        else:
            metrics["correct_answers" if answer == expected else "wrong_answers"] += 1
        if answer is None:
            metrics["needs_review_or_clarification"] += 1
        if result["status"] == "clarification":
            metrics["clarification_proposals"] += 1
        rows.append({"id": item["id"], "kind": item["kind"],
                     "expected": expected, "status": result["status"], "answer": answer})
    return {"metrics": dict(sorted(metrics.items())), "cases": rows}


def guard_cases(cases):
    return [{"reports": case["reports"], "expected": case["expected"]} for case in cases]


def qualify(airi, item, protected):
    first = airi.assess(item["arrival"]["reports"])
    if first["status"] != "clarification":
        raise RuntimeError(f"Expected a mapping question for {item['id']}: {first}")
    reports = guard_cases(item["qualification"])
    activated = airi.activate(first["proposal"], item["confirmed_shipped_label"], reports,
                              protected=guard_cases(protected))
    if activated["status"] != "activated":
        raise RuntimeError(f"Airi rejected the mapping for {item['id']}: {activated}")
    return {
        "question": first["question"],
        "observed_report": item["arrival"]["reports"][0],
        "confirmed_shipped_label": item["confirmed_shipped_label"],
        "approval": activated["status"],
        "arrival_before": first["status"],
        "labelled_reference_cases": len(reports),
        "replayed_historical_cases": activated["historical_cases"],
    }


def run(corpus):
    from repair import FormatRepair

    rounds = []
    original = ConfiguredParser()
    maintained = original
    protected = corpus["protected"]
    checks = protected[:]
    with TemporaryDirectory(prefix="airi-format-repair-") as folder:
        state = Path(folder) / "formats.sqlite"
        adaptive = FormatRepair(state)
        before = evaluate(adaptive.assess, protected)
        for index, item in enumerate(corpus["updates"]):
            receipt = qualify(adaptive, item, protected if index == 0 else [])
            adaptive = FormatRepair(state)
            checks += item["qualification"]
            maintained = maintained.configure(item["arrival"]["reports"][0],
                                             item["confirmed_shipped_label"],
                                             checks)
            arrival = adaptive.assess(item["arrival"]["reports"])
            receipt.update({
                "arrival_after": arrival["status"],
                "arrival_expected": item["arrival"]["expected"],
                "arrival_answer": arrival.get("answer"),
            })
            rounds.append({
                "id": item["id"],
                "repair": receipt,
                "final": {
                    "airi": evaluate(adaptive.assess, item["final"]),
                    "maintained_script": evaluate(maintained.assess, item["final"]),
                    "frozen_script": evaluate(original.assess, item["final"]),
                },
            })
        after = evaluate(adaptive.assess, protected)
    interventions = len(rounds)
    return {
        "corpus_sha256": CORPUS_SHA256,
        "protected": {
            "cases": len(protected),
            "before": before["metrics"],
            "after": after["metrics"],
            "changed_decisions": sum(
                left != right for left, right in zip(before["cases"], after["cases"], strict=True)
            ),
        },
        "human_work_proxy": {
            "airi_confirmed_label_meanings": interventions,
            "maintained_script_confirmed_label_meanings": interventions,
            "airi_manual_rule_entries": 0,
            "maintained_script_manual_rule_entries": interventions,
            "shared_labelled_qualification_cases": sum(
                len(item["qualification"]) for item in corpus["updates"]
            ),
        },
        "rounds": rounds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = (HERE / "CASES.json").read_bytes()
    if sha256(data).hexdigest() != CORPUS_SHA256:
        raise ValueError("The frozen corpus SHA-256 does not match")
    result = run(json.loads(data))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    totals = {}
    for update in result["rounds"]:
        for method, trial in update["final"].items():
            totals.setdefault(method, Counter()).update(trial["metrics"])
    print(json.dumps({"test_totals": totals, "protected": result["protected"],
                      "human_work_proxy": result["human_work_proxy"]}, indent=2))


if __name__ == "__main__":
    main()
