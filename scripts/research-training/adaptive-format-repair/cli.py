"""Read reports and approve a proposed format from local JSON files."""

import argparse
import json
import sqlite3
from pathlib import Path

from repair import FormatRepair


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True, help="SQLite registry path")
    commands = parser.add_subparsers(dest="command", required=True)
    assess = commands.add_parser("assess", help="assess reports or request clarification")
    assess.add_argument("reports", type=Path, help="JSON array of report strings")
    activate = commands.add_parser("activate", help="verify and activate a clarified format")
    activate.add_argument("assessment", type=Path, help="JSON clarification returned by assess")
    activate.add_argument("shipped_label", help="field giving the units shipped, confirmed by the operator")
    activate.add_argument("qualification", type=Path, help="JSON array of independently labeled checks")
    activate.add_argument("--protected", type=Path, help="extra labeled checks for the existing version")
    args = parser.parse_args(argv)

    try:
        if args.command == "assess":
            reports = read_json(args.reports)
            if not isinstance(reports, list) or any(not isinstance(report, str) for report in reports):
                parser.error("reports must be a JSON array of strings")
            result = FormatRepair(args.state).assess(reports)
        else:
            assessment = read_json(args.assessment)
            if (not isinstance(assessment, dict) or assessment.get("status") != "clarification"
                    or not isinstance(assessment.get("proposal"), dict)):
                parser.error("assessment must contain a clarification proposal returned by assess")
            qualification = read_json(args.qualification)
            if not isinstance(qualification, list):
                parser.error("qualification must be a JSON array of labeled checks")
            protected = read_json(args.protected) if args.protected else None
            result = FormatRepair(args.state).activate(
                assessment["proposal"], args.shipped_label, qualification, protected=protected
            )
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.error(str(error))

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["status"] == "rejected" else 0


if __name__ == "__main__":
    raise SystemExit(main())
