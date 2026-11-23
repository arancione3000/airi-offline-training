"""Adapt a local reconciliation rule to a new pair of quantity labels."""
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
import unicodedata


BASE_FORMAT = ('spedite', 'stornate')
MAX_FORMATS = 32
MAX_REPORTS = 16
MAX_REPORT_LENGTH = 256
MAX_CHECKS = 64
FIELD = re.compile(r'[ \t]*([^:=;\r\n]+?)\s*[:=][ \t]*([0-9]{1,7})[ \t]*')
SEPARATOR = re.compile(r';|\r?\n')
INITIAL = {'revision': 0, 'formats': [BASE_FORMAT], 'checks': []}


def _label(value):
    if not isinstance(value, str) or len(value) > 64:
        return None
    name = unicodedata.normalize('NFC', ' '.join(value.split()).casefold())
    if not 1 <= len(name) <= 48 or not any(char.isalpha() for char in name):
        return None
    if not all(char.isalpha() or char in ' _-' for char in name):
        return None
    return name


def _parse(report):
    if not isinstance(report, str) or len(report) > MAX_REPORT_LENGTH:
        return None
    pieces = SEPARATOR.split(report.strip(' \t\r\n'))
    if len(pieces) != 2:
        return None
    fields = {}
    for piece in pieces:
        match = FIELD.fullmatch(piece)
        if match is None:
            return None
        name = _label(match[1])
        value = int(match[2])
        if name is None or name in fields or value > 1_000_000:
            return None
        fields[name] = value
    return fields


def _review(reason, revision):
    return {'status': 'review', 'reason': reason, 'revision': revision}


def _assess(reports, formats, revision):
    if not isinstance(reports, list):
        return _review('malformed_report', revision)
    if not reports:
        return _review('no_reports', revision)
    if len(reports) > MAX_REPORTS:
        return _review('too_many_reports', revision)

    known = {frozenset(pair): pair for pair in formats}
    evidence = []
    for index, report in enumerate(reports):
        fields = _parse(report)
        if fields is None:
            return _review('malformed_report', revision)
        roles = known.get(frozenset(fields))
        if roles is None:
            if len(reports) > 1:
                return _review('unrecognized_format_among_multiple', revision)
            labels = sorted(fields)
            return {
                'status': 'clarification',
                'question': f'Quale campo indica le unità spedite, «{labels[0]}» o «{labels[1]}»?',
                'revision': revision,
                'proposal': {'labels': labels, 'revision': revision, 'report': report},
            }
        shipped, cancelled = roles
        evidence.append({
            'report_index': index, 'shipped_label': shipped, 'cancelled_label': cancelled,
            'shipped': fields[shipped], 'cancelled': fields[cancelled],
        })

    operands = {(row['shipped'], row['cancelled']) for row in evidence}
    if len(operands) != 1:
        return _review('conflicting_reports', revision)
    shipped, cancelled = operands.pop()
    return {'status': 'answered', 'answer': shipped - cancelled, 'revision': revision, 'sources': evidence}


def _check(row, formats, revision):
    result = _assess(row['reports'], formats, revision)
    expected = row['expected']
    if expected is None:
        return result['status'] == 'review'
    return result['status'] == 'answered' and result['answer'] == expected


def _valid_case(row):
    return (
        isinstance(row, dict)
        and set(row) == {'reports', 'expected'}
        and isinstance(row['reports'], list)
        and len(row['reports']) <= MAX_REPORTS
        and all(isinstance(report, str) and len(report) <= MAX_REPORT_LENGTH for report in row['reports'])
        and (row['expected'] is None or type(row['expected']) is int)
    )


def _qualify(rows, proposal, formats, revision):
    if not isinstance(rows, list) or not 12 <= len(rows) <= MAX_CHECKS:
        return 'qualification_size'
    labels = proposal['labels']
    first_fields = _parse(proposal['report'])
    first_pair = tuple(first_fields[label] for label in labels)
    pairs, cases = set(), set()
    negatives = 0
    cross_format_conflict = False
    for row in rows:
        if not _valid_case(row):
            return 'invalid_qualification_case'
        reports = row['reports']
        parsed = [_parse(report) for report in reports]
        case_key = frozenset(
            ('fields', tuple(sorted(fields.items()))) if fields is not None
            else ('invalid', ' '.join(report.casefold().split()))
            for report, fields in zip(reports, parsed, strict=True)
        )
        if case_key in cases:
            return 'duplicate_qualification_case'
        cases.add(case_key)
        if not _check(row, formats, revision):
            return 'qualification_mismatch'
        if row['expected'] is None:
            negatives += 1
            if _assess(reports, formats, revision).get('reason') == 'conflicting_reports':
                signatures = {frozenset(fields) for fields in parsed}
                cross_format_conflict |= len(signatures) > 1 and frozenset(labels) in signatures
        elif len(reports) == 1:
            fields = parsed[0]
            if fields is not None and set(fields) == set(labels):
                pair = tuple(fields[label] for label in labels)
                if pair != first_pair:
                    pairs.add(pair)
    if len(pairs) < 8 or negatives < 4:
        return 'insufficient_distinct_qualification'
    if not any(left != right for left, right in pairs):
        return 'ambiguous_qualification'
    if not cross_format_conflict:
        return 'missing_cross_format_conflict'
    return None


class FormatRepair:
    """Independent, bounded workflow with transactional activation and a persisted replay set."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        with closing(self._connect()) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS registry (id INTEGER PRIMARY KEY CHECK(id=1), state TEXT NOT NULL)')
            db.execute('INSERT OR IGNORE INTO registry VALUES (1, ?)', (json.dumps(INITIAL),))

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    @staticmethod
    def _load(db):
        return json.loads(db.execute('SELECT state FROM registry WHERE id=1').fetchone()[0])

    def assess(self, reports):
        with closing(self._connect()) as db:
            state = self._load(db)
        return _assess(reports, state['formats'], state['revision'])

    def activate(self, proposal, shipped_label, qualification, *, protected=None):
        # Serialize writers from validation through commit, including expected-revision checks.
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            state = self._load(db)
            revision = state['revision']
            if not isinstance(proposal, dict) or type(proposal.get('revision')) is not int:
                return {'status': 'rejected', 'reason': 'invalid_proposal', 'revision': revision}
            if proposal['revision'] != revision:
                return {'status': 'rejected', 'reason': 'stale_proposal', 'revision': revision}
            original = _assess([proposal.get('report')], state['formats'], revision)
            if original.get('proposal') != proposal:
                return {'status': 'rejected', 'reason': 'invalid_proposal', 'revision': revision}
            if len(state['formats']) >= MAX_FORMATS:
                return {'status': 'rejected', 'reason': 'format_limit', 'revision': revision}
            shipped = _label(shipped_label)
            if shipped not in proposal['labels']:
                return {'status': 'rejected', 'reason': 'invalid_mapping', 'revision': revision}
            cancelled = next(label for label in proposal['labels'] if label != shipped)
            formats = [*state['formats'], (shipped, cancelled)]
            error = _qualify(qualification, proposal, formats, revision + 1)
            protected = [] if protected is None else protected
            if error is None and (
                not isinstance(protected, list) or len(protected) > MAX_CHECKS
                or any(not _valid_case(row) or not _check(row, state['formats'], revision) for row in protected)
            ):
                error = 'invalid_protected_cases'
            if error is None and any(
                not _check(row, formats, revision + 1) for row in [*state['checks'], *protected]
            ):
                error = 'historical_regression'
            if error is not None:
                return {'status': 'rejected', 'reason': error, 'revision': revision}
            updated = {
                'revision': revision + 1, 'formats': formats,
                'checks': [*state['checks'], *protected, *qualification],
            }
            # Escapes preserve malformed Unicode in review fixtures without exposing it to SQLite's encoder.
            db.execute('UPDATE registry SET state=? WHERE id=1', (json.dumps(updated),))
            return {
                'status': 'activated', 'revision': revision + 1, 'mapping': {'shipped': shipped, 'cancelled': cancelled},
                'qualification_cases': len(qualification), 'historical_cases': len(state['checks']) + len(protected),
            }
