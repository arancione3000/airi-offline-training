"""Experimental retention of verified data-only programs under exact task IDs."""

from contextlib import closing
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3


TASK_ID = re.compile(r"[a-zA-Z][a-zA-Z0-9_.-]{0,63}\Z", re.ASCII)
MAX_SKILLS = 64
MAX_CASES = 128
MAX_VALUE_BYTES = 16_384
MAX_STATE_BYTES = 16 * 1024 * 1024


class RegistryError(ValueError):
    """The registry is unavailable, invalid, or bound to another runtime."""


def _json_shape(value, depth=0):
    if depth > 32:
        raise ValueError("JSON nesting limit exceeded")
    kind = type(value)
    if kind in (str, int, bool, type(None)):
        return
    if kind is float and math.isfinite(value):
        return
    if kind is list:
        for item in value:
            _json_shape(item, depth + 1)
        return
    if kind is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json_shape(item, depth + 1)
        return
    raise ValueError("expected JSON data")


def _canonical(value, *, bounded=False):
    _json_shape(value)
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    if bounded and len(encoded) > MAX_VALUE_BYTES:
        raise ValueError("JSON value exceeds the byte limit")
    return encoded


def _digest(encoded):
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _case_keys(rows, minimum):
    if type(rows) is not list or not minimum <= len(rows) <= MAX_CASES:
        raise ValueError("invalid number of cases")
    keys = set()
    for row in rows:
        if type(row) is not dict or set(row) != {"input", "expected"}:
            raise ValueError("each case needs an input and an expected result")
        key = _canonical(row["input"], bounded=True)
        _canonical(row["expected"], bounded=True)
        if key in keys:
            raise ValueError("duplicate input")
        keys.add(key)
    return keys


class ProgramRegistry:
    """Store and verify append-only programs using a caller-bound interpreter.

    The trainer supplies independent expected values. The registry does not
    infer a program or interpret natural language. SQLite serialization and a
    payload digest protect transaction integrity and catch accidental edits.
    """

    def __init__(self, path, *, validate_program, execute_program, runtime_id):
        if type(runtime_id) is not str or not 1 <= len(runtime_id) <= 128:
            raise ValueError("runtime_id must be a bounded nonempty string")
        if not callable(validate_program) or not callable(execute_program):
            raise TypeError("program validator and interpreter must be callable")
        self.path = Path(path)
        self.runtime_id = runtime_id
        self.validate_program = validate_program
        self.execute_program = execute_program
        empty = _canonical({"schema": 1, "runtime_id": runtime_id, "skills": {}})
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS bounded_registry "
                "(id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL, digest TEXT NOT NULL)"
            )
            db.execute("INSERT OR IGNORE INTO bounded_registry VALUES (1, ?, ?)", (empty, _digest(empty)))
            self._load(db)

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def _load(self, db):
        row = db.execute("SELECT payload, digest FROM bounded_registry WHERE id=1").fetchone()
        if (row is None or type(row[0]) is not str or len(row[0]) > MAX_STATE_BYTES
                or _digest(row[0]) != row[1]):
            raise RegistryError("registry payload digest mismatch")
        try:
            state = json.loads(row[0])
        except ValueError as error:
            raise RegistryError("invalid registry JSON") from error
        if (type(state) is not dict or set(state) != {"schema", "runtime_id", "skills"}
                or state["schema"] != 1 or type(state["skills"]) is not dict
                or len(state["skills"]) > MAX_SKILLS):
            raise RegistryError("invalid registry schema")
        if state["runtime_id"] != self.runtime_id:
            raise RegistryError("registry belongs to a different runtime")
        return state["skills"]

    def _validate(self, program):
        if type(program) is not dict:
            raise ValueError("invalid program")
        encoded = _canonical(program, bounded=True)
        if self.validate_program(json.loads(encoded)) is not True:
            raise ValueError("invalid program")
        return encoded

    def _answer(self, program, value):
        try:
            return self.execute_program(deepcopy(program), deepcopy(value))
        except ValueError:
            return None

    def _matches(self, program, cases):
        for case in cases:
            actual = self._answer(program, case["input"])
            if _canonical(actual, bounded=True) != _canonical(case["expected"], bounded=True):
                return False
        return True

    def acquire(self, task_id, program, training, qualification, *, expected_revision=None):
        """Admit a candidate after new qualification and prior replay."""
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            skills = self._load(db)
            revision = len(skills)

            def reject(reason):
                return {"status": "rejected", "reason": reason, "revision": revision}

            if type(task_id) is not str or not TASK_ID.fullmatch(task_id):
                return reject("invalid_task_id")
            if expected_revision is not None and (
                type(expected_revision) is not int or revision != expected_revision
            ):
                return reject("stale_revision")
            try:
                encoded_program = self._validate(program)
                program = json.loads(encoded_program)
                train_keys = _case_keys(training, 2)
                qualify_keys = _case_keys(qualification, 3)
                encoded_training = _canonical(training)
                encoded_qualification = _canonical(qualification)
            except ValueError:
                return reject("invalid_candidate_or_cases")
            training = json.loads(encoded_training)
            qualification = json.loads(encoded_qualification)
            if train_keys & qualify_keys:
                return reject("overlapping_inputs")
            if sum(case["expected"] is not None for case in qualification) < 3:
                return reject("too_few_qualification_answers")

            evidence = {
                "program_sha256": _digest(encoded_program),
                "training_sha256": _digest(encoded_training),
                "qualification_sha256": _digest(encoded_qualification),
            }
            existing = skills.get(task_id)
            if existing is not None:
                provenance = existing["provenance"]
                if all(provenance[key] == value for key, value in evidence.items()):
                    return {"status": "unchanged", "revision": revision, "provenance": dict(provenance)}
                return reject("existing_task")
            if revision >= MAX_SKILLS:
                return reject("skill_limit")
            try:
                if not self._matches(program, training) or not self._matches(program, qualification):
                    return reject("candidate_mismatch")
            except ValueError:
                return reject("invalid_candidate_output")

            replayed_cases = 0
            for stored in skills.values():
                try:
                    self._validate(stored["program"])
                    if not self._matches(stored["program"], stored["qualification"]):
                        return reject("historical_regression")
                except ValueError:
                    return reject("historical_regression")
                replayed_cases += len(stored["qualification"])

            provenance = {
                **evidence, "task_id": task_id, "admitted_revision": revision + 1,
                "runtime_id": self.runtime_id,
            }
            provenance["certificate_sha256"] = _digest(_canonical(provenance))
            skills[task_id] = {"program": program, "qualification": qualification, "provenance": provenance}
            payload = _canonical({"schema": 1, "runtime_id": self.runtime_id, "skills": skills})
            if len(payload) > MAX_STATE_BYTES:
                return reject("registry_size_limit")
            db.execute("UPDATE bounded_registry SET payload=?, digest=? WHERE id=1", (payload, _digest(payload)))
            return {
                "status": "activated", "revision": revision + 1,
                "replayed_cases": replayed_cases, "provenance": dict(provenance),
            }

    def assess(self, task_id, value):
        """Execute a retained procedure addressed by its exact task ID."""
        if type(task_id) is not str or not TASK_ID.fullmatch(task_id):
            raise ValueError("invalid task ID")
        with closing(self._connect()) as db:
            skills = self._load(db)
        result = {"task_id": task_id, "revision": len(skills)}
        stored = skills.get(task_id)
        if stored is None:
            return {**result, "status": "unknown_skill"}
        try:
            if _digest(self._validate(stored["program"])) != stored["provenance"]["program_sha256"]:
                raise RegistryError("program digest mismatch")
        except (ValueError, KeyError, TypeError) as error:
            raise RegistryError("invalid stored program") from error
        _canonical(value, bounded=True)
        answer = self._answer(stored["program"], value)
        _canonical(answer, bounded=True)
        result["provenance"] = dict(stored["provenance"])
        if answer is None:
            return {**result, "status": "review", "reason": "incompatible_input"}
        return {**result, "status": "answered", "answer": answer}
