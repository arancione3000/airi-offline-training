"""Opt-in execution of requalified arithmetic procedures on explicit requests."""
from pathlib import Path
import re
import sys
from threading import Lock

from flexible_skills import _QuestionRoutes
from skill_bank import RegistryError
import skill_programs


_TASK = re.compile(r'Esegui procedura ([a-z][a-z0-9_]{0,47})\.')
_PREFIX = 'Valori procedura: '
_FORMAT = {'operation': 'add', 'source': {'prefix': _PREFIX, 'suffix': '.', 'separator': ' e '}}


def open_verified_procedures(state_path, reference_path, pin):
    study = str(Path(__file__).resolve().parents[1] / 'research-training/procedural-capacity-growth')
    if study not in sys.path:
        sys.path.insert(0, study)
    from live_bank import VerifiedProcedures
    return VerifiedProcedures.open(state_path, reference_path, pin)


class _RefreshingBank:
    def __init__(self, state_path, reference_path, pin):
        self.state_path = Path(state_path)
        self.reference_path = reference_path
        self.pin = pin
        self.lock = Lock()
        self.signature = self._signature()
        self.bank = open_verified_procedures(self.state_path, reference_path, pin)

    def _signature(self):
        status = self.state_path.stat()
        return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns

    def snapshot(self):
        with self.lock:
            try:
                current = self._signature()
                if current != self.signature:
                    updated = open_verified_procedures(self.state_path, self.reference_path, self.pin)
                    if not set(updated.task_ids).issuperset(self.bank.task_ids):
                        raise ValueError('procedure snapshot dropped an approved task')
                    # Bind to the identity captured before opening. If a writer
                    # published concurrently, the next request checks again.
                    self.bank, self.signature = updated, current
            except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
                raise RegistryError('verified procedure snapshot is unavailable') from error
            return self.bank


def follow_verified_procedures(state_path, reference_path, pin):
    """Reload atomically replaced snapshots under the original reference pin."""
    return _RefreshingBank(state_path, reference_path, pin)


class _ProcedureRouter:
    def __init__(self, bank):
        self.bank = bank

    def predict(self, messages):
        if (not isinstance(messages, (list, tuple)) or not messages or not isinstance(messages[-1], dict)
                or not isinstance(messages[-1].get('content'), str)):
            return None
        task = _TASK.fullmatch(messages[-1]['content'])
        if task is None:
            return None
        sources = skill_programs._eligible(messages)
        matches = [(index, text) for index, text in (sources or ()) if text.startswith(_PREFIX)]
        if len(matches) != 1:
            return None
        index, text = matches[0]
        extracted = skill_programs._extract(_FORMAT, text)
        if extracted is None:
            return None
        value, operands = extracted
        name = task.group(1)
        bank = self.bank.snapshot() if isinstance(self.bank, _RefreshingBank) else self.bank
        answer = bank.execute(name, *(item['value'] for item in operands))
        if answer is None:
            return None
        return {
            'answer': answer,
            'source': 'certified_procedure',
            'operation': name,
            'certificate_sha256': bank.fingerprint,
            'provenance': {
                'message_index': index, 'role': 'user', 'start': 0, 'end': len(text), 'span': text,
                'value_start': len(_PREFIX), 'value_end': len(text) - 1, 'value_span': value,
                'operands': operands,
            },
        }


class _Priority:
    def __init__(self, first, next_router):
        self.first = first
        self.next_router = next_router

    def predict(self, messages):
        answer = self.first.predict(messages)
        return answer if answer is not None else self.next_router.predict(messages)


def with_verified_procedures(router, bank):
    if bank is None:
        return router
    procedure = _ProcedureRouter(bank)
    if isinstance(router, _QuestionRoutes):
        fallback = procedure if router.semantic is None else _Priority(router.semantic, procedure)
        return _QuestionRoutes(router.aliases, fallback)
    if callable(getattr(router, 'match', None)):
        return _QuestionRoutes(router, procedure)
    if router is None:
        return procedure
    return _Priority(router, procedure)
