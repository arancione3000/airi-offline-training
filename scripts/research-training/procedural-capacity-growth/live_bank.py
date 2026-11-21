"""Read-only execution of requalified procedural growth snapshots."""
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from cycle import INPUTS, digest, engine_fingerprint, requalify, restore_state, trusted_tasks

_INPUT_INDEX = MappingProxyType({point: index for index, point in enumerate(INPUTS)})
_DOMAIN_SHA256 = digest(INPUTS)


@dataclass(frozen=True, slots=True)
class VerifiedProcedures:
    """An immutable, requalified snapshot. Reopen explicitly to pick up changes."""

    fingerprint: str
    _outputs: Mapping[str, tuple[int, ...]]

    @classmethod
    def open(cls, state_path: Path, reference_path: Path, pin: str) -> 'VerifiedProcedures':
        state_path, reference_path = Path(state_path), Path(reference_path)
        if state_path.resolve() == reference_path.resolve():
            raise ValueError('state and reference require separate paths')
        for path in (state_path, reference_path):
            if not path.is_file():
                raise FileNotFoundError(f'regular snapshot file required: {path}')

        engine_id = engine_fingerprint()
        try:
            tasks = trusted_tasks(reference_path, pin)
            state = restore_state(state_path, engine_id, required=True)
            engine, graph, approved, _ = requalify(state, tasks)
        except RecursionError as error:
            raise ValueError('procedure snapshot is nested too deeply') from error

        committed = engine.tables
        outputs = {name: committed[node] for name, node in sorted(approved.items())}
        fingerprint = digest({
            'engine_sha256': engine_id,
            'domain_sha256': _DOMAIN_SHA256,
            'graph': graph,
            'endpoints': {name: [approved[name], digest(values)] for name, values in outputs.items()},
        })
        return cls(fingerprint, MappingProxyType(outputs))

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(self._outputs)

    def execute(self, task: str, a: int, b: int) -> str | None:
        if type(task) is not str or type(a) is not int or type(b) is not int:
            return None
        outputs = self._outputs.get(task)
        index = _INPUT_INDEX.get((a, b))
        if outputs is None or index is None:
            return None
        return str(outputs[index])
