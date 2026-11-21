"""Transactional verification for a finite-domain graph of arithmetic procedures."""
import operator
from threading import Lock
from types import MappingProxyType

MAX_NODES = 128
MAX_INPUTS = 8192
MAX_MAGNITUDE = 10**9


def trunc_quotient(a, b):
    magnitude = abs(a) // abs(b)
    return -magnitude if (a < 0) != (b < 0) else magnitude


def euclidean_quotient(a, b):
    return a // b if b > 0 else -((-a) // b)


OPERATIONS = MappingProxyType({
    'add': (2, operator.add),
    'subtract': (2, operator.sub),
    'multiply': (2, operator.mul),
    'floor_quotient': (2, operator.floordiv),
    'trunc_quotient': (2, trunc_quotient),
    'euclidean_quotient': (2, euclidean_quotient),
    'absolute': (1, abs),
    'negate': (1, operator.neg),
    'one': (0, lambda: 1),
})
OPERANDS = ('a', 'b')


def ordered_nodes(graph):
    if type(graph) is not dict or len(graph) > MAX_NODES:
        raise ValueError('invalid graph size or type')
    nodes = {}
    for name, node in graph.items():
        if type(name) is not str or not 0 < len(name) <= 128 or name in OPERANDS:
            raise ValueError('invalid or reserved node name')
        if type(node) is not dict or set(node) != {'op', 'deps'}:
            raise ValueError(f'invalid definition: {name}')
        op, deps = node['op'], node['deps']
        if (type(op) is not str or op not in OPERATIONS or type(deps) not in (list, tuple)
                or len(deps) != OPERATIONS[op][0] or any(type(dep) is not str for dep in deps)):
            raise ValueError(f'invalid operation or arity: {name}')
        nodes[name] = (op, tuple(deps))

    ordered = {}
    visiting = set()

    def visit(name):
        if name in ordered or name in OPERANDS:
            return
        if name in visiting:
            raise ValueError(f'graph contains a cycle at: {name}')
        if name not in nodes:
            raise ValueError(f'unknown dependency: {name}')
        visiting.add(name)
        for dep in nodes[name][1]:
            visit(dep)
        visiting.remove(name)
        ordered[name] = nodes[name]

    for name in nodes:
        visit(name)
    return ordered


class Verifier:
    def __init__(self, inputs, mode='semantic'):
        if mode not in ('full', 'structural', 'semantic'):
            raise ValueError('unsupported verification mode')
        if not isinstance(inputs, (tuple, list)) or not 0 < len(inputs) <= MAX_INPUTS:
            raise ValueError('invalid input domain')
        normalized = []
        for pair in inputs:
            if (type(pair) not in (tuple, list) or len(pair) != 2
                    or any(type(x) is not int or abs(x) > MAX_MAGNITUDE for x in pair)):
                raise ValueError('invalid input pair')
            normalized.append(tuple(pair))
        if len(set(normalized)) != len(normalized):
            raise ValueError('duplicate input pair')
        self._mode = mode
        self._inputs = tuple(normalized)
        self._fixed = {name: tuple(x[i] for x in self._inputs) for i, name in enumerate(OPERANDS)}
        self._tables = dict(self._fixed)
        self._nodes = {}
        self._reference = {}
        self._version = 0
        self._lock = Lock()

    @property
    def version(self):
        with self._lock:
            return self._version

    @property
    def tables(self):
        with self._lock:
            return MappingProxyType({name: table for name, table in self._tables.items() if name not in OPERANDS})

    def _evaluate(self, op, arguments, report):
        function = OPERATIONS[op][1]
        rows = zip(*arguments, strict=True) if arguments else (() for _ in self._inputs)
        result = []
        for index, values in enumerate(rows):
            report['evaluated_cells'] += 1
            try:
                value = function(*values)
            except ZeroDivisionError:
                return None, index
            if abs(value) > MAX_MAGNITUDE:
                return None, index
            result.append(value)
        return tuple(result), None

    def update(self, graph, expected):
        with self._lock:
            return self._update(graph, expected)

    def _update(self, graph, expected):
        nodes = ordered_nodes(graph)
        if type(expected) is not dict or expected.keys() != nodes.keys():
            raise ValueError('reference must cover exactly the proposed graph')
        expected = dict(expected)
        size = len(self._inputs)
        for name, values in expected.items():
            if (type(values) is not tuple or len(values) != size
                    or (values is not self._reference.get(name) and
                        any(type(v) is not int or abs(v) > MAX_MAGNITUDE for v in values))):
                raise ValueError(f'invalid reference table: {name}')

        report = {'status': 'rejected', 'version': self._version, 'evaluated_nodes': 0,
                  'evaluated_cells': 0, 'reused_nodes': 0, 'compared_reference_cells': 0,
                  'compared_dependency_cells': 0, 'mismatch': None}
        tables = dict(self._fixed)
        dirty = set()
        for name, (op, deps) in nodes.items():
            same_node = self._nodes.get(name) == (op, deps)
            arguments = tuple(tables[dep] for dep in deps)
            reuse = False
            if same_node and self._mode == 'structural':
                reuse = dirty.isdisjoint(deps)
            elif same_node and self._mode == 'semantic':
                reuse = True
                for current, dep in zip(arguments, deps, strict=True):
                    previous = self._tables[dep]
                    if current is previous:
                        continue
                    report['compared_dependency_cells'] += size
                    if current != previous:
                        reuse = False
                        break
            reference = expected[name]
            if reuse:
                report['reused_nodes'] += 1
                result = self._tables[name]
            else:
                dirty.add(name)
                report['evaluated_nodes'] += 1
                result, failure = self._evaluate(op, arguments, report)
                if failure is not None:
                    report['mismatch'] = {'node': name, 'index': failure, 'input': self._inputs[failure],
                                          'expected': reference[failure], 'actual': None}
                    return report
            if not reuse or reference is not self._reference.get(name):
                report['compared_reference_cells'] += size
                if result != reference:
                    failure = next(i for i, (value, gold) in enumerate(zip(result, reference, strict=True))
                                   if value != gold)
                    report['mismatch'] = {'node': name, 'index': failure, 'input': self._inputs[failure],
                                          'expected': reference[failure], 'actual': result[failure]}
                    return report
            tables[name] = result

        self._tables = tables
        self._nodes = nodes
        self._reference = dict(expected)
        self._version += 1
        report.update(status='accepted', version=self._version)
        return report
