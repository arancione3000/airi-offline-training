"""Learn and replay one bounded arithmetic convention for a trusted workflow graph."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACT_SHA256 = '2fc8d30c0777520bbd0dafbea1211cdb0a6d5a7de9524f21ea08282a438900b8'
ENGINE_SHA256 = sha256(Path(__file__).read_bytes()).hexdigest()
MAX_RECEIPT_BYTES = 8192
MAX_OBSERVATIONS = 32


def encode(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':')) + '\n').encode()


def load_contract(path=HERE / 'contract.json'):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != CONTRACT_SHA256:
        raise ValueError('workflow contract differs from the pinned specification')
    return json.loads(raw)


CONTRACT = load_contract()
WORKFLOWS = tuple(CONTRACT['workflows'])
POLICIES = tuple(CONTRACT['policies'])
PROBES = tuple((a, b) for a in (-7, -1, 1, 7) for b in (-3, -2, 2, 3))


def valid_operands(a, b):
    bounds = CONTRACT['domain']
    return (type(a) is int and type(b) is int and bounds['numerator'][0] <= a <= bounds['numerator'][1]
            and bounds['divisor'][0] <= b <= bounds['divisor'][1] and b != 0)


def outputs(policy, a, b):
    if policy not in POLICIES:
        raise ValueError('policy is outside the contract')
    q, r = divmod(a, b)
    if policy == 'truncation' and r and (a < 0) != (b < 0):
        q += 1
    if policy == 'euclidean' and r < 0:
        q += 1
    product = q * b
    rest = a - product
    return {'q': str(q), 'r': str(rest), 'product': str(product), 'magnitude': str(abs(rest)),
            'exact': 'true' if rest == 0 else 'false', 'next': str(product + b)}


def predict(policy, workflow, a, b):
    return outputs(policy, a, b)[CONTRACT['workflows'][workflow]]


def partition(policies, workflow, a, b):
    buckets = {}
    for policy in policies:
        buckets.setdefault(predict(policy, workflow, a, b), []).append(policy)
    return tuple(tuple(bucket) for bucket in buckets.values())


def resolved(policies, workflows=WORKFLOWS):
    """A probe disagreement witnesses ambiguity; the trial checks completeness."""
    return tuple(workflow for workflow in workflows
                 if all(len(partition(policies, workflow, a, b)) == 1 for a, b in PROBES))


def next_probe(policies, workflows=WORKFLOWS):
    known = set(resolved(policies, workflows))
    best, score = None, None
    for workflow in workflows:
        if workflow in known:
            continue
        for a, b in PROBES:
            buckets = partition(policies, workflow, a, b)
            if len(buckets) < 2:
                continue
            # Each branch receives weight proportional to its number of candidates.
            gain = sum(len(bucket) * (len(resolved(bucket, workflows)) - len(known)) for bucket in buckets)
            worst = max(map(len, buckets))
            rank = (gain, -worst)
            if score is None or rank > score:
                best, score = (workflow, a, b), rank
    return best


def replay(observations):
    if type(observations) is not list or len(observations) > MAX_OBSERVATIONS:
        raise ValueError('invalid observations')
    candidates = POLICIES
    seen = set()
    for row in observations:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError('invalid observation row')
        workflow, a, b, answer = row
        if (type(workflow) is not str or workflow not in WORKFLOWS or not valid_operands(a, b)
                or type(answer) is not str or len(answer) > 12 or (workflow, a, b) in seen):
            raise ValueError('invalid or repeated observation')
        seen.add((workflow, a, b))
        candidates = tuple(p for p in candidates if predict(p, workflow, a, b) == answer)
        if not candidates:
            raise ValueError('observations contradict every policy in the contract')
    return candidates


@dataclass(frozen=True)
class Certified:
    policies: tuple
    fingerprint: str

    def execute(self, workflow, a, b):
        if workflow not in WORKFLOWS or not valid_operands(a, b):
            return None
        answers = {predict(policy, workflow, a, b) for policy in self.policies}
        return answers.pop() if len(answers) == 1 else None


def issue(observations):
    policy_set = replay(observations)
    evidence = {'contract_sha256': CONTRACT_SHA256, 'observations': observations}
    return {'evidence': evidence, 'engine_sha256': ENGINE_SHA256, 'policies': list(policy_set), 'schema': 1}


def evidence_digest(receipt):
    return sha256(encode(receipt['evidence'])).hexdigest()


def verify(raw, expected_evidence_sha256):
    if len(raw) > MAX_RECEIPT_BYTES:
        raise ValueError('receipt too large')
    try:
        receipt = json.loads(raw)
        evidence = receipt['evidence']
        if evidence_digest(receipt) != expected_evidence_sha256:
            raise ValueError('evidence digest differs from the independently supplied digest')
        expected = issue(evidence['observations'])
        if raw != encode(expected):
            raise ValueError('receipt differs from a replay under this contract and executor')
    except (KeyError, TypeError, RecursionError) as exc:
        raise ValueError('invalid receipt schema') from exc
    return Certified(tuple(expected['policies']), sha256(encode(expected)).hexdigest())
