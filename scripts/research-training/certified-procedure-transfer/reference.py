"""Independent integer policy oracle for the bounded research experiment."""
from decimal import Context, Decimal, ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, localcontext


POLICIES = ("floor", "truncation", "euclidean")
WORKFLOWS = (
    "quotient",
    "remainder",
    "completed_amount",
    "remaining_magnitude",
    "exact_division",
    "next_boundary",
)


def answer(policy: str, workflow: str, a: int, b: int) -> str:
    """Return the declared workflow output on signed inputs in the test domain."""
    if policy not in POLICIES or workflow not in WORKFLOWS:
        raise ValueError("unknown policy or workflow")
    if type(a) is not int or type(b) is not int or not (-64 <= a <= 64 and 0 < abs(b) <= 16):
        raise ValueError("expected a in [-64, 64] and a nonzero b in [-16, 16]")

    # Euclidean division rounds down for positive divisors and up for negative
    # divisors, ensuring that the remainder is always nonnegative.
    rounding = {
        "floor": ROUND_FLOOR,
        "truncation": ROUND_DOWN,
        "euclidean": ROUND_FLOOR if b > 0 else ROUND_CEILING,
    }[policy]
    # On this domain, nonintegral ratios are at least 1/16 from the nearest
    # integer. The precision also makes integer ratios exact.
    with localcontext(Context(prec=40)):
        ratio = Decimal(a) / Decimal(b)
        quotient = int(ratio.to_integral_value(rounding=rounding))
    if workflow == "quotient":
        return str(quotient)

    completed = b * quotient
    remainder = a - completed
    outputs = {
        "remainder": str(remainder),
        "completed_amount": str(completed),
        "remaining_magnitude": str(abs(remainder)),
        "exact_division": "true" if completed == a else "false",
        "next_boundary": str(completed + b),
    }
    return outputs[workflow]
