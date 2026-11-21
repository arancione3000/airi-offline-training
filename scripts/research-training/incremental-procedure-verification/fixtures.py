"""Signed integer workflow references and a deterministic stream of candidate edits.

The reference tables use Decimal rounding and formulae for the desired workflows.
They never inspect a proposed operation or recursively evaluate a candidate graph.
"""
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, localcontext


INPUTS = tuple((a, b) for a in range(-64, 65) for b in range(-16, 17) if b)


def _reference_row(a, b):
    with localcontext() as context:
        context.prec = 40
        ratio = Decimal(a) / Decimal(b)
        floor = int(ratio.to_integral_value(rounding=ROUND_FLOOR))
        trunc = int(ratio.to_integral_value(rounding=ROUND_DOWN))
        euclidean_rounding = ROUND_FLOOR if b > 0 else ROUND_CEILING
        euclidean = int(ratio.to_integral_value(rounding=euclidean_rounding))

    floor_amount = b * floor
    trunc_amount = b * trunc
    euclidean_amount = b * euclidean
    floor_remainder = a - floor_amount
    trunc_remainder = a - trunc_amount
    euclidean_remainder = a - euclidean_amount
    next_amount = b * (floor + 1)
    previous_amount = b * (floor - 1)
    return {
        'one_unit': 1,
        'negative_unit': -1,
        'negative_divisor': -b,
        'two_units': 2,
        'floor_q': floor,
        'trunc_q': trunc,
        'euclidean_q': euclidean,
        'completed_floor': floor_amount,
        'negative_completed_floor': -floor_amount,
        'remainder_floor': floor_remainder,
        'magnitude_floor': abs(floor_remainder),
        'incremented_q': floor + 1,
        'next_boundary': next_amount,
        'headroom': next_amount - a,
        'headroom_magnitude': abs(next_amount - a),
        'restored_dividend': a,
        'double_completed': 2 * floor_amount,
        'step_width': b,
        'decremented_q': floor - 1,
        'previous_boundary': previous_amount,
        'negative_previous_boundary': -previous_amount,
        'remaining_after_previous': a - previous_amount,
        'completed_trunc': trunc_amount,
        'remainder_trunc': trunc_remainder,
        'magnitude_trunc': abs(trunc_remainder),
        'completed_euclidean': euclidean_amount,
        'remainder_euclidean': euclidean_remainder,
        'magnitude_euclidean': euclidean_remainder,
        'floor_gap_squared': floor_remainder ** 2,
        'euclidean_gap_squared': euclidean_remainder ** 2,
        'trunc_gap_squared': trunc_remainder ** 2,
        'reconstructed_difference': a - b,
        'gap_difference': abs(floor_remainder) - euclidean_remainder,
    }


def _reference_columns():
    rows = tuple(_reference_row(a, b) for a, b in INPUTS)
    return {name: tuple(row[name] for row in rows) for name in rows[0]}


def _node(op, *dependencies):
    return {'op': op, 'deps': list(dependencies)}


def _initial_graph():
    return {
        'one_unit': _node('one'),
        'negative_unit': _node('negate', 'one_unit'),
        'negative_divisor': _node('negate', 'b'),
        'two_units': _node('add', 'one_unit', 'one_unit'),
        'direction_unit': _node('multiply', 'one_unit', 'one_unit'),
        'floor_q': _node('floor_quotient', 'a', 'b'),
        'trunc_q': _node('trunc_quotient', 'a', 'b'),
        'euclidean_q': _node('euclidean_quotient', 'a', 'b'),
        'completed_floor': _node('multiply', 'floor_q', 'b'),
        'negative_completed_floor': _node('negate', 'completed_floor'),
        'remainder_floor': _node('subtract', 'a', 'completed_floor'),
        'magnitude_floor': _node('absolute', 'remainder_floor'),
        'incremented_q': _node('add', 'floor_q', 'direction_unit'),
        'next_boundary': _node('multiply', 'incremented_q', 'b'),
        'headroom': _node('subtract', 'next_boundary', 'a'),
        'headroom_magnitude': _node('absolute', 'headroom'),
        'restored_dividend': _node('add', 'completed_floor', 'remainder_floor'),
        'double_completed': _node('add', 'completed_floor', 'completed_floor'),
        'step_width': _node('subtract', 'next_boundary', 'completed_floor'),
        'decremented_q': _node('subtract', 'floor_q', 'direction_unit'),
        'previous_boundary': _node('multiply', 'decremented_q', 'b'),
        'negative_previous_boundary': _node('negate', 'previous_boundary'),
        'remaining_after_previous': _node('subtract', 'a', 'previous_boundary'),
        'completed_trunc': _node('multiply', 'trunc_q', 'b'),
        'remainder_trunc': _node('subtract', 'a', 'completed_trunc'),
        'magnitude_trunc': _node('absolute', 'remainder_trunc'),
        'completed_euclidean': _node('multiply', 'euclidean_q', 'b'),
        'remainder_euclidean': _node('subtract', 'a', 'completed_euclidean'),
        'magnitude_euclidean': _node('absolute', 'remainder_euclidean'),
    }


def scenarios():
    """Yield (name, graph, expected-by-node integer tables, should_accept)."""
    reference = _reference_columns()
    graph = _initial_graph()

    def snapshot(name, proposed, direction=1, valid=True, overrides=None):
        columns = {**reference, 'direction_unit': (direction,) * len(INPUTS)}
        if overrides:
            columns.update(overrides)
        return (
            name,
            {node: _node(recipe['op'], *recipe['deps']) for node, recipe in proposed.items()},
            {node: columns[node] for node in proposed},
            valid,
        )

    yield snapshot('baseline', graph)

    updates = (
        ('commute_completed_product', {'completed_floor': _node('multiply', 'b', 'floor_q')}),
        ('express_negation_as_product', {'negative_completed_floor': _node('multiply', 'completed_floor', 'negative_unit')}),
        ('subtract_via_signed_sum', {'remainder_floor': _node('add', 'a', 'negative_completed_floor')}),
        ('commute_reconstruction', {'restored_dividend': _node('add', 'remainder_floor', 'completed_floor')}),
        ('combine_duplicate_terms', {'double_completed': _node('multiply', 'two_units', 'completed_floor')}),
        ('use_original_step_width', {'step_width': _node('multiply', 'one_unit', 'b')}),
        ('arithmetic_for_next_amount', {'next_boundary': _node('add', 'completed_floor', 'b')}),
        ('use_nonnegative_euclidean_remainder', {'magnitude_euclidean': _node('multiply', 'remainder_euclidean', 'one_unit')}),
        ('express_previous_negation_as_product', {'negative_previous_boundary': _node('multiply', 'previous_boundary', 'negative_unit')}),
        ('arithmetic_for_previous_amount', {'previous_boundary': _node('add', 'completed_floor', 'negative_divisor')}),
        ('previous_remainder_as_sum', {'remaining_after_previous': _node('add', 'a', 'negative_previous_boundary')}),
    )
    for name, edits in updates:
        graph = {**graph, **edits}
        yield snapshot(name, graph)
        if name == 'subtract_via_signed_sum':
            wrong = {**graph, 'floor_q': _node('trunc_quotient', 'a', 'b')}
            yield snapshot('reject_foreign_rounding_convention', wrong, valid=False)

    # A private sign convention changes; the two workflows consuming it
    # compensate. Their outputs and all public workflow tables are invariant.
    graph = {**graph,
             'direction_unit': _node('multiply', 'negative_unit', 'one_unit'),
             'incremented_q': _node('subtract', 'floor_q', 'direction_unit'),
             'decremented_q': _node('add', 'floor_q', 'direction_unit')}
    yield snapshot('switch_internal_direction_convention', graph, direction=-1)

    wrong = {**graph, 'next_boundary': _node('subtract', 'completed_floor', 'b')}
    yield snapshot('reject_reversed_step_width', wrong, direction=-1, valid=False)

    graph = {**graph,
             'floor_gap_squared': _node('multiply', 'remainder_floor', 'remainder_floor'),
             'euclidean_gap_squared': _node('multiply', 'remainder_euclidean', 'remainder_euclidean'),
             'reconstructed_difference': _node('subtract', 'restored_dividend', 'b')}
    yield snapshot('add_residue_diagnostics', graph, direction=-1)

    graph = {**graph,
             'trunc_gap_squared': _node('multiply', 'remainder_trunc', 'remainder_trunc'),
             'gap_difference': _node('subtract', 'magnitude_floor', 'magnitude_euclidean')}
    yield snapshot('add_comparison_workflows', graph, direction=-1)

    # Same definitions under a changed specification must be reverified.
    wrong_reference = tuple(-value for value in reference['gap_difference'])
    yield snapshot('reject_unimplemented_specification_change', graph, direction=-1,
                   valid=False, overrides={'gap_difference': wrong_reference})

    graph = {node: recipe for node, recipe in graph.items() if node != 'reconstructed_difference'}
    yield snapshot('retire_unused_workflow', graph, direction=-1)

    graph = {**graph, 'completed_trunc': _node('multiply', 'b', 'trunc_q'),
             'completed_floor': _node('multiply', 'floor_q', 'b')}
    yield snapshot('restore_prior_product_and_commute_truncation', graph, direction=-1)

    graph = {**graph, 'next_boundary': _node('multiply', 'incremented_q', 'b'),
             'previous_boundary': _node('multiply', 'decremented_q', 'b')}
    yield snapshot('restore_explicit_boundary_definitions', graph, direction=-1)
