"""Broad in-sample diagnostics, excluded from sampling and recipe decisions."""
def diagnostic_rows(rows):
    selected=[]
    from composition_data import sampling_pools
    for lang,pools in sampling_pools(rows).items():
        for key,values in sorted(pools.items()):
            # Prefer v2 examples; no fixed four-letter micro-drill.
            values=[r for r in values if 'length_bucket' in r] or values
            selected+=values[:2]
    return selected

def recipe_phase(elapsed):
    import math
    if not isinstance(elapsed,(float,int)) or not math.isfinite(elapsed) or elapsed<0:
        raise ValueError('invalid compute duration')
    return 0 if elapsed<14400 else 1 if elapsed<28800 else 2
