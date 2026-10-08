"""Offline quality comparisons; no live promotion or validation threshold changes."""
def improved(before,after):
    return (after['legacy_correct']>=before['legacy_correct']
            and after['exact_correct']>=before['exact_correct']
            and all(after['nll'][lang]<before['nll'][lang]*.995 for lang in ['it','en'])
            and after['repetition']<=max(.2,before['repetition']))

def recover_best(stagnant_rounds):
    return stagnant_rounds>=2
