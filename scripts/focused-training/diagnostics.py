"""In-sample memorization is diagnostic evidence, never a heldout reward."""
def diagnostic_rows(rows):
    selected=[]
    for lang in ('it','en'):
        for family in ('foundation-copy','foundation-recall'):
            for value in ('a','b','c','d'):
                matches=[r for r in rows if r['language']==lang and r['family']==family and r['expected']==value]
                if not matches:raise ValueError('missing protected diagnostic training example')
                selected.append(matches[0])
    return selected

def diagnostic_phase(phase,updates,correct,probes):
    if probes<=0 or not 0<=correct<=probes:raise ValueError('invalid diagnostic counts')
    if correct/probes>=.75:return 2
    if phase==0 and updates>=4096 and correct/probes<.5:return 1
    return phase
