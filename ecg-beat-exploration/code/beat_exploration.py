import numpy as np

def consecutive_adjacent_mean(values, source_indices):
    values=np.asarray(values,float); source=np.asarray(source_indices,int)
    if len(values)<2: return float('nan')
    keep=np.diff(source)==1
    diffs=np.abs(np.diff(values))[keep]
    return float(np.mean(diffs)) if diffs.size else float('nan')

def patient_auc(scores, labels):
    scores=np.asarray(scores,float); labels=np.asarray(labels,int)
    pos=scores[labels==1]; neg=scores[labels==0]
    if not len(pos) or not len(neg): return float('nan')
    return float(np.mean((pos[:,None]>neg[None,:]) + 0.5*(pos[:,None]==neg[None,:])))
