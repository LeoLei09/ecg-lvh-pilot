from dataclasses import dataclass
import numpy as np

def build_template_contributors(meta, train_ids, label): return sorted([int(e) for e,v in meta.items() if int(e) in train_ids and v['role']=='prototype_pool' and v['label']==label])
def feature_groups(cols):
 cols=list(cols); return {'A':[c for c in cols if c.startswith('A_')], 'B':[c for c in cols if c.startswith('B_') and 'stepamp' not in c.lower()], 'C':[c for c in cols if c.startswith('C_') or 'stepamp' in c.lower()]}
@dataclass
class ShuffleResult: vectors: np.ndarray; source_indices: np.ndarray; pair_mask: np.ndarray
def shuffle_patient_vectors(vectors, source_indices, rng):
 o=rng.permutation(len(vectors)); return ShuffleResult(np.asarray(vectors)[o],np.asarray(source_indices).copy(),np.diff(np.asarray(source_indices))==1)
def valid_pair_mask(source_indices): return np.diff(np.asarray(source_indices))==1
def peak_time_seconds(sample, fs): return float(sample)/float(fs)

