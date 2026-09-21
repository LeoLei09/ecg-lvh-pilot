import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beat_exploration import consecutive_adjacent_mean, patient_auc

def test_adjacent_skips_nonconsecutive_source_indices():
    values=np.array([0.1,0.2,0.9,0.4])
    source=np.array([0,1,3,4])
    assert np.isclose(consecutive_adjacent_mean(values, source), (0.1+0.5)/2)

def test_patient_auc_rank_formula():
    assert np.isclose(patient_auc(np.array([0.1,0.9,0.2,0.8]), np.array([0,1,0,1])), 1.0)
