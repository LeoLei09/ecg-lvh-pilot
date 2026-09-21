import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixed_analysis import build_template_contributors, feature_groups, shuffle_patient_vectors, valid_pair_mask, peak_time_seconds

def test_template_contributors_are_train_only():
    m = {1:{"role":"prototype_pool","label":"LVH"}, 2:{"role":"prototype_pool","label":"LVH"}, 3:{"role":"prototype_pool","label":"NORM"}}
    assert build_template_contributors(m, {1,3}, "LVH") == [1]

def test_step_amplitude_is_C_not_B():
    groups=feature_groups(["A_x","B_amp_V1_mean","B_template_V1_mean","C_adjacent_waveform_V1_mean","C_adjacent_template_distance_V1_mean","C_stepamp_V1_mean"])
    assert all("stepamp" not in x.lower() for x in groups["B"])
    assert "C_stepamp_V1_mean" in groups["C"]

def test_shuffle_preserves_ab_and_pair_count():
    v=np.arange(4*2).reshape(4,2)
    s=shuffle_patient_vectors(v, np.array([11,12,14,15]), np.random.default_rng(7))
    assert s.vectors.shape == v.shape
    assert len(s.pair_mask)==3 and int(s.pair_mask.sum())==2

def test_peak_time_uses_sample_position():
    assert peak_time_seconds(250,500)==0.5
