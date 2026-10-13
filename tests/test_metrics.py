import numpy as np
from src.evaluation.bootstrap import paired_stratified_bootstrap_test

def test_bootstrap_identical_models():
    y = np.array([0]*50 + [1]*50)
    preds = np.random.uniform(0, 1, 100)
    mean_gain, ci_low, ci_high, p_val = paired_stratified_bootstrap_test(y, preds, preds, n_boot=100)
    assert np.isclose(mean_gain, 0.0, atol=1e-5)
