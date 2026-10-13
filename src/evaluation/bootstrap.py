import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Tuple

def paired_stratified_bootstrap_test(
    y_true: np.ndarray,
    preds_model_a: np.ndarray,
    preds_model_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42
) -> Tuple[float, float, float, float]:
    """Computes paired stratified bootstrap hypothesis test between two models.
    
    Returns:
        (mean_gain, ci_low, ci_high, p_value)
    """
    np.random.seed(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    delta_aurocs = []
    
    for _ in range(n_boot):
        b_pos = np.random.choice(pos_idx, size=len(pos_idx), replace=True)
        b_neg = np.random.choice(neg_idx, size=len(neg_idx), replace=True)
        boot_idx = np.concatenate([b_pos, b_neg])
        
        auc_a = roc_auc_score(y_true[boot_idx], preds_model_a[boot_idx])
        auc_b = roc_auc_score(y_true[boot_idx], preds_model_b[boot_idx])
        delta_aurocs.append(auc_b - auc_a)
        
    delta_aurocs = np.array(delta_aurocs)
    mean_gain = float(np.mean(delta_aurocs))
    ci_low = float(np.percentile(delta_aurocs, 2.5))
    ci_high = float(np.percentile(delta_aurocs, 97.5))
    p_value = float(np.mean(delta_aurocs <= 0.0))
    
    return mean_gain, ci_low, ci_high, p_value
