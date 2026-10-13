"""Evaluation metrics for multi-task regulatory prediction and differential plasticity."""

from typing import Dict
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_multitask_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray) -> Dict[str, float]:
    """Computes lineage-specific and macro AUROC / AUPRC on (N, 2) target matrices."""
    k562_roc = float(roc_auc_score(y_true[:, 0], y_pred_probs[:, 0]))
    k562_prc = float(average_precision_score(y_true[:, 0], y_pred_probs[:, 0]))

    hepg2_roc = float(roc_auc_score(y_true[:, 1], y_pred_probs[:, 1]))
    hepg2_prc = float(average_precision_score(y_true[:, 1], y_pred_probs[:, 1]))

    return {
        "macro_auroc": (k562_roc + hepg2_roc) / 2.0,
        "macro_auprc": (k562_prc + hepg2_prc) / 2.0,
        "k562_auroc": k562_roc,
        "k562_auprc": k562_prc,
        "hepg2_auroc": hepg2_roc,
        "hepg2_auprc": hepg2_prc,
    }


def compute_differential_plasticity_metrics(
    categories: np.ndarray, y_pred_probs: np.ndarray
) -> Dict[str, float]:
    """Isolates plastic loci ([1,0] vs [0,1]) and measures discriminative delta accuracy.
    
    Category values must contain 'k562_specific' and 'hepg2_specific'.
    """
    mask = np.isin(categories, ["k562_specific", "hepg2_specific"])
    if np.sum(mask) == 0:
        return {"diff_auroc": 0.0, "n_diff": 0}

    diff_cats = categories[mask]
    diff_probs = y_pred_probs[mask]

    # Binary label: 1 if K562-specific, 0 if HepG2-specific
    y_diff = (diff_cats == "k562_specific").astype(int)
    # Differential score: ΔScore = P(K562) - P(HepG2)
    delta_scores = diff_probs[:, 0] - diff_probs[:, 1]

    diff_auroc = float(roc_auc_score(y_diff, delta_scores))
    return {
        "diff_auroc": diff_auroc,
        "n_diff": int(np.sum(mask)),
        "n_k562_specific": int(np.sum(y_diff == 1)),
        "n_hepg2_specific": int(np.sum(y_diff == 0)),
    }
