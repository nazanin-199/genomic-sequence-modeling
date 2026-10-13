"""CLI evaluation script: calculates test holdout metrics, differential plasticity, and bootstraps."""

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from src.data.dataset import MultiLineageGenomicDataset, collate_genomic_batch
from src.evaluation.bootstrap import paired_stratified_bootstrap_test
from src.evaluation.metrics import (
    compute_differential_plasticity_metrics,
    compute_multitask_metrics,
)
from src.models.cnn import MultiTaskGenomicCNN
from src.models.transformer import MultiTaskCNNTransformer


def main():
    parser = argparse.ArgumentParser(description="Evaluate regulatory models on holdout sets.")
    parser.add_argument("--config", default="configs/config.yaml", type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--baseline-checkpoint", default=None, type=str, help="Optional 2nd model for paired test")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu")
    df = pd.read_parquet(cfg["data"].get("target_file", "ctcf_multilineage_benchmark_500bp.parquet"))

    test_df = df[df["chrom"].isin(cfg["data"]["splits"]["test"])].reset_index(drop=True)
    test_ds = MultiLineageGenomicDataset(
        test_df["seq"].tolist(), test_df[["label_k562", "label_hepg2"]].values
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, collate_fn=collate_genomic_batch
    )

    # Instantiate model
    if cfg["model"]["type"] == "transformer":
        model = MultiTaskCNNTransformer(
            d_model=cfg["model"]["d_model"],
            nhead=cfg["model"]["nhead"],
            num_layers=cfg["model"]["num_layers"],
            dim_feedforward=cfg["model"]["dim_feedforward"],
        )
    else:
        model = MultiTaskGenomicCNN()

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            out = torch.sigmoid(model(xb.to(device)))
            all_preds.append(out.cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_targets = test_df[["label_k562", "label_hepg2"]].values

    # 1. Multi-task metrics
    mt_metrics = compute_multitask_metrics(all_targets, all_preds)
    print("\n================ TEST SET METRICS (chr21, chr22) ================")
    for k, v in mt_metrics.items():
        print(f"  {k:18s}: {v:.4f}")

    # 2. Differential plasticity metrics
    diff_metrics = compute_differential_plasticity_metrics(test_df["category"].values, all_preds)
    print("\n=========== DIFFERENTIAL REGULATORY PLASTICITY ===========")
    for k, v in diff_metrics.items():
        print(f"  {k:18s}: {v if isinstance(v, int) else f'{v:.4f}'}")

    # 3. Optional Paired Bootstrap
    if args.baseline_checkpoint:
        base_model = MultiTaskGenomicCNN().to(device)
        base_model.load_state_dict(torch.load(args.baseline_checkpoint, map_location=device))
        base_model.eval()
        base_preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                base_preds.append(torch.sigmoid(base_model(xb.to(device))).cpu().numpy())
        base_preds = np.vstack(base_preds)

        diff_mask = test_df["category"].isin(["k562_specific", "hepg2_specific"]).values
        y_diff = (test_df.loc[diff_mask, "category"] == "k562_specific").astype(int).values
        scores_a = base_preds[diff_mask, 0] - base_preds[diff_mask, 1]
        scores_b = all_preds[diff_mask, 0] - all_preds[diff_mask, 1]

        mean_gain, low, high, p_val = paired_stratified_bootstrap_test(y_diff, scores_a, scores_b)
        print("\n============= PAIRED STRATIFIED BOOTSTRAP =============")
        print(f"  Mean Gain (Model - Base): +{mean_gain:.4f}")
        print(f"  95% Confidence Interval : [{low:.4f}, {high:.4f}]")
        print(f"  Empirical p-value       : {p_val:.4f}")


if __name__ == "__main__":
    main()
