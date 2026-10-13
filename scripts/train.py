import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from src.data.dataset import MultiLineageGenomicDataset, collate_genomic_batch
from src.models.cnn import MultiTaskGenomicCNN
from src.models.transformer import MultiTaskCNNTransformer
from sklearn.metrics import roc_auc_score

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml", type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['experiment']['device'] if torch.cuda.is_available() else 'cpu')
    df = pd.read_parquet("ctcf_multilineage_benchmark_500bp.parquet")

    train_df = df[df['chrom'].isin(cfg['data']['splits']['train'])]
    val_df = df[df['chrom'].isin(cfg['data']['splits']['val'])]

    train_ds = MultiLineageGenomicDataset(train_df['seq'].tolist(), train_df[['label_k562', 'label_hepg2']].values)
    val_ds = MultiLineageGenomicDataset(val_df['seq'].tolist(), val_df[['label_k562', 'label_hepg2']].values)

    train_loader = DataLoader(train_ds, batch_size=cfg['data']['batch_size'], shuffle=True, collate_fn=collate_genomic_batch)
    val_loader = DataLoader(val_ds, batch_size=cfg['data']['batch_size'], shuffle=False, collate_fn=collate_genomic_batch)

    model = MultiTaskCNNTransformer(
        d_model=cfg['model']['d_model'],
        nhead=cfg['model']['nhead'],
        num_layers=cfg['model']['num_layers'],
        dim_feedforward=cfg['model']['dim_feedforward']
    ).to(device) if cfg['model']['type'] == 'transformer' else MultiTaskGenomicCNN().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()

    print(f"Training {cfg['model']['type']} on {device}...")
    for epoch in range(1, cfg['training']['epochs'] + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = torch.sigmoid(model(xb.to(device)))
                val_preds.append(out.cpu().numpy())
                val_targets.append(yb.numpy())
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        roc = (roc_auc_score(val_targets[:, 0], val_preds[:, 0]) + roc_auc_score(val_targets[:, 1], val_preds[:, 1])) / 2.0
        print(f"Epoch {epoch:02d} | Train Loss: {total_loss / len(train_ds):.4f} | Val Macro AUROC: {roc:.4f}")

    torch.save(model.state_dict(), f"checkpoints/{cfg['model']['type']}_best.pt")
    print("Model saved to checkpoints/")

if __name__ == "__main__":
    import numpy as np
    train()
