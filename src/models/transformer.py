import torch
import torch.nn as nn

class MultiTaskCNNTransformer(nn.Module):
    """Downsampling 1D Conv Stem + Multi-Head Self-Attention for Multi-Task Regulatory Prediction."""
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        max_tokens: int = 64,
        num_tasks: int = 2
    ):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=4),  # 500 -> 125
            
            nn.Conv1d(64, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)   # 125 -> 62
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, max_tokens, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_tasks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.conv_stem(x).permute(0, 2, 1)
        b, seq_len, _ = tokens.shape
        tokens = tokens + self.pos_embedding[:, :seq_len, :]
        
        trans_out = self.transformer(tokens)
        
        avg_pool = trans_out.mean(dim=1)
        max_pool = trans_out.max(dim=1)[0]
        rep = torch.cat([avg_pool, max_pool], dim=-1)
        return self.head(rep)
