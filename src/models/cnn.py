import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskGenomicCNN(nn.Module):
    """Hierarchical 3-layer 1D CNN baseline with wide motif discovery kernels."""
    def __init__(self, num_tasks: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=19, padding=9)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(4, stride=4)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4, stride=4)
        
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.global_max = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_tasks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        
        avg_p = self.global_avg(x).squeeze(-1)
        max_p = self.global_max(x).squeeze(-1)
        rep = torch.cat([avg_p, max_p], dim=1)
        return self.fc(rep)
