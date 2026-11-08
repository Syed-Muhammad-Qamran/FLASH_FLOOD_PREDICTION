"""
Model A: Cloudburst CNN-LSTM Hybrid & Logistic Baseline (model_a.py)
===================================================================
Mechanistic flash flood model for extreme precipitation / convective bursts.
Spatial Branch: 2D-CNN over 3x3 multi-point grid patch.
Temporal Branch: 1-layer LSTM over 72-hour lookback.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve
from torch import nn

# Enforced architectural constraints (§4.3)
MAX_CONV_BLOCKS = 2
MAX_CONV_FILTERS = 32
MAX_LSTM_LAYERS = 1
MAX_LSTM_UNITS = 64


class CloudburstCNN_LSTM(nn.Module):
    """
    Hybrid Neural Architecture for Cloudburst Detection:
      - Spatial Branch: 2D ConvNet over 3x3 meteorological patch (precipitation, temp, freezing lvl).
      - Temporal Branch: 1-layer LSTM over 72-hour temporal sequence.
      - Fusion Head: Concatenation -> Dense -> Sigmoid.
    """
    def __init__(
        self,
        in_spatial_channels: int = 3,
        in_temporal_features: int = 12,
        conv_filters: int = 32,
        lstm_units: int = 64,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        assert conv_filters <= MAX_CONV_FILTERS, f"Conv filters exceed limit of {MAX_CONV_FILTERS}"
        assert lstm_units <= MAX_LSTM_UNITS, f"LSTM units exceed limit of {MAX_LSTM_UNITS}"

        # 1. Spatial 2D-CNN Branch (3x3 grid)
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(in_spatial_channels, 16, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, conv_filters, kernel_size=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 2. Temporal 1-Layer LSTM Branch (72h lookback)
        self.lstm = nn.LSTM(
            input_size=in_temporal_features,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True
        )

        # 3. Dense Head
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(conv_filters + lstm_units, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x_temporal: torch.Tensor, x_spatial: torch.Tensor) -> torch.Tensor:
        """
        x_temporal: (Batch, 72, 12)
        x_spatial: (Batch, 3, 3, 3)
        returns: (Batch,) probabilities in [0.0, 1.0]
        """
        # Spatial embedding
        cnn_out = self.spatial_cnn(x_spatial).squeeze(-1).squeeze(-1)  # (Batch, 32)
        # Temporal embedding
        _, (h_n, _) = self.lstm(x_temporal)
        lstm_out = h_n[-1]  # (Batch, 64)
        # Fused prediction
        combined = torch.cat([cnn_out, lstm_out], dim=-1)
        return self.head(combined).squeeze(-1)


class LogisticRegressionBaselineA:
    """Mandatory non-deep baseline for Model A (§4.1)."""
    def __init__(self):
        self.clf = LogisticRegression(max_iter=300, random_state=42)

    def fit(self, X_temporal_mean: np.ndarray, y: np.ndarray) -> None:
        self.clf.fit(X_temporal_mean, y)

    def predict_proba(self, X_temporal_mean: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X_temporal_mean)[:, 1]


def evaluate_model_a_metrics(y_true: np.ndarray, y_prob: np.ndarray, cutoff: float = 0.50) -> dict[str, float]:
    """Computes POD, FAR, CSI, PR-AUC for Model A."""
    y_pred = (y_prob >= cutoff).astype(int)
    hits = int(np.sum((y_pred == 1) & (y_true == 1)))
    misses = int(np.sum((y_pred == 0) & (y_true == 1)))
    false_alarms = int(np.sum((y_pred == 1) & (y_true == 0)))

    pod = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0.0

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recall, precision)) if len(recall) > 1 else 0.0

    return {
        "pod": round(float(pod), 4),
        "far": round(float(far), 4),
        "csi": round(float(csi), 4),
        "pr_auc": round(float(pr_auc), 4),
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms
    }
