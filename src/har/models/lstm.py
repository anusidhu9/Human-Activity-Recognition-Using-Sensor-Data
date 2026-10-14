"""LSTM for raw inertial windows.

Input  : (batch, 128, 9)  - the same tensor the CNN gets
Output : (batch, 6)       - class logits

The window is read step by step (50 Hz samples); the recurrent state carries
the motion history. The sequence summary fed to the classifier is either the
final hidden state (``pool="last"``) or the mean of all hidden states
(``pool="mean"``).
"""

from __future__ import annotations

import torch
from torch import nn

from har.config import NUM_CLASSES, LSTMConfig


class LSTMClassifier(nn.Module):
    def __init__(self, in_channels: int = 9, num_classes: int = NUM_CLASSES,
                 hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = False, pool: str = "last"):
        super().__init__()
        if pool not in ("last", "mean"):
            raise ValueError("pool must be 'last' or 'mean'")
        self.pool = pool
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(in_channels, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=bidirectional)
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, (h_n, _) = self.lstm(x)  # outputs: (B, T, H*dirs), h_n: (layers*dirs, B, H)
        if self.pool == "mean":
            summary = outputs.mean(dim=1)
        elif self.bidirectional:  # last layer, forward and backward directions
            summary = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            summary = h_n[-1]
        return self.classifier(self.dropout(summary))


def build_lstm(cfg: LSTMConfig, in_channels: int = 9) -> LSTMClassifier:
    return LSTMClassifier(in_channels=in_channels, hidden_size=cfg.hidden_size,
                          num_layers=cfg.num_layers, dropout=cfg.dropout,
                          bidirectional=cfg.bidirectional, pool=cfg.pool)
