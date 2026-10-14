"""1D convolutional network for raw inertial windows.

Input  : (batch, 128, 9)  - time steps x channels, the same tensor the LSTM gets
Output : (batch, 6)       - class logits (softmax is applied outside the model)

Architecture: three Conv1d blocks (conv -> batch norm -> ReLU -> [max-pool] ->
dropout) that learn local motion patterns along time, then global average
pooling over time so the classifier is independent of exact event position.
"""

from __future__ import annotations

import torch
from torch import nn

from har.config import NUM_CLASSES, CNNConfig


class CNN1D(nn.Module):
    def __init__(self, in_channels: int = 9, num_classes: int = NUM_CLASSES,
                 channels: tuple[int, ...] = (64, 128, 128), kernel_size: int = 5,
                 dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        for i, out in enumerate(channels):
            layers += [
                nn.Conv1d(prev, out, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out),
                nn.ReLU(),
            ]
            if i < len(channels) - 1:  # downsample 128 -> 64 -> 32, keep the last resolution
                layers.append(nn.MaxPool1d(2))
            layers.append(nn.Dropout(dropout))
            prev = out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T) for Conv1d
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def build_cnn(cfg: CNNConfig, in_channels: int = 9) -> CNN1D:
    return CNN1D(in_channels=in_channels, channels=tuple(cfg.channels),
                 kernel_size=cfg.kernel_size, dropout=cfg.dropout)
