"""
Model definitions for temperature forecasting.

This module provides PyTorch models for several neural network
architectures: Multi‑Layer Perceptron (MLP), one‑dimensional
Convolutional Neural Network (CNN1D), Long Short‑Term Memory (LSTM),
and a placeholder for Extended LSTM (XLSTM).  Each model exposes
forward() that takes a batch of sequences and produces a batch of
temperature predictions.
"""

import torch
from torch import nn


class MLPModel(nn.Module):
    """Feed‑forward neural network for next‑step prediction.

    The input sequences are flattened into a single vector per sample.  A
    configurable number of hidden layers with ReLU activation can be
    specified via the hidden_dims list.  Dropout is applied between
    layers if specified.
    """

    def __init__(self, seq_len: int, input_dim: int, output_dim: int,
                 hidden_dims: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.flatten_dim = seq_len * input_dim
        layers = []
        in_dim = self.flatten_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        out = self.net(x)
        return out


class CNN1DModel(nn.Module):
    """One‑dimensional CNN model for sequence to one‑step prediction.

    This model applies convolution along the temporal axis.  The input
    tensor shape should be (batch, seq_len, input_dim).  We transpose
    it to (batch, input_dim, seq_len) to use as the channel dimension
    for conv1d.  The output of the conv layers is flattened and passed
    through fully connected layers to produce the final output.
    """

    def __init__(self, seq_len: int, input_dim: int, output_dim: int,
                 num_filters: list[int], kernel_sizes: list[int],
                 fc_dims: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        assert len(num_filters) == len(kernel_sizes), "num_filters and kernel_sizes must be same length"
        conv_layers = []
        in_channels = input_dim
        current_seq_len = seq_len
        for nf, ks in zip(num_filters, kernel_sizes):
            conv_layers.append(nn.Conv1d(in_channels, nf, kernel_size=ks))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = nf
            current_seq_len = (current_seq_len - ks + 1) // 2  # approximate output length after conv + pool
        self.conv = nn.Sequential(*conv_layers)
        # compute flattened dimension
        self.flatten_dim = in_channels * max(1, current_seq_len)
        fc_layers = []
        in_dim = self.flatten_dim
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(in_dim, fc_dim))
            fc_layers.append(nn.ReLU())
            if dropout > 0.0:
                fc_layers.append(nn.Dropout(dropout))
            in_dim = fc_dim
        fc_layers.append(nn.Linear(in_dim, output_dim))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        out = self.fc(x)
        return out


class LSTMModel(nn.Module):
    """Multi‑layer LSTM for sequence to next‑step prediction."""

    def __init__(self, input_dim: int, hidden_dim: int,
                 num_layers: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        # take last layer's hidden state
        last_hidden = h_n[-1]
        out = self.fc(last_hidden)
        return out


class XLSTMModel(nn.Module):
    """Placeholder for an Extended LSTM (xLSTM) model.

    This implementation reuses the standard LSTM from PyTorch but
    exposes a similar interface so that it can be replaced with a
    full xLSTM implementation later.  See instructions.md for guidance
    on extending this class to implement the exponential gating and
    modified memory structures described in the xLSTM literature.
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 num_layers: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.fc(last_hidden)
        return out


def build_model(model_cfg: dict, input_dim: int, output_dim: int, seq_len: int) -> nn.Module:
    """Factory function to instantiate a model based on configuration."""
    model_type = model_cfg["type"].lower()
    if model_type == "mlp":
        hidden_dims = model_cfg.get("hidden_dims", [64, 64])
        dropout = model_cfg.get("dropout", 0.0)
        return MLPModel(seq_len=seq_len, input_dim=input_dim,
                        output_dim=output_dim, hidden_dims=hidden_dims,
                        dropout=dropout)
    elif model_type == "cnn1d":
        num_filters = model_cfg.get("num_filters", [32, 64])
        kernel_sizes = model_cfg.get("kernel_sizes", [3, 3])
        fc_dims = model_cfg.get("fc_dims", [64])
        dropout = model_cfg.get("dropout", 0.0)
        return CNN1DModel(seq_len=seq_len, input_dim=input_dim,
                          output_dim=output_dim, num_filters=num_filters,
                          kernel_sizes=kernel_sizes, fc_dims=fc_dims,
                          dropout=dropout)
    elif model_type == "lstm":
        hidden_dim = model_cfg.get("hidden_dim", 64)
        num_layers = model_cfg.get("num_layers", 2)
        dropout = model_cfg.get("dropout", 0.0)
        return LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim,
                         num_layers=num_layers, output_dim=output_dim,
                         dropout=dropout)
    elif model_type == "xlstm":
        hidden_dim = model_cfg.get("hidden_dim", 64)
        num_layers = model_cfg.get("num_layers", 2)
        dropout = model_cfg.get("dropout", 0.0)
        return XLSTMModel(input_dim=input_dim, hidden_dim=hidden_dim,
                          num_layers=num_layers, output_dim=output_dim,
                          dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")