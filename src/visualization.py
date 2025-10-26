"""
Visualization utilities for model evaluation outputs.

Currently provides helper routines to create scatter plots comparing
ground-truth values to predicted values and annotate the coefficient
of determination (R²).  The functions operate with Matplotlib in
non-interactive (Agg) mode so they can be used in headless training
and inference environments.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=C0413


def ensure_directory(path: str) -> None:
    """Create directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def plot_truth_vs_prediction(
    ground_truth,
    predictions,
    out_path: str,
    title: str = "",
    xlabel: str = "Ground Truth",
    ylabel: str = "Prediction",
) -> float:
    """Plot scatter of predictions vs ground-truth and annotate R².

    Args:
        ground_truth: array-like of reference values.
        predictions: array-like of predicted values.
        out_path: destination path for the PNG image.
        title: optional plot title.
        xlabel: label for x-axis.
        ylabel: label for y-axis.

    Returns:
        The computed coefficient of determination (R²).
    """
    gt_flat = np.asarray(ground_truth, dtype=np.float64).reshape(-1)
    pred_flat = np.asarray(predictions, dtype=np.float64).reshape(-1)

    if gt_flat.size == 0:
        raise ValueError("No data provided to plot_truth_vs_prediction.")

    ss_res = np.sum((pred_flat - gt_flat) ** 2)
    ss_tot = np.sum((gt_flat - gt_flat.mean()) ** 2)
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot

    min_val = float(np.min([gt_flat.min(), pred_flat.min()]))
    max_val = float(np.max([gt_flat.max(), pred_flat.max()]))
    if max_val == min_val:
        buffer = 1.0
    else:
        buffer = 0.05 * (max_val - min_val)
    x_vals = np.array([min_val - buffer, max_val + buffer])

    ensure_directory(os.path.dirname(out_path) or ".")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(gt_flat, pred_flat, alpha=0.35, s=18, color="#1f77b4", label="Predictions")
    ax.plot(x_vals, x_vals, color="#d62728", linestyle="--", linewidth=1.5, label="Ideal y=x")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(x_vals[0], x_vals[1])
    ax.set_ylim(x_vals[0], x_vals[1])
    ax.text(
        0.05,
        0.95,
        f"$R^2 = {r2:.4f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return float(r2)


__all__ = ["plot_truth_vs_prediction", "ensure_directory"]
