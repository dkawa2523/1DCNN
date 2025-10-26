"""
Visualization utilities for model evaluation outputs.

Provides helper routines to create scatter plots comparing ground-truth
values to predictions, time-series overlays, and aggregate metric
visualisations.  Everything operates with Matplotlib in Agg mode so the
functions can be used in headless training and inference environments.
"""

from __future__ import annotations

import math
import os
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

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
    """Plot scatter of predictions vs ground-truth and annotate R²."""
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


def plot_time_series_overlay(
    actual_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    target_cols: Sequence[str],
    out_path: str,
    time_column: str | None = "time",
    highlight_seq_len: int | None = None,
) -> None:
    """Plot time-series overlays of ground-truth and predicted values."""
    ensure_directory(os.path.dirname(out_path) or ".")

    if time_column and time_column in actual_df.columns:
        time_values = actual_df[time_column].to_numpy()
    else:
        time_values = np.arange(len(actual_df))

    n_targets = len(target_cols)
    n_cols = 2 if n_targets > 1 else 1
    n_rows = math.ceil(n_targets / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows), sharex=True)
    if n_rows * n_cols == 1:
        axes = np.array([axes])  # type: ignore
    axes = axes.flatten()

    for idx, col in enumerate(target_cols):
        ax = axes[idx]
        pred_col = f"pred_{col}"
        actual_series = actual_df[col].to_numpy()
        pred_series = predicted_df[pred_col].to_numpy()
        ax.plot(time_values, actual_series, label="Ground Truth", linewidth=1.2)
        ax.plot(time_values, pred_series, label="Prediction", linewidth=1.2, linestyle="--")
        ax.set_title(col)
        ax.set_xlabel("Time")
        ax.set_ylabel("Temperature")
        ax.grid(True, linestyle="--", alpha=0.4)
        if highlight_seq_len and highlight_seq_len > 0:
            ax.axvspan(time_values[0], time_values[min(len(time_values) - 1, highlight_seq_len - 1)],
                       color="#dddddd", alpha=0.2, label="Warm-up")

    for idx in range(n_targets, len(axes)):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if len(axes) > 0 and handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_metric_summary(metrics: Iterable[dict], out_path: str) -> None:
    """Create a summary bar chart for MAE and RMSE across files."""
    metrics_list = list(metrics)
    if not metrics_list:
        return

    df = pd.DataFrame(metrics_list)
    ensure_directory(os.path.dirname(out_path) or ".")

    df_sorted_mae = df.sort_values("mae", ascending=False)
    df_sorted_rmse = df.sort_values("rmse", ascending=False)

    height = max(4, 0.4 * len(df))
    fig, axes = plt.subplots(1, 2, figsize=(12, height), sharey=False)

    axes[0].barh(df_sorted_mae["file"], df_sorted_mae["mae"], color="#1f77b4")
    axes[0].set_title("MAE by File")
    axes[0].set_xlabel("MAE")
    axes[0].invert_yaxis()
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].barh(df_sorted_rmse["file"], df_sorted_rmse["rmse"], color="#ff7f0e")
    axes[1].set_title("RMSE by File")
    axes[1].set_xlabel("RMSE")
    axes[1].invert_yaxis()
    axes[1].grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


__all__ = [
    "ensure_directory",
    "plot_truth_vs_prediction",
    "plot_time_series_overlay",
    "plot_metric_summary",
]
