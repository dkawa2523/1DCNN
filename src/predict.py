"""
Inference script for trained temperature forecasting models.

This script loads a trained model and normalisation statistics, reads
CSV files from a test directory, and performs sequential prediction of
temperatures over time.  For each time step, the previous predicted
temperatures are fed back into the model along with the control
inputs.  The complete predicted time series is saved to an output
directory, and if ground truth temperature values are available in
the CSV files, error metrics are computed.

Usage:
    python predict.py --config path/to/config_pred.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "NumPy が見つかりません。仮想環境を有効化し、`uv pip install -r requirements.txt` を実行してください。"
    ) from exc

try:
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "pandas が見つかりません。仮想環境を有効化し、`uv pip install -r requirements.txt` を実行してください。"
    ) from exc

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "PyTorch が見つかりません。仮想環境を有効化し、`uv pip install -r requirements.txt` を実行してください。"
    ) from exc

from .data_loader import read_csv_files, NormalisationStats
from .models import build_model
from .visualization import (
    plot_metric_summary,
    plot_time_series_overlay,
    plot_truth_vs_prediction,
)
from .config_loader import load_config


def load_normalisation_stats(path: str) -> NormalisationStats:
    with open(path, "r") as f:
        data = json.load(f)
    return NormalisationStats(mean=data["mean"], std=data["std"])


def predict_file(df: pd.DataFrame, model: torch.nn.Module,
                 norm_stats: NormalisationStats,
                 seq_len: int,
                 input_cols: List[str],
                 target_cols: List[str],
                 device: torch.device) -> pd.DataFrame:
    """Perform sequential prediction on a single DataFrame.

    Args:
        df: input DataFrame with time, control inputs and temperatures.
        model: trained PyTorch model in eval mode.
        norm_stats: normalisation statistics to apply/invert.
        seq_len: length of input sequence required by the model.
        input_cols: list of input column names (in order).
        target_cols: list of target column names (in order).
        device: computation device.

    Returns:
        DataFrame with an additional set of columns named like
        "pred_{target_col}" containing the predicted temperature series.
    """
    # make a copy to avoid modifying original
    df_norm = df.copy()
    # ensure columns exist
    for col in target_cols:
        if col not in df_norm.columns:
            raise ValueError(f"Target column {col} missing in input data")
    # apply normalisation
    norm_cols = list(set(input_cols + target_cols))
    norm_stats.apply(df_norm, norm_cols)

    # allocate prediction array
    pred_temps = np.zeros((len(df_norm), len(target_cols)), dtype=np.float32)

    # prepare initial sequence from ground truth for first seq_len steps
    # Each element is a vector of input_cols; includes heater, brine, plasma and normalized temperatures
    if len(df_norm) < seq_len + 1:
        raise ValueError("Input sequence shorter than sequence length + 1")

    # Copy initial target columns for first seq_len steps (already normalised)
    # We'll fill pred_temps for first seq_len with ground truth to facilitate sliding
    for i in range(seq_len):
        pred_temps[i] = df_norm[target_cols].iloc[i].to_numpy()

    # For step-by-step prediction
    model.eval()
    with torch.no_grad():
        # iterate over time starting from seq_len (predict at index seq_len)
        for t in range(seq_len, len(df_norm)):
            # build input sequence (seq_len rows) of input_cols
            # the sequence consists of rows t-seq_len to t-1
            seq_inputs = []
            for i in range(t - seq_len, t):
                row = []
                for col in input_cols:
                    if col in target_cols:
                        # use predicted temperatures for previous steps
                        idx_target = target_cols.index(col)
                        row.append(pred_temps[i][idx_target])
                    else:
                        row.append(df_norm[col].iloc[i])
                seq_inputs.append(row)
            seq_inputs = np.array(seq_inputs, dtype=np.float32)  # shape (seq_len, len(input_cols))
            x = torch.tensor(seq_inputs, dtype=torch.float32).unsqueeze(0).to(device)
            out = model(x).cpu().numpy().squeeze()  # (len(target_cols),)
            # store prediction
            pred_temps[t] = out

    # invert normalisation for predicted temps
    # create DataFrame for predicted values
    pred_temps_denorm = pred_temps.copy()
    # invert using normalization stats
    for j, col in enumerate(target_cols):
        mean = norm_stats.mean[col]
        std = norm_stats.std[col]
        pred_temps_denorm[:, j] = pred_temps_denorm[:, j] * (std + 1e-8) + mean

    # build output DataFrame
    pred_df = pd.DataFrame(pred_temps_denorm, columns=[f"pred_{col}" for col in target_cols])
    result_df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
    return result_df


def main(config_path: str) -> None:
    config = load_config(config_path)

    test_dir = config["data"]["dataset_dir"]
    input_cols = config["data"]["input_columns"]
    target_cols = config["data"]["target_columns"]
    seq_len = config["data"].get("sequence_length", 1)
    model_dir = config["model"]["model_dir"]
    config_used_path = os.path.join(model_dir, "config_used.yaml")
    resolved_model_cfg = dict(config["model"])
    if os.path.exists(config_used_path):
        try:
            saved_cfg = load_config(config_used_path)
            saved_model_cfg = saved_cfg.get("model", {})
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: failed to read saved model config ({exc}); using config_pred settings.")
            saved_model_cfg = {}
        for key, value in saved_model_cfg.items():
            if key not in resolved_model_cfg:
                resolved_model_cfg[key] = value
        if "type" in saved_model_cfg and resolved_model_cfg.get("type") != saved_model_cfg["type"]:
            print(f"Info: overriding model type '{resolved_model_cfg.get('type')}' with "
                  f"saved type '{saved_model_cfg['type']}' from {config_used_path}.")
            resolved_model_cfg["type"] = saved_model_cfg["type"]
    model_type = resolved_model_cfg["type"]
    hidden_dim = resolved_model_cfg.get("hidden_dim", 64)
    num_layers = resolved_model_cfg.get("num_layers", 2)
    dropout = resolved_model_cfg.get("dropout", 0.0)
    # output directory
    out_dir = config["output"].get("prediction_dir", "predictions")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load normalisation stats
    norm_stats_path = os.path.join(model_dir, "norm_stats.json")
    norm_stats = load_normalisation_stats(norm_stats_path)

    # instantiate model and load weights
    input_dim = len(input_cols)
    output_dim = len(target_cols)
    model_cfg = resolved_model_cfg
    model = build_model(model_cfg, input_dim=input_dim, output_dim=output_dim, seq_len=seq_len)
    model_path = os.path.join(model_dir, "best_model.pth")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    metrics_records: List[Dict[str, float]] = []

    # read CSV files
    datasets = read_csv_files(test_dir)
    # process each file
    for path, df in datasets:
        out_df = predict_file(df, model, norm_stats, seq_len, input_cols, target_cols, device)
        filename = os.path.basename(path)
        out_path = os.path.join(out_dir, f"pred_{filename}")
        out_df.to_csv(out_path, index=False)
        print(f"Saved predictions to {out_path}")
        # if ground truth exists, compute error metrics
        if set(target_cols).issubset(df.columns):
            # compute MAE/RMSE from seq_len onwards (since first seq_len points are from truth)
            gt = df[target_cols].to_numpy()[seq_len:]
            pred = out_df[[f"pred_{col}" for col in target_cols]].to_numpy()[seq_len:]
            mae = np.mean(np.abs(pred - gt))
            rmse = np.sqrt(np.mean((pred - gt) ** 2))
            print(f"File {path} MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            scatter_path = os.path.join(out_dir, f"scatter_{os.path.splitext(filename)[0]}.png")
            r2 = plot_truth_vs_prediction(
                gt,
                pred,
                scatter_path,
                title=f"Prediction Scatter: {filename}",
                xlabel="Ground Truth Temperature",
                ylabel="Predicted Temperature",
            )
            print(f"Saved scatter plot to {scatter_path} (R^2={r2:.4f})")
            overlay_path = os.path.join(out_dir, f"timeseries_{os.path.splitext(filename)[0]}.png")
            plot_time_series_overlay(
                actual_df=df.reset_index(drop=True),
                predicted_df=out_df,
                target_cols=target_cols,
                out_path=overlay_path,
                highlight_seq_len=seq_len,
            )
            metrics_records.append({
                "file": filename,
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2),
            })

    if metrics_records:
        metrics_df = pd.DataFrame(metrics_records)
        metrics_csv_path = os.path.join(out_dir, "prediction_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        summary_plot_path = os.path.join(out_dir, "prediction_metrics.png")
        plot_metric_summary(metrics_records, summary_plot_path)
        print(f"Saved prediction metrics to {metrics_csv_path} and {summary_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sequential predictions with trained model")
    parser.add_argument("--config", type=str, required=True, help="Path to config_pred.yaml")
    args = parser.parse_args()
    main(args.config)
