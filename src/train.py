"""
Training script for the temperature forecasting models.

This script loads configuration from a YAML file, prepares the training
and validation datasets, initialises the chosen model architecture,
trains the model and saves the trained weights along with
normalisation statistics.  It also supports early stopping based on
validation loss.

Usage:
    python train.py --config path/to/config_train.yaml

See README.md for detailed instructions on configuration parameters.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=C0413

import numpy as np

from .data_loader import (prepare_datasets, prepare_full_dataset,
                          NormalisationStats, read_csv_files)
from .models import build_model
from .predict import predict_file
from .config_loader import load_config, dump_config
from .visualization import plot_truth_vs_prediction, ensure_directory


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device,
             norm_stats: NormalisationStats | None = None,
             target_cols: List[str] | None = None,
             denormalize: bool = False,
             return_outputs: bool = False) -> Dict[str, float]:
    """Evaluate model on loader and return metrics dict.

    Args:
        model: trained model.
        loader: dataloader to iterate over.
        criterion: loss function used for optimisation (typically MSE).
        device: torch device.
        norm_stats: normalisation stats used to optionally denormalise outputs.
        target_cols: order of target columns for denormalisation.
        denormalize: whether to convert predictions/targets back to original scale.
        return_outputs: if True, include predictions and targets (np.ndarray) in the result.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = criterion(out, y)
            running_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            preds.append(out.detach().cpu())
            targets.append(y.detach().cpu())

    if total_samples == 0:
        raise ValueError("Evaluation loader yielded no samples.")

    preds_tensor = torch.cat(preds, dim=0)
    targets_tensor = torch.cat(targets, dim=0)

    mse_norm = running_loss / total_samples
    pred_np = preds_tensor.numpy()
    target_np = targets_tensor.numpy()

    if denormalize and norm_stats is not None and target_cols is not None:
        pred_np = norm_stats.invert(pred_np.copy(), target_cols)
        target_np = norm_stats.invert(target_np.copy(), target_cols)
    else:
        pred_np = pred_np.copy()
        target_np = target_np.copy()

    diff = pred_np - target_np
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    target_flat = target_np.reshape(-1)
    pred_flat = pred_np.reshape(-1)
    ss_tot = float(np.sum((target_flat - target_flat.mean()) ** 2))
    if ss_tot == 0.0:
        r2 = 1.0
    else:
        ss_res = float(np.sum((pred_flat - target_flat) ** 2))
        r2 = 1.0 - ss_res / ss_tot

    metrics: Dict[str, float] = {
        "loss": float(mse_norm),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "samples": float(total_samples),
    }

    if return_outputs:
        metrics["predictions"] = pred_np
        metrics["targets"] = target_np

    return metrics


def write_log_artifacts(log_records, artifact_dir: str) -> None:
    """Persist training history as JSON/CSV and accompanying plots."""
    if not log_records:
        return

    ensure_directory(artifact_dir)

    json_path = os.path.join(artifact_dir, "train_log.json")
    with open(json_path, "w") as lf:
        json.dump(log_records, lf, indent=2)

    csv_path = os.path.join(artifact_dir, "train_log.csv")
    fieldnames = list(log_records[0].keys())
    with open(csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_records)

    # Prepare data for plots
    epochs = [entry["epoch"] for entry in log_records]
    train_loss = [entry["train_loss"] for entry in log_records]
    val_loss = [entry["val_loss"] for entry in log_records]
    val_mae = [entry["val_mae"] for entry in log_records]
    val_rmse = [entry["val_rmse"] for entry in log_records]

    # Plot training vs validation loss
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_loss, marker="o", label="Train Loss")
    ax.plot(epochs, val_loss, marker="o", label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training vs Validation Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    loss_plot_path = os.path.join(artifact_dir, "loss_curve.png")
    fig.savefig(loss_plot_path, dpi=200)
    plt.close(fig)

    # Plot validation metrics
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, val_mae, marker="s", label="Validation MAE")
    ax.plot(epochs, val_rmse, marker="^", label="Validation RMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error")
    ax.set_title("Validation Error Metrics")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    metrics_plot_path = os.path.join(artifact_dir, "validation_metrics.png")
    fig.savefig(metrics_plot_path, dpi=200)
    plt.close(fig)

    print(f"Saved training history to {json_path} and {csv_path}")
    print(f"Saved plots to {loss_plot_path} and {metrics_plot_path}")


def run_cross_validation(config: Dict,
                         device: torch.device,
                         artifact_dir: str) -> None:
    """Run K-fold cross validation based on configuration."""
    cv_cfg = config.get("training", {}).get("cross_validation", {})
    num_folds = int(cv_cfg.get("num_folds", 0))
    if num_folds <= 1:
        return

    print(f"Running {num_folds}-fold cross validation...")
    dataset, norm_stats = prepare_full_dataset(config)
    n_samples = len(dataset)
    if n_samples < num_folds:
        print(f"Not enough samples ({n_samples}) for {num_folds}-fold cross validation. Skipping CV.")
        return

    indices = np.arange(n_samples)
    shuffle = bool(cv_cfg.get("shuffle", True))
    seed = cv_cfg.get("random_seed", 42)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    fold_sizes = np.full(num_folds, n_samples // num_folds, dtype=int)
    fold_sizes[: n_samples % num_folds] += 1

    batch_size = config["model"].get("batch_size", 32)
    learning_rate = config["model"].get("learning_rate", 1e-3)
    weight_decay = config["model"].get("weight_decay", 0.0)
    num_epochs = config["model"].get("num_epochs", 100)
    patience = config["model"].get("early_stopping_patience", 10)

    input_dim = len(config["data"]["input_columns"])
    output_dim = len(config["data"]["target_columns"])
    target_cols = config["data"]["target_columns"]
    seq_len = config["data"].get("sequence_length", 1)

    cv_dir = os.path.join(artifact_dir, "cv")
    ensure_directory(cv_dir)

    cv_results: List[Dict[str, float]] = []
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    current = 0
    for fold_idx in range(num_folds):
        start = current
        stop = start + fold_sizes[fold_idx]
        val_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        current = stop

        train_subset = Subset(dataset, train_indices.tolist())
        val_subset = Subset(dataset, val_indices.tolist())

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = build_model(config["model"], input_dim=input_dim, output_dim=output_dim, seq_len=seq_len)
        model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        fold_dir = os.path.join(cv_dir, f"fold_{fold_idx + 1}")
        ensure_directory(fold_dir)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = copy.deepcopy(model.state_dict())
        log_records = []

        for epoch in range(1, num_epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device,
                                   norm_stats=norm_stats,
                                   target_cols=target_cols,
                                   denormalize=True,
                                   return_outputs=False)

            log_entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_mse": val_metrics["mse"],
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_r2": val_metrics["r2"],
            }
            log_records.append(log_entry)

            print(f"[CV {fold_idx + 1}/{num_folds}] Epoch {epoch}: train_loss={train_loss:.6f}, "
                  f"val_loss={val_metrics['loss']:.6f}, val_mse={val_metrics['mse']:.6f}, "
                  f"val_rmse={val_metrics['rmse']:.6f}, val_mae={val_metrics['mae']:.6f}, "
                  f"val_r2={val_metrics['r2']:.6f}")

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[CV {fold_idx + 1}] Early stopping at epoch {epoch}")
                    break

        model.load_state_dict(best_state)
        final_metrics = evaluate(model, val_loader, criterion, device,
                                 norm_stats=norm_stats,
                                 target_cols=target_cols,
                                 denormalize=True,
                                 return_outputs=True)

        predictions = final_metrics.pop("predictions")
        targets = final_metrics.pop("targets")
        fold_number = fold_idx + 1
        final_metrics.update({
            "fold": fold_number,
            "num_train_samples": float(len(train_subset)),
            "num_val_samples": float(len(val_subset)),
        })
        cv_results.append(final_metrics)
        all_preds.append(predictions)
        all_targets.append(targets)

        write_log_artifacts(log_records, fold_dir)
        scatter_path = os.path.join(fold_dir, f"scatter_fold_{fold_number}.png")
        plot_truth_vs_prediction(
            targets,
            predictions,
            scatter_path,
            title=f"CV Fold {fold_number} Predictions",
            xlabel="Ground Truth Temperature",
            ylabel="Predicted Temperature",
        )
        print(f"[CV {fold_number}] Metrics: MSE={final_metrics['mse']:.6f}, RMSE={final_metrics['rmse']:.6f}, "
              f"MAE={final_metrics['mae']:.6f}, R^2={final_metrics['r2']:.6f}; scatter saved to {scatter_path}")

    if not cv_results:
        return

    metrics_keys = ["mse", "rmse", "mae", "r2"]
    aggregate = {}
    for key in metrics_keys:
        values = [res[key] for res in cv_results]
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    summary_payload = {
        "fold_metrics": cv_results,
        "aggregate": aggregate,
    }
    cv_json_path = os.path.join(cv_dir, "cv_metrics.json")
    with open(cv_json_path, "w") as jf:
        json.dump(summary_payload, jf, indent=2)

    cv_csv_path = os.path.join(cv_dir, "cv_metrics.csv")
    fieldnames = ["fold", "mse", "rmse", "mae", "r2", "loss", "num_train_samples", "num_val_samples"]
    with open(cv_csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for res in cv_results:
            writer.writerow({key: res.get(key, "") for key in fieldnames})
        for label, stat_key in [("mean", "mean"), ("std", "std"), ("min", "min"), ("max", "max")]:
            row = {key: "" for key in fieldnames}
            row["fold"] = label
            for metric in metrics_keys:
                row[metric] = aggregate[metric][stat_key]
            writer.writerow(row)

    fold_numbers = [res["fold"] for res in cv_results]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    plots = [("MSE", "mse"), ("RMSE", "rmse"), ("MAE", "mae"), ("R^2", "r2")]
    for ax, (title, key) in zip(axes.flat, plots):
        ax.plot(fold_numbers, [res[key] for res in cv_results], marker="o")
        ax.set_xlabel("Fold")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    cv_plot_path = os.path.join(cv_dir, "cv_metrics.png")
    fig.savefig(cv_plot_path, dpi=200)
    plt.close(fig)

    overall_predictions = np.concatenate(all_preds, axis=0)
    overall_targets = np.concatenate(all_targets, axis=0)
    overall_scatter_path = os.path.join(cv_dir, "scatter_overall.png")
    overall_r2 = plot_truth_vs_prediction(
        overall_targets,
        overall_predictions,
        overall_scatter_path,
        title=f"CV Scatter ({num_folds}-fold)",
        xlabel="Ground Truth Temperature",
        ylabel="Predicted Temperature",
    )

    print(f"Saved CV metrics to {cv_json_path} and {cv_csv_path}")
    print(f"Saved CV plots to {cv_plot_path} and {overall_scatter_path} (R^2={overall_r2:.6f})")
    print("[CV] Aggregate metrics:")
    for key, stats in aggregate.items():
        print(f"  {key.upper()}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
              f"min={stats['min']:.6f}, max={stats['max']:.6f}")


def run_standard_training(config: Dict,
                          device: torch.device,
                          artifact_dir: str,
                          output_dir: str) -> NormalisationStats:
    """Carry out standard training (single split) and persist best model."""
    train_ds, val_ds, norm_stats = prepare_datasets(config, mode="train")
    batch_size = config["model"].get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_dim = len(config["data"]["input_columns"])
    output_dim = len(config["data"]["target_columns"])
    target_cols = config["data"]["target_columns"]
    seq_len = config["data"].get("sequence_length", 1)

    model = build_model(config["model"], input_dim=input_dim, output_dim=output_dim, seq_len=seq_len)
    model.to(device)

    criterion = nn.MSELoss()
    learning_rate = config["model"].get("learning_rate", 1e-3)
    weight_decay = config["model"].get("weight_decay", 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    num_epochs = config["model"].get("num_epochs", 100)
    patience = config["model"].get("early_stopping_patience", 10)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    log_records = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device,
                               norm_stats=norm_stats,
                               target_cols=target_cols,
                               denormalize=True,
                               return_outputs=False)

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_r2": val_metrics["r2"],
        }
        log_records.append(log_entry)

        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_metrics['loss']:.6f}, "
              f"val_mse={val_metrics['mse']:.6f}, val_rmse={val_metrics['rmse']:.6f}, "
              f"val_mae={val_metrics['mae']:.6f}, val_r2={val_metrics['r2']:.6f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch

            model_path = os.path.join(output_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            stats_path = os.path.join(output_dir, "norm_stats.json")
            with open(stats_path, "w") as sf:
                json.dump({"mean": norm_stats.mean, "std": norm_stats.std}, sf, indent=2)
            cfg_out_path = os.path.join(output_dir, "config_used.yaml")
            dump_config(config, cfg_out_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    write_log_artifacts(log_records, artifact_dir)

    model.load_state_dict(best_state)
    final_val_metrics = evaluate(model, val_loader, criterion, device,
                                 norm_stats=norm_stats,
                                 target_cols=target_cols,
                                 denormalize=True,
                                 return_outputs=False)
    print(f"Best epoch: {best_epoch} | val_loss={best_val_loss:.6f}, "
          f"val_mse={final_val_metrics['mse']:.6f}, val_rmse={final_val_metrics['rmse']:.6f}, "
          f"val_mae={final_val_metrics['mae']:.6f}, val_r2={final_val_metrics['r2']:.6f}")

    return norm_stats


def evaluate_on_test(config: dict,
                     norm_stats: NormalisationStats,
                     device: torch.device,
                     artifact_dir: str) -> None:
    """Optionally evaluate the trained model on a test dataset and save scatter plots."""
    test_dir = config["data"].get("test_dataset_dir")
    if not test_dir:
        return
    if not os.path.isdir(test_dir):
        print(f"Test dataset directory '{test_dir}' not found. Skipping test evaluation.")
        return

    input_cols = config["data"].get("test_input_columns", config["data"]["input_columns"])
    target_cols = config["data"].get("test_target_columns", config["data"]["target_columns"])
    seq_len = config["data"].get("sequence_length", 1)
    model_dir = config["output"].get("model_dir", "models")
    model_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"Best model not found at '{model_path}'. Skipping test evaluation.")
        return

    # instantiate and load best model
    model_cfg = config["model"]
    model = build_model(model_cfg,
                        input_dim=len(input_cols),
                        output_dim=len(target_cols),
                        seq_len=seq_len)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    datasets = read_csv_files(test_dir)
    if not datasets:
        print(f"No CSV files found in '{test_dir}'. Skipping test evaluation.")
        return

    test_scatter_dir = os.path.join(artifact_dir, "test_scatter")
    ensure_directory(test_scatter_dir)

    metrics = []
    all_gt = []
    all_pred = []

    for path, df in datasets:
        pred_df = predict_file(df, model, norm_stats, seq_len, input_cols, target_cols, device)
        gt = df[target_cols].to_numpy()[seq_len:]
        pred = pred_df[[f"pred_{col}" for col in target_cols]].to_numpy()[seq_len:]
        if gt.size == 0:
            print(f"File {path} too short for evaluation (sequence_length={seq_len}). Skipping.")
            continue

        mae = float(np.mean(np.abs(pred - gt)))
        rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
        base_name = os.path.splitext(os.path.basename(path))[0]
        scatter_path = os.path.join(test_scatter_dir, f"scatter_{base_name}.png")
        r2 = plot_truth_vs_prediction(
            gt,
            pred,
            scatter_path,
            title=f"Test Scatter: {base_name}",
            xlabel="Ground Truth Temperature",
            ylabel="Predicted Temperature",
        )
        print(f"[Test] {path} MAE={mae:.4f}, RMSE={rmse:.4f}, R^2={r2:.4f}; saved {scatter_path}")
        metrics.append({"file": os.path.basename(path), "mae": mae, "rmse": rmse, "r2": r2})
        all_gt.append(gt)
        all_pred.append(pred)

    if not metrics:
        print("No test metrics were computed.")
        return

    overall_gt = np.concatenate(all_gt, axis=0)
    overall_pred = np.concatenate(all_pred, axis=0)
    overall_mae = float(np.mean(np.abs(overall_pred - overall_gt)))
    overall_rmse = float(np.sqrt(np.mean((overall_pred - overall_gt) ** 2)))
    overall_scatter_path = os.path.join(test_scatter_dir, "scatter_overall.png")
    overall_r2 = plot_truth_vs_prediction(
        overall_gt,
        overall_pred,
        overall_scatter_path,
        title="Test Scatter: All Files",
        xlabel="Ground Truth Temperature",
        ylabel="Predicted Temperature",
    )
    print(f"[Test] Aggregate MAE={overall_mae:.4f}, RMSE={overall_rmse:.4f}, R^2={overall_r2:.4f}; "
          f"saved {overall_scatter_path}")

    metrics.append({"file": "_aggregate", "mae": overall_mae, "rmse": overall_rmse, "r2": overall_r2})

    metrics_json_path = os.path.join(artifact_dir, "test_metrics.json")
    with open(metrics_json_path, "w") as jf:
        json.dump(metrics, jf, indent=2)
    metrics_csv_path = os.path.join(artifact_dir, "test_metrics.csv")
    fieldnames = ["file", "mae", "rmse", "r2"]
    with open(metrics_csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    print(f"Saved test metrics to {metrics_json_path} and {metrics_csv_path}")


def main(config_path: str) -> None:
    # load configuration
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = config["output"].get("model_dir", "models")
    ensure_directory(output_dir)
    logs_dir_cfg = config["output"].get("logs_dir")
    artifact_dir = logs_dir_cfg if logs_dir_cfg else output_dir
    if logs_dir_cfg:
        ensure_directory(logs_dir_cfg)
    run_cross_validation(config, device, artifact_dir)
    norm_stats = run_standard_training(config, device, artifact_dir, output_dir)
    evaluate_on_test(config, norm_stats, device, artifact_dir)
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train temperature forecasting model")
    parser.add_argument("--config", type=str, required=True, help="Path to config_train.yaml")
    args = parser.parse_args()
    main(args.config)
