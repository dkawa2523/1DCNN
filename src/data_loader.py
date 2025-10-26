"""
Data loading and preprocessing utilities for temperature forecasting.

This module provides functions to read CSV files containing simulated or
real thermal data, construct sequences suitable for machine learning models,
and apply normalisation.  The CSV files should contain a time column,
control inputs (heater, brine, plasma_heat) and one or more temperature
columns (e.g. temp_p0, temp_p1, ...).  See README for the expected
format.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class NormalisationStats:
    """Simple container for mean and std of each column."""
    mean: Dict[str, float] = field(default_factory=dict)
    std: Dict[str, float] = field(default_factory=dict)

    def apply(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Normalise specified columns in the DataFrame in place."""
        for col in columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found while applying normalisation.")
            df[col] = (df[col] - self.mean[col]) / (self.std[col] + 1e-8)
        return df

    def invert(self, data: np.ndarray, columns: List[str]) -> np.ndarray:
        """Undo normalisation on a batch of data (NumPy array)."""
        # data is shape [batch, num_features]
        for idx, col in enumerate(columns):
            data[:, idx] = data[:, idx] * (self.std[col] + 1e-8) + self.mean[col]
        return data


class TempDataset(Dataset):
    """PyTorch Dataset for sequences of temperature data.

    Each item returns a tuple (sequence, target) where:
        - sequence: Tensor of shape (seq_len, num_features).
        - target: Tensor of shape (num_targets,) (temperatures at next time step).

    The dataset is built by sliding a window of length seq_len over the time
    series.  The target is the vector of temperatures at the next time step.
    """

    def __init__(self,
                 sequences: List[np.ndarray],
                 targets: List[np.ndarray]):
        self.sequences = sequences
        self.targets = targets

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        tgt = torch.tensor(self.targets[idx], dtype=torch.float32)
        return seq, tgt


def parse_filename(filename: str) -> Dict[str, float]:
    """Extract brine, heater, plasma and initial temperature from the file name.

    Expected format: temp_{brine}_{heater}_{plasma}_{initial_temp}.csv
    The values may be floats (for brine, initial_temp) or integers.
    Returns a dictionary with keys 'brine', 'heater', 'plasma', 'initial_temp'.
    """
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    # remove leading prefix 'temp_' if present
    if name.startswith("temp_"):
        parts = name[len("temp_"):].split("_")
    else:
        parts = name.split("_")
    if len(parts) < 4:
        raise ValueError(f"File name {filename} does not contain four parameters")
    brine_str, heater_str, plasma_str, init_str = parts[:4]
    brine = float(brine_str)
    heater = float(heater_str)
    plasma = float(plasma_str)
    init_temp = float(init_str)
    return {"brine": brine, "heater": heater, "plasma": plasma, "initial_temp": init_temp}


def _load_new_format_csv(path: str, const_features: Dict[str, float]) -> pd.DataFrame:
    """Load CSV files that follow the headerless temp_{...}.csv convention."""
    df = pd.read_csv(path, header=None)
    if df.empty:
        raise ValueError(f"{path} is empty.")
    n_cols = df.shape[1]
    if n_cols < 2:
        raise ValueError(f"{path} does not contain enough columns.")
    time_col = ["time"]
    temp_cols = [f"temp_p{i}" for i in range(n_cols - 1)]
    df.columns = time_col + temp_cols
    for key, val in const_features.items():
        df[key] = float(val)
    return df


def _load_legacy_csv(path: str) -> pd.DataFrame:
    """Load legacy CSV files that include headers for inputs."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty.")
    df = df.copy()
    if "plasma" not in df.columns and "plasma_heat" in df.columns:
        df.rename(columns={"plasma_heat": "plasma"}, inplace=True)
    # derive initial_temp if not provided
    temp_cols = [col for col in df.columns if col.startswith("temp")]
    if not temp_cols:
        raise ValueError(f"{path} does not contain temperature columns.")
    if "initial_temp" not in df.columns:
        first_row_numeric = pd.to_numeric(df.loc[0, temp_cols], errors="coerce")
        init_temp = float(first_row_numeric.mean())
        df["initial_temp"] = init_temp
    # ensure numeric types for all non-time columns
    for col in df.columns:
        if col == "time":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df.isna().any().any():
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        if df.empty:
            raise ValueError(f"{path} only contained invalid (non-numeric) values.")
    return df


def read_csv_files(directory: str) -> List[Tuple[str, pd.DataFrame]]:
    """Read all CSV files in the given directory.

    CSV files are expected to have no header.  The first column is the
    time index and the remaining columns are temperature values for
    measurement points.  The file name encodes constant features such
    as brine, heater, plasma and initial temperature, which are added
    to each row of the resulting DataFrame.

    Legacy CSV files that already contain headers (`time, heater, brine, ...`)
    are also supported.  In that case constant features are read from the
    columns rather than the file name.

    Returns:
        A list of tuples (path, dataframe) for each successfully parsed file.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Dataset directory '{directory}' does not exist.")

    paths = sorted(glob.glob(os.path.join(directory, "*.csv")))
    dataframes: List[Tuple[str, pd.DataFrame]] = []
    for path in paths:
        try:
            const_features = parse_filename(path)
        except ValueError:
            const_features = None

        try:
            if const_features:
                df = _load_new_format_csv(path, const_features)
            else:
                df = _load_legacy_csv(path)
        except ValueError as exc:
            print(f"Skipping {path}: {exc}")
            continue

        dataframes.append((path, df))

    if not dataframes:
        raise ValueError(f"No usable CSV files were found in '{directory}'.")
    return dataframes


def compute_normalisation_stats(dfs: List[pd.DataFrame], columns: List[str]) -> NormalisationStats:
    """Compute mean and std for each specified column across all DataFrames."""
    concat = pd.concat([df[columns] for df in dfs], ignore_index=True)
    mean = concat.mean().to_dict()
    std = concat.std().replace(0, 1e-8).to_dict()  # avoid division by zero
    return NormalisationStats(mean=mean, std=std)


def build_sequences(df: pd.DataFrame,
                    seq_len: int,
                    input_cols: List[str],
                    target_cols: List[str],
                    step_size: int = 1) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Create sequences and targets from a single DataFrame.

    Args:
        df: input DataFrame containing sequential data.
        seq_len: length of each input sequence.
        input_cols: names of input columns (e.g., heater, brine, plasma_heat, temp_p0, ...).
        target_cols: names of target columns (temperature columns).
        step_size: number of steps to shift the window between sequences.

    Returns:
        sequences: list of arrays with shape (seq_len, num_features).
        targets: list of arrays with shape (num_targets,).
    """
    sequences: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    n_rows = len(df)
    last_index = n_rows - seq_len - 1  # ensure target exists at i+seq_len
    for start in range(0, last_index + 1, step_size):
        end = start + seq_len
        seq = df[input_cols].iloc[start:end].to_numpy()
        tgt = df[target_cols].iloc[end].to_numpy()
        sequences.append(seq)
        targets.append(tgt)
    return sequences, targets


def prepare_datasets(config: Dict, mode: str = "train") -> Tuple[TempDataset, Optional[TempDataset], NormalisationStats]:
    """Prepare datasets for training or inference.

    Args:
        config: dictionary loaded from config YAML containing data parameters.
        mode: either "train" or "eval".  When "train" this function splits
              data into training and validation sets.  When "eval" returns a
              dataset without splitting.

    Returns:
        train_dataset: instance of TempDataset with sequences for training.
        val_dataset: instance of TempDataset for validation (None if mode != 'train').
        norm_stats: normalisation statistics computed on training data.
    """
    dataset_dir = config["data"]["dataset_dir"]
    input_cols = config["data"]["input_columns"]
    target_cols = config["data"]["target_columns"]
    seq_len = config["data"].get("sequence_length", 1)
    step_size = config["data"].get("step_size", 1)

    # read CSV files
    dfs_with_paths = read_csv_files(dataset_dir)
    dfs = [df for _, df in dfs_with_paths]

    # compute normalisation on full dataset (inputs and targets) using training mode only
    norm_cols = list(set(input_cols + target_cols))
    norm_stats = compute_normalisation_stats(dfs, norm_cols)

    # apply normalisation to each DataFrame
    for df in dfs:
        norm_stats.apply(df, norm_cols)

    # build sequences and targets from each DataFrame
    all_sequences: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    for df in dfs:
        seqs, tgts = build_sequences(df, seq_len, input_cols, target_cols, step_size)
        all_sequences.extend(seqs)
        all_targets.extend(tgts)

    # convert to numpy arrays
    all_sequences = np.array(all_sequences, dtype=np.float32)
    all_targets = np.array(all_targets, dtype=np.float32)

    if mode == "train":
        # shuffle
        indices = np.arange(len(all_sequences))
        np.random.shuffle(indices)
        all_sequences = all_sequences[indices]
        all_targets = all_targets[indices]
        # split
        test_ratio = config["data"].get("validation_ratio", 0.2)
        n_val = int(len(all_sequences) * test_ratio)
        val_sequences = all_sequences[:n_val]
        val_targets = all_targets[:n_val]
        train_sequences = all_sequences[n_val:]
        train_targets = all_targets[n_val:]
        train_ds = TempDataset(train_sequences, train_targets)
        val_ds = TempDataset(val_sequences, val_targets)
        return train_ds, val_ds, norm_stats
    else:
        ds = TempDataset(all_sequences, all_targets)
        return ds, None, norm_stats


def prepare_full_dataset(config: Dict) -> Tuple[TempDataset, NormalisationStats]:
    """Prepare the full dataset without splitting for cross-validation."""
    dataset_dir = config["data"]["dataset_dir"]
    input_cols = config["data"]["input_columns"]
    target_cols = config["data"]["target_columns"]
    seq_len = config["data"].get("sequence_length", 1)
    step_size = config["data"].get("step_size", 1)

    dfs_with_paths = read_csv_files(dataset_dir)
    dfs = [df for _, df in dfs_with_paths]

    norm_cols = list(set(input_cols + target_cols))
    norm_stats = compute_normalisation_stats(dfs, norm_cols)
    for df in dfs:
        norm_stats.apply(df, norm_cols)

    all_sequences: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    for df in dfs:
        seqs, tgts = build_sequences(df, seq_len, input_cols, target_cols, step_size)
        all_sequences.extend(seqs)
        all_targets.extend(tgts)

    all_sequences_np = np.array(all_sequences, dtype=np.float32)
    all_targets_np = np.array(all_targets, dtype=np.float32)
    dataset = TempDataset(all_sequences_np, all_targets_np)
    return dataset, norm_stats
