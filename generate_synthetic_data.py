"""
Synthetic dataset generator for the temperature prediction problem.

This script produces a set of CSV files representing different
experimental conditions for a simplified thermal system.  The thermal
system consists of multiple measurement points distributed across the
Cell.  At each time step the temperatures of all points are updated
based on the current heater power, brine temperature, plasma heating
input and the previous temperatures.  Each CSV file contains a time
column, the control inputs (heater, brine and plasma), and a column
for each measurement point's temperature.  See the README for more
details.

The generated data is not meant to reflect any particular physical
process but instead provides a controlled environment for testing the
machine learning pipeline.  Feel free to adjust the parameters in
generate_synthetic_data() to explore different behaviours.
"""

import argparse
import os
from typing import List

import numpy as np
import pandas as pd


def generate_condition(num_steps: int,
                       n_points: int,
                       base_temp: float,
                       heater_profile: np.ndarray,
                       brine_profile: np.ndarray,
                       plasma_profile: np.ndarray,
                       coeffs: np.ndarray,
                       ambient: float = 20.0,
                       dt: float = 1.0) -> pd.DataFrame:
    """Generate a single condition with synthetic dynamics.

    Args:
        num_steps: number of time steps to simulate.
        n_points: number of measurement points.
        base_temp: initial temperature of each point (with small noise added).
        heater_profile: 1D array of length num_steps with heater power values.
        brine_profile: 1D array of length num_steps with brine values.
        plasma_profile: 1D array of length num_steps with plasma heat values.
        coeffs: (n_points, 4) array with per‑point coefficients for heater,
                brine, plasma and cooling effect.
        ambient: ambient temperature used for cooling term.
        dt: time step size.

    Returns:
        DataFrame with columns: time, heater, brine, plasma_heat, temp_p0, ...
    """
    # ensure floating dtype so that noise and continuous updates are allowed even if base_temp is int
    temps = np.full((num_steps + 1, n_points), base_temp, dtype=np.float64)
    temps[0] += np.random.normal(scale=0.5, size=n_points)  # small noise on initial temps
    time = np.arange(num_steps + 1) * dt

    # simulate dynamics
    for t in range(num_steps):
        for j in range(n_points):
            k_h, k_b, k_p, k_cool = coeffs[j]
            # temperature change due to heater, brine and plasma inputs
            delta = (
                k_h * heater_profile[t]
                + k_b * brine_profile[t]
                + k_p * plasma_profile[t]
                - k_cool * (temps[t, j] - ambient)
            ) * dt
            # small coupling to neighbouring points
            neighbor_term = 0.0
            if j > 0:
                neighbor_term += 0.05 * (temps[t, j - 1] - temps[t, j])
            if j < n_points - 1:
                neighbor_term += 0.05 * (temps[t, j + 1] - temps[t, j])
            temps[t + 1, j] = temps[t, j] + delta + neighbor_term
        # add process noise
        temps[t + 1] += np.random.normal(scale=0.05, size=n_points)

    data = {
        "time": time,
        "heater": np.concatenate([heater_profile, [heater_profile[-1]]]),
        "brine": np.concatenate([brine_profile, [brine_profile[-1]]]),
        "plasma_heat": np.concatenate([plasma_profile, [plasma_profile[-1]]]),
    }
    # add temperature columns
    for j in range(n_points):
        data[f"temp_p{j}"] = temps[:, j]
    df = pd.DataFrame(data)
    return df


def generate_synthetic_data(out_dir: str,
                            n_conditions_train: int = 3,
                            n_conditions_test: int = 1,
                            num_steps: int = 200,
                            n_points: int = 5,
                            seed: int = 42) -> None:
    """(Deprecated) Generate random training and test CSV files with headers.

    This function is kept for backward compatibility but is no longer used
    in the updated dataset format.  See ``generate_combination_dataset`` for
    generating headerless datasets with condition encoded in the file name.
    """
    rng = np.random.default_rng(seed)
    train_dir = os.path.join(out_dir, "train")
    test_dir = os.path.join(out_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    def create_profiles():
        # use smooth random walk for control inputs
        h = np.cumsum(rng.normal(scale=0.1, size=num_steps))
        h = (h - np.min(h)) / (np.max(h) - np.min(h))
        b = np.cumsum(rng.normal(scale=0.1, size=num_steps))
        b = (b - np.min(b)) / (np.max(b) - np.min(b))
        p = np.cumsum(rng.normal(scale=0.1, size=num_steps))
        p = (p - np.min(p)) / (np.max(p) - np.min(p))
        return h, b, p

    # create training conditions
    for idx in range(n_conditions_train):
        coeffs = rng.uniform(low=[0.1, 0.05, 0.05, 0.01], high=[0.5, 0.3, 0.3, 0.05], size=(n_points, 4))
        heater, brine, plasma = create_profiles()
        df = generate_condition(num_steps=num_steps,
                                n_points=n_points,
                                base_temp=25 + rng.uniform(-2, 2),
                                heater_profile=heater,
                                brine_profile=brine,
                                plasma_profile=plasma,
                                coeffs=coeffs,
                                ambient=20.0,
                                dt=1.0)
        file_path = os.path.join(train_dir, f"condition_train_{idx}.csv")
        df.to_csv(file_path, index=False)
        print(f"Generated training condition CSV: {file_path}")

    # create test conditions
    for idx in range(n_conditions_test):
        coeffs = rng.uniform(low=[0.1, 0.05, 0.05, 0.01], high=[0.5, 0.3, 0.3, 0.05], size=(n_points, 4))
        heater, brine, plasma = create_profiles()
        df = generate_condition(num_steps=num_steps,
                                n_points=n_points,
                                base_temp=25 + rng.uniform(-2, 2),
                                heater_profile=heater,
                                brine_profile=brine,
                                plasma_profile=plasma,
                                coeffs=coeffs,
                                ambient=20.0,
                                dt=1.0)
        file_path = os.path.join(test_dir, f"condition_test_{idx}.csv")
        df.to_csv(file_path, index=False)
        print(f"Generated test condition CSV: {file_path}")


def generate_combination_dataset(out_dir: str,
                                 brine_values: List[float],
                                 heater_values: List[float],
                                 plasma_values: List[float],
                                 init_temps: List[float],
                                 num_steps: int = 200,
                                 n_points: int = 5,
                                 seed: int = 42) -> None:
    """Generate a dataset for all combinations of brine, heater, plasma and initial temperature.

    Files are saved with the pattern ``temp_{brine}_{heater}_{plasma}_{init_temp}.csv``.
    Each file contains no header; the first column is time and the remaining columns
    are temperature values for measurement points.  Constant features (brine,
    heater, plasma, initial_temp) are encoded in the file name.

    Args:
        out_dir: directory to store the generated CSV files (no subdirectories are used).
        brine_values: list of brine settings.
        heater_values: list of heater settings.
        plasma_values: list of plasma settings.
        init_temps: list of initial temperatures.
        num_steps: number of time steps per file.
        n_points: number of measurement points.
        seed: random seed.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    # iterate over all combinations
    for brine in brine_values:
        for heater in heater_values:
            for plasma in plasma_values:
                for init_temp in init_temps:
                    # generate random coefficients per measurement point
                    coeffs = rng.uniform(low=[0.1, 0.05, 0.05, 0.01], high=[0.5, 0.3, 0.3, 0.05], size=(n_points, 4))
                    # create profiles as constant for entire series (since control values are constant in these files)
                    heater_profile = np.full(num_steps, heater)
                    brine_profile = np.full(num_steps, brine)
                    plasma_profile = np.full(num_steps, plasma)
                    df = generate_condition(num_steps=num_steps,
                                            n_points=n_points,
                                            base_temp=init_temp,
                                            heater_profile=heater_profile,
                                            brine_profile=brine_profile,
                                            plasma_profile=plasma_profile,
                                            coeffs=coeffs,
                                            ambient=20.0,
                                            dt=1.0)
                    # df currently has columns: time, heater, brine, plasma_heat, temp_p*
                    # we only keep time and temperature columns
                    temp_cols = [col for col in df.columns if col.startswith("temp_p")]
                    minimal_df = pd.concat([df[["time"]], df[temp_cols]], axis=1)
                    # save without header
                    filename = f"temp_{brine}_{heater}_{plasma}_{init_temp}.csv"
                    file_path = os.path.join(out_dir, filename)
                    minimal_df.to_csv(file_path, header=False, index=False)
                    print(f"Generated combination CSV: {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic temperature data")
    parser.add_argument("out_dir", type=str, default="data",
                        nargs="?", help="Base output directory")
    parser.add_argument("--train", type=int, default=3, help="Number of training conditions")
    parser.add_argument("--test", type=int, default=1, help="Number of test conditions")
    parser.add_argument("--steps", type=int, default=200, help="Number of time steps per condition")
    parser.add_argument("--points", type=int, default=5, help="Number of measurement points")
    args = parser.parse_args()
    generate_synthetic_data(out_dir=args.out_dir,
                            n_conditions_train=args.train,
                            n_conditions_test=args.test,
                            num_steps=args.steps,
                            n_points=args.points)