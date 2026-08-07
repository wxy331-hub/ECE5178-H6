"""Validate a Lab 1 automarker CSV and plot sim-to-real performance."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TARGET = np.array([0.5, 0.5], dtype=float)
REQUIRED_COLUMNS = ("sim_x", "sim_y", "real_x", "real_y")
EXPECTED_ROWS = 100


def read_submission(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        rows = list(reader)

    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected 100 data rows, found {len(rows)}")

    data = {
        column: np.asarray([float(row[column]) for row in rows], dtype=float)
        for column in REQUIRED_COLUMNS
    }
    if not all(np.isfinite(values).all() for values in data.values()):
        raise ValueError("CSV contains NaN or infinite values")
    return data


def calculate_metrics(data: dict[str, np.ndarray]) -> dict[str, float]:
    sim = np.column_stack([data["sim_x"], data["sim_y"]])
    real = np.column_stack([data["real_x"], data["real_y"]])
    return {
        "sim_final_distance": float(np.linalg.norm(sim[-1] - TARGET)),
        "real_final_distance": float(np.linalg.norm(real[-1] - TARGET)),
        "trajectory_rmse": float(np.sqrt(np.mean(np.sum((sim - real) ** 2, axis=1)))),
    }


def save_plot(data: dict[str, np.ndarray], output_path: Path) -> None:
    sim = np.column_stack([data["sim_x"], data["sim_y"]])
    real = np.column_stack([data["real_x"], data["real_y"]])
    point_error = np.linalg.norm(sim - real, axis=1)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(sim[:, 0], sim[:, 1], label="Simulation", linewidth=2)
    axes[0].plot(real[:, 0], real[:, 1], label="Real robot", linewidth=2)
    axes[0].scatter([0.0], [0.0], marker="o", color="black", label="Start")
    axes[0].scatter([TARGET[0]], [TARGET[1]], marker="*", s=160, color="gold", label="Target")
    axes[0].set_title("Lab 1 trajectories")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(np.arange(1, len(point_error) + 1), point_error, color="tab:red")
    axes[1].axhline(0.20, color="black", linestyle="--", label="RMSE threshold reference")
    axes[1].set_title("Pointwise sim-to-real error")
    axes[1].set_xlabel("Control step")
    axes[1].set_ylabel("Position error (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check and plot a Lab 1 submission CSV")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()

    data = read_submission(args.csv_path)
    metrics = calculate_metrics(data)
    plot_path = args.plot or args.csv_path.with_name(f"{args.csv_path.stem}_analysis.png")
    save_plot(data, plot_path)

    checks = {
        "Simulation final distance <= 0.10 m": metrics["sim_final_distance"] <= 0.10,
        "Real final distance <= 0.10 m": metrics["real_final_distance"] <= 0.10,
        "Trajectory RMSE <= 0.20 m": metrics["trajectory_rmse"] <= 0.20,
    }
    print(f"Simulation final distance: {metrics['sim_final_distance']:.4f} m")
    print(f"Real final distance:       {metrics['real_final_distance']:.4f} m")
    print(f"Trajectory RMSE:           {metrics['trajectory_rmse']:.4f} m")
    for label, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {label}")
    print(f"Plot written to: {plot_path.resolve()}")
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
