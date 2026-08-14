"""Validate and score an ECE5178 Lab 2 automarker CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EXPECTED_COLUMNS = (
    "sim_x",
    "sim_y",
    "real_x",
    "real_y",
    "P_xx",
    "P_xy",
    "P_yy",
)
EXPECTED_ROWS = 200
CHI_SQUARE_95_2DOF = 5.991464547107979
MEAN_MAHALANOBIS_LIMIT = 4.0
PASS_RATE_LIMIT = 0.90


class SubmissionValidationError(ValueError):
    """Raised when a CSV does not satisfy the Lab 2 submission contract."""


@dataclass(frozen=True)
class AnalysisResult:
    mahalanobis_sq: np.ndarray
    mean_mahalanobis: float
    chi_square_pass_rate: float
    position_rmse: float
    min_covariance_eigenvalue: float

    @property
    def mean_passed(self) -> bool:
        return self.mean_mahalanobis <= MEAN_MAHALANOBIS_LIMIT

    @property
    def coverage_passed(self) -> bool:
        return self.chi_square_pass_rate >= PASS_RATE_LIMIT

    @property
    def passed(self) -> bool:
        return self.mean_passed and self.coverage_passed


def load_submission(path: Path) -> np.ndarray:
    """Load a CSV after strictly validating its schema and numeric values."""

    if not path.is_file():
        raise SubmissionValidationError(f"file does not exist: {path}")

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as error:
            raise SubmissionValidationError("CSV is empty") from error

        if header != EXPECTED_COLUMNS:
            raise SubmissionValidationError(
                f"columns must be exactly {list(EXPECTED_COLUMNS)}, got {list(header)}"
            )

        rows: list[list[float]] = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(EXPECTED_COLUMNS):
                raise SubmissionValidationError(
                    f"line {line_number} has {len(row)} columns; "
                    f"expected {len(EXPECTED_COLUMNS)}"
                )
            try:
                rows.append([float(value) for value in row])
            except ValueError as error:
                raise SubmissionValidationError(
                    f"line {line_number} contains a non-numeric value"
                ) from error

    if len(rows) != EXPECTED_ROWS:
        raise SubmissionValidationError(
            f"CSV must contain exactly {EXPECTED_ROWS} data rows; got {len(rows)}"
        )

    values = np.asarray(rows, dtype=np.float64)
    if values.shape != (EXPECTED_ROWS, len(EXPECTED_COLUMNS)):
        raise SubmissionValidationError(f"invalid data shape: {values.shape}")
    if not np.all(np.isfinite(values)):
        bad_rows = np.flatnonzero(~np.isfinite(values).all(axis=1)) + 2
        raise SubmissionValidationError(
            f"NaN or infinite value on CSV line(s): {bad_rows.tolist()}"
        )
    return values


def position_covariances(values: np.ndarray) -> np.ndarray:
    """Reconstruct the symmetric 2x2 posterior position covariances."""

    covariances = np.empty((values.shape[0], 2, 2), dtype=np.float64)
    covariances[:, 0, 0] = values[:, 4]
    covariances[:, 0, 1] = values[:, 5]
    covariances[:, 1, 0] = values[:, 5]
    covariances[:, 1, 1] = values[:, 6]
    return covariances


def analyze_values(values: np.ndarray) -> AnalysisResult:
    """Calculate the two published automarker metrics."""

    values = np.asarray(values, dtype=np.float64)
    expected_shape = (EXPECTED_ROWS, len(EXPECTED_COLUMNS))
    if values.shape != expected_shape:
        raise SubmissionValidationError(
            f"values must have shape {expected_shape}; got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise SubmissionValidationError("values contain NaN or infinity")

    covariances = position_covariances(values)
    eigenvalues = np.linalg.eigvalsh(covariances)
    min_eigenvalue = float(np.min(eigenvalues))
    if min_eigenvalue <= 0.0:
        bad_rows = np.flatnonzero(np.min(eigenvalues, axis=1) <= 0.0) + 2
        raise SubmissionValidationError(
            "position covariance must be positive definite; invalid CSV "
            f"line(s): {bad_rows.tolist()}"
        )

    errors = values[:, 0:2] - values[:, 2:4]
    mahalanobis_sq = np.array(
        [
            float(error @ np.linalg.solve(covariance, error))
            for error, covariance in zip(errors, covariances, strict=True)
        ],
        dtype=np.float64,
    )

    return AnalysisResult(
        mahalanobis_sq=mahalanobis_sq,
        mean_mahalanobis=float(np.mean(mahalanobis_sq)),
        chi_square_pass_rate=float(
            np.mean(mahalanobis_sq <= CHI_SQUARE_95_2DOF)
        ),
        position_rmse=float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))),
        min_covariance_eigenvalue=min_eigenvalue,
    )


def analyze_submission(path: Path) -> tuple[np.ndarray, AnalysisResult]:
    values = load_submission(path)
    return values, analyze_values(values)


def save_plot(values: np.ndarray, result: AnalysisResult, output: Path) -> None:
    """Save a trajectory and consistency diagnostic plot."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(values[:, 0], values[:, 1], label="Simulator truth")
    axes[0].plot(values[:, 2], values[:, 3], label="EKF estimate")
    axes[0].set_title("Position trajectories")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    steps = np.arange(1, EXPECTED_ROWS + 1)
    axes[1].plot(steps, result.mahalanobis_sq, label="Position statistic")
    axes[1].axhline(
        CHI_SQUARE_95_2DOF,
        color="tab:red",
        linestyle="--",
        label="95% chi-square gate",
    )
    axes[1].set_title("Position consistency")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Squared Mahalanobis distance")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def print_result(path: Path, result: AnalysisResult) -> None:
    print(f"CSV: {path}")
    print(f"Rows: {EXPECTED_ROWS}")
    print(
        "Mean squared Mahalanobis distance: "
        f"{result.mean_mahalanobis:.4f} "
        f"({_status(result.mean_passed)}; <= {MEAN_MAHALANOBIS_LIMIT:.1f})"
    )
    print(
        "Chi-square pass rate: "
        f"{result.chi_square_pass_rate:.3f} "
        f"({_status(result.coverage_passed)}; >= {PASS_RATE_LIMIT:.2f})"
    )
    print(f"Position RMSE: {result.position_rmse:.4f} m")
    print(
        "Minimum covariance eigenvalue: "
        f"{result.min_covariance_eigenvalue:.6g}"
    )
    print(f"Overall: {_status(result.passed)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Lab 2 CSV")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--plot",
        nargs="?",
        const="auto",
        help="save diagnostics, optionally to the supplied PNG path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        values, result = analyze_submission(args.csv_path)
    except SubmissionValidationError as error:
        print(f"INVALID: {error}", file=sys.stderr)
        return 2

    print_result(args.csv_path, result)
    if args.plot is not None:
        plot_path = (
            args.csv_path.with_name(f"{args.csv_path.stem}_analysis.png")
            if args.plot == "auto"
            else Path(args.plot)
        )
        save_plot(values, result, plot_path)
        print(f"Plot: {plot_path}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
