"""Check a Lab 2 hardware run from its diagnostic log.

The submission metrics say whether the filter was consistent; they do not say
whether it was fed real sensors.  A run can pass both thresholds while the
speed "measurement" is still the command echoed back.  This reads
``logs/lab2_diagnostics.csv`` and reports, per check, what the log proves and
what to change when it fails.

Usage:  python labs/lab2/check_run.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

try:
    from . import lab2
except ImportError:
    import lab2

LAB_DIR = Path(__file__).resolve().parent
LOG_PATH = LAB_DIR.parents[1] / "logs" / "lab2_diagnostics.csv"

EXPECTED_STEPS = 200
DT = 0.1
CHI2_95_2DOF = 5.991464547107979

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "n/a "


def load(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} is empty")
    columns = rows[0].keys()
    return {
        name: np.array([float(row[name]) for row in rows], dtype=np.float64)
        for name in columns
    }


class Report:
    def __init__(self) -> None:
        self.failures: list[tuple[str, str]] = []
        self.warnings: list[tuple[str, str]] = []

    def add(self, status: str, title: str, detail: str, fix: str = "") -> None:
        print(f"  [{status}] {title}")
        print(f"         {detail}")
        if status == FAIL:
            self.failures.append((title, fix))
        elif status == WARN:
            self.warnings.append((title, fix))


def check_completeness(log: dict[str, np.ndarray], report: Report) -> None:
    steps = len(log["step"])
    if steps == EXPECTED_STEPS:
        report.add(PASS, "Run length", f"{steps} steps recorded.")
    else:
        report.add(
            FAIL,
            "Run length",
            f"{steps} steps, expected {EXPECTED_STEPS}.",
            "The run was interrupted. The submission CSV is only written after "
            "all 200 updates, so re-run before submitting.",
        )


def check_timing(log: dict[str, np.ndarray], report: Report) -> None:
    intervals = np.diff(log["t_rel"])
    if intervals.size == 0:
        return
    median = float(np.median(intervals))
    worst = float(np.max(np.abs(intervals - DT)))
    detail = (
        f"median period {median * 1000:.1f} ms, worst deviation "
        f"{worst * 1000:.1f} ms (EKF assumes {DT * 1000:.0f} ms)."
    )
    if abs(median - DT) <= 0.01:
        report.add(PASS, "Control period", detail)
    else:
        report.add(
            FAIL,
            "Control period",
            detail,
            f"EKF hard-codes dt={DT}. A median period this far off scales every "
            "predicted displacement. Usually the BLE reads are blocking longer "
            "than the loop budget: check for Bluetooth interference, or lower "
            "N_STEPS rather than silently accepting the drift.",
        )


def _valid_fraction(values: np.ndarray) -> float:
    return float(np.mean(np.isfinite(values)))


def check_speed_source(log: dict[str, np.ndarray], report: Report) -> None:
    encoder = log["enc_speed_m_s"]
    available = _valid_fraction(encoder)
    if available == 0.0:
        report.add(
            FAIL,
            "Speed from encoders",
            "enc_speed_m_s is empty for every step.",
            "info['velocity'] never arrived, so the filter silently fell back "
            "to the observation, which on hardware is the command echoed back. "
            "Check that the run really used hardware (no --sim) and that the "
            "velocity sensor stream is enabled in the wrapper.",
        )
        return

    z_speed, commanded = log["z_speed"], log["speed_cmd"]
    finite = np.isfinite(z_speed) & np.isfinite(encoder)
    echoes = float(np.mean(np.isclose(z_speed[finite], commanded[finite])))
    matches_encoder = float(np.mean(np.isclose(z_speed[finite], encoder[finite])))

    detail = (
        f"encoder present on {available:.0%} of steps; the value fed to the "
        f"filter matched the encoder on {matches_encoder:.0%} and the command "
        f"on {echoes:.0%}."
    )
    if matches_encoder >= 0.9:
        report.add(PASS, "Speed from encoders", detail)
    elif echoes >= 0.9:
        report.add(
            FAIL,
            "Speed from encoders",
            detail,
            "The filter is still being fed its own command. "
            "robot_speed_measurement() returned None, so check the shape of "
            "info['velocity'] against what that function expects.",
        )
    else:
        report.add(WARN, "Speed from encoders", detail,
                   "Mixed sources; inspect enc_speed_m_s for dropped frames.")


def _wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def check_heading_source(log: dict[str, np.ndarray], report: Report) -> None:
    """Rebuild the heading from the raw yaw and see whether the filter used it.

    Comparing z_heading against obs_heading cannot answer this: on hardware
    the observation *is* the commanded heading, and a robot that tracks its
    command well makes an honest IMU reading agree with it.  Replaying
    ImuHeadingSensor's arithmetic on the logged yaw is a direct test instead.
    """

    yaw = log["imu_yaw_deg"]
    available = _valid_fraction(yaw)
    if available == 0.0:
        report.add(
            FAIL,
            "Heading from IMU",
            "imu_yaw_deg is empty for every step.",
            "info['orientation'] never arrived, so heading fell back to the "
            "observation, which on hardware echoes the command. Confirm the "
            "attitude stream is enabled for this toy type.",
        )
        return

    z_heading = log["z_heading"]
    usable = np.isfinite(yaw) & np.isfinite(z_heading)
    yaw_seen, z_seen = yaw[usable], z_heading[usable]
    turned = _wrap(np.deg2rad(yaw_seen - yaw_seen[0]))

    errors = {
        sign: float(np.max(np.abs(_wrap(_wrap(z_seen[0] + sign * turned) - z_seen))))
        for sign in (1.0, -1.0)
    }
    sign, error = min(errors.items(), key=lambda item: item[1])

    if error < 1e-6:
        report.add(
            PASS,
            "Heading from IMU",
            f"IMU yaw present on {available:.0%} of steps, and the heading fed "
            f"to the filter reproduces it exactly (sign {sign:+.0f}).",
        )
        return

    observed = log["obs_heading"][usable]
    fell_back = float(np.mean(np.isclose(z_seen, observed)))
    report.add(
        FAIL,
        "Heading from IMU",
        f"IMU yaw present on {available:.0%} of steps, but the heading fed to "
        f"the filter departs from it by up to {np.degrees(error):.1f} deg; it "
        f"matched the plain observation on {fell_back:.0%} of steps.",
        "The IMU source was disabled mid-run, which the guard only does after "
        "10 consecutive steps of >120 deg disagreement. Check the next line "
        "for the sign, and look for the IMU_YAW_SIGN warning in the console.",
    )


def check_imu_sign(log: dict[str, np.ndarray], report: Report) -> None:
    yaw, commanded = log["imu_yaw_deg"], log["heading_cmd"]
    finite = np.isfinite(yaw) & np.isfinite(commanded)
    if finite.sum() < 20:
        report.add(SKIP, "IMU yaw sign", "Not enough IMU samples to judge.")
        return

    measured = np.unwrap(np.deg2rad(yaw[finite]))
    expected = np.unwrap(commanded[finite])
    if np.ptp(expected) < np.deg2rad(45.0):
        report.add(SKIP, "IMU yaw sign",
                   "The robot barely turned; the sign cannot be judged.")
        return

    slope = float(np.polyfit(expected, measured, 1)[0])
    configured = float(lab2.IMU_YAW_SIGN)
    detail = (
        f"IMU yaw tracked the commanded heading with slope {slope:+.2f}; "
        f"IMU_YAW_SIGN is {configured:+.0f}."
    )
    if abs(slope) > 0.5 and np.sign(slope) == np.sign(configured):
        report.add(PASS, "IMU yaw sign", detail + " They agree.")
    elif abs(slope) > 0.5:
        report.add(
            FAIL,
            "IMU yaw sign",
            detail + " They disagree.",
            f"Set IMU_YAW_SIGN = {-configured:+.1f} in labs/lab2/lab2.py and "
            "re-run. This run's heading fell back to the observation once the "
            "guard tripped, so its estimate is not the one the fix produces.",
        )
    else:
        report.add(
            WARN,
            "IMU yaw sign",
            detail + " Too weak to call.",
            "The robot may not have followed the commanded heading. Check the "
            "trajectory plot before trusting either sign.",
        )


def check_consistency(log: dict[str, np.ndarray], report: Report) -> None:
    nis = log["nis"][np.isfinite(log["nis"])]
    if nis.size == 0:
        return
    mean_nis = float(np.mean(nis))
    inside = float(np.mean(nis <= CHI2_95_2DOF))
    detail = (
        f"mean NIS {mean_nis:.2f} (theory 2.00 for 2 DoF); "
        f"{inside:.0%} within the 95% gate."
    )
    if 1.0 <= mean_nis <= 4.0 and inside >= 0.90:
        report.add(PASS, "Filter consistency", detail)
    elif mean_nis > 4.0 or inside < 0.90:
        report.add(
            FAIL,
            "Filter consistency",
            detail,
            "The filter is overconfident: real errors exceed what P claims. "
            "Increase REAL_PROCESS_NOISE in lab2.py, or R in EKF.py if the "
            "innovations (nu_heading, nu_speed) are noisy rather than biased. "
            "A constant non-zero innovation is a bias, not noise -- fix the "
            "measurement instead of inflating the covariance.",
        )
    else:
        report.add(
            WARN,
            "Filter consistency",
            detail,
            "The filter is pessimistic: it passes the thresholds but claims "
            "more uncertainty than it has. REAL_PROCESS_NOISE can come down, "
            "which is exactly what its comment asks for after a hardware run.",
        )


def check_innovation_bias(log: dict[str, np.ndarray], report: Report) -> None:
    for name, column in (("heading", "nu_heading"), ("speed", "nu_speed")):
        values = log[column][np.isfinite(log[column])]
        if values.size == 0:
            continue
        mean, spread = float(np.mean(values)), float(np.std(values))
        detail = f"{name} innovation mean {mean:+.4f}, sd {spread:.4f}."
        if spread > 0 and abs(mean) > 2.0 * spread:
            report.add(
                FAIL,
                f"Innovation bias ({name})",
                detail,
                "The innovation sits off zero by more than twice its own "
                "spread, so the model and the measurement disagree "
                "systematically. For speed this points at speed_gain in "
                "MODEL_CONFIG; re-estimate it by differentiating the locator "
                "track rather than widening R.",
            )
        else:
            report.add(PASS, f"Innovation bias ({name})", detail)


def main() -> int:
    if not LOG_PATH.exists():
        raise SystemExit(f"No diagnostic log at {LOG_PATH}. Run lab2.py first.")

    log = load(LOG_PATH)
    hardware = _valid_fraction(log["imu_yaw_deg"]) > 0.0 or _valid_fraction(
        log["enc_speed_m_s"]
    ) > 0.0

    print(f"\nDiagnostic log: {LOG_PATH}")
    print(f"Mode detected : {'HARDWARE' if hardware else 'SIMULATION'}\n")
    if not hardware:
        print("  This log has no hardware sensor columns, so the sensor-source")
        print("  checks below cannot say anything. Run without --sim.\n")

    report = Report()
    check_completeness(log, report)
    check_timing(log, report)
    if hardware:
        check_speed_source(log, report)
        check_heading_source(log, report)
        check_imu_sign(log, report)
    check_innovation_bias(log, report)
    check_consistency(log, report)

    print()
    if report.failures:
        print(f"{len(report.failures)} check(s) failed. What to change:\n")
        for title, fix in report.failures:
            print(f"  {title}:\n    {fix}\n")
        return 1
    if report.warnings:
        print(f"All checks passed, {len(report.warnings)} with reservations:\n")
        for title, fix in report.warnings:
            print(f"  {title}:\n    {fix}\n")
        return 0
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
