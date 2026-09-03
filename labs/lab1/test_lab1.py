"""Offline regression tests for the Lab 1 controller and submission pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import analyze_lab1
import lab1
from dynamics import MODEL_CONFIG, dynamics, wrap_angle


def test_target_matches_the_assignment_specification() -> None:
    """instructions.md requires (0.5 m, 0.5 m) relative to the start."""

    np.testing.assert_allclose(analyze_lab1.TARGET, [0.5, 0.5])


def test_controller_and_marker_share_one_target() -> None:
    """A second copy of TARGET once let the two disagree silently."""

    assert lab1.TARGET is analyze_lab1.TARGET


def test_submission_shape_constants_agree() -> None:
    assert lab1.N_STEPS == analyze_lab1.EXPECTED_ROWS == 100
    assert lab1.REQUIRED_COLUMNS == ("sim_x", "sim_y", "real_x", "real_y")


def test_minimum_speed_clears_the_model_deadband() -> None:
    """A floor inside the deadband is commanded but produces no motion."""

    deadband = MODEL_CONFIG["command_deadband_m_s"]
    assert lab1.ControllerConfig().min_speed > deadband

    # The floor must actually move the robot from rest.
    state = np.array([0.0, 0.0, 0.0, 0.0])
    action = np.array([lab1.ControllerConfig().min_speed, 0.0])
    assert dynamics(state, action)[3] > 0.0


def test_config_rejects_a_floor_inside_the_deadband() -> None:
    with pytest.raises(ValueError, match="deadband"):
        lab1.ControllerConfig(min_speed=0.02)


def test_controller_stops_inside_the_stop_tolerance() -> None:
    controller = lab1.PositionPDController()
    near_target = np.array(
        [lab1.TARGET[0], lab1.TARGET[1] - 0.01, 0.3, 0.0], dtype=np.float32
    )
    action = controller.compute(near_target)
    assert action[0] == pytest.approx(0.0)
    assert action[1] == pytest.approx(0.3, abs=1e-6)


def test_controller_points_at_the_target_using_the_y_forward_convention() -> None:
    """Heading 0 is +y and heading +pi/2 is +x in the supplied environment."""

    controller = lab1.PositionPDController()
    # Target is due +y from the robot.
    action = controller.compute(
        np.array([lab1.TARGET[0], lab1.TARGET[1] - 0.4, 0.0, 0.0], dtype=np.float32)
    )
    assert action[1] == pytest.approx(0.0, abs=1e-6)

    controller = lab1.PositionPDController()
    # Target is due +x from the robot.
    action = controller.compute(
        np.array([lab1.TARGET[0] - 0.4, lab1.TARGET[1], 0.0, 0.0], dtype=np.float32)
    )
    assert action[1] == pytest.approx(np.pi / 2.0, abs=1e-6)


def test_simulation_reaches_the_target_within_the_marker_threshold() -> None:
    with lab1.open_sim_env(False) as environment:
        trajectory = lab1.run_simulation(environment, render=False)

    assert len(trajectory) == lab1.N_STEPS
    final_distance = float(np.linalg.norm(trajectory[-1] - lab1.TARGET))
    assert final_distance <= analyze_lab1.FINAL_DISTANCE_LIMIT


def test_simulation_does_not_stall_at_the_deadband_radius() -> None:
    """Regression: min_speed inside the deadband stalled at deadband / kp."""

    stall_radius = (
        MODEL_CONFIG["command_deadband_m_s"] / lab1.ControllerConfig().kp
    )
    with lab1.open_sim_env(False) as environment:
        trajectory = lab1.run_simulation(environment, render=False)

    final_distance = float(np.linalg.norm(trajectory[-1] - lab1.TARGET))
    assert final_distance < 0.5 * stall_radius


def test_write_submission_round_trips_through_the_validator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lab1, "LAB_DIR", tmp_path)
    sim = [np.array([0.005 * i, 0.005 * i]) for i in range(lab1.N_STEPS)]
    real = [np.array([0.005 * i + 0.01, 0.005 * i]) for i in range(lab1.N_STEPS)]

    output = lab1.write_submission("33377006", sim, real)
    loaded_sim, loaded_real = analyze_lab1.read_submission(output)

    assert output.name == "33377006_lab1.csv"
    assert loaded_sim.shape == (lab1.N_STEPS, 2)
    np.testing.assert_allclose(loaded_sim, np.asarray(sim))
    np.testing.assert_allclose(loaded_real, np.asarray(real))
    assert not list(tmp_path.glob("*.tmp"))


def test_write_submission_rejects_a_short_trajectory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lab1, "LAB_DIR", tmp_path)
    short = [np.zeros(2)] * 10
    full = [np.zeros(2)] * lab1.N_STEPS

    with pytest.raises(ValueError, match="exactly 100"):
        lab1.write_submission("33377006", short, full)
    assert not list(tmp_path.iterdir())


def test_write_submission_rejects_non_numeric_student_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lab1, "LAB_DIR", tmp_path)
    rows = [np.zeros(2)] * lab1.N_STEPS

    with pytest.raises(ValueError, match="digits only"):
        lab1.write_submission("abc", rows, rows)


def test_write_submission_rejects_non_finite_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lab1, "LAB_DIR", tmp_path)
    rows = [np.zeros(2)] * lab1.N_STEPS
    bad = list(rows)
    bad[5] = np.array([np.nan, 0.0])

    with pytest.raises(ValueError, match="NaN or infinite"):
        lab1.write_submission("33377006", bad, rows)
    assert not list(tmp_path.iterdir())


def test_validator_rejects_a_wrong_row_count(tmp_path: Path) -> None:
    output = tmp_path / "short_lab1.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(analyze_lab1.REQUIRED_COLUMNS)
        writer.writerow([0.0, 0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="100 data rows"):
        analyze_lab1.read_submission(output)


def test_submitted_csv_still_passes_every_marker_threshold() -> None:
    """Guards the recorded hardware run against an accidental overwrite."""

    submission = Path(__file__).with_name("33377006_lab1.csv")
    if not submission.is_file():
        pytest.skip("no hardware submission present")

    sim, real = analyze_lab1.read_submission(submission)
    limit = analyze_lab1.FINAL_DISTANCE_LIMIT
    assert float(np.linalg.norm(sim[-1] - analyze_lab1.TARGET)) <= limit
    assert float(np.linalg.norm(real[-1] - analyze_lab1.TARGET)) <= limit
    rmse = float(np.sqrt(np.mean(np.sum((sim - real) ** 2, axis=1))))
    assert rmse <= analyze_lab1.TRAJECTORY_RMSE_LIMIT


def test_wrap_angle_normalises_to_the_half_open_interval() -> None:
    assert wrap_angle(3.0 * np.pi) == pytest.approx(-np.pi)
    assert wrap_angle(-3.0 * np.pi) == pytest.approx(-np.pi)
    assert wrap_angle(0.5) == pytest.approx(0.5)
