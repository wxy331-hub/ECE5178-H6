"""Offline regression tests for the Lab 2 EKF pipeline."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

import analyze_lab2
import lab2
from EKF import EKF, MODEL_CONFIG, _dynamics_float64, dynamics, wrap_angle


LAB1_DYNAMICS_PATH = Path(__file__).parents[1] / "lab1" / "dynamics.py"


def _load_lab1_dynamics_module():
    spec = importlib.util.spec_from_file_location(
        "lab1_dynamics_for_test", LAB1_DYNAMICS_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Lab 1 dynamics")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dynamics_uses_lab1_equations_with_lab2_speed_calibration() -> None:
    lab1 = _load_lab1_dynamics_module()
    assert MODEL_CONFIG["speed_gain"] == pytest.approx(2.47)
    assert MODEL_CONFIG["speed_gain"] < lab1.MODEL_CONFIG["speed_gain"]

    # Equal parameters must produce identical outputs, demonstrating that only
    # the calibrated constant changed and the Lab 1 model equations did not.
    lab1.MODEL_CONFIG["speed_gain"] = MODEL_CONFIG["speed_gain"]
    rng = np.random.default_rng(5178)

    for _ in range(1000):
        state = np.array(
            [
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-0.5, 0.5),
            ]
        )
        action = np.array(
            [rng.uniform(-0.6, 0.6), rng.uniform(-np.pi, np.pi)]
        )
        np.testing.assert_array_equal(
            dynamics(state, action), lab1.dynamics(state, action)
        )


def test_process_jacobian_matches_directional_difference() -> None:
    state = np.array([0.2, -0.1, 0.4, 0.08])
    action = np.array([0.12, 1.1])
    direction = np.array([0.3, -0.5, 0.2, 0.7])
    epsilon = 1e-6

    jacobian = EKF.process_jacobian(state, action)
    actual = (
        _dynamics_float64(state + epsilon * direction, action)
        - _dynamics_float64(state, action)
    )
    actual[2] = wrap_angle(actual[2])
    predicted = epsilon * jacobian @ direction

    np.testing.assert_allclose(actual, predicted, atol=1e-10, rtol=1e-5)


def test_update_wraps_heading_innovation_and_ignores_position() -> None:
    initial_state = np.array([0.0, 0.0, np.pi - 0.01, 0.0])
    first = EKF(initial_state=initial_state)
    second = EKF(initial_state=initial_state)

    first.update(np.array([100.0, -100.0, -np.pi + 0.01, 0.0, 0.0]))
    second.update(np.array([-50.0, 80.0, -np.pi + 0.01, 0.0, 1.0]))

    assert first.last_innovation[0] == pytest.approx(0.02)
    np.testing.assert_allclose(first.state_est, second.state_est)
    np.testing.assert_allclose(first.P, second.P)


def test_covariance_remains_symmetric_positive_definite() -> None:
    ekf = EKF()
    rng = np.random.default_rng(22)

    for _ in range(500):
        action = np.array(
            [rng.uniform(-0.15, 0.15), rng.uniform(-np.pi, np.pi)]
        )
        ekf.predict(action)
        measurement = np.array(
            [
                wrap_angle(ekf.state_est[2] + rng.normal(0.0, 0.025)),
                ekf.state_est[3] + rng.normal(0.0, 0.025),
            ]
        )
        ekf.update(measurement)

        np.testing.assert_allclose(ekf.P, ekf.P.T, atol=1e-12)
        assert np.linalg.eigvalsh(ekf.P).min() > 0.0
        assert np.all(np.isfinite(ekf.state_est))


def test_scripted_actions_form_four_square_legs() -> None:
    initial_heading = 0.3
    expected_relative = (0.0, np.pi / 2.0, np.pi, -np.pi / 2.0)
    for step, relative_heading in zip(
        (0, 50, 100, 150), expected_relative, strict=True
    ):
        action = lab2.scripted_action(step, initial_heading)
        assert action[0] == pytest.approx(0.10)
        assert action[1] == pytest.approx(
            wrap_angle(initial_heading + relative_heading)
        )


def test_full_simulation_and_csv_analysis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lab2.time, "sleep", lambda _: None)
    monkeypatch.setattr(lab2, "LAB_DIR", tmp_path)

    with lab2.open_sim_env(False) as environment:
        records = lab2.run_experiment(
            environment,
            None,
            render=False,
            teleop=False,
            seed=5178,
        )

    output = lab2.write_submission("33377006", records)
    values, result = analyze_lab2.analyze_submission(output)

    assert output.name == "33377006_lab2.csv"
    assert values.shape == (200, 7)
    assert np.all(np.isfinite(values))
    assert result.mean_mahalanobis <= 4.0
    assert result.chi_square_pass_rate >= 0.90
    assert result.min_covariance_eigenvalue > 0.0


def test_default_noise_is_the_smallest_100_seed_pass_candidate() -> None:
    ekf = EKF()
    assert ekf.Q[0, 0] == pytest.approx(3.125e-6)
    assert ekf.Q[1, 1] == pytest.approx(3.125e-6)
    assert ekf.R[0, 0] == pytest.approx(0.025**2)
    assert ekf.R[1, 1] == pytest.approx(0.025**2)


def test_terminal_metrics_include_values_thresholds_and_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = lab2.EstimateRecord(0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1)
    lab2.print_submission_metrics([record] * lab2.N_STEPS)
    output = capsys.readouterr().out

    assert "Mean Mahalanobis distance of position error (sim - real)" in output
    assert "0.0000 (PASS; required <= 4.0)" in output
    assert "Chi-square pass rate (2 DoF, 95% gate)" in output
    assert "1.000 (PASS; required >= 0.90)" in output


def test_stop_motion_stops_robot_before_visualization_hold() -> None:
    sim_env = Mock()
    robot_env = Mock()

    lab2.stop_motion(sim_env, robot_env)

    robot_env.emergency_stop.assert_called_once_with()
    sim_env.emergency_stop.assert_called_once_with()


def test_real_process_noise_reflects_hardware_residual() -> None:
    default = EKF().Q
    assert lab2.REAL_PROCESS_NOISE[0, 0] == pytest.approx(
        10.0 * default[0, 0]
    )
    assert lab2.REAL_PROCESS_NOISE[1, 1] == pytest.approx(
        10.0 * default[1, 1]
    )


def test_csv_validator_rejects_wrong_row_count(tmp_path: Path) -> None:
    output = tmp_path / "short_lab2.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(analyze_lab2.EXPECTED_COLUMNS)
        writer.writerow([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1])

    with pytest.raises(
        analyze_lab2.SubmissionValidationError,
        match="exactly 200",
    ):
        analyze_lab2.load_submission(output)


def test_csv_validator_rejects_non_positive_covariance(tmp_path: Path) -> None:
    output = tmp_path / "invalid_covariance_lab2.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(analyze_lab2.EXPECTED_COLUMNS)
        writer.writerows(
            [[0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.1]]
            * analyze_lab2.EXPECTED_ROWS
        )

    values = analyze_lab2.load_submission(output)
    with pytest.raises(
        analyze_lab2.SubmissionValidationError,
        match="positive definite",
    ):
        analyze_lab2.analyze_values(values)
