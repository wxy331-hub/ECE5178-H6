"""Offline regression tests for the Lab 2 EKF pipeline."""

from __future__ import annotations

import csv
import importlib.util
import sys
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


def test_dynamics_equations_match_lab1_under_lab1_parameters() -> None:
    """Same equations as the submitted model; parameters re-calibrated.

    instructions.md asks for two things that collide once a Lab 1 parameter is
    contradicted by measurement: use the Lab 1 motion model, and calibrate the
    dynamics against the real robot.  The resolution is that the *equations*
    stay identical -- a filter solving different equations would describe a
    different robot -- while parameters may be re-estimated, which is what the
    calibration requirement asks for.

    So this feeds Lab 1's own parameters into Lab 2's dynamics and demands
    bit-identical output: it pins the equations without freezing the numbers.
    See test_the_deadband_was_re_estimated_from_hardware for the numbers.
    """

    lab1 = _load_lab1_dynamics_module()
    saved = dict(MODEL_CONFIG)
    MODEL_CONFIG.update(lab1.MODEL_CONFIG)
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
        try:
            np.testing.assert_array_equal(
                dynamics(state, action), lab1.dynamics(state, action)
            )
        except BaseException:
            MODEL_CONFIG.clear()
            MODEL_CONFIG.update(saved)
            raise
    MODEL_CONFIG.clear()
    MODEL_CONFIG.update(saved)


def test_the_deadband_was_re_estimated_from_hardware() -> None:
    """Record which parameters diverged from Lab 1, and why.

    Lab 1's deadband of 0.0322 is contradicted twice over: its own run stalled
    0.0633 m short while still commanding 0.0506 m/s, and at a 0.07 command the
    encoders read 0.0318 m/s, implying 0.0582.  The gain is untouched, so the
    two labs still agree on how command maps to speed once the robot moves.
    """

    lab1 = _load_lab1_dynamics_module()
    assert MODEL_CONFIG["speed_gain"] == pytest.approx(
        lab1.MODEL_CONFIG["speed_gain"]
    )
    assert MODEL_CONFIG["command_deadband_m_s"] > lab1.MODEL_CONFIG[
        "command_deadband_m_s"
    ]
    assert 0.045 <= MODEL_CONFIG["command_deadband_m_s"] <= 0.065
    assert MODEL_CONFIG["dt"] > lab1.MODEL_CONFIG["dt"]  # measured 104.5 ms


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
    legs = lab2.N_STEPS // lab2.LEG_STEPS
    checkpoints = [leg * lab2.LEG_STEPS for leg in range(legs)]
    for step, relative_heading in zip(
        checkpoints, [expected_relative[i % 4] for i in range(legs)], strict=True
    ):
        action = lab2.scripted_action(step, initial_heading)
        assert action[0] == pytest.approx(lab2.SCRIPT_SPEED)
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


def test_observation_range_must_cover_the_model_response() -> None:
    """The command limit and the sensor range are different quantities.

    Both environments call the parameter ``vel_limit``, and using one value
    for both silently saturated every simulated speed reading.

    This deliberately does not use SCRIPT_SPEED.  The two quantities have to
    stay separate whatever speed the script happens to command, so the bound
    that matters is the fastest command the hardware accepts: if the model
    outruns the range even there, tying them together is wrong in general.
    Anchoring the assertion to the working point instead would let a slower
    script silently satisfy it while the underlying confusion remained.
    """

    steady_at_cap = MODEL_CONFIG["speed_gain"] * (
        lab2.COMMAND_SPEED_LIMIT - MODEL_CONFIG["command_deadband_m_s"]
    )
    assert steady_at_cap > lab2.COMMAND_SPEED_LIMIT  # why it mattered
    assert lab2.SIM_SPEED_LIMIT > steady_at_cap  # why it is fixed now


def test_simulated_speed_readings_are_no_longer_clipped() -> None:
    """Drive at the command cap, not at SCRIPT_SPEED.

    The point is that the observation range has to cover the model's response
    to the fastest command the hardware accepts.  Testing at the scripted
    speed instead ties the assertion to the working point: after the deadband
    was re-estimated, raw 8 settles at 0.14 m/s and the old assertion failed
    even though nothing about the clipping bug had changed.
    """

    fastest = np.array([lab2.COMMAND_SPEED_LIMIT, 0.0], dtype=np.float32)
    with lab2.open_sim_env(False) as environment:
        environment.reset(seed=5178)
        speeds = [float(environment.step(fastest)[0][3]) for _ in range(45)]

    # Under the old range every steady-state reading came back pinned to the
    # limit; the true response has to be observable instead.
    assert max(speeds) > lab2.COMMAND_SPEED_LIMIT
    assert max(speeds) < lab2.SIM_SPEED_LIMIT


def test_hardware_keeps_the_scale_the_wrapper_converts_commands_with() -> None:
    """Robot builds the raw byte as speed_cmd / vel_limit * raw_speed_limit.

    Widening it for the hardware environment would quietly slow the robot
    down: a 0.10 command would send 3/15 instead of 10/15.
    """

    environment = lab2.make_real_env(Mock())
    assert environment.vel_limit == pytest.approx(lab2.COMMAND_SPEED_LIMIT)
    assert environment.vel_limit == pytest.approx(0.15)


def wrap_angle_degrees(degrees: float) -> float:
    """Wrap to [-180, 180), the range the IMU attitude sensor reports."""

    return (degrees + 180.0) % 360.0 - 180.0


def _hardware_info(
    *,
    vx: float = 0.0,
    vy: float = 18.24,
    speed_cmd: float = 0.10,
    yaw: float = 0.0,
) -> dict:
    """An info dict shaped like the one Robot.step() returns."""

    return {
        "state_odom": np.zeros(4, dtype=np.float32),
        "state_true": np.zeros(4, dtype=np.float32),
        "velocity": {"x": vx, "y": vy},
        "orientation": {"yaw": yaw, "pitch": 0.0, "roll": 0.0},
        "gyroscope": {"x": 0.0, "y": 0.0, "z": 0.0},
        "speed_cmd": speed_cmd,
        "heading_cmd": 0.0,
    }


def _honest_yaw(heading_rad: float, origin: float = 0.0,
                initial_rad: float = 0.0) -> float:
    """The yaw a correctly-signed IMU reports when the robot holds a heading.

    ImuHeadingSensor rebuilds heading as ``initial + SIGN * turned``, so an
    honest sensor whose zero sits at ``origin`` reports
    ``origin + SIGN * (heading - initial)``.  Deriving the test data from
    IMU_YAW_SIGN keeps these tests about the logic -- zero cancellation,
    the transient guard, the fallback -- rather than about one sign that the
    hardware later contradicted.
    """

    turned = np.degrees(wrap_angle(heading_rad - initial_rad))
    return wrap_angle_degrees(origin + lab2.IMU_YAW_SIGN * turned)


def _fake_robot() -> Mock:
    robot = Mock()
    observation = np.array([0.0, 0.0, 0.0, 0.10, 0.0], dtype=np.float32)
    robot.reset.return_value = (observation, _hardware_info())

    def step(action):
        # A well-behaved robot: its IMU follows the commanded heading, while
        # observation[2] stays at zero the way a command echo would.
        info = _hardware_info(yaw=_honest_yaw(float(action[1])))
        return observation, None, False, False, info

    robot.step.side_effect = step
    return robot


def test_hardware_speed_comes_from_the_encoder_not_the_command() -> None:
    """api.get_speed() returns the commanded target, so obs[3] is an echo.

    In the 2026-08-14 log the speed column equals speed_cmd in all 200 rows.
    The encoder reading in info["velocity"] is the actual sensor.
    """

    observation = np.array([0.0, 0.0, 0.3, 0.10, 0.0])
    z = lab2.extract_measurement(observation, _hardware_info(vx=3.0, vy=4.0))

    assert z[1] == pytest.approx(0.05)  # hypot(3, 4) cm/s, not the command
    assert z[0] == pytest.approx(0.3)  # heading still taken from the obs


def test_hardware_speed_takes_its_sign_from_the_command() -> None:
    z = lab2.extract_measurement(
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        _hardware_info(vx=0.0, vy=-8.0, speed_cmd=-0.10),
    )
    assert z[1] == pytest.approx(-0.08)


@pytest.mark.parametrize(
    "velocity", [None, {}, "unavailable", {"x": float("nan"), "y": 0.0}]
)
def test_measurement_falls_back_when_the_velocity_frame_drops(velocity) -> None:
    """A dropped BLE sensor frame must never inject NaN into the filter."""

    observation = np.array([0.0, 0.0, 0.2, 0.11, 0.0])
    info = _hardware_info()
    info["velocity"] = velocity

    z = lab2.extract_measurement(observation, info)

    assert np.all(np.isfinite(z))
    assert z[1] == pytest.approx(0.11)


def test_simulation_measurement_is_the_noisy_observation() -> None:
    """Simulated observations are truth plus noise, so they are legitimate."""

    observation = np.array([1.0, 2.0, 0.4, 0.09, 0.0])
    np.testing.assert_allclose(
        lab2.extract_measurement(observation), [0.4, 0.09]
    )


@pytest.mark.parametrize("imu_origin", [0.0, 137.0, -95.5, 179.0])
def test_imu_heading_cancels_an_unknown_zero_point(imu_origin: float) -> None:
    """Only yaw differences are used, so the IMU's own origin never matters."""

    sensor = lab2.ImuHeadingSensor(initial_heading=0.0)

    first = sensor.measure(_hardware_info(yaw=imu_origin), 0.0)
    assert first == pytest.approx(0.0)

    turned = sensor.measure(
        _hardware_info(yaw=_honest_yaw(np.pi / 2.0, origin=imu_origin)),
        np.pi / 2.0,
    )
    assert turned == pytest.approx(np.pi / 2.0)


def test_imu_heading_survives_a_normal_turn_transient() -> None:
    """A 90 degree command step needs six steps at the turn limit.

    That transient must not be mistaken for a sign error.
    """

    sensor = lab2.ImuHeadingSensor(initial_heading=0.0)
    sensor.measure(_hardware_info(yaw=0.0), 0.0)
    step_degrees = np.degrees(MODEL_CONFIG["max_turn_rate_rad_s"] * 0.1)

    turned = 0.0
    for _ in range(12):
        turned = min(90.0, turned + step_degrees)
        yaw = _honest_yaw(np.deg2rad(turned))
        assert sensor.measure(_hardware_info(yaw=yaw), np.pi / 2.0) is not None

    assert not sensor.disabled


def test_imu_heading_disables_itself_when_the_sign_is_wrong(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A mirrored yaw reads ~180 degrees off for a whole leg, not 6 steps."""

    sensor = lab2.ImuHeadingSensor(initial_heading=0.0)
    sensor.measure(_hardware_info(yaw=0.0), 0.0)

    # The robot really turns +90; an IMU wired the other way reports the
    # mirror of what IMU_YAW_SIGN expects, whichever sign that constant holds.
    mirrored = -_honest_yaw(np.pi / 2.0)
    result: float | None = 0.0
    for _ in range(lab2.IMU_DISAGREEMENT_PATIENCE + 1):
        result = sensor.measure(_hardware_info(yaw=mirrored), np.pi / 2.0)

    assert result is None
    assert sensor.disabled
    assert "IMU_YAW_SIGN is probably wrong" in capsys.readouterr().out


@pytest.mark.parametrize(
    "orientation", [None, {}, "unavailable", {"yaw": float("nan")}]
)
def test_imu_heading_falls_back_when_the_yaw_frame_drops(orientation) -> None:
    sensor = lab2.ImuHeadingSensor(initial_heading=0.0)
    info = _hardware_info()
    info["orientation"] = orientation

    assert sensor.measure(info, 0.0) is None


def test_extract_measurement_prefers_the_imu_over_the_heading_echo() -> None:
    sensor = lab2.ImuHeadingSensor(initial_heading=0.0)
    # obs[2] is the command echo; 0.9 rad is deliberately not the true heading.
    observation = np.array([0.0, 0.0, 0.9, 0.10, 0.0])

    lab2.extract_measurement(
        observation,
        _hardware_info(yaw=0.0),
        heading_sensor=sensor,
        commanded_heading=0.0,
    )
    z = lab2.extract_measurement(
        observation,
        _hardware_info(yaw=_honest_yaw(np.deg2rad(45.0))),
        heading_sensor=sensor,
        commanded_heading=np.pi / 4.0,
    )

    assert z[0] == pytest.approx(np.pi / 4.0)
    assert z[0] != pytest.approx(0.9)


def test_echoing_the_command_biases_the_speed_innovation() -> None:
    """Why the encoder matters, reproduced from the filter alone.

    Feeding a command back as the measurement while the model predicts the
    robot's response to that same command leaves a one-sided innovation
    instead of zero-mean noise.  Its size is the gap between command and
    modelled response, so it scales with the working point -- the assertion is
    therefore against that gap, not against a fixed multiple of sigma.
    """

    command = lab2.COMMAND_SPEED_LIMIT
    modelled = MODEL_CONFIG["speed_gain"] * (
        command - MODEL_CONFIG["command_deadband_m_s"]
    )
    ekf = EKF(initial_state=np.zeros(4))
    action = np.array([command, 0.0])
    innovations = []
    for _ in range(80):
        ekf.predict(action)
        ekf.update(np.array([0.0, command]))  # the command, echoed back
        innovations.append(float(ekf.last_innovation[1]))

    settled = float(np.mean(innovations[-20:]))
    assert settled < 0.0  # the echo always reads slower than the response
    assert abs(settled) > 0.5 * (modelled - command)


def test_hardware_run_feeds_the_estimate_back_to_the_robot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2026-08-14 log has NaN est_*/cov_* because this call was missing."""

    monkeypatch.setattr(lab2.time, "sleep", lambda _: None)
    robot = _fake_robot()

    with lab2.open_sim_env(False) as environment:
        lab2.run_experiment(
            environment, robot, render=False, teleop=False, seed=5178
        )

    assert robot.update_estimate.call_count == lab2.N_STEPS + 1
    mean, covariance = robot.update_estimate.call_args[0]
    assert np.all(np.isfinite(mean)) and mean.shape == (4,)
    assert np.all(np.isfinite(covariance)) and covariance.shape == (4, 4)


def test_hardware_run_tracks_heading_through_the_imu_not_the_echo(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """observation[2] stays at 0 while the square turns a full 360 degrees.

    Ending the run near 0 would mean the filter followed the echo; following
    the IMU means ending on the fourth leg's -90 degrees.
    """

    monkeypatch.setattr(lab2.time, "sleep", lambda _: None)
    diagnostics = lab2.DiagnosticLog()

    with lab2.open_sim_env(False) as environment:
        lab2.run_experiment(
            environment,
            _fake_robot(),
            render=False,
            teleop=False,
            seed=5178,
            diagnostics=diagnostics,
        )

    assert "IMU_YAW_SIGN is probably wrong" not in capsys.readouterr().out

    column = lab2.DiagnosticLog.COLUMNS.index("z_heading")
    assert diagnostics.rows[-1][column] == pytest.approx(-np.pi / 2.0)


def test_diagnostic_log_records_timing_and_raw_sensors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lab2.time, "sleep", lambda _: None)
    diagnostics = lab2.DiagnosticLog()

    with lab2.open_sim_env(False) as environment:
        lab2.run_experiment(
            environment,
            None,
            render=False,
            teleop=False,
            seed=5178,
            diagnostics=diagnostics,
        )

    output = diagnostics.write(tmp_path / "diagnostics.csv")
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert tuple(rows[0]) == lab2.DiagnosticLog.COLUMNS
    assert len(rows) == lab2.N_STEPS + 1
    assert not list(tmp_path.glob("*.tmp"))

    values = np.genfromtxt(output, delimiter=",", names=True)
    # Timing must be monotonic so control-period drift is measurable.
    assert np.all(np.diff(values["t_rel"]) > 0.0)
    for column in ("nis", "nu_speed", "ekf_x", "P_xx"):
        assert np.all(np.isfinite(values[column])), column


def test_diagnostic_write_failure_does_not_mask_the_real_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Raising from the finally block would hide why the run actually died.

    The diagnostic log is written on the way out of a failed run, so a disk
    error there must not replace the dropped-BLE exception in flight.
    """

    monkeypatch.setattr(
        sys,
        "argv",
        ["lab2.py", "--sim", "--no-render", "--student-id", "33377006"],
    )
    monkeypatch.setattr(lab2.time, "sleep", lambda _: None)

    def record_one_row_then_fail(sim_env, robot_env, *, diagnostics=None, **_):
        if diagnostics is not None:
            diagnostics.rows.append((0.0,) * len(lab2.DiagnosticLog.COLUMNS))
        raise RuntimeError("BLE link dropped")

    def fail_to_write(self, path):
        raise OSError("disk full")

    monkeypatch.setattr(lab2, "run_experiment", record_one_row_then_fail)
    monkeypatch.setattr(lab2.DiagnosticLog, "write", fail_to_write)

    with pytest.raises(RuntimeError, match="BLE link dropped"):
        lab2.main()

    assert "could not write the diagnostic log" in capsys.readouterr().out


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
