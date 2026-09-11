"""Offline tests for the Lab 3 planner, controller and control loop."""

from __future__ import annotations

import csv

import numpy as np
import pytest

import lab3
from Planner import Planner
from lab3 import ARRIVAL_TOLERANCE, DT, GOAL_TOLERANCE, START_POSITION, Controller
from EKF import MODEL_CONFIG, wrap_angle
from sphero_env.envs.custom_maze_full import (
    GRID_RESOLUTION,
    GOAL_CELL,
    START_CELL,
    build_occupancy_grid,
    cell_to_world,
)

GOAL = np.array([0.5, 0.5])
# The A* optimum on this maze: eight 0.25 m plate-to-plate moves.
OPTIMAL_LENGTH = 2.0
FINAL_DISTANCE_LIMIT = 0.10
PATH_DEVIATION_LIMIT = 0.20


@pytest.fixture(scope="module")
def occupancy():
    return build_occupancy_grid()


@pytest.fixture(scope="module")
def planner(occupancy):
    return Planner(map=occupancy, dt=DT, grid_resolution=GRID_RESOLUTION)


@pytest.fixture(scope="module")
def run():
    """One headless simulation run, shared by the loop tests."""
    env = lab3.make_sim_env(render=False)
    try:
        records = lab3.control_loop(env, render=False, verbose=False)
    finally:
        env.close()
    return np.asarray(records, dtype=np.float64)


def environment_occupied(occupancy, position):
    """Reimplement SpheroEnv._is_collision so the planner is checked, not trusted."""
    height, width = occupancy.shape
    column = int(np.floor(position[0] / GRID_RESOLUTION + width / 2.0))
    row = int(np.floor(-position[1] / GRID_RESOLUTION + height / 2.0))
    if not (0 <= row < height and 0 <= column < width):
        return False
    return occupancy[row, column] == 1


def deviation_from(points, polyline):
    worst = 0.0
    for point in points:
        best = np.inf
        for a, b in zip(polyline, polyline[1:]):
            segment = b - a
            length2 = float(segment @ segment)
            t = 0.0 if length2 == 0.0 else float(np.clip((point - a) @ segment / length2, 0.0, 1.0))
            best = min(best, float(np.linalg.norm(point - (a + t * segment))))
        worst = max(worst, best)
    return worst


# ---------------- constants ----------------

def test_dt_matches_the_calibrated_model():
    # The Lab 2 EKF rejects any other value, so a drifting constant here would
    # fail at construction rather than silently mispredict.
    assert DT == MODEL_CONFIG["dt"]


def test_start_position_is_the_maze_start_plate():
    assert np.allclose(START_POSITION, cell_to_world(START_CELL))
    assert np.allclose(START_POSITION, (-0.5, -0.5))


def test_goal_matches_the_environment_and_the_maze():
    env = lab3.make_sim_env(render=False)
    try:
        assert np.allclose(env.goal_pos, cell_to_world(GOAL_CELL))
        assert np.allclose(env.goal_pos, GOAL)
        assert env.grid_resolution == GRID_RESOLUTION
    finally:
        env.close()


def test_arrival_tolerance_leaves_margin_under_the_marked_limit():
    assert ARRIVAL_TOLERANCE < GOAL_TOLERANCE <= FINAL_DISTANCE_LIMIT


# ---------------- planner geometry ----------------

def test_grid_and_world_round_trip(planner):
    for cell in ((1, 9), (9, 1), (5, 5), (1, 3)):
        assert planner.world_to_grid(planner.grid_to_world(cell)) == cell


def test_grid_to_world_agrees_with_the_maze_helper(planner):
    for logical in (START_CELL, GOAL_CELL, (2, 2), (3, 1)):
        occ_cell = (2 * logical[1] + 1, 2 * logical[0] + 1)
        assert np.allclose(planner.grid_to_world(occ_cell), cell_to_world(logical))


# ---------------- planner output ----------------

def test_plan_is_optimal_length(planner):
    waypoints = planner.plan(np.array([*START_POSITION, 0.0, 0.0]), GOAL)
    length = sum(
        float(np.linalg.norm(b - a)) for a, b in zip(waypoints, waypoints[1:])
    )
    assert length == pytest.approx(OPTIMAL_LENGTH, abs=1e-6)


def test_plan_starts_at_the_robot_and_ends_on_the_goal(planner):
    waypoints = planner.plan(np.array([*START_POSITION, 0.0, 0.0]), GOAL)
    assert np.allclose(waypoints[0], START_POSITION)
    assert np.allclose(waypoints[-1], GOAL)


def test_plan_never_crosses_a_wall(planner, occupancy):
    waypoints = planner.plan(np.array([*START_POSITION, 0.0, 0.0]), GOAL)
    # The robot drives the straight line between waypoints, so the segments
    # matter as much as the endpoints.
    for a, b in zip(waypoints, waypoints[1:]):
        for t in np.linspace(0.0, 1.0, 41):
            assert not environment_occupied(occupancy, a + (b - a) * t)


def test_waypoint_spacing_is_one_plate(planner):
    waypoints = planner.plan(np.array([*START_POSITION, 0.0, 0.0]), GOAL)
    legs = [float(np.linalg.norm(b - a)) for a, b in zip(waypoints, waypoints[1:])]
    assert max(legs) <= 2.0 * GRID_RESOLUTION + 1e-9


def test_plan_recovers_from_an_estimate_inside_a_wall(planner, occupancy):
    # Drift can put the estimate in a wall; refusing to plan would strand the
    # robot, so the planner snaps to the nearest free cell instead.
    inside_wall = np.array([-0.375, -0.375])
    assert environment_occupied(occupancy, inside_wall)
    assert not planner.is_clear(inside_wall)
    waypoints = planner.plan(np.array([*inside_wall, 0.0, 0.0]), GOAL)
    assert np.allclose(waypoints[-1], GOAL)
    for point in waypoints:
        assert not environment_occupied(occupancy, point)


def test_is_clear_matches_the_environment(planner, occupancy):
    for position in ((-0.5, -0.5), (0.5, 0.5), (-0.375, -0.375), (0.0, 0.0)):
        point = np.array(position)
        assert planner.is_clear(point) == (not environment_occupied(occupancy, point))


def test_planner_handles_a_generated_maze():
    # instructions.md points at MazeGenerator for testing other maps, so the
    # planner must not be tied to the custom maze's helpers.
    from sphero_env.envs.maze_generator import MazeGenerator

    grid, _, _ = MazeGenerator(width=15, height=15, seed=0).generate()
    other = Planner(map=grid, dt=DT, grid_resolution=0.1)
    start = other.grid_to_world(other._nearest_free(other.world_to_grid(np.zeros(2))))
    goal = other.grid_to_world(other._nearest_free(other.world_to_grid(np.array([-0.5, -0.5]))))
    waypoints = other.plan(np.array([*start, 0.0, 0.0]), goal)
    assert len(waypoints) >= 2
    assert np.allclose(waypoints[-1], goal)


# ---------------- controller ----------------

def test_controller_uses_the_lab_heading_convention():
    # Heading zero is +y and pi/2 is +x, so a waypoint due east asks for pi/2.
    controller = Controller(dt=DT)
    action = controller.compute_action(np.array([0.0, 0.0, np.pi / 2, 0.0]),
                                       np.array([1.0, 0.0]))
    assert action[1] == pytest.approx(np.pi / 2)

    action = controller.compute_action(np.array([0.0, 0.0, 0.0, 0.0]),
                                       np.array([0.0, 1.0]))
    assert action[1] == pytest.approx(0.0)


def test_controller_turns_before_it_drives():
    controller = Controller(dt=DT)
    # Facing +y, waypoint due east: a 90 degree error, so no translation yet.
    action = controller.compute_action(np.array([0.0, 0.0, 0.0, 0.0]),
                                       np.array([1.0, 0.0]))
    assert action[0] == 0.0

    # Aligned, so cruise.
    action = controller.compute_action(np.array([0.0, 0.0, 0.0, 0.0]),
                                       np.array([0.0, 1.0]))
    assert action[0] == pytest.approx(lab3.CRUISE_SPEED)


def test_controller_speed_clears_the_model_deadband():
    # Below the deadband the model produces no motion at all, so a cruise
    # command under it would stall the run.
    assert lab3.CRUISE_SPEED > MODEL_CONFIG["command_deadband_m_s"]


def test_controller_heading_error_wraps():
    controller = Controller(dt=DT)
    # Facing just under +pi with the waypoint just over it: a 2 degree error,
    # not a 358 degree one.
    action = controller.compute_action(
        np.array([0.0, 0.0, wrap_angle(np.pi - np.deg2rad(1.0)), 0.0]),
        np.array([-0.01, -1.0]),
    )
    assert action[0] == pytest.approx(lab3.CRUISE_SPEED)


# ---------------- closed loop ----------------

def test_run_reaches_the_goal(run):
    assert np.linalg.norm(run[-1, 0:2] - GOAL) <= FINAL_DISTANCE_LIMIT
    assert np.linalg.norm(run[-1, 2:4] - GOAL) <= FINAL_DISTANCE_LIMIT


def test_run_stays_on_the_optimal_path(run, planner):
    optimal = np.asarray(
        planner.plan(np.array([*START_POSITION, 0.0, 0.0]), GOAL), dtype=np.float64
    )
    assert deviation_from(run[:, 0:2], optimal) <= PATH_DEVIATION_LIMIT
    assert deviation_from(run[:, 2:4], optimal) <= PATH_DEVIATION_LIMIT


def test_run_never_enters_a_wall(run, occupancy):
    assert not any(environment_occupied(occupancy, point) for point in run[:, 0:2])


def test_run_starts_on_the_start_plate(run):
    assert np.linalg.norm(run[0, 0:2] - START_POSITION) < 2.0 * GRID_RESOLUTION


def test_run_ends_when_the_goal_is_reached(run):
    # Every row is a step of the run itself; nothing pads it to a fixed length.
    assert len(run) < lab3.MAX_STEPS
    assert np.linalg.norm(run[-1, 2:4] - GOAL) <= lab3.ARRIVAL_TOLERANCE


def test_replanning_recovers_from_a_jump_in_the_estimate(monkeypatch):
    """A jolted estimate must not strand the run."""
    base = lab3.Estimator

    class JoltingEKF(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._calls = 0

        def predict(self, action):
            result = super().predict(action)
            self._calls += 1
            if self._calls == 60:
                self.state_est[0] += 0.20
            return result

    monkeypatch.setattr(lab3, "Estimator", JoltingEKF)
    env = lab3.make_sim_env(render=False)
    try:
        records = lab3.control_loop(env, max_steps=1200, render=False, verbose=False)
    finally:
        env.close()
    estimate = np.asarray(records, dtype=np.float64)[-1, 2:4]
    assert np.linalg.norm(estimate - GOAL) <= FINAL_DISTANCE_LIMIT


# ---------------- submission ----------------

def test_submission_csv_has_the_required_shape(run, tmp_path, monkeypatch):
    monkeypatch.setattr(lab3, "LAB_DIR", tmp_path)
    path = lab3.write_submission([tuple(row) for row in run])

    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))

    assert tuple(rows[0]) == lab3.CSV_COLUMNS
    assert len(rows) - 1 == len(run)
    values = np.asarray([[float(v) for v in row] for row in rows[1:]])
    assert values.shape[1] == 4
    assert np.all(np.isfinite(values))
    assert np.allclose(values, run, atol=1e-6)


# ---------------- hardware path ----------------
#
# The hardware branch cannot be exercised without a robot, so these drive it
# through a mock shaped like Robot.step()'s return.  They check the wiring --
# which sensor feeds which term, what triggers a replan -- not the physics.

from unittest.mock import Mock  # noqa: E402

import Estimator as Estimator_module  # noqa: E402
import lab2  # noqa: E402

# The speed the calibrated model produces for a CRUISE_SPEED command, in the
# cm/s the API reports.
ENCODER_CM_S = 100.0 * MODEL_CONFIG["speed_gain"] * (
    lab3.CRUISE_SPEED - MODEL_CONFIG["command_deadband_m_s"]
)


def _wrap_degrees(angle):
    return (angle + 180.0) % 360.0 - 180.0


def _honest_yaw(heading_rad, initial_rad=0.0):
    """What a correctly-signed IMU reports while the robot holds a heading."""
    turned = np.degrees(wrap_angle(heading_rad - initial_rad))
    return _wrap_degrees(lab2.IMU_YAW_SIGN * turned)


def _hardware_info(yaw=0.0, speed_cm_s=0.0, speed_cmd=0.0, odom=None):
    if odom is None:
        odom = np.zeros(4, dtype=np.float32)
    return {
        "state_odom": np.asarray(odom, dtype=np.float32),
        "state_true": np.zeros(4, dtype=np.float32),
        "velocity": {"x": 0.0, "y": speed_cm_s},
        "orientation": {"yaw": yaw, "pitch": 0.0, "roll": 0.0},
        "gyroscope": {"x": 0.0, "y": 0.0, "z": 0.0},
        "speed_cmd": speed_cmd,
        "heading_cmd": 0.0,
    }


def _fake_robot(collide_at=(), stall_from=None, speed_scale=1.0):
    """A robot whose IMU, encoder and locator all agree with the command.

    observation[2] stays at zero throughout, the way api.get_heading()'s echo
    behaves, so any test that sees the filter track the commanded heading is
    seeing it come from the IMU.

    The locator has to move, because the filter now measures it: a mock whose
    state_odom stays at the origin is a robot pinned in place, and the filter
    is right to refuse to believe it went anywhere.  ``stall_from`` pins it
    from that step on, for the tests that want exactly that.

    ``speed_scale`` makes the robot faster or slower than the model, so it
    reaches each corner at a different step from the simulator -- the
    situation the 12:29 hardware run was in.
    """
    robot = Mock()
    state = {"step": 0, "xy": np.zeros(2)}
    # What the calibrated model does with a cruise command, scaled.
    speed = speed_scale * MODEL_CONFIG["speed_gain"] * (
        lab3.CRUISE_SPEED - MODEL_CONFIG["command_deadband_m_s"]
    )
    robot.reset.return_value = (
        np.zeros(5, dtype=np.float32),
        _hardware_info(),
    )

    def step(action):
        speed_cmd, heading = float(action[0]), float(action[1])
        driving = speed_cmd > MODEL_CONFIG["command_deadband_m_s"]
        stalled = stall_from is not None and state["step"] >= stall_from
        moving = driving and not stalled
        if moving:
            state["xy"] = state["xy"] + speed * lab3.DT * np.array(
                [np.sin(heading), np.cos(heading)]
            )
        collided = state["step"] in collide_at
        observation = np.array(
            [0.0, 0.0, 0.0, 0.0, 1.0 if collided else 0.0], dtype=np.float32
        )
        info = _hardware_info(
            yaw=_honest_yaw(heading),
            speed_cm_s=100.0 * speed if moving else 0.0,
            speed_cmd=speed_cmd,
            odom=[state["xy"][0], state["xy"][1], heading, speed if moving else 0.0],
        )
        state["step"] += 1
        return observation, 0.0, False, False, info

    robot.step.side_effect = step
    return robot


def _run_hardware(monkeypatch, robot, max_steps=600):
    monkeypatch.setattr(lab3.time, "sleep", lambda _: None)
    env = lab3.make_sim_env(render=False)
    try:
        records = lab3.control_loop(
            env, robot, max_steps=max_steps, render=False, verbose=False
        )
    finally:
        env.close()
    return np.asarray(records, dtype=np.float64)


def test_hardware_uses_the_throttled_robot():
    # The base Robot writes heading and speed separately, which measured a
    # 209 ms period against the 105 ms the model is calibrated for.
    environment = lab3.make_real_env(Mock())
    assert isinstance(environment, lab2.ThrottledRobot)
    assert environment.dt == DT
    # vel_limit doubles as the raw-speed scale on hardware.
    assert environment.vel_limit == lab3.COMMAND_SPEED_LIMIT
    assert environment.raw_speed_limit == lab3.RAW_SPEED_LIMIT


def test_hardware_run_reaches_the_goal(monkeypatch):
    run = _run_hardware(monkeypatch, _fake_robot())
    assert np.linalg.norm(run[-1, 2:4] - GOAL) <= FINAL_DISTANCE_LIMIT
    assert np.linalg.norm(run[-1, 0:2] - GOAL) <= FINAL_DISTANCE_LIMIT


def test_hardware_steps_the_robot_with_the_same_action_as_the_simulator(monkeypatch):
    robot = _fake_robot()
    run = _run_hardware(monkeypatch, robot)
    # One robot step per logged row, and the simulator supplies sim_x/sim_y.
    assert robot.step.call_count == len(run)
    assert robot.emergency_stop.called


def test_hardware_heading_comes_from_the_imu_not_the_echo(monkeypatch):
    """observation[2] is a command echo, so the filter must not follow it."""
    robot = _fake_robot()
    run = _run_hardware(monkeypatch, robot)
    # If the filter had believed the echo it would still think it faces +y and
    # would never have turned east towards the goal.
    assert run[-1, 2] > 0.0


def test_hardware_speed_comes_from_the_encoder(monkeypatch):
    # extract_measurement converts the API's cm/s into the m/s the filter uses.
    info = _hardware_info(speed_cm_s=ENCODER_CM_S, speed_cmd=lab3.CRUISE_SPEED)
    z = lab3.extract_measurement(np.zeros(5, dtype=np.float32), info)
    assert z[3] == pytest.approx(ENCODER_CM_S / 100.0)


def test_hardware_collision_triggers_a_replan(monkeypatch):
    """A collision means a wall the plan did not have, so the path is stale."""
    plain = _run_hardware(monkeypatch, _fake_robot())
    bumped = _run_hardware(monkeypatch, _fake_robot(collide_at=(30, 31, 32)))
    # The run still finishes, and the collision costs it steps rather than
    # stranding it.
    assert np.linalg.norm(bumped[-1, 2:4] - GOAL) <= FINAL_DISTANCE_LIMIT
    assert len(bumped) >= len(plain)


def test_hardware_uses_the_measured_noise_matrices(monkeypatch):
    """Simulation and hardware are different noise regimes."""
    captured = {}
    base = lab3.Estimator

    class RecordingEKF(base):
        def __init__(self, *args, **kwargs):
            captured["process"] = kwargs.get("process_noise")
            captured["measurement"] = kwargs.get("measurement_noise")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(lab3, "Estimator", RecordingEKF)
    _run_hardware(monkeypatch, _fake_robot(), max_steps=3)
    assert np.allclose(captured["process"], lab2.REAL_PROCESS_NOISE)
    assert np.allclose(captured["measurement"], Estimator_module.REAL_MEASUREMENT_NOISE)

    captured.clear()
    env = lab3.make_sim_env(render=False)
    try:
        lab3.control_loop(env, max_steps=3, render=False, verbose=False)
    finally:
        env.close()
    assert captured["process"] is None
    assert np.allclose(captured["measurement"], Estimator_module.SIM_MEASUREMENT_NOISE)


def test_pygame_is_initialised_before_the_event_queue_is_polled(monkeypatch):
    """_poll_for_abort() on an uninitialised pygame raises, so render() goes first."""
    order = []
    monkeypatch.setattr(lab3, "_poll_for_abort", lambda: order.append("poll"))
    env = lab3.make_sim_env(render=False)
    monkeypatch.setattr(env, "render", lambda: order.append("render"))
    try:
        lab3.control_loop(env, max_steps=2, render=True, verbose=False)
    finally:
        env.close()
    assert order[0] == "render"


def test_records_survive_an_abort(monkeypatch):
    """A hardware run costs a pairing and a placement; losing its log is expensive."""
    records = []

    def abort_after_five():
        if len(records) >= 5:
            raise lab3.ExperimentAborted("test")

    monkeypatch.setattr(lab3, "_poll_for_abort", abort_after_five)
    env = lab3.make_sim_env(render=False)
    try:
        with pytest.raises(lab3.ExperimentAborted):
            lab3.control_loop(env, render=True, verbose=False, records=records)
    finally:
        env.close()
    assert len(records) == 5


def test_main_writes_the_csv_after_an_abort(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lab3, "LAB_DIR", tmp_path)
    # managed_env logs to logs/lab3_sim.csv relative to the working directory,
    # which from the repository root is where a hardware run's log lives.
    monkeypatch.chdir(tmp_path)
    seen = {"n": 0}

    def abort_after_five():
        seen["n"] += 1
        if seen["n"] > 5:
            raise lab3.ExperimentAborted("test")

    monkeypatch.setattr(lab3, "_poll_for_abort", abort_after_five)
    lab3.main(["--sim"])

    output = capsys.readouterr().out
    assert "Run aborted" in output
    written = tmp_path / f"{lab3.STUDENT_ID}_lab3.csv"
    assert written.exists()
    with open(written, newline="") as handle:
        assert len(list(csv.reader(handle))) == 6  # header plus five steps


# ---------------- position measurement ----------------

def test_estimator_measures_the_whole_state():
    estimator = Estimator_module.Estimator(initial_state=np.zeros(4))
    assert estimator.H.shape == (4, 4)
    assert np.allclose(estimator.H, np.eye(4))


def test_estimator_wraps_the_heading_innovation():
    """The heading is third in this measurement vector, not first as in Lab 2."""
    estimator = Estimator_module.Estimator(
        initial_state=np.array([0.0, 0.0, np.pi - 0.01, 0.0])
    )
    # Measured just the other side of +pi: the real innovation is 0.02 rad.
    estimator.update(np.array([0.0, 0.0, -np.pi + 0.01, 0.0]))
    assert abs(estimator.last_innovation[2]) == pytest.approx(0.02, abs=1e-6)


def test_simulation_measurement_is_the_observation():
    observation = np.array([0.1, -0.2, 0.3, 0.04, 0.0], dtype=np.float32)
    z = lab3.extract_measurement(observation)
    assert np.allclose(z, [0.1, -0.2, 0.3, 0.04], atol=1e-6)


def test_hardware_position_is_offset_onto_the_start_plate():
    """The locator zeroes where the robot was reset, not at the world origin."""
    info = _hardware_info(odom=[0.25, -0.10, 0.0, 0.0])
    z = lab3.extract_measurement(
        np.zeros(5, dtype=np.float32), info, position_offset=START_POSITION
    )
    assert z[0] == pytest.approx(START_POSITION[0] + 0.25)
    assert z[1] == pytest.approx(START_POSITION[1] - 0.10)


def test_a_stalled_robot_does_not_run_away_in_the_estimate(monkeypatch):
    """The whole reason the locator is measured.

    Replaying the three hardware logs, the Lab 2 filter's estimate ran 0.11 m,
    0.50 m and 1.19 m past the locator, in proportion to how often the robot
    was commanded to move while the encoder read zero -- it integrates the
    command. Lab 3's limit is 0.10 m, so the position has to be measured.
    """
    run = _run_hardware(monkeypatch, _fake_robot(stall_from=40), max_steps=140)
    estimates = run[:, 2:4]
    # Give the filter a few steps to settle after the stall begins.
    drift = float(np.linalg.norm(estimates[-1] - estimates[50]))
    assert drift < 0.05, f"estimate drifted {drift:.3f} m while the robot was stuck"


def test_the_filter_would_have_run_away_without_the_position(monkeypatch):
    """The same stall, measured the Lab 2 way, to show the fix is the position."""
    from EKF import EKF as Lab2EKF

    class HeadingSpeedOnly(Lab2EKF):
        """Lab 2's measurement model behind Lab 3's constructor signature."""

        def update(self, measurement):
            return super().update(np.asarray(measurement, dtype=np.float64)[2:4])

    def build(*args, **kwargs):
        kwargs["measurement_noise"] = lab2.REAL_MEASUREMENT_NOISE
        return HeadingSpeedOnly(*args, **kwargs)

    monkeypatch.setattr(lab3, "Estimator", build)
    run = _run_hardware(monkeypatch, _fake_robot(stall_from=40), max_steps=140)
    estimates = run[:, 2:4]
    drift = float(np.linalg.norm(estimates[-1] - estimates[50]))
    assert drift > 0.20, f"expected dead reckoning to run away, drifted only {drift:.3f} m"


def _flaky_robot(stall_every):
    """Pins the robot for `stall_every` steps out of every ten."""
    robot = Mock()
    state = {"step": 0, "xy": np.zeros(2)}
    speed = MODEL_CONFIG["speed_gain"] * (
        lab3.CRUISE_SPEED - MODEL_CONFIG["command_deadband_m_s"]
    )
    robot.reset.return_value = (np.zeros(5, dtype=np.float32), _hardware_info())

    def step(action):
        speed_cmd, heading = float(action[0]), float(action[1])
        driving = speed_cmd > MODEL_CONFIG["command_deadband_m_s"]
        stalled = stall_every and state["step"] % 10 < stall_every
        moving = driving and not stalled
        if moving:
            state["xy"] = state["xy"] + speed * lab3.DT * np.array(
                [np.sin(heading), np.cos(heading)]
            )
        info = _hardware_info(
            yaw=_honest_yaw(heading),
            speed_cm_s=100.0 * speed if moving else 0.0,
            speed_cmd=speed_cmd,
            odom=[state["xy"][0], state["xy"][1], heading, speed if moving else 0.0],
        )
        state["step"] += 1
        return np.zeros(5, dtype=np.float32), 0.0, False, False, info

    robot.step.side_effect = step
    return robot


def test_a_stalling_robot_does_not_send_the_simulator_off_course(monkeypatch):
    """Both CSV columns are marked, and the simulator cannot stall on its own.

    The filter measures the locator, so a stuck robot stops advancing and the
    controller keeps aiming at the same waypoint.  The simulator, stepped with
    that same action, has nothing stopping it -- it drives straight through the
    waypoint and away.  At the 23% stall rate the 12:47 hardware log measured,
    that put sim_x/sim_y about a metre from the goal while the estimate was
    within 0.03 m.  Feeding the encoder's verdict to the simulator fixes it.
    """
    monkeypatch.setattr(lab3.time, "sleep", lambda _: None)
    for stall_every in (1, 2, 3, 5):
        env = lab3.make_sim_env(render=False)
        try:
            records = lab3.control_loop(
                env, _flaky_robot(stall_every), max_steps=600,
                render=False, verbose=False,
            )
        finally:
            env.close()
        arr = np.asarray(records, dtype=np.float64)
        sim_error = float(np.linalg.norm(arr[-1, 0:2] - GOAL))
        real_error = float(np.linalg.norm(arr[-1, 2:4] - GOAL))
        assert sim_error <= FINAL_DISTANCE_LIMIT, (
            f"{stall_every * 10}% stalls left the simulator {sim_error:.3f} m out"
        )
        assert real_error <= FINAL_DISTANCE_LIMIT


def test_the_simulator_is_only_held_back_when_the_encoder_says_so(monkeypatch):
    """A robot that moves normally must not have its simulator throttled."""
    monkeypatch.setattr(lab3.time, "sleep", lambda _: None)
    robot = _fake_robot()
    env = lab3.make_sim_env(render=False)
    try:
        records = lab3.control_loop(
            env, robot, max_steps=600, render=False, verbose=False
        )
    finally:
        env.close()
    sim = np.asarray(records, dtype=np.float64)[:, 0:2]
    travelled = float(np.sum(np.linalg.norm(np.diff(sim, axis=0), axis=1)))
    # The plan is 2.0 m; turning in place costs some of it but not most.
    assert travelled > 1.5, f"simulator only covered {travelled:.3f} m"


# ---------------- simulator follows its own plan ----------------
#
# On the 12:29 hardware run the simulator was stepped with the robot's actions,
# so it turned whenever the robot reached a corner.  At the first corner it had
# already overshot by 0.05 m, then grazed a wall and fell 0.11 m behind, and
# the next turn sent it into that wall for good.  The simulator has to decide
# for itself when it has reached a corner.

import inspect  # noqa: E402


@pytest.mark.parametrize("speed_scale", [0.7, 1.3])
def test_simulator_turns_at_its_own_corners(monkeypatch, planner, occupancy, speed_scale):
    """0.7 is the Lab 2 ratio of measured to modelled speed; 1.3 the reverse."""
    run = _run_hardware(monkeypatch, _fake_robot(speed_scale=speed_scale), max_steps=1500)
    sim, real = run[:, 0:2], run[:, 2:4]
    optimal = np.asarray(
        planner.plan(np.array([*START_POSITION, 0.0, 0.0]), GOAL), dtype=np.float64
    )

    assert np.linalg.norm(sim[-1] - GOAL) <= FINAL_DISTANCE_LIMIT
    assert np.linalg.norm(real[-1] - GOAL) <= FINAL_DISTANCE_LIMIT
    assert deviation_from(sim, optimal) <= PATH_DEVIATION_LIMIT
    assert deviation_from(real, optimal) <= PATH_DEVIATION_LIMIT
    assert not any(environment_occupied(occupancy, point) for point in sim)


def test_a_robot_waiting_at_the_goal_does_not_freeze_the_simulator(monkeypatch):
    """The faster ball arrives first and holds; the run must wait for the other.

    A robot holding still reads zero on its encoder, which is not a stall --
    it was not asked to move.  Treating it as one would hold the simulator
    back forever and the run would never end.
    """
    run = _run_hardware(monkeypatch, _fake_robot(speed_scale=1.3), max_steps=1500)
    sim_error = np.linalg.norm(run[:, 0:2] - GOAL, axis=1)
    real_error = np.linalg.norm(run[:, 2:4] - GOAL, axis=1)
    real_arrived = int(np.argmax(real_error <= lab3.ARRIVAL_TOLERANCE + 1e-9))
    assert real_error[real_arrived] <= lab3.ARRIVAL_TOLERANCE + 1e-9
    # The robot got there first, and the run went on until the simulator did.
    assert sim_error[real_arrived] > lab3.ARRIVAL_TOLERANCE
    assert sim_error[-1] <= FINAL_DISTANCE_LIMIT


def test_there_is_no_fixed_step_limit():
    """Lab 3 asks for every step from start to finish, not a set number."""
    assert lab3.parse_args([]).steps == lab3.MAX_STEPS
    default = inspect.signature(lab3.control_loop).parameters["max_steps"].default
    assert default == lab3.MAX_STEPS


@pytest.mark.parametrize("speed_scale", [0.7, 1.3])
def test_simulator_waits_for_robot_at_corners(monkeypatch, speed_scale):
    """A leading simulator holds its heading until the robot reaches that corner."""
    monkeypatch.setattr(lab3.time, "sleep", lambda _: None)
    env = lab3.make_sim_env(render=False)
    samples = []
    original_step = env.step

    def record_step(action):
        result = original_step(action)
        samples.append((action.copy(), result[4]["state_true"].copy()))
        return result

    monkeypatch.setattr(env, "step", record_step)
    try:
        records = lab3.control_loop(
            env, _fake_robot(speed_scale=speed_scale), max_steps=1500,
            render=False, verbose=False,
        )
    finally:
        env.close()

    track = np.asarray(records)
    actions = np.asarray([sample[0] for sample in samples])
    states = np.asarray([sample[1] for sample in samples])
    waits = 0
    for corner in ([-0.5, 0.0], [-0.25, 0.0], [-0.25, 0.5]):
        sim_ready = np.flatnonzero(
            np.linalg.norm(track[:, :2] - corner, axis=1) <= lab3.WAYPOINT_TOLERANCE
        )
        real_ready = np.flatnonzero(
            np.linalg.norm(track[:, 2:] - corner, axis=1) <= lab3.WAYPOINT_TOLERANCE
        )
        assert sim_ready.size and real_ready.size
        if sim_ready[0] < real_ready[0]:
            waiting = actions[sim_ready[0] + 1:real_ready[0] + 1]
            assert np.all(waiting[:, 0] == 0.0), f"did not wait at {corner}"
            heading = states[sim_ready[0], 2]
            assert all(abs(wrap_angle(a[1] - heading)) < 1e-6 for a in waiting)
            waits += 1
    if speed_scale < 1.0:
        assert waits > 0
    assert np.linalg.norm(track[-1, :2] - GOAL) <= FINAL_DISTANCE_LIMIT
    assert np.linalg.norm(track[-1, 2:] - GOAL) <= FINAL_DISTANCE_LIMIT


def test_corner_wait_remembers_robot_arrival_after_replanning(planner):
    start = np.array([*START_POSITION, 0.0, 0.0])
    robot = lab3.WaypointFollower(planner, Controller(), GOAL, start)
    sim = lab3.WaypointFollower(planner, Controller(), GOAL, start, wait_for=robot)
    straight = np.array([-0.5, -0.25, 0.0, 0.0])
    corner = np.array([-0.5, 0.0, 0.0, 0.0])

    sim.advance(start, False)
    sim.advance(straight, False)
    assert sim.target == 2  # Straight waypoints do not wait for the robot.
    sim.advance(corner, False)
    np.testing.assert_array_equal(sim.action(corner), [0.0, 0.0])

    robot.advance(start, False)
    robot.advance(straight, False)
    sim.advance(corner, False)
    assert sim.target == 2  # Reaching a different waypoint cannot release the wait.

    robot.advance(corner, False)
    robot.advance(np.array([-0.4, 0.0, np.pi / 2, 0.0]), True)
    assert robot.replans == 1
    sim.advance(np.array([-0.5, 0.01, 0.0, 0.0]), False)
    assert sim.target == 3
    assert sim.action(corner)[1] == pytest.approx(np.pi / 2)
