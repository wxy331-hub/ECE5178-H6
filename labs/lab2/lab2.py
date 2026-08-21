"""Run the 200-step Sphero Lab 2 EKF experiment over a square trajectory.

``--sim`` uses simulator observations; a hardware run measures heading and
speed from the robot's own sensors and keeps the simulator as ground truth.
The automarker CSV is written only after all 200 updates succeed.
"""

from __future__ import annotations

import argparse
import csv
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np
import pygame

from sphero_env.envs import SpheroEnv
from sphero_env.robot.connect import scan_and_connect
from sphero_env.robot.robot import Robot
from sphero_unsw.sphero_edu import SpheroEduAPI

try:
    from .EKF import EKF, MODEL_CONFIG, dynamics, wrap_angle
    from .analyze_lab2 import analyze_values, save_plot
except ImportError:
    from EKF import EKF, MODEL_CONFIG, dynamics, wrap_angle
    from analyze_lab2 import analyze_values, save_plot


DT = MODEL_CONFIG["dt"]  # 0.105: what the hardware loop actually achieves
N_STEPS = 200

# Both environments call their speed cap ``vel_limit``, but they mean
# different things by it.  The hardware wrapper also uses it to scale the raw
# 0-15 speed byte (speed_raw = speed_cmd / vel_limit * raw_speed_limit), so
# this one must stay at 0.15 or the robot drives at the wrong speed.
COMMAND_SPEED_LIMIT = 0.15
# The simulator's is an observation range, and it has to cover what the model
# can produce: the original 0.10 command settled at 0.182 m/s, so a 0.15
# range clipped every speed reading.
SIM_SPEED_LIMIT = MODEL_CONFIG["max_speed_m_s"]

RAW_SPEED_LIMIT = 15

# Speed commanded along every leg of the square.  Two things bound it.  The
# hardware quantises the command to an integer raw unit
# (raw = int(speed_cmd / COMMAND_SPEED_LIMIT * RAW_SPEED_LIMIT)), so only
# multiples of 0.01 reach the robot unchanged.  And the lab1 run put the real
# deadband near 0.05 -- well above the 0.0322 the model assumes -- so a
# command close to that stalls the robot instead of slowing it.
# raw 10, which the 200-step run on 2026-08-21 completed successfully.  It
# sits 4.6 counts above the re-estimated deadband of 0.0537; raw 8 would be
# only 2.6 and drives at 0.051 m/s, close enough to the stalling point that
# uneven floor stops the robot outright.  That run still stalled on 28% of its
# steps, which is the floor rather than the command -- the arena has high
# spots -- so the process noise below has to absorb it.
SCRIPT_SPEED = 0.10

# Geometry of the yellow lane on the lab track, estimated from the rulers in
# the 2026-08-21 photograph.  These are the straight runs between corners and
# the corner radius, not the outside dimensions: the lap works out at 1.67 m
# and the path spans 0.44 x 0.45 m, which matches the measured track.
#
# Driving the lane instead of a square removes the pivot that cost the robot
# its speed at every corner.  A 0.065 m radius at 0.09 m/s needs 1.39 rad/s,
# which is 8.4 degrees per step against a demonstrated 45 -- the robot turns
# while still driving, so there is no low-speed window for a high spot to
# catch.
TRACK_STRAIGHT_X = 0.31
TRACK_STRAIGHT_Y = 0.32
TRACK_RADIUS = 0.065

LAB_DIR = Path(__file__).resolve().parent
CSV_COLUMNS = (
    "sim_x",
    "sim_y",
    "real_x",
    "real_y",
    "P_xx",
    "P_xy",
    "P_yy",
)

# Ten times the simulator's position Q.  The evidence for inflating it (short
# hardware legs) has since been traced to the command echo and the clipped
# observation range, both now fixed, so re-estimate this from
# logs/lab2_diagnostics.csv after the next hardware run.  Deflating it
# beforehand would be guesswork in the optimistic direction, which is the
# dangerous one for a consistency metric.
REAL_PROCESS_NOISE = np.diag([3.125e-5, 3.125e-5, 3.0e-3, 2.5e-5])

# Hardware measurement noise, taken from the innovations of the 2026-08-21 run
# rather than from the simulator's settings.  The heading innovation had a
# standard deviation of 0.10 rad, four times the 0.025 the simulator assumes,
# because that reading carries the robot's real yaw wander on an uneven floor
# and not just sensor noise.  Replaying the log offline, this alone brings the
# mean NIS from 34.9 to 4.0 and the share inside the 95% gate from 2% to 90%;
# the heading process noise above finishes it at 2.0 and 96%.
REAL_MEASUREMENT_NOISE = np.diag([1.03e-2, 1.08e-3])

# Sign of the IMU yaw relative to set_heading().  Both are documented as
# increasing clockwise seen from above, so +1 was expected -- but regressing
# the measured yaw against the commanded heading over the 2026-08-21 run gives
# a slope of -0.97, so the documentation does not hold for this robot.  The
# guard in ImuHeadingSensor caught it mid-run and fell back to the observation.
IMU_YAW_SIGN = -1.0
# A scripted heading step is 90 degrees and the 7.5 rad/s turn limit clears
# it in two steps, so disagreement past this margin is a sign error rather
# than a turn transient.
IMU_HEADING_DISAGREEMENT_LIMIT = np.deg2rad(120.0)
IMU_DISAGREEMENT_PATIENCE = 10


class EstimateRecord(NamedTuple):
    """One posterior estimate aligned with one simulator ground-truth step."""

    sim_x: float
    sim_y: float
    real_x: float
    real_y: float
    p_xx: float
    p_xy: float
    p_yy: float


class ExperimentAborted(RuntimeError):
    """Raised when the operator closes the window or presses Q."""


class TeleopController:
    """Translate pygame keyboard state into safe Sphero actions."""

    def __init__(self, initial_heading: float) -> None:
        self.heading = wrap_angle(initial_heading)
        self.speed = 0.08

    def action(self, robot_env: Robot | None) -> np.ndarray:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise ExperimentAborted("window closed")
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_q:
                raise ExperimentAborted("Q pressed")
            if event.key == pygame.K_SPACE:
                if robot_env is not None:
                    robot_env.emergency_stop()
                self.speed = 0.0
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.speed = min(COMMAND_SPEED_LIMIT, self.speed + 0.01)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.speed = max(0.0, self.speed - 0.01)

        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_a]:
            self.heading = wrap_angle(self.heading - 0.10)
        if pressed[pygame.K_d]:
            self.heading = wrap_angle(self.heading + 0.10)

        commanded_speed = 0.0
        if pressed[pygame.K_w]:
            commanded_speed = self.speed
        elif pressed[pygame.K_s]:
            commanded_speed = -self.speed

        return np.array([commanded_speed, self.heading], dtype=np.float32)


def make_sim_env(render: bool) -> SpheroEnv:
    return SpheroEnv(
        dt=DT,
        max_steps=N_STEPS,
        vel_limit=SIM_SPEED_LIMIT,
        world_width=5.0,
        world_height=5.0,
        goal_pos=(0.5, 0.5),
        goal_tolerance=0.1,
        occupancy_grid=None,
        dynamics=dynamics,
        obs_noise_std_pos=0.05,
        process_noise_std_speed=0.005,
        process_noise_std_heading=0.01,
        obs_noise_std_vel=0.025,
        render_mode="human" if render else None,
        window_size=(800, 800),
    )


class ThrottledRobot(Robot):
    """One BLE write per step instead of two, without touching the framework.

    ``Robot.set_heading_and_speed`` calls ``set_heading`` and then
    ``set_speed``, and both end in the same
    ``ToyUtil.roll_start(heading, speed)`` command.  The first one carries the
    *previous* speed and is overwritten by the second a full BLE round trip
    later, so whenever the heading is unchanged it buys nothing.

    It costs a lot.  The 2026-08-21 run measured a 209 ms control period
    against the 100 ms the filter assumes, which doubles every predicted
    displacement, and 49 of every 50 steps hold the heading constant.

    Overriding this one method leaves src/sphero_env untouched.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_heading_deg: int | None = None

    def set_heading_and_speed(self, heading_deg: float, speed: int) -> None:
        commanded = int(heading_deg)
        with self._lock:
            if commanded != self._last_heading_deg:
                self.api.set_heading(commanded)
                self._last_heading_deg = commanded
            self.api.set_speed(int(np.clip(speed, 0, 255)))


def make_real_env(api: SpheroEduAPI) -> Robot:
    return ThrottledRobot(
        api=api,
        dt=DT,
        max_steps=N_STEPS,
        vel_limit=COMMAND_SPEED_LIMIT,
        raw_speed_limit=RAW_SPEED_LIMIT,
        world_width=5.0,
        world_height=5.0,
        goal_pos=(0.5, 0.5),
        goal_tolerance=0.1,
        obs_noise_std_pos=0.05,
        obs_noise_std_vel=0.025,
        render_mode=None,
    )


def _retryable_ble_error(error: BaseException) -> bool:
    if isinstance(error, TimeoutError):
        return True
    return isinstance(error, OSError) and getattr(error, "winerror", None) in {
        -2147023673,
        1223,
    }


def connect_with_retry(stack: ExitStack, selected_toy: object) -> SpheroEduAPI:
    """Retry known transient Windows BLE connection failures."""

    last_error: BaseException | None = None
    for attempt in range(1, 4):
        try:
            return stack.enter_context(SpheroEduAPI(selected_toy))
        except (TimeoutError, OSError) as error:
            if not _retryable_ble_error(error):
                raise
            last_error = error
            if attempt < 3:
                print(f"Bluetooth connection failed ({attempt}/3); retrying...")
                time.sleep(3.0)
    raise RuntimeError("Bluetooth connection failed after 3 attempts") from last_error


@contextmanager
def open_sim_env(render: bool) -> Iterator[SpheroEnv]:
    env = make_sim_env(render)
    try:
        yield env
    finally:
        env.emergency_stop()
        env.close()


@contextmanager
def open_real_env() -> Iterator[Robot]:
    with ExitStack() as stack:
        selected_toy, _ = scan_and_connect()
        api = connect_with_retry(stack, selected_toy)
        print(f"Connected to: {selected_toy.name}")
        env = make_real_env(api)
        diagnostic_path = LAB_DIR.parents[1] / "logs" / "lab2_robot.csv"
        env.set_log_path(str(diagnostic_path))
        env.start_logging()
        try:
            yield env
        finally:
            env.stop_logging()
            env.emergency_stop()
            env.close()


def _track_segments() -> tuple[tuple[float, float], ...]:
    """The lane as (arc length, heading change) pairs, one lap, clockwise."""

    corner = 0.5 * np.pi * TRACK_RADIUS
    quarter = np.pi / 2.0
    return (
        (TRACK_STRAIGHT_Y, 0.0), (corner, quarter),
        (TRACK_STRAIGHT_X, 0.0), (corner, quarter),
        (TRACK_STRAIGHT_Y, 0.0), (corner, quarter),
        (TRACK_STRAIGHT_X, 0.0), (corner, quarter),
    )


TRACK_LAP = sum(length for length, _ in _track_segments())


def track_heading(distance: float) -> float:
    """Heading relative to the start, at ``distance`` metres along the lane.

    Straights hold their heading; corners turn linearly with arc length, which
    is what a constant-radius turn at constant speed does.  Laps repeat.
    """

    remaining = distance % TRACK_LAP
    heading = 0.0
    for length, turn in _track_segments():
        if remaining <= length:
            return heading + (turn * remaining / length if turn else 0.0)
        remaining -= length
        heading += turn
    return heading


def scripted_action(travelled: float, initial_heading: float) -> np.ndarray:
    """Point along the lane at ``travelled`` metres of measured progress.

    The argument is distance actually covered, not ``step * speed * dt``.
    Those differ whenever the robot fails to move: on 2026-08-21 it stalled on
    33% of its steps against high spots in the floor, and a clock-driven
    schedule kept advancing the corner while the robot stood still, so the
    turn was commanded after 0.219 m of real motion instead of the 0.32 m
    straight.  Feeding measured progress in makes a stall cost time but not
    position -- the robot resumes the corner exactly where it left off.
    """

    if travelled < 0.0:
        raise ValueError("travelled must not be negative")

    # Stop after one lap.  The assignment wants 200 steps of data, not 200
    # steps of driving, and how far 200 steps carry the robot depends on how
    # much of that time it spends stuck: 1.08 laps clean, 0.72 at the 33%
    # stall rate measured on 2026-08-21.  Holding still once the lap closes
    # keeps the path to exactly one lap either way, and the filter still has
    # something to do -- a stationary robot is a prediction it should get
    # right.
    completed = travelled >= TRACK_LAP
    speed = 0.0 if completed else SCRIPT_SPEED
    heading = wrap_angle(initial_heading + track_heading(min(travelled, TRACK_LAP)))
    return np.array([speed, heading], dtype=np.float32)


def _poll_for_abort() -> None:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            raise ExperimentAborted("window closed")
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            raise ExperimentAborted("Q pressed")


def _align_simulator_heading(sim_env: SpheroEnv, heading: float) -> None:
    """Align the simulator frame with the real robot's reset heading."""

    if sim_env.state_true is None or sim_env.state_odom is None:
        raise RuntimeError("simulator must be reset before heading alignment")
    aligned_heading = wrap_angle(heading)
    sim_env.state_true[2] = aligned_heading
    sim_env.state_odom[2] = aligned_heading


def robot_speed_measurement(info: dict) -> float | None:
    """Return the encoder speed in m/s, or ``None`` if the stream is not ready.

    ``api.get_velocity()`` is the only real speed sensor on the hardware:
    ``api.get_speed()`` returns the value the last ``set_speed()`` wrote, so
    the observation's speed entry is the command echoed back.  Feeding that
    back as ``z`` made the filter chase its own input.

    Only the magnitude is used, since the wrapper does not document whether
    ``x``/``y`` are body or world axes; the sign comes from the command.
    """

    velocity = info.get("velocity")
    if not isinstance(velocity, dict):
        return None
    if "x" not in velocity and "y" not in velocity:
        # An empty dict is a dropped frame, not a robot standing still.
        return None
    try:
        vx = float(velocity.get("x", 0.0))
        vy = float(velocity.get("y", 0.0))
    except (TypeError, ValueError):
        return None

    speed = float(np.hypot(vx, vy)) / 100.0  # the API reports cm/s
    if not np.isfinite(speed):
        return None

    commanded = float(info.get("speed_cmd", 0.0))
    return -speed if commanded < 0.0 else speed


class ImuHeadingSensor:
    """Heading measurement derived from the IMU yaw.

    ``api.get_heading()`` echoes the command, so the observation's heading is
    not a measurement either.  ``info["orientation"]["yaw"]`` is a genuine
    gyroscope-derived attitude, but its zero point and sign were both unknown.

    The zero is removed exactly: the first reading becomes the reference and
    only differences from it are used.  The sign is assumed (see
    ``IMU_YAW_SIGN``) but a wrong guess is detectable, because it holds the
    measured heading ~180 degrees from the command for a whole leg.  That
    disables this source in favour of the observation, so guessing wrong costs
    one warning rather than the run.
    """

    def __init__(self, initial_heading: float) -> None:
        self.initial_heading = float(initial_heading)
        self.reference_yaw: float | None = None
        self.consecutive_disagreements = 0
        self.disabled = False

    @staticmethod
    def _yaw_degrees(info: dict) -> float | None:
        orientation = info.get("orientation")
        if not isinstance(orientation, dict) or "yaw" not in orientation:
            return None
        try:
            yaw = float(orientation["yaw"])
        except (TypeError, ValueError):
            return None
        return yaw if np.isfinite(yaw) else None

    def measure(self, info: dict, commanded_heading: float) -> float | None:
        """Return a heading in radians, or ``None`` to use the observation."""

        if self.disabled:
            return None
        yaw = self._yaw_degrees(info)
        if yaw is None:
            return None

        if self.reference_yaw is None:
            self.reference_yaw = yaw
            return self.initial_heading

        turned = wrap_angle(np.deg2rad(yaw - self.reference_yaw))
        heading = wrap_angle(self.initial_heading + IMU_YAW_SIGN * turned)

        disagreement = abs(wrap_angle(heading - commanded_heading))
        if disagreement <= IMU_HEADING_DISAGREEMENT_LIMIT:
            self.consecutive_disagreements = 0
            return heading

        self.consecutive_disagreements += 1
        if self.consecutive_disagreements < IMU_DISAGREEMENT_PATIENCE:
            return heading

        self.disabled = True
        print(
            "WARNING: IMU heading disagreed with the command by more than "
            f"{np.degrees(IMU_HEADING_DISAGREEMENT_LIMIT):.0f} deg for "
            f"{IMU_DISAGREEMENT_PATIENCE} consecutive steps. Falling back to "
            "the observation; IMU_YAW_SIGN is probably wrong. Check "
            "imu_heading_deg against heading_cmd in the diagnostic log."
        )
        return None


def extract_measurement(
    observation: np.ndarray,
    info: dict | None = None,
    *,
    heading_sensor: ImuHeadingSensor | None = None,
    commanded_heading: float | None = None,
) -> np.ndarray:
    """Build the ``[heading, speed]`` measurement for one EKF update.

    In simulation the observation is ground truth plus Gaussian noise, so it
    is used as-is.  On hardware both entries come from real sensors instead:
    speed from the wheel encoders, heading from the IMU yaw.  Either falls
    back to the observation if its sensor frame is missing, so a dropped BLE
    packet degrades the measurement rather than stalling the filter.
    """

    z = np.array([float(observation[2]), float(observation[3])], dtype=np.float64)
    if info is None:
        return z

    measured_speed = robot_speed_measurement(info)
    if measured_speed is not None:
        z[1] = measured_speed

    if heading_sensor is not None and commanded_heading is not None:
        measured_heading = heading_sensor.measure(info, commanded_heading)
        if measured_heading is not None:
            z[0] = measured_heading
    return z


def _record(sim_info: dict, ekf: EKF) -> EstimateRecord:
    truth = np.asarray(sim_info["state_true"], dtype=np.float64)
    position_covariance = ekf.P[:2, :2]
    return EstimateRecord(
        sim_x=float(truth[0]),
        sim_y=float(truth[1]),
        real_x=float(ekf.state_est[0]),
        real_y=float(ekf.state_est[1]),
        p_xx=float(position_covariance[0, 0]),
        p_xy=float(position_covariance[0, 1]),
        p_yy=float(position_covariance[1, 1]),
    )


class DiagnosticLog:
    """Per-step log of raw sensors, EKF internals and control timing.

    Three tuning questions can only be settled with the robot in front of you,
    and one 200-step run answers all three: which frame ``get_velocity()``
    reports in, the zero and sign of the IMU yaw, and the real measurement
    noise that ``R`` currently only assumes.  The step time is logged because
    ``EKF`` hard-codes ``dt = 0.1``, so any drift in the real control period
    goes straight into the predicted displacement.

    ``t_rel`` uses ``perf_counter``: ``monotonic`` and ``time`` resolve to
    15.625 ms on Windows, which would swamp the jitter being measured.
    ``t_wall`` keeps that coarse resolution on purpose, to line this log up
    against the robot's own CSV.
    """

    COLUMNS = (
        "t_wall",
        "t_rel",
        "step",
        "speed_cmd",
        "heading_cmd",
        "obs_heading",
        "obs_speed",
        "enc_vx_cm_s",
        "enc_vy_cm_s",
        "enc_speed_m_s",
        "imu_yaw_deg",
        "gyro_z_deg_s",
        "z_heading",
        "z_speed",
        "odom_x",
        "odom_y",
        "ekf_x",
        "ekf_y",
        "ekf_heading",
        "ekf_speed",
        "nu_heading",
        "nu_speed",
        "nis",
        "P_xx",
        "P_xy",
        "P_yy",
    )

    def __init__(self) -> None:
        self.rows: list[tuple[float, ...]] = []
        self._start = time.perf_counter()

    @staticmethod
    def _component(source: object, *names: str) -> float:
        """Read the first present key, tolerating absent sensor streams."""

        if not isinstance(source, dict):
            return float("nan")
        for name in names:
            if name in source:
                try:
                    return float(source[name])
                except (TypeError, ValueError):
                    return float("nan")
        return float("nan")

    def record(
        self,
        step: int,
        action: np.ndarray,
        observation: np.ndarray,
        info: dict | None,
        ekf: EKF,
        measurement: np.ndarray | None = None,
    ) -> None:
        nan = float("nan")
        info = info or {}
        velocity = info.get("velocity")
        encoder_speed = robot_speed_measurement(info) if info else None

        innovation = ekf.last_innovation
        innovation_covariance = ekf.last_innovation_covariance
        try:
            nis = float(
                innovation @ np.linalg.solve(innovation_covariance, innovation)
            )
        except np.linalg.LinAlgError:
            nis = nan

        self.rows.append(
            (
                time.time(),
                time.perf_counter() - self._start,
                float(step),
                float(action[0]),
                float(action[1]),
                float(observation[2]),
                float(observation[3]),
                self._component(velocity, "x"),
                self._component(velocity, "y"),
                nan if encoder_speed is None else encoder_speed,
                self._component(info.get("orientation"), "yaw"),
                self._component(info.get("gyroscope"), "z", "yaw"),
                nan if measurement is None else float(measurement[0]),
                nan if measurement is None else float(measurement[1]),
                float(info["state_odom"][0]) if "state_odom" in info else nan,
                float(info["state_odom"][1]) if "state_odom" in info else nan,
                float(ekf.state_est[0]),
                float(ekf.state_est[1]),
                float(ekf.state_est[2]),
                float(ekf.state_est[3]),
                float(innovation[0]),
                float(innovation[1]),
                nis,
                float(ekf.P[0, 0]),
                float(ekf.P[0, 1]),
                float(ekf.P[1, 1]),
            )
        )

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.COLUMNS)
            writer.writerows(self.rows)
        temporary.replace(path)
        return path


def run_experiment(
    sim_env: SpheroEnv,
    robot_env: Robot | None,
    *,
    render: bool,
    teleop: bool,
    seed: int,
    diagnostics: DiagnosticLog | None = None,
) -> list[EstimateRecord]:
    """Run exactly 200 predict/update cycles and return posterior records."""

    sim_observation, _ = sim_env.reset(seed=seed)
    if robot_env is None:
        initial_heading = float(sim_observation[2])
        initial_speed = float(sim_observation[3])
    else:
        _, robot_info = robot_env.reset(seed=seed)
        initial_heading = float(robot_info["state_odom"][2])
        initial_speed = float(robot_info["state_odom"][3])

    _align_simulator_heading(sim_env, initial_heading)

    initial_state = np.array(
        [0.0, 0.0, initial_heading, initial_speed], dtype=np.float64
    )
    hardware = robot_env is not None
    ekf = EKF(
        dt=DT,
        initial_state=initial_state,
        process_noise=REAL_PROCESS_NOISE if hardware else None,
        measurement_noise=REAL_MEASUREMENT_NOISE if hardware else None,
    )
    sim_env.update_estimate(ekf.state_est, ekf.P)
    if robot_env is not None:
        # Without this the robot's own log records the belief as NaN.
        robot_env.update_estimate(ekf.state_est, ekf.P)
    if render:
        # Initialise pygame before polling keyboard/window events.
        sim_env.render()

    controller = TeleopController(initial_heading) if teleop else None
    # Hardware only: in simulation the observation already is a measurement.
    heading_sensor = (
        ImuHeadingSensor(initial_heading) if robot_env is not None else None
    )
    records: list[EstimateRecord] = []
    # Lane progress, measured rather than assumed; see scripted_action.
    travelled = 0.0
    previous_position: np.ndarray | None = None
    # perf_counter, not monotonic: the latter quantises to 15.625 ms here, so
    # pacing a 100 ms loop with it injects up to 16% period error by itself.
    next_tick = time.perf_counter()

    print(
        f"Starting {N_STEPS} steps; hardware speed limit is "
        f"{RAW_SPEED_LIMIT}/255."
    )
    for step in range(N_STEPS):
        if controller is not None:
            action = controller.action(robot_env)
        else:
            if render:
                _poll_for_abort()
            action = scripted_action(travelled, initial_heading)

        sim_observation, _, _, _, sim_info = sim_env.step(action)
        if robot_env is None:
            sensor_observation, sensor_info = sim_observation, None
        else:
            sensor_observation, _, _, _, sensor_info = robot_env.step(action)

        measurement = extract_measurement(
            sensor_observation,
            sensor_info,
            heading_sensor=heading_sensor,
            commanded_heading=float(action[1]),
        )
        ekf.predict(action)
        ekf.update(measurement)
        sim_env.update_estimate(ekf.state_est, ekf.P)
        if robot_env is not None:
            robot_env.update_estimate(ekf.state_est, ekf.P)
        records.append(_record(sim_info, ekf))
        if diagnostics is not None:
            diagnostics.record(
                step,
                action,
                sensor_observation,
                sensor_info if sensor_info is not None else sim_info,
                ekf,
                measurement,
            )

        # Advance lane progress by what actually moved.  Hardware uses the
        # locator, which reports nothing while the robot is stuck; simulation
        # uses its own truth so both environments follow one schedule.
        source = sim_info if robot_env is None else sensor_info
        position = np.asarray(
            source["state_true" if robot_env is None else "state_odom"][:2],
            dtype=np.float64,
        )
        if previous_position is not None:
            travelled += float(np.linalg.norm(position - previous_position))
        previous_position = position

        if render:
            sim_env.render()
        if (step + 1) % 50 == 0:
            print(
                f"Completed {step + 1}/{N_STEPS} steps, "
                f"{travelled:.2f}/{TRACK_LAP:.2f} m of the lap"
            )

        next_tick += DT
        time.sleep(max(0.0, next_tick - time.perf_counter()))

    if len(records) != N_STEPS:
        raise RuntimeError(f"expected {N_STEPS} records, got {len(records)}")
    return records


def write_submission(
    student_id: str, records: list[EstimateRecord]
) -> Path:
    """Atomically write a validated 200-row automarker CSV."""

    if not student_id.isdigit():
        raise ValueError("student_id must contain digits only")
    if len(records) != N_STEPS:
        raise ValueError(f"submission requires exactly {N_STEPS} records")

    values = np.asarray(records, dtype=float)
    if values.shape != (N_STEPS, len(CSV_COLUMNS)):
        raise ValueError("submission rows have an invalid shape")
    if not np.all(np.isfinite(values)):
        raise ValueError("submission contains NaN or infinite values")

    output = LAB_DIR / f"{student_id}_lab2.csv"
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(values)
    temporary.replace(output)
    return output


def print_submission_metrics(records: list[EstimateRecord]):
    """Print the two published automarker metrics and thresholds."""

    values = np.asarray(records, dtype=float)
    result = analyze_values(values)
    mean_status = "PASS" if result.mean_passed else "FAIL"
    coverage_status = "PASS" if result.coverage_passed else "FAIL"

    print("\nLab 2 assessment metrics")
    print(
        "Mean Mahalanobis distance of position error (sim - real): "
        f"{result.mean_mahalanobis:.4f} "
        f"({mean_status}; required <= 4.0)"
    )
    print(
        "Chi-square pass rate (2 DoF, 95% gate): "
        f"{result.chi_square_pass_rate:.3f} "
        f"({coverage_status}; required >= 0.90)"
    )
    return result


def hold_visualization(sim_env: SpheroEnv) -> None:
    """Keep the final trajectory visible until Q or window close."""

    print("Simulation complete. Press Q or close the window to exit.")
    while True:
        try:
            _poll_for_abort()
        except ExperimentAborted:
            return
        sim_env.render()
        time.sleep(0.05)


def stop_motion(sim_env: SpheroEnv, robot_env: Robot | None) -> None:
    """Stop both systems immediately while keeping their contexts open."""

    if robot_env is not None:
        robot_env.emergency_stop()
    sim_env.emergency_stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 2 EKF experiment")
    parser.add_argument("--sim", action="store_true", help="simulation only")
    parser.add_argument("--no-render", action="store_true", help="disable animation")
    parser.add_argument("--teleop", action="store_true", help="use keyboard control")
    parser.add_argument(
        "--hold",
        action="store_true",
        help="keep the final visualization open until Q or window close",
    )
    parser.add_argument("--seed", type=int, default=5178, help="simulation seed")
    parser.add_argument(
        "--student-id",
        required=True,
        help="digits used for the studentid_lab2.csv filename",
    )
    args = parser.parse_args()
    if not args.student_id.isdigit():
        parser.error("--student-id must contain digits only")
    if args.teleop and args.no_render:
        parser.error("--teleop requires the pygame window")
    if args.hold and args.no_render:
        parser.error("--hold requires the pygame window")
    return args


def main() -> None:
    args = parse_args()
    render = not args.no_render
    diagnostics = DiagnosticLog()
    diagnostic_path = LAB_DIR.parents[1] / "logs" / "lab2_diagnostics.csv"

    try:
        with open_sim_env(render) as sim_env:
            if args.sim:
                try:
                    records = run_experiment(
                        sim_env,
                        None,
                        render=render,
                        teleop=args.teleop,
                        seed=args.seed,
                        diagnostics=diagnostics,
                    )
                finally:
                    stop_motion(sim_env, None)
            else:
                # Exit this context immediately after step 200, so the robot
                # and its BLE connection are closed before --hold.
                with open_real_env() as robot_env:
                    try:
                        records = run_experiment(
                            sim_env,
                            robot_env,
                            render=render,
                            teleop=args.teleop,
                            seed=args.seed,
                            diagnostics=diagnostics,
                        )
                    finally:
                        stop_motion(sim_env, robot_env)
                print("Robot stopped; Bluetooth connection closed.")

            output = write_submission(args.student_id, records)
            evidence = "simulation-only" if args.sim else "robot + simulator"
            print(f"Automarker CSV ({evidence}): {output}")
            result = print_submission_metrics(records)
            plot_path = output.with_name(f"{output.stem}_analysis.png")
            save_plot(np.asarray(records, dtype=float), result, plot_path)
            print(f"Analysis graph: {plot_path}")
            if args.hold:
                hold_visualization(sim_env)
    except ExperimentAborted as error:
        raise SystemExit(f"Experiment aborted: {error}; no CSV written") from error
    finally:
        # Written even on an abort: a run that had to be stopped is exactly
        # the one whose sensor trace is worth reading.  The failure is
        # swallowed because raising here would replace the exception already
        # propagating, hiding the real fault behind a disk error.
        if diagnostics.rows:
            try:
                diagnostics.write(diagnostic_path)
                print(f"Diagnostic log: {diagnostic_path}")
            except OSError as error:
                print(f"WARNING: could not write the diagnostic log: {error}")

if __name__ == "__main__":
    main()
