# Import necessary libraries
from sphero_env.robot.connect import scan_and_connect
from sphero_unsw.sphero_edu import SpheroEduAPI
from sphero_env.robot.robot import Robot
from sphero_env.envs import SpheroEnv

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from Planner import *
from sphero_env.envs.custom_maze_full import (
    GOAL_CELL,
    GRID_RESOLUTION,
    START_CELL,
    build_occupancy_grid,
    cell_to_world,
)

from contextlib import ExitStack, contextmanager

# Lab 2 owns the calibrated motion model, the filter built on it, and the
# hardware plumbing that was debugged on the robot.  Importing rather than
# copying keeps one definition of each.
LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB_DIR.parent / "lab2"))
from EKF import MODEL_CONFIG, dynamics, wrap_angle  # noqa: E402
from Estimator import (  # noqa: E402
    REAL_MEASUREMENT_NOISE,
    SIM_MEASUREMENT_NOISE,
    Estimator,
)
from lab2 import (  # noqa: E402
    REAL_PROCESS_NOISE,
    ExperimentAborted,
    ImuHeadingSensor,
    ThrottledRobot,
    connect_with_retry,
    robot_speed_measurement,
    stop_motion,
)
# Private to Lab 2, but it is the abort gesture the operator already knows and
# it doubles as the pygame event pump the window needs to stay responsive.
from lab2 import _poll_for_abort  # noqa: E402

LAB1_SEED = 0
# Lab 3 sets no step count: the CSV holds every step from start to goal.  This
# is only the skeleton's backstop for a run that never arrives; pressing Q or
# closing the window ends one sooner.
MAX_STEPS = 5000
map = build_occupancy_grid()

STUDENT_ID = "33377006"
CSV_COLUMNS = ("sim_x", "sim_y", "real_x", "real_y")

# 0.105 s, the control period ThrottledRobot actually achieves; the Lab 2
# filter refuses any other value because the model was calibrated at it.
DT = MODEL_CONFIG["dt"]
# The maze starts the robot on the plate at logical (0, 4), not at the origin.
START_POSITION = cell_to_world(START_CELL)
GOAL_TOLERANCE = 0.1
# Leave an intermediate waypoint early: the next one lies straight ahead, so
# there is nothing to gain from closing the last few centimetres of this one.
WAYPOINT_TOLERANCE = 0.05
# The final waypoint is the goal, and the run is marked on how close the robot
# stops to it.  Stopping at GOAL_TOLERANCE would score right on the 0.10 m
# limit with nothing left for hardware error.
ARRIVAL_TOLERANCE = 0.03

# A 0.15 command maps to raw 15. The calibrated model predicts a steady
# speed of 1.957 * (0.15 - 0.0537) ~= 0.19 m/s at that command.
CRUISE_SPEED = 0.15

# Below this the encoder is reporting a robot that is not moving, whatever it
# was told to do.  Same threshold the Lab 2 diagnostics used to count stalls.
STALL_SPEED = 0.01

# Hardware only, from Lab 2: vel_limit doubles as the scale for the raw speed
# byte there, so it has to stay 0.15.
COMMAND_SPEED_LIMIT = 0.15
RAW_SPEED_LIMIT = 15


### Custom dynamics function for the Sphero robot - replace this with the one you developed in Lab 1
### Replaced: `dynamics` is imported from Lab 2 above.  The skeleton's version
### read action[1] as a turn rate, but SpheroEnv defines the action as
### [speed, absolute heading] and clips it against that action space.


class Controller:
    def __init__(self, dt=0.1, cruise_speed=CRUISE_SPEED,
                 heading_tolerance=np.deg2rad(15.0)):
        self.dt = dt
        self.cruise_speed = cruise_speed
        self.heading_tolerance = heading_tolerance

    def compute_action(self, state, waypoint):
        """
        Fill in this function to implement a simple controller that computes the action based on the current state and the waypoint.
        """
        offset = (
            np.asarray(waypoint, dtype=np.float64)[:2]
            - np.asarray(state, dtype=np.float64)[:2]
        )
        # Heading zero points along +y and pi/2 along +x, so x is the first
        # argument to atan2 here.
        heading_cmd = float(np.arctan2(offset[0], offset[1]))

        # Turn on the spot before translating.  A corridor is one 0.25 m plate
        # wide and the model keeps driving while it turns, so cornering under
        # power would cut across the inside wall.
        heading_error = wrap_angle(heading_cmd - float(state[2]))
        speed = 0.0 if abs(heading_error) > self.heading_tolerance else self.cruise_speed

        return np.array([speed, heading_cmd], dtype=np.float64)


class WaypointFollower:
    """Follow own waypoints; optionally wait for another follower at corners."""

    def __init__(self, planner, controller, goal, state, *, wait_for=None):
        self.planner = planner
        self.controller = controller
        self.goal = goal
        self.waypoints = planner.plan(state, goal)
        self.target = 0
        self.replans = 0
        self.arrived = False
        self.wait_for = wait_for
        self.reached_waypoints = set()
        self.waiting_heading = None

    def _resume_target(self, position):
        """Which waypoint of a fresh plan to drive at first.

        The plan starts on the cell the estimate snapped to.  When the estimate
        has strayed into the band the occupancy grid gives a wall, that cell
        centre is beside the robot, not ahead of it, and driving back to it
        turns the robot away from the way it was going.  Beside a wall that is
        a loop -- the 2026-09-18 run went round it for about 3 s on plate
        (3, 0): turn to the centre, turn back, drift into the band, replan,
        turn to the centre.

        So the start is skipped when the robot is level with it or past it
        along the first leg, and only while the robot is inside that leg's
        corridor.  The walls beside a leg sit one grid cell from its centre
        line; from further out, the straight run to the second waypoint can
        cut through a wall the plan went round.

        The corridor is the full cell, not the cell less the ball's radius.
        From the outer part of it the ball can clip a wall end on the way;
        but an estimate that far out is as often a locator reading 9-12 cm
        wide of a robot still on the centre line -- the 2026-09-18 runs saw
        errors that size -- and sending that one back to the start restores
        the loop.  A clipped wall end is a bump to recover from; the loop is
        the end of the run.
        """
        if len(self.waypoints) < 2:
            return 0
        start = np.asarray(self.waypoints[0], dtype=np.float64)
        leg = np.asarray(self.waypoints[1], dtype=np.float64) - start
        direction = leg / np.linalg.norm(leg)
        offset = np.asarray(position, dtype=np.float64)[:2] - start
        along = float(offset @ direction)
        across = abs(float(offset[0] * direction[1] - offset[1] * direction[0]))
        in_corridor = across < self.planner.grid_resolution
        return 1 if along > -WAYPOINT_TOLERANCE and in_corridor else 0

    def _peer_reached_here_or_later(self):
        """True once the other follower has reached this waypoint or a later one.

        A replan restarts the other follower's path wherever it then is, so a
        robot that replans from beyond this corner never records the corner
        itself and a plain membership test would hold the simulator here for
        the rest of the run.  Its arrival at any later waypoint of this path
        shows it got past the corner anyway.  On an ordinary run it records
        waypoints in path order, so this answers exactly as that test did.
        """
        reached = self.wait_for.reached_waypoints
        return any(tuple(point) in reached for point in self.waypoints[self.target:])

    def action(self, state):
        if self.waiting_heading is not None:
            return np.array([0.0, self.waiting_heading], dtype=np.float64)
        if self.arrived:
            # Wait at the goal for the other ball, facing the way it came in.
            return np.array([0.0, float(state[2])], dtype=np.float64)
        return self.controller.compute_action(state, self.waypoints[self.target])

    def advance(self, state, collided):
        """Update progress after a step; True once the goal has been reached."""
        if self.arrived:
            return True
        position = np.asarray(state, dtype=np.float64)[:2]

        # Replan when the run stops matching the plan: a collision means the
        # ball met a wall the plan said was not there, and a position inside a
        # wall means the waypoints ahead were computed for a pose it is no
        # longer in.  Either way the old path is stale.
        if collided or not self.planner.is_clear(position):
            self.waypoints = self.planner.plan(state, self.goal)
            self.target = self._resume_target(position)
            self.replans += 1
            self.waiting_heading = None

        final = self.target == len(self.waypoints) - 1
        tolerance = ARRIVAL_TOLERANCE if final else WAYPOINT_TOLERANCE
        at_waypoint = float(np.linalg.norm(position - self.waypoints[self.target])) <= tolerance
        if at_waypoint or self.waiting_heading is not None:
            waypoint = tuple(self.waypoints[self.target])
            # The four-connected plan turns where adjacent legs are perpendicular or opposite.
            corner = 0 < self.target < len(self.waypoints) - 1 and np.dot(
                self.waypoints[self.target] - self.waypoints[self.target - 1],
                self.waypoints[self.target + 1] - self.waypoints[self.target],
            ) <= 0.0
            if self.wait_for is not None and corner and not self._peer_reached_here_or_later():
                if self.waiting_heading is None:
                    self.waiting_heading = float(state[2])
                return False
            self.waiting_heading = None
            # Coordinates survive replanning and remember a robot that already passed.
            self.reached_waypoints.add(waypoint)
            if final:
                self.arrived = True
            else:
                self.target += 1
        return self.arrived


def make_sim_env(render=True):
    return SpheroEnv(
        dt=DT,
        max_steps=MAX_STEPS,
        vel_limit=COMMAND_SPEED_LIMIT,
        world_width=1.25,
        world_height=1.25,
        goal_pos=(0.5, 0.5),
        goal_tolerance=GOAL_TOLERANCE,
        occupancy_grid=map,
        grid_resolution=GRID_RESOLUTION,
        dynamics=dynamics,
        obs_noise_std_pos=0.05,
        process_noise_std_speed=0.005,
        process_noise_std_heading=0.01,
        obs_noise_std_vel=0.025,
        render_mode="human" if render else None,
        window_size=(800, 800),
    )


class Lab3Robot(ThrottledRobot):
    """ThrottledRobot that stops before it turns on the spot.

    ThrottledRobot writes the heading first and the speed second, which suits
    a heading that changes while driving.  It does not suit the controller's
    turn on the spot: the SDK's set_heading() sends the new heading together
    with the speed it last sent, so a robot told to stop and face a new way
    first rolls off that way at cruise speed, for one BLE round trip, before
    the zero speed reaches it.  Next to a wall that is a lurch toward or along
    the wall every time the controller stops to realign.

    Stopping first is the same two writes in the other order, so it costs no
    BLE time; every other command goes out exactly as ThrottledRobot sends it.
    """

    def set_heading_and_speed(self, heading_deg, speed):
        commanded = int(heading_deg)
        if int(np.clip(speed, 0, 255)) == 0 and commanded != self._last_heading_deg:
            # Not through super(): the lock is not re-entrant.
            with self._lock:
                self.api.set_speed(0)
                self.api.set_heading(commanded)
                self._last_heading_deg = commanded
            return
        super().set_heading_and_speed(heading_deg, speed)


def make_real_env(api):
    # ThrottledRobot, not Robot: the base class writes heading and speed
    # separately every step, which measured a 209 ms period against the 105 ms
    # the model is calibrated for.  Lab3Robot adds stopping before a turn.
    return Lab3Robot(
        api=api,
        dt=DT,
        max_steps=MAX_STEPS,
        vel_limit=COMMAND_SPEED_LIMIT,
        raw_speed_limit=RAW_SPEED_LIMIT,
        world_width=1.25,
        world_height=1.25,
        goal_pos=(0.5, 0.5),
        goal_tolerance=GOAL_TOLERANCE,
        render_mode=None,
        window_size=(800, 800),
    )


@contextmanager
def managed_env(sim: bool, render: bool = True):
    """Yield ``(sim_env, robot_env)``; ``robot_env`` is None in simulation.

    The simulator runs in both modes.  It owns the occupancy grid and the goal,
    and its ground truth supplies sim_x/sim_y for the automarker.  In hardware
    mode, the simulator and robot use their own waypoint followers and actions.
    """
    sim_env = make_sim_env(render=render)
    sim_env.set_log_path("logs/lab3_sim.csv")
    sim_env.start_logging()
    try:
        if sim:
            print("Simulation only: no robot, no Bluetooth.")
            yield sim_env, None
        else:
            # The simulator window opens in this mode too, so say which mode
            # this is before the scan rather than leaving the window to imply
            # it.  The scanner reprompts forever if nothing is found; Ctrl+C
            # is the way out.
            print(
                "Hardware mode: scanning for the robot. "
                "Pass --sim to run without one, or press Ctrl+C to stop."
            )
            with ExitStack() as stack:
                selected_toy, _ = scan_and_connect()
                print(f"Selected: {selected_toy.name}")

                api = connect_with_retry(stack, selected_toy)
                real_env = make_real_env(api)
                real_env.set_log_path("logs/lab3_real.csv")

                real_env.start_logging()
                try:
                    yield sim_env, real_env
                finally:
                    real_env.close()
                    real_env.stop_logging()
    finally:
        sim_env.stop_logging()
        sim_env.close()


def extract_measurement(observation, info=None, *, heading_sensor=None,
                        commanded_heading=None, position_offset=None):
    """Build the ``[x, y, heading, speed]`` measurement for one update.

    In simulation the observation is the state plus Gaussian noise, so it is
    the measurement.  On hardware every entry comes from a real sensor
    instead: position from the locator, speed from the wheel encoders, heading
    from the IMU yaw.  Each falls back to the observation when its frame is
    missing, so a dropped BLE packet degrades the measurement rather than
    stalling the filter.
    """
    z = np.array(
        [float(observation[0]), float(observation[1]),
         float(observation[2]), float(observation[3])],
        dtype=np.float64,
    )
    if info is None:
        return z

    # The locator zeroes wherever the robot was reset, so it reports
    # displacement from the start plate rather than a world position.
    locator = info.get("state_odom")
    if locator is not None and position_offset is not None:
        z[0:2] = (
            np.asarray(locator, dtype=np.float64)[:2]
            + np.asarray(position_offset, dtype=np.float64)
        )

    measured_speed = robot_speed_measurement(info)
    if measured_speed is not None:
        z[3] = measured_speed

    if heading_sensor is not None and commanded_heading is not None:
        measured_heading = heading_sensor.measure(info, commanded_heading)
        if measured_heading is not None:
            z[2] = measured_heading
    return z


class DiagnosticLog:
    """Per-step record of what the hardware loop sensed and decided.

    The 2026-09-18 run could not be taken apart afterwards: lab3_real.csv holds
    the command echoed back as heading and speed, and nothing of the collision
    flag, the IMU, the encoder or the replans.  This keeps them.  It copies
    only values the loop has already computed -- no sensor is read and no
    measurement taken a second time -- so logging cannot change what a run
    does.  Positions are in the world frame, angles in degrees.

    Each run writes its own file, named for when it started.  The runs worth
    investigating are the ones that go wrong, and the usual response to one
    is to run again straight away, which would overwrite a single shared log.
    """

    COLUMNS = (
        "step", "t_s", "speed_cmd", "heading_cmd_deg",
        "collision", "encoder_speed", "imu_yaw_deg", "gyro_z_deg_s",
        "imu_pitch_deg", "imu_roll_deg", "vel_x_cm_s", "vel_y_cm_s",
        "locator_x", "locator_y",
        "z_x", "z_y", "z_heading_deg", "z_speed", "imu_heading_disabled",
        "est_x", "est_y", "est_heading_deg", "est_speed",
        "stalled", "clear", "target", "target_x", "target_y", "replans",
        "sim_target", "sim_waiting",
    )

    def __init__(self):
        self.rows = []
        self._start = time.perf_counter()
        self.path = Path("logs") / f"lab3_diagnostics_{time.strftime('%m%d_%H%M%S')}.csv"

    @staticmethod
    def _reading(source, *names):
        """The first of ``names`` present in a sensor dict, else NaN."""
        if isinstance(source, dict):
            for name in names:
                if name in source:
                    try:
                        return float(source[name])
                    except (TypeError, ValueError):
                        break
        return float("nan")

    def record(self, *, step, action, collided, info, encoder_speed, measurement,
               imu_disabled, estimate, stalled, clear, target, target_xy,
               replans, sim_target, sim_waiting):
        nan = float("nan")
        locator = info.get("state_odom")
        locator_xy = (
            (nan, nan) if locator is None
            else np.asarray(locator, dtype=np.float64)[:2] + np.asarray(START_POSITION)
        )
        self.rows.append((
            float(step), time.perf_counter() - self._start,
            float(action[0]), float(np.degrees(action[1])),
            float(collided),
            nan if encoder_speed is None else float(encoder_speed),
            self._reading(info.get("orientation"), "yaw"),
            self._reading(info.get("gyroscope"), "z", "yaw"),
            # Tilt shows a slope on the top row, and the encoder's own vector
            # gives a direction of travel that locator packet timing cannot
            # smear -- the two things a skew along a wall could be.
            self._reading(info.get("orientation"), "pitch"),
            self._reading(info.get("orientation"), "roll"),
            self._reading(info.get("velocity"), "x"),
            self._reading(info.get("velocity"), "y"),
            float(locator_xy[0]), float(locator_xy[1]),
            float(measurement[0]), float(measurement[1]),
            float(np.degrees(measurement[2])), float(measurement[3]),
            float(imu_disabled),
            float(estimate[0]), float(estimate[1]),
            float(np.degrees(estimate[2])), float(estimate[3]),
            float(stalled), float(clear),
            float(target), float(target_xy[0]), float(target_xy[1]),
            float(replans), float(sim_target), float(sim_waiting),
        ))

    def write(self, path=None):
        path = Path(self.path if path is None else path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.COLUMNS)
            writer.writerows(self.rows)
        temporary.replace(path)
        return path


def start_at(control_env, position, heading=0.0):
    """Place the simulator on the maze's start plate.

    SpheroEnv.reset() always starts at the origin, but the start plate sits at
    (-0.5, -0.5); without this the state and the occupancy grid disagree.
    """
    start = np.array([position[0], position[1], heading], dtype=np.float32)
    control_env.state_true[0:3] = start
    control_env.state_odom[0:3] = start


def control_loop(sim_env, robot_env=None, max_steps=MAX_STEPS, render=True,
                 verbose=True, records=None, diagnostics=None):
    """Drive the maze. ``records`` may be passed in so an aborted run still logs.

    ``diagnostics``, a DiagnosticLog, collects a row per step on hardware.
    """
    hardware = robot_env is not None
    if records is None:
        records = []

    sim_obs, _ = sim_env.reset(seed=LAB1_SEED)
    if hardware:
        _, robot_info = robot_env.reset(seed=LAB1_SEED)
        # The robot is placed on the start plate facing wherever it settled.
        initial_heading = float(robot_info["state_odom"][2])
    else:
        initial_heading = 0.0
    start_at(sim_env, START_POSITION, initial_heading)

    controller = Controller(dt=sim_env.dt)
    planner = Planner(
        map=sim_env.occupancy_grid,
        dt=sim_env.dt,
        grid_resolution=sim_env.grid_resolution,
    )

    # The controller steers on the filter's estimate, not on the raw noisy
    # observation, so one state feeds planning, control and the log.
    ekf = Estimator(
        dt=DT,
        initial_state=np.array(
            [START_POSITION[0], START_POSITION[1], initial_heading, 0.0],
            dtype=np.float64,
        ),
        process_noise=REAL_PROCESS_NOISE if hardware else None,
        measurement_noise=REAL_MEASUREMENT_NOISE if hardware else SIM_MEASUREMENT_NOISE,
    )
    # api.get_heading() echoes the command, so on hardware the heading has to
    # come from the IMU yaw instead of from the observation.
    heading_sensor = ImuHeadingSensor(initial_heading) if hardware else None

    goal = np.asarray(sim_env.goal_pos, dtype=np.float64)
    real_loop = WaypointFollower(planner, controller, goal, ekf.state_est)
    # In simulation there is one ball and the filter tracks it, so one loop
    # drives it.  On hardware the simulator is a second ball with a loop of its
    # own, waiting at corners until the robot has reached that corner or a
    # point beyond it.
    sim_loop = (
        WaypointFollower(planner, controller, goal, sim_env.state_true, wait_for=real_loop)
        if hardware else real_loop
    )

    if render:
        # Initialise pygame before polling its event queue.
        sim_env.render()

    # perf_counter, not monotonic: the latter quantises to 15.625 ms here.
    next_tick = time.perf_counter()

    for step in range(max_steps):
        if render:
            # Also drains the pygame queue; without it the window stops
            # repainting and Windows marks it "not responding" mid-run.
            _poll_for_abort()

        action = real_loop.action(ekf.state_est)
        # What that action aimed at, for the diagnostic log: advance() may
        # replace the waypoints before the row is written.
        target = real_loop.target
        target_xy = real_loop.waypoints[target]

        if hardware:
            sensor_obs, _, _, _, sensor_info = robot_env.step(action)
            sim_action = sim_loop.action(sim_env.state_true)
            # The model has no notion of a wheel spinning against a bump, so a
            # robot that was told to move and did not holds the simulator back
            # too, keeping the two balls in step.  A robot told to stop --
            # turning on the spot, or waiting at the goal -- reads zero as well
            # but is not stalled; holding the simulator then would freeze it.
            measured_speed = robot_speed_measurement(sensor_info)
            stalled = (
                action[0] > MODEL_CONFIG["command_deadband_m_s"]
                and measured_speed is not None
                and abs(measured_speed) < STALL_SPEED
            )
            if stalled:
                sim_action[0] = 0.0
        else:
            sim_action, sensor_info = action, None

        sim_obs, _, terminated, truncated, sim_info = sim_env.step(sim_action)
        if not hardware:
            sensor_obs = sim_obs

        ekf.predict(action)
        # Taken once and kept: the IMU heading sensor has state of its own, so
        # measuring a second time for the log would change the next reading.
        measurement = extract_measurement(
            sensor_obs,
            sensor_info,
            heading_sensor=heading_sensor,
            commanded_heading=float(action[1]),
            position_offset=START_POSITION,
        )
        ekf.update(measurement)
        sim_env.update_estimate(ekf.state_est, ekf.P)
        if hardware:
            robot_env.update_estimate(ekf.state_est, ekf.P)

        records.append(
            (
                float(sim_info["state_true"][0]),
                float(sim_info["state_true"][1]),
                float(ekf.state_est[0]),
                float(ekf.state_est[1]),
            )
        )

        real_done = real_loop.advance(ekf.state_est, sensor_obs[4] > 0.5)
        sim_done = (
            sim_loop.advance(sim_info["state_true"], sim_obs[4] > 0.5)
            if hardware else real_done
        )
        if hardware and diagnostics is not None:
            diagnostics.record(
                step=step,
                action=action,
                collided=sensor_obs[4] > 0.5,
                info=sensor_info,
                encoder_speed=measured_speed,
                measurement=measurement,
                imu_disabled=heading_sensor.disabled,
                estimate=ekf.state_est,
                stalled=stalled,
                clear=planner.is_clear(ekf.state_est[:2]),
                target=target,
                target_xy=target_xy,
                replans=real_loop.replans,
                sim_target=sim_loop.target,
                sim_waiting=sim_loop.waiting_heading is not None,
            )
        # The run ends when both balls are at the goal; whichever gets there
        # first waits for the other, so both CSV columns finish on it.
        if real_done and sim_done:
            if verbose:
                replans = f"{real_loop.replans} replans"
                if hardware:
                    replans += f", simulator {sim_loop.replans}"
                print(f"Goal reached after {step + 1} steps, {replans}.")
            break

        if render:
            sim_env.render()

        if terminated or truncated:
            if verbose:
                print(f"Run stopped after {step + 1} steps (terminated={terminated}).")
            break

        if hardware:
            next_tick += DT
            time.sleep(max(0.0, next_tick - time.perf_counter()))
    else:
        if verbose:
            print(
                f"Step budget of {max_steps} used up "
                f"{float(np.linalg.norm(ekf.state_est[:2] - goal)):.3f} m from the goal."
            )

    sim_env.emergency_stop()
    if hardware:
        robot_env.emergency_stop()
    return records


def write_submission(records, student_id=STUDENT_ID):
    """Write the automarker CSV: simulator truth against the filter's estimate."""
    path = LAB_DIR / f"{student_id}_lab3.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for row in records:
            writer.writerow(f"{value:.6f}" for value in row)
    return path


def print_metrics(records):
    """Report the four numbers the automarker checks, against the A* optimum."""
    data = np.asarray(records, dtype=np.float64)
    goal = np.asarray(cell_to_world(GOAL_CELL), dtype=np.float64)
    optimal = np.asarray(
        Planner(map=map, dt=DT, grid_resolution=GRID_RESOLUTION).plan(
            np.array([START_POSITION[0], START_POSITION[1], 0.0, 0.0]), goal
        ),
        dtype=np.float64,
    )

    def deviation(track):
        worst = 0.0
        for point in track:
            best = np.inf
            for a, b in zip(optimal, optimal[1:]):
                segment = b - a
                length2 = float(segment @ segment)
                t = 0.0 if length2 == 0.0 else float(
                    np.clip((point - a) @ segment / length2, 0.0, 1.0)
                )
                best = min(best, float(np.linalg.norm(point - (a + t * segment))))
            worst = max(worst, best)
        return worst

    print(f"{'metric':<32}{'value':>10}{'limit':>10}  result")
    for name, value, limit in (
        ("final distance to goal (sim)", float(np.linalg.norm(data[-1, 0:2] - goal)), 0.10),
        ("final distance to goal (real)", float(np.linalg.norm(data[-1, 2:4] - goal)), 0.10),
        ("distance to optimal path (sim)", deviation(data[:, 0:2]), 0.20),
        ("distance to optimal path (real)", deviation(data[:, 2:4]), 0.20),
    ):
        print(f"{name:<32}{value:>10.4f}{limit:>10.2f}  {'PASS' if value <= limit else 'FAIL'}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Lab 3 maze navigation. Without --sim the robot is scanned "
                    "for over Bluetooth and driven alongside the simulator."
    )
    parser.add_argument(
        "--sim", action="store_true",
        help="simulation only; skip the Bluetooth scan and the robot",
    )
    parser.add_argument("--no-render", action="store_true", help="disable animation")
    parser.add_argument(
        "--steps", type=int, default=MAX_STEPS,
        help=f"safety cap only; Lab 3 sets no step count (default {MAX_STEPS})",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    render = not args.no_render
    # Held out here so an abort or a crash still writes what the run produced;
    # a hardware run costs a Bluetooth pairing and a robot placement to repeat.
    records = []
    diagnostics = None if args.sim else DiagnosticLog()

    try:
        with managed_env(args.sim, render=render) as (sim_env, robot_env):
            try:
                control_loop(
                    sim_env, robot_env, max_steps=args.steps,
                    render=render, records=records, diagnostics=diagnostics,
                )
            finally:
                stop_motion(sim_env, robot_env)
    except ExperimentAborted as error:
        print(f"Run aborted: {error}")
    finally:
        try:
            if records:
                path = write_submission(records)
                print(f"Wrote {len(records)} rows to {path}")
                print_metrics(records)
        finally:
            # The diagnostic log is the evidence for whatever went wrong, so it
            # is written even when the submission could not be -- a CSV held
            # open in Excel, say.  Failing to write it is reported, never
            # raised over an error already on its way out.
            if diagnostics is not None and diagnostics.rows:
                try:
                    path = diagnostics.write()
                except OSError as error:
                    print(f"Could not write the diagnostic log: {error}")
                else:
                    print(f"Wrote {len(diagnostics.rows)} diagnostic rows to {path}")


if __name__ == "__main__":
    main()
