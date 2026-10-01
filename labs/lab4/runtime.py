"""Shared plumbing for Lab 4: the maze, the robot, the safety layer, the run loop.

Lab 3 already drives this maze on the real robot -- calibrated model, filter,
IMU heading, BLE throttling, stopping before a turn -- so Lab 4 imports it the
way Lab 3 imports Lab 2 and replaces only what decides the action.  Lab 3's
own control_loop, managed_env and write_submission are not used: they carry
the planner-driven navigation and Lab 3's output paths.
"""

from collections import Counter, deque
import sys
import time
from pathlib import Path

import numpy as np

LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB_DIR.parent / "lab3"))
import lab3  # noqa: E402  (puts labs/lab2 on the path as well)
from lab3 import (  # noqa: E402
    ARRIVAL_TOLERANCE,
    CRUISE_SPEED,
    DT,
    STALL_SPEED,
    START_POSITION,
    WAYPOINT_TOLERANCE,
    extract_measurement,
    start_at,
)
from EKF import MODEL_CONFIG, dynamics, wrap_angle  # noqa: E402
from Estimator import (  # noqa: E402
    REAL_MEASUREMENT_NOISE,
    SIM_MEASUREMENT_NOISE,
    Estimator,
)
from lab2 import (  # noqa: E402
    REAL_PROCESS_NOISE,
    ImuHeadingSensor,
    _poll_for_abort,
    robot_speed_measurement,
)
from Planner import Planner  # noqa: E402
from sphero_env.envs.custom_maze_full import (  # noqa: E402
    BLOCKED_EDGES,
    GOAL_CELL,
    GRID_RESOLUTION,
    cell_to_world,
)

GOAL = np.asarray(cell_to_world(GOAL_CELL), dtype=np.float64)
MAX_TURN_RATE = MODEL_CONFIG["max_turn_rate_rad_s"]
# Sphero BOLT, 73 mm across.
BALL_RADIUS = 0.0365
# The controller Lab 3 proved on the robot turns on the spot past this error.
TURN_GATE = np.deg2rad(15.0)
# Lab 4 sets no step count; the skeleton's cap.  The slowest run so far, the
# 2026-09-18 12:54 one, took 388.
MAX_STEPS = 500
# Measured speeds reach ~0.19 m/s at the 0.15 cruise command, so the
# simulator's observation must not clip them to the command limit.
SPEED_OBSERVATION_LIMIT = 0.5
# Steps without a usable IMU yaw before the run stops, about 1 s -- the same
# patience Lab 2 gives an IMU that disagrees with the command.  Past it the
# filter's heading is only the commanded heading echoed back.
YAW_PATIENCE = 10

PLANNER = Planner(map=lab3.map, dt=DT, grid_resolution=GRID_RESOLUTION)


# ---------------- maze geometry ----------------

def _wall_segments():
    """The maze's walls as the thin segments they are on the floor.

    The occupancy grid draws each wall as a band one grid cell wide, which
    leaves 6.25 cm either side of a plate's centre line -- less than the
    locator's everyday error.  The safety checks use the real walls instead.
    """
    half = GRID_RESOLUTION
    segments = []
    for a, b in BLOCKED_EDGES:
        mid = (np.asarray(cell_to_world(a)) + np.asarray(cell_to_world(b))) / 2.0
        if a[1] == b[1]:  # neighbours in one row: a wall running north-south
            segments.append((mid - (0.0, half), mid + (0.0, half)))
        else:
            segments.append((mid - (half, 0.0), mid + (half, 0.0)))
    edge = 0.625
    corners = [(-edge, -edge), (edge, -edge), (edge, edge), (-edge, edge)]
    for p, q in zip(corners, corners[1:] + corners[:1]):
        segments.append((np.asarray(p), np.asarray(q)))
    starts = np.array([s for s, _ in segments], dtype=np.float64)
    ends = np.array([e for _, e in segments], dtype=np.float64)
    return starts, ends


WALL_A, WALL_B = _wall_segments()


def _point_to_walls(p):
    """Distance from each wall segment to point(s) p, shape (..., walls)."""
    p = np.asarray(p, dtype=np.float64)[..., None, :]
    ab = WALL_B - WALL_A
    t = np.clip(((p - WALL_A) * ab).sum(-1) / (ab * ab).sum(-1), 0.0, 1.0)
    return np.linalg.norm(p - (WALL_A + t[..., None] * ab), axis=-1)


def wall_clearance(p):
    """Distance from a point to the nearest wall."""
    return float(_point_to_walls(p).min())


def run_clearance(p, q):
    """Closest a straight run from p to q comes to any wall; 0 if it crosses one."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    d, e = q - p, WALL_B - WALL_A
    denom = d[0] * e[:, 1] - d[1] * e[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((WALL_A - p)[:, 0] * e[:, 1] - (WALL_A - p)[:, 1] * e[:, 0]) / denom
        u = ((WALL_A - p)[:, 0] * d[1] - (WALL_A - p)[:, 1] * d[0]) / denom
    if np.any((np.abs(denom) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)):
        return 0.0
    # Two segments that do not cross are closest at an endpoint of one of them.
    run_ends = float(_point_to_walls(np.vstack([p, q])).min())
    return min(run_ends, _segment_ends_to_run(p, q))


def _segment_ends_to_run(p, q):
    """Distance from every wall endpoint to the run p-q."""
    points = np.vstack([WALL_A, WALL_B])
    d = q - p
    length2 = float(d @ d)
    if length2 == 0.0:
        return float(np.linalg.norm(points - p, axis=1).min())
    t = np.clip((points - p) @ d / length2, 0.0, 1.0)
    return float(np.linalg.norm(points - (p + t[:, None] * d), axis=1).min())


def keeps_clear(p, q):
    """True if a run from p to q stays off the walls.

    It may not come within the ball's radius of a wall -- except that a robot
    already that close may still move, as long as the run takes it no closer.
    """
    return run_clearance(p, q) >= min(BALL_RADIUS, wall_clearance(p)) - 1e-9


# ---------------- actions ----------------

def to_env_action(state, action):
    """The policy's [speed, turn rate] as the environment's [speed, heading].

    SpheroEnv and the robot take an absolute heading.  The turn is taken from
    the current estimate every step, never accumulated from the last command,
    so a heading the robot did not reach is not compounded.
    """
    speed, rate = float(action[0]), float(action[1])
    return np.array([speed, wrap_angle(float(state[2]) + rate * DT)], dtype=np.float64)


class SafetyLayer:
    """Checks every action the policy proposes before it reaches the motors.

    It only ever slows or stops the ball; it never steers it anywhere.  The
    navigation is the learned policy's, and the count of steps this layer
    changed is reported so that stays true.
    """

    # About 0.3 s still after a collision before the policy drives again.
    PAUSE_STEPS = 3
    # The model keeps rolling after a zero command; 0.6 s covers it.
    BRAKE_STEPS = 6
    # Five collisions inside two seconds is a ball pinned against a wall, not
    # a bump on the way past.  Starting values, not yet checked on the robot.
    COLLISION_LIMIT = 5
    COLLISION_WINDOW = 20

    def __init__(self):
        self.pause = 0
        self.decisions = 0
        self.interventions = 0
        self.reasons = Counter()
        self.recent = deque(maxlen=self.COLLISION_WINDOW)

    @property
    def pinned(self):
        return sum(self.recent) >= self.COLLISION_LIMIT

    def filter(self, state, action, collided=False):
        """Return (action to apply, reason it was changed or None)."""
        self.decisions += 1
        speed, rate = float(action[0]), float(action[1])
        reason = None
        if not (np.isfinite(speed) and np.isfinite(rate)):
            speed, rate, reason = 0.0, 0.0, "non-finite"
        speed = float(np.clip(speed, 0.0, CRUISE_SPEED))
        rate = float(np.clip(rate, -MAX_TURN_RATE, MAX_TURN_RATE))
        self.recent.append(bool(collided))
        if collided:
            self.pause = self.PAUSE_STEPS
        if self.pause > 0:
            self.pause -= 1
            speed, rate, reason = 0.0, 0.0, reason or "collision-pause"
        else:
            if speed > 0.0 and abs(rate) * DT > TURN_GATE:
                # Driving through a large turn cuts across the inside wall.
                speed, reason = 0.0, "turn-before-drive"
            if not self._stops_clear(state, speed, rate):
                # A zero command does not stop a rolling ball, and turning
                # while it rolls on swings it toward whatever is beside it.
                # Stop driving first; if the turn is still what makes it
                # unsafe, hold the heading and brake straight.
                if speed > 0.0 and self._stops_clear(state, 0.0, rate):
                    speed = 0.0
                else:
                    speed, rate = 0.0, 0.0
                reason = "brake"
        if reason is not None:
            self.interventions += 1
            self.reasons[reason] += 1
        return np.array([speed, rate], dtype=np.float64), reason

    def _stops_clear(self, state, speed, rate):
        """Would one step of this action, then a stop, keep the ball off the walls?"""
        s = np.asarray(state, dtype=np.float64)
        heading = wrap_angle(float(s[2]) + rate * DT)
        path = [s[:2].copy()]
        s = np.asarray(dynamics(s, np.array([speed, heading])), dtype=np.float64)
        path.append(s[:2].copy())
        for _ in range(self.BRAKE_STEPS):
            s = np.asarray(dynamics(s, np.array([0.0, heading])), dtype=np.float64)
            path.append(s[:2].copy())
        start = path[0]
        allowed = min(BALL_RADIUS, wall_clearance(start)) - 1e-9
        return all(run_clearance(a, b) >= allowed for a, b in zip(path, path[1:]))


# ---------------- environments ----------------

def scaled_dynamics(scale):
    """The calibrated model with the response above the deadband scaled.

    SpheroEnv calls a custom dynamics function directly and never applies its
    own speed_scale, so a faster or slower robot has to be modelled here.
    """
    deadband = MODEL_CONFIG["command_deadband_m_s"]

    def model(state, action):
        speed = float(action[0])
        if speed > deadband:
            speed = deadband + (speed - deadband) * scale
        return dynamics(state, np.array([speed, float(action[1])]))

    return model


def make_sim_env(render=False, speed_scale=1.0):
    env = lab3.make_sim_env(render=render)
    low, high = env.observation_space.low.copy(), env.observation_space.high.copy()
    low[3], high[3] = -SPEED_OBSERVATION_LIMIT, SPEED_OBSERVATION_LIMIT
    env.observation_space = type(env.observation_space)(low=low, high=high, dtype=np.float32)
    if speed_scale != 1.0:
        env.dynamics = scaled_dynamics(speed_scale)
    return env


make_real_env = lab3.make_real_env


# ---------------- the run ----------------

def _state_row(prefix, state):
    return {f"{prefix}_x": float(state[0]), f"{prefix}_y": float(state[1]),
            f"{prefix}_heading": float(state[2]), f"{prefix}_speed": float(state[3])}


def run_episode(policy_fn, sim_env, robot_env=None, *, max_steps=MAX_STEPS,
                render=False, pos_offset=(0.0, 0.0), heading_offset=0.0,
                records=None, step_log=None, verbose=False):
    """Drive the maze with ``policy_fn(state) -> [speed, turn_rate]``.

    In simulation there is one ball: the filter tracks it from the
    simulator's noisy observation, and the policy steers on the filter.  On
    hardware the robot is steered on its filter and the simulator is a second
    ball steered on its own true state; both use the same policy, each with
    its own safety layer, and the run ends when both have arrived.

    ``pos_offset``/``heading_offset`` bias the simulated observation, for
    training on a locator that is out.  ``records`` and ``step_log`` may be
    passed in so an aborted run still keeps what it did.
    """
    hardware = robot_env is not None
    records = [] if records is None else records

    sim_env.reset(seed=lab3.LAB1_SEED)
    if hardware:
        _, robot_info = robot_env.reset(seed=lab3.LAB1_SEED)
        initial_heading = float(robot_info["state_odom"][2])
    else:
        initial_heading = 0.0
    start_at(sim_env, START_POSITION, initial_heading)

    ekf = Estimator(
        dt=DT,
        initial_state=np.array([*START_POSITION, initial_heading, 0.0], dtype=np.float64),
        process_noise=REAL_PROCESS_NOISE if hardware else None,
        measurement_noise=REAL_MEASUREMENT_NOISE if hardware else SIM_MEASUREMENT_NOISE,
    )
    heading_sensor = ImuHeadingSensor(initial_heading) if hardware else None
    robot_guard, sim_guard = SafetyLayer(), SafetyLayer()
    robot_done = sim_done = False
    robot_collided = sim_collided = False
    yaw_missing = 0
    stop_reason = "max_steps"
    if render:
        sim_env.render()
    next_tick = time.perf_counter()
    start = next_tick

    for step in range(max_steps):
        if render:
            _poll_for_abort()

        state = ekf.state_est.copy()
        proposed = np.zeros(2) if robot_done else np.asarray(policy_fn(state), dtype=np.float64)
        applied, reason = robot_guard.filter(state, proposed, robot_collided)
        env_action = to_env_action(state, applied)

        if hardware:
            sensor_obs, _, _, _, sensor_info = robot_env.step(env_action)
            sim_state = np.asarray(sim_env.state_true[:4], dtype=np.float64)
            sim_proposed = np.zeros(2) if sim_done else np.asarray(policy_fn(sim_state), dtype=np.float64)
            sim_applied, _ = sim_guard.filter(sim_state, sim_proposed, sim_collided)
            sim_obs, _, _, _, sim_info = sim_env.step(to_env_action(sim_state, sim_applied))
            sim_collided = bool(sim_obs[4] > 0.5)
        else:
            sim_obs, _, _, _, sim_info = sim_env.step(env_action)
            sensor_obs = np.array(sim_obs, dtype=np.float64)
            sensor_obs[0:2] += pos_offset
            sensor_obs[2] = wrap_angle(sensor_obs[2] + heading_offset)
            sensor_info = None
        robot_collided = bool(sensor_obs[4] > 0.5)

        # A ball held by a wall or a seam did not move, whatever it was told.
        # Predicting the commanded motion anyway walks the estimate on past
        # the obstacle -- on to a corner the ball never reached -- and the
        # policy steers from the estimate straight back into the wall.
        # Hardware sees the hold on the encoder, the simulator reports it.
        measured_speed = robot_speed_measurement(sensor_info) if hardware else None
        stalled = (
            applied[0] > MODEL_CONFIG["command_deadband_m_s"]
            and measured_speed is not None
            and abs(measured_speed) < STALL_SPEED
        )
        held = robot_collided or stalled
        ekf.predict(np.array([0.0, env_action[1]]) if held else env_action)
        ekf.update(extract_measurement(
            sensor_obs, sensor_info, heading_sensor=heading_sensor,
            commanded_heading=float(env_action[1]), position_offset=START_POSITION,
        ))
        sim_env.update_estimate(ekf.state_est, ekf.P)
        if hardware:
            robot_env.update_estimate(ekf.state_est, ekf.P)

        truth = sim_info["state_true"]
        records.append((float(truth[0]), float(truth[1]),
                        float(ekf.state_est[0]), float(ekf.state_est[1])))
        if step_log is not None:
            step_log.append({
                "step": step, "t_s": time.perf_counter() - start,
                **_state_row("obs", state),
                "policy_speed": float(proposed[0]), "policy_rate": float(proposed[1]),
                "applied_speed": float(applied[0]), "applied_rate": float(applied[1]),
                "applied_heading": float(env_action[1]),
                "collision": float(robot_collided), "safety": reason or "",
                **_state_row("next", ekf.state_est),
            })

        robot_done = robot_done or float(np.linalg.norm(ekf.state_est[:2] - GOAL)) <= ARRIVAL_TOLERANCE
        if hardware:
            sim_done = sim_done or float(np.linalg.norm(truth[:2] - GOAL)) <= ARRIVAL_TOLERANCE
        if robot_done and (sim_done or not hardware):
            stop_reason = "arrived"
            break
        if robot_guard.pinned or sim_guard.pinned:
            stop_reason = "repeated_collision"
            break
        if hardware:
            valid_yaw = ImuHeadingSensor._yaw_degrees(sensor_info) is not None
            yaw_missing = 0 if valid_yaw else yaw_missing + 1
            if heading_sensor.disabled or yaw_missing >= YAW_PATIENCE:
                stop_reason = "imu_lost"
                break
        if render:
            sim_env.render()
        if hardware:
            next_tick += DT
            time.sleep(max(0.0, next_tick - time.perf_counter()))

    sim_env.emergency_stop()
    if hardware:
        robot_env.emergency_stop()
    if verbose:
        print(f"{stop_reason} after {len(records)} steps; safety changed "
              f"{robot_guard.interventions}/{robot_guard.decisions} robot actions "
              f"{dict(robot_guard.reasons)}")
    return {
        "records": records, "steps": len(records), "stop_reason": stop_reason,
        "robot_done": robot_done, "sim_done": sim_done or not hardware,
        "interventions": robot_guard.interventions, "decisions": robot_guard.decisions,
        "reasons": dict(robot_guard.reasons),
        "sim_interventions": sim_guard.interventions,
    }


# ---------------- scoring ----------------

OPTIMAL_PATH = np.asarray(
    PLANNER.plan(np.array([*START_POSITION, 0.0, 0.0]), GOAL), dtype=np.float64
)


def path_deviation(track):
    """Largest distance from any point of a track to the optimal A* path."""
    worst = 0.0
    for point in np.asarray(track, dtype=np.float64):
        best = np.inf
        for a, b in zip(OPTIMAL_PATH, OPTIMAL_PATH[1:]):
            segment = b - a
            t = float(np.clip((point - a) @ segment / (segment @ segment), 0.0, 1.0))
            best = min(best, float(np.linalg.norm(point - (a + t * segment))))
        worst = max(worst, best)
    return worst


def metrics(records):
    """The four numbers the automarker checks."""
    data = np.asarray(records, dtype=np.float64)
    return {
        "final_sim": float(np.linalg.norm(data[-1, 0:2] - GOAL)),
        "final_real": float(np.linalg.norm(data[-1, 2:4] - GOAL)),
        "path_sim": path_deviation(data[:, 0:2]),
        "path_real": path_deviation(data[:, 2:4]),
    }


LIMITS = {"final_sim": 0.10, "final_real": 0.10, "path_sim": 0.20, "path_real": 0.20}


def passes(record_metrics):
    return all(record_metrics[k] <= LIMITS[k] for k in LIMITS)
