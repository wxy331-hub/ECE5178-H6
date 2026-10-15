"""Project race: drive any start (plate, heading) to any goal plate in the Lab 3/4 maze.

    .venv/Scripts/python.exe project/race.py --start A1 --heading 0 --goal E5 --sim
    .venv/Scripts/python.exe project/race.py --start A1 --heading 90 --goal E5 --raw-speed 20

Plates are A1..E5 (letters west to east, digits south to north; A1 is
(-0.5, -0.5)) or 'x,y' in metres.  Headings are 0/90/180/270, clockwise from
north.

The stack is Lab 4's, minus the network: the plan picks the plates (fewest,
then fewest turns), Lab 3's waypoint follower walks them, Lab 2's filter
tracks the robot, and Lab 4's safety layer and heading hold sit between the
decision and the motors.  Nothing under labs/ is changed.

On the robot every position, heading and command is in maze coordinates;
the only rotation is at the robot's boundary.  Before each run the robot's
aim is reset with it facing the start heading, so the firmware's 0 degrees is
that heading, and the locator is zeroed there.  A displacement q the locator
reports (SDK frame, robot forward = +y) is then

    p_maze = p_start + C(theta0) q,   C = [[cos t0, sin t0], [-sin t0, cos t0]]

and a maze heading h goes out as h - theta0.
"""

import argparse
import csv
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "labs" / "lab4"))
import runtime as R  # noqa: E402  (puts labs/lab3 and labs/lab2 on the path)
from mission import make_mission, plan_cells, plate_name, count_turns, world_to_cell, reachable_from  # noqa: E402
from sphero_env.envs.custom_maze_full import cell_to_world  # noqa: E402

lab3 = R.lab3
DT = R.DT
CRUISE_SPEED = R.CRUISE_SPEED
wrap_angle = R.wrap_angle

TIME_LIMIT_S = 45.0
RAW_SPEED = 20             # what cruising goes out as on the robot (2026-10-02)
# Past this heading error the robot turns on the spot whatever lies ahead:
# setting off the wrong way, or turning round, must not creep forward.
TURN_ROUND = np.deg2rad(100.0)
# Stopped: estimated speed this low for this many steps after the brake.
STOPPED_SPEED = 0.02
STOPPED_STEPS = 5
BRAKE_TIMEOUT_S = 2.0
# The whole ball on the goal plate: centre within this of the plate's centre
# on both axes (0.125 m half plate less the 0.0365 m radius).
WHOLE_BALL_IN = 0.125 - R.BALL_RADIUS


# ---------------- frames ----------------

def sdk_to_maze_rotation(theta0):
    c, s = np.cos(theta0), np.sin(theta0)
    return np.array([[c, s], [-s, c]])


def locator_to_maze(locator_xy, origin_xy, start_xy, theta0):
    """Maze position (m) from a locator reading (m, SDK frame) and the one taken at the start."""
    q = np.asarray(locator_xy, dtype=np.float64) - np.asarray(origin_xy, dtype=np.float64)
    return np.asarray(start_xy, dtype=np.float64) + sdk_to_maze_rotation(theta0) @ q


def maze_to_sdk_heading(heading, theta0):
    return wrap_angle(float(heading) - float(theta0))


# ---------------- planning and following ----------------

class RoutePlanner(R.Planner):
    """Lab 3's planner, but the route is the fewest-turn shortest one from here.

    Lab 3's WaypointFollower calls plan() at the start and whenever it
    replans; is_clear() and the grid are Lab 3's, unchanged.  The route is
    planned from the plate the estimate is on, even when that is in a wall
    band; only an estimate off the maze, or on a plate with no way to the
    goal, falls back to Lab 3's snapping plan.
    """

    def plan(self, state, goal):
        here, goal_cell = world_to_cell(state[:2]), world_to_cell(goal)
        if here is None or goal_cell is None or goal_cell not in reachable_from(here):
            return super().plan(state, goal)
        heading_deg = int(round(np.degrees(float(state[2])) / 90.0)) % 4 * 90
        cells = plan_cells(here, heading_deg, goal_cell)
        points = [np.asarray(cell_to_world(c), dtype=np.float64) for c in cells]
        points[-1] = np.asarray(goal, dtype=np.float64)
        return points


class DriveController:
    """Heads for the waypoint and always drives, as Lab 4's teacher does.

    The safety layer then decides how a turn past 15 degrees is taken: on the
    spot with a wall close ahead, slowly while driving on open floor (the
    second corner's slope, 2026-10-02).  Only a heading error past TURN_ROUND
    -- setting off the wrong way, turning round -- stops here first.
    """

    def compute_action(self, state, waypoint):
        offset = np.asarray(waypoint, dtype=np.float64)[:2] - np.asarray(state, dtype=np.float64)[:2]
        heading = float(np.arctan2(offset[0], offset[1]))
        error = abs(wrap_angle(heading - float(state[2])))
        return np.array([0.0 if error > TURN_ROUND else CRUISE_SPEED, heading], dtype=np.float64)


class RaceSafety(R.SafetyLayer):
    """Lab 4's safety layer, predicting a driving action held for two model steps.

    A step that sends a new heading lasts ~200 ms on the robot, not 105 ms,
    and whether a heading is resent depends on the last one sent, which this
    layer does not see; one step under-predicted how far a turn carries the
    ball (Codex review of the Lab 4 experiment, 2026-10-07).

    The model is the one calibrated at raw 15, fed the bare command although
    cruising goes out as raw 20.  Scaling the command by 20/15 looked
    consistent, but replaying the five raw-20 runs of 2026-10-02 the unscaled
    model tracked the measured speed and position better (speed RMSE 0.084
    against 0.129 m/s): on that floor raw 20 is not a third faster.
    """

    def _stops_clear(self, state, speed, rate):
        s = np.asarray(state, dtype=np.float64)
        heading = wrap_angle(float(s[2]) + rate * DT)
        path = [s[:2].copy()]
        for _ in range(2 if speed > 0.0 else 1):
            s = np.asarray(R.dynamics(s, np.array([speed, heading])), dtype=np.float64)
            path.append(s[:2].copy())
        for _ in range(self.BRAKE_STEPS):
            s = np.asarray(R.dynamics(s, np.array([0.0, heading])), dtype=np.float64)
            path.append(s[:2].copy())
        allowed = min(R.BALL_RADIUS, R.wall_clearance(path[0])) - 1e-9
        return all(R.run_clearance(a, b) >= allowed for a, b in zip(path, path[1:]))


def hold_heading_safely(guard, state, applied, last_sent):
    """The action to send: the held heading, unless holding it is unsafe."""
    requested = R.to_env_action(state, applied)
    held = R.hold_heading(requested, last_sent)
    if held is requested:
        return requested
    held_rate = wrap_angle(float(held[1]) - float(state[2])) / DT
    return held if guard._stops_clear(state, float(applied[0]), held_rate) else requested


def whole_ball_on_goal(position, goal_xy):
    d = np.abs(np.asarray(position, dtype=np.float64)[:2] - np.asarray(goal_xy, dtype=np.float64))
    return bool(d.max() <= WHOLE_BALL_IN)


# ---------------- the robot ----------------

class RaceRobot(lab3.Lab3Robot):
    """Lab 3's robot, sending heading and speed together in one packet.

    The SDK's set_heading() and set_speed() each send a drive command that
    carries the other value as last stored, so changing both took two
    packets with a half-old command between them: cruising into a slow turn
    went first at cruise speed on the new heading (Lab 3's order), or slowly
    on the old heading (speed first) -- moves the safety layer never checked
    (Codex reviews, 2026-10-09).  drive_with_heading() takes both at once, so
    one packet goes out per step and no intermediate command exists.

    The SDK's own copies of heading and speed are kept in step, so that its
    stop_roll()/set_speed(0) at the end do not swing the robot back to a
    stale heading.
    """

    def __init__(self, *args, toy=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._toy = toy

    def set_heading_and_speed(self, heading_deg, speed):
        from sphero_unsw.utils import ToyUtil

        commanded, raw = int(heading_deg) % 360, int(np.clip(speed, 0, 255))
        with self._lock:
            ToyUtil.roll_start(self._toy, commanded, raw)
            self.api._SpheroEduAPI__heading = commanded
            self.api._SpheroEduAPI__speed = raw
            self._last_heading_deg = commanded


def make_race_robot(api, toy):
    """lab3.make_real_env, with RaceRobot in place of Lab3Robot."""
    return RaceRobot(
        api=api, dt=DT, max_steps=lab3.MAX_STEPS, vel_limit=lab3.COMMAND_SPEED_LIMIT,
        raw_speed_limit=lab3.RAW_SPEED_LIMIT, world_width=1.25, world_height=1.25,
        goal_pos=(0.5, 0.5), goal_tolerance=lab3.GOAL_TOLERANCE, render_mode=None,
        window_size=(800, 800), toy=toy,
    )


READY_TIMEOUT_S = 4.0
READY_LOCATOR_CM = 1.0     # a reset locator reads within this of zero
READY_FRAMES = 3           # sensor frames to wait for after each reset step


class FrameCounter:
    """Counts the frames the robot streams; a getter alone only shows the SDK's cache."""

    def __init__(self):
        self.count = 0
        self._lock = threading.Lock()

    def __call__(self, _data):
        with self._lock:
            self.count += 1


def wait_for_frames(counter, frames, deadline):
    """True once ``frames`` new frames have arrived, False at the deadline."""
    target = counter.count + frames
    while counter.count < target:
        if time.perf_counter() > deadline:
            return False
        time.sleep(0.02)
    return True


def ready_robot(robot_env, toy, mission):
    """Make the robot's frame the start's: aim and locator reset where it stands.

    Called with the robot still, placed on the start facing the start heading.
    reset_aim makes that direction the firmware's 0 degrees; the SDK keeps its
    own heading variable, which stop_roll(heading=0) brings into line; the
    locator is zeroed last.

    The SDK's getters return its cache, which can hold readings from before
    the reset, and reading it again only reads the same cache.  So readings
    are taken only after READY_FRAMES frames have streamed in since the reset
    -- counted by a listener on the robot's sensor stream -- and the reference
    yaw and locator origin are taken last, after robot_env.reset() has let the
    robot settle.  Returns the locator origin (m) and that yaw.
    """
    from sphero_unsw.utils import ToyUtil

    api = robot_env.api
    counter = FrameCounter()
    toy.sensor_control.add_sensor_data_listener(counter)
    try:
        deadline = time.perf_counter() + READY_TIMEOUT_S
        ToyUtil.set_locator_flags(toy, False)
        api.reset_aim()
        api.stop_roll(heading=0)
        ToyUtil.reset_locator(toy)
        if not wait_for_frames(counter, READY_FRAMES, deadline):
            raise RuntimeError("the robot stopped streaming sensor data; reconnect and try again")
        robot_env.reset()
        if not wait_for_frames(counter, READY_FRAMES, time.perf_counter() + READY_TIMEOUT_S):
            raise RuntimeError("the robot stopped streaming sensor data; reconnect and try again")
        location = api.get_location()
        yaw = R.ImuHeadingSensor._yaw_degrees({"orientation": api.get_orientation()})
    finally:
        toy.sensor_control.remove_sensor_data_listener(counter)
    if not isinstance(location, dict) or yaw is None:
        raise RuntimeError("no locator or IMU reading after the reset; try again")
    origin = np.array([float(location.get("x", np.inf)), float(location.get("y", np.inf))]) / 100.0
    if not np.all(np.abs(origin) < READY_LOCATOR_CM / 100.0):
        raise RuntimeError(f"the locator did not reset ({origin * 100} cm); keep the robot still and try again")
    return origin, yaw


def robot_measurement(info, robot_env, origin, mission, heading_sensor, commanded_heading, fallback):
    """[x, y, heading, speed] in maze coordinates from one hardware step."""
    z = np.asarray(fallback, dtype=np.float64).copy()
    locator = info.get("state_odom") if isinstance(info, dict) else None
    if locator is None:
        locator = robot_env.state_odom
    z[0:2] = locator_to_maze(np.asarray(locator, dtype=np.float64)[:2], origin,
                             mission.start_xy, mission.start_heading)
    speed = R.robot_speed_measurement(info)
    if speed is not None:
        z[3] = speed
    heading = heading_sensor.measure(info, commanded_heading)
    if heading is not None:
        z[2] = heading
    return z


# ---------------- one run ----------------

def run_race(mission, env=None, robot_env=None, *, toy=None, raw_speed=RAW_SPEED, render=False,
             step_log=None, verbose=True, time_limit=TIME_LIMIT_S, max_steps=2000):
    """Drive one mission, in simulation (env) or on the robot (robot_env).

    Returns a dict: outcome ('arrived', 'stopped-off-goal', 'unsettled',
    'time-limit', 'repeated-collision', 'imu-lost', 'max-steps'), elapsed seconds, steps,
    the final estimate, and in simulation the true final position.
    """
    hardware = robot_env is not None
    theta0 = mission.start_heading
    start_state = np.array([*mission.start_xy, theta0, 0.0], dtype=np.float64)

    if hardware:
        robot_env.raw_speed_limit = raw_speed
        origin, reference_yaw = ready_robot(robot_env, toy, mission)
        heading_sensor = R.ImuHeadingSensor(theta0)
        # The IMU's reference is the yaw read while the robot stood on the
        # start heading.  Left to the first step, it would be read after the
        # first command -- often a turn on the spot here -- and every heading
        # after would carry whatever part of that turn was already done.
        heading_sensor.measure({"orientation": {"yaw": reference_yaw}}, theta0)
    else:
        env.reset(seed=lab3.LAB1_SEED)
        lab3.start_at(env, mission.start_xy, theta0)
        if render:
            env.render()

    ekf = R.Estimator(
        dt=DT, initial_state=start_state,
        process_noise=R.REAL_PROCESS_NOISE if hardware else None,
        measurement_noise=R.REAL_MEASUREMENT_NOISE if hardware else R.SIM_MEASUREMENT_NOISE,
    )
    planner = RoutePlanner(map=lab3.map, dt=DT, grid_resolution=R.GRID_RESOLUTION)
    follower = lab3.WaypointFollower(planner, DriveController(), mission.goal_xy, start_state)
    # The plan starts on the start plate's centre, where the robot already
    # is; steering at a point it stands on would aim it anywhere.
    follower.advance(start_state, False)
    guard = RaceSafety()
    sent_heading = None
    collided = False
    braking_since = None
    stopped_steps = 0
    yaw_missing = 0
    outcome = "max-steps"

    start = time.perf_counter()
    next_tick = start
    step = 0
    try:
        for step in range(max_steps):
            elapsed = time.perf_counter() - start if hardware else step * DT
            if elapsed > time_limit:
                outcome = "time-limit"
                break
            if render and not hardware:
                R._poll_for_abort()
            state = ekf.state_est.copy()

            if braking_since is None:
                speed, heading = follower.action(state)
                rate = float(np.clip(wrap_angle(heading - state[2]) / DT, -R.MAX_TURN_RATE, R.MAX_TURN_RATE))
                applied, reason = guard.filter(state, np.array([speed, rate]), collided)
                env_action = hold_heading_safely(guard, state, applied, sent_heading)
            else:
                # Braking on the goal: zero speed on the last heading sent, so
                # a wandering estimate does not turn the ball on the plate.
                applied, reason = np.zeros(2), None
                env_action = np.array([0.0, sent_heading if sent_heading is not None else state[2]])
            sent_heading = float(env_action[1])

            if hardware:
                sdk_action = np.array([env_action[0], maze_to_sdk_heading(env_action[1], theta0)])
                obs, _, _, _, info = robot_env.step(sdk_action)
                collided = bool(obs[4] > 0.5)
                z = robot_measurement(info, robot_env, origin, mission, heading_sensor,
                                      float(env_action[1]), ekf.state_est)
                measured_speed = R.robot_speed_measurement(info)
            else:
                obs, _, _, _, info = env.step(env_action)
                collided = bool(obs[4] > 0.5)
                z = np.asarray(obs[:4], dtype=np.float64)
                measured_speed = None
            # The clock again after the step: a BLE call can block.
            elapsed = time.perf_counter() - start if hardware else (step + 1) * DT

            stalled = (applied[0] > R.MODEL_CONFIG["command_deadband_m_s"]
                       and measured_speed is not None and abs(measured_speed) < R.STALL_SPEED)
            ekf.predict(np.array([0.0, env_action[1]]) if (collided or stalled) else env_action)
            ekf.update(z)
            estimate = ekf.state_est
            if braking_since is None:
                follower.advance(estimate, collided)

            if step_log is not None:
                row = {"step": step, "t_s": round(time.perf_counter() - start, 4) if hardware else round(elapsed, 4),
                       "x": estimate[0], "y": estimate[1], "heading": estimate[2], "speed": estimate[3],
                       "cmd_speed": applied[0], "cmd_heading_maze": env_action[1],
                       "target": follower.target, "collision": int(collided), "safety": reason or "",
                       "braking": int(braking_since is not None)}
                if hardware:
                    row["cmd_heading_sdk"] = sdk_action[1]
                else:
                    row["true_x"], row["true_y"] = float(env.state_true[0]), float(env.state_true[1])
                step_log.append(row)

            if elapsed > time_limit:
                outcome = "time-limit"
                break
            # Arrival: brake within GOAL_STOP_RADIUS, then watch the ball stop.
            # Stopped is the measured speed -- the encoders on the robot, the
            # truth in simulation -- not the filter's, which a zero command
            # pulls to zero while the ball still rolls.
            if braking_since is None and np.linalg.norm(estimate[:2] - mission.goal_xy) <= R.GOAL_STOP_RADIUS:
                braking_since = elapsed
            if braking_since is not None:
                rolling = measured_speed if hardware else float(env.state_true[3])
                still = rolling is not None and abs(rolling) <= STOPPED_SPEED
                stopped_steps = stopped_steps + 1 if still else 0
                if stopped_steps >= STOPPED_STEPS:
                    outcome = "arrived" if whole_ball_on_goal(estimate, mission.goal_xy) else "stopped-off-goal"
                    break
                if elapsed - braking_since > BRAKE_TIMEOUT_S:
                    outcome = "unsettled"   # braked, never seen still
                    break
            if guard.pinned:
                outcome = "repeated-collision"
                break
            if hardware:
                valid_yaw = R.ImuHeadingSensor._yaw_degrees(info) is not None
                yaw_missing = 0 if valid_yaw else yaw_missing + 1
                if heading_sensor.disabled or yaw_missing >= R.YAW_PATIENCE:
                    outcome = "imu-lost"
                    break
                next_tick += DT
                time.sleep(max(0.0, next_tick - time.perf_counter()))
            elif render:
                env.render()
    finally:
        if hardware:
            robot_env.api.set_speed(0)

    result = {"outcome": outcome, "steps": step + 1,
              "elapsed_s": (time.perf_counter() - start) if hardware else (step + 1) * DT,
              "estimate": ekf.state_est.copy(), "replans": follower.replans}
    if not hardware:
        result["true_xy"] = np.asarray(env.state_true[:2], dtype=np.float64).copy()
        result["true_on_goal"] = whole_ball_on_goal(result["true_xy"], mission.goal_xy)
    if verbose:
        where = result.get("true_xy", result["estimate"][:2])
        print(f"{outcome} after {result['elapsed_s']:.1f} s ({result['steps']} steps), "
              f"ended at ({where[0]:+.3f}, {where[1]:+.3f}), {result['replans']} replans")
    return result


def write_log(rows, mission):
    if not rows:
        return None
    (sc, sr), (gc, gr) = mission.start_cell, mission.goal_cell
    path = Path("logs") / (f"project_{time.strftime('%m%d_%H%M%S')}_"
                           f"{sc}{sr}-h{mission.start_heading_deg}-{gc}{gr}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for row in rows:
        fields += [k for k in row if k not in fields]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Race one route through the Lab 3/4 maze.")
    parser.add_argument("--start", required=True, help="start plate (A1..E5) or 'x,y' in metres")
    parser.add_argument("--heading", type=int, required=True, help="start heading: 0, 90, 180 or 270")
    parser.add_argument("--goal", required=True, help="goal plate (A1..E5) or 'x,y' in metres")
    parser.add_argument("--sim", action="store_true", help="simulation only")
    parser.add_argument("--raw-speed", type=int, default=20, help="raw Sphero speed for cruising")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args(argv)

    mission = make_mission(args.start, args.heading, args.goal)
    cells = plan_cells(mission.start_cell, mission.start_heading_deg, mission.goal_cell)
    print(mission.describe())
    print(f"route {' '.join(plate_name(c) for c in cells)}: {len(cells) - 1} plates, "
          f"{count_turns(cells, mission.start_heading_deg)} turns")

    rows = []
    try:
        if args.sim:
            env = R.make_sim_env(render=not args.no_render)
            try:
                run_race(mission, env, render=not args.no_render, step_log=rows)
            finally:
                env.close()
        else:
            from lab2 import connect_with_retry
            from sphero_env.robot.connect import scan_and_connect

            with ExitStack() as stack:
                toy, _ = scan_and_connect()
                print(f"Selected: {toy.name}")
                api = connect_with_retry(stack, toy)
                robot_env = make_race_robot(api, toy)
                input(f"Place the robot on {plate_name(mission.start_cell)} facing "
                      f"{mission.start_heading_deg} deg (tail light behind it), keep it still, then press Enter: ")
                try:
                    run_race(mission, robot_env=robot_env, toy=toy, raw_speed=args.raw_speed, step_log=rows)
                finally:
                    robot_env.close()
    except KeyboardInterrupt:
        print("Stopped by the operator.")
    finally:
        path = write_log(rows, mission)
        if path:
            print(f"Wrote {len(rows)} steps to {path}")


if __name__ == "__main__":
    main()
