"""Fast race: straight legs at a high raw speed, corners as single heading steps.

    .venv/Scripts/python.exe project/fast_race.py --start "0,4" --heading 0 --goal "4,0"
    .venv/Scripts/python.exe project/fast_race.py --start "0,4" --heading 0 --goal "4,0" --raw 40 --turn-lead 10

What the robot showed on 2026-10-09 (logs/calibration/1009_11*), at raw 40:
- cruising 0.30-0.35 m/s; ~0.55 s from the first command to moving;
- told to stop at ~0.3 m/s it slides 7-13 cm;
- told to turn 90 degrees in one step while rolling, the firmware brakes,
  pivots and drives on: the course is round in ~0.45 s, the ball having gone
  9-11 cm on and ~7 cm into the new direction, and is back to ~0.4 m/s.
  Swinging the heading gradually (180 or 360 deg/s) was slower and went
  wider (15-20 cm on).
- with no drive command for ~2 s the firmware brakes, so one goes out every
  CONTROL_PERIOD whatever happens.

So a route is a list of straight legs between the plan's turning points.
Each leg is driven at --raw on the leg's heading, nudged back toward its
centre line; --turn-lead cm before a turning point the heading steps to the
next leg's; --stop-lead cm before the goal the robot is told to stop.  Only
the locator and velocity stream are used (no filter): at 50 ms frames they
are fresh enough, and the position is projected ahead by the command delay.

Frames as in race.py: the aim and locator are reset where the robot stands,
facing the start heading, and every reading is turned by that heading.
"""

import argparse
import csv
import sys
import threading
import time
from collections import deque
from contextlib import ExitStack
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
from mission import make_mission, plan_cells, plate_name, count_turns  # noqa: E402
from race import FrameCounter, wait_for_frames, sdk_to_maze_rotation, whole_ball_on_goal  # noqa: E402
from sphero_env.envs.custom_maze_full import cell_to_world  # noqa: E402

CONTROL_PERIOD_S = 0.1
SENSOR_INTERVAL_MS = 50
TIME_LIMIT_S = 45.0
LATENCY_S = 0.05          # the age of a 50 ms frame: the turn data already include the command delay
# 1.0 deg/cm and an 8 deg cap could not pull the ball off the north wall of
# the top row in 1.5 s (121047): it stayed 10 cm north of the line.
STEER_GAIN_DEG_PER_CM = 1.5
STEER_LIMIT_DEG = 15.0
# After a corner the heading is held at exactly the new leg's until the ball
# really moves along it.  On 2026-10-09 the line correction swung the
# command 78 -> 102 degrees through each turn; a heading that keeps changing
# is the slow, wide kind of turn (15-20 cm on, as the gradual turns of the
# calibration), and twice at the third corner the ball ran on north into
# the boundary wall.
ALIGNED_DEG = 15.0
ALIGN_TIMEOUT_S = 1.2
# The full --turn-lead suits 0.25-0.40 m/s: with it the last eight corners
# (124624-125001, boost held until facing) ended 3.4 cm short to 6.3 cm past
# the corner line.  A slow ball runs on less: at 0.15 m/s the same lead
# turned it 6.9 cm early into a wall end (124624).  So the lead shrinks below
# LEAD_FULL_FROM and grows above LEAD_SPEED.  The readings swing +-0.1 m/s
# frame to frame (one high reading made 124852 turn 4 cm early onto the
# slope's bad side), so the speed is a mean over SPEED_MEAN_STEPS steps.
LEAD_FULL_FROM = 0.25
LEAD_SPEED = 0.40
SPEED_MEAN_STEPS = 3
# The plate north of the second corner rises (lab4 found it on 10-02).  At
# raw 40 the ball came out of the corner step slow, stalled on it and twice
# rolled back down at 0.5 m/s, once sat there 4 s (122022, 122137).  So
# while the ball moves along its leg slower than BOOST_BELOW the raw speed is
# raised, up to --boost more at a standstill or rolling back.
BOOST_BELOW = 0.25
# ...but only once the robot faces the commanded heading.  At a corner the
# ball is still rolling the old way, so it is "slow along its leg" from the
# moment the heading steps; boosted to raw 114-120 there the robot never
# pivoted (yaw stayed 90-130 deg off the command) and drove on south-west
# off the map (123959, 124110).  The yaw runs opposite to the SDK heading:
# told 90 the robot reads about -90.
FACING_DEG = 30.0
STOPPED_CM_S = 2.0
STOPPED_FOR_S = 0.3
STOP_TIMEOUT_S = 2.5


def legs_of(cells):
    """Turning points of a plate route, as world points: start, each corner, goal."""
    pts = [np.asarray(cell_to_world(c), dtype=float) for c in cells]
    kept = [pts[0]]
    for a, b, c in zip(pts, pts[1:], pts[2:]):
        if not np.allclose((b - a) / np.linalg.norm(b - a), (c - b) / np.linalg.norm(c - b)):
            kept.append(b)
    kept.append(pts[-1])
    return kept


def heading_of(vector):
    """Maze heading (rad, clockwise from north) of a direction."""
    return float(np.arctan2(vector[0], vector[1]))


def boosted(raw, forward, facing_error_deg, args):
    """Raw speed with the slope boost for a ball moving ``forward`` m/s along its leg.

    No boost unless the robot faces within FACING_DEG of the command (an
    unknown yaw, NaN, counts as not facing it).
    """
    if raw <= 0 or not abs(facing_error_deg) <= FACING_DEG:
        return raw
    return raw + int(round(args.boost * np.clip((BOOST_BELOW - forward) / BOOST_BELOW, 0.0, 1.0)))


def command_for(p, v, legs, leg, turning, args):
    """(maze heading deg, raw speed, leg index, stopping?, turning?) for the projected position.

    ``turning`` is True from a leg change until the ball moves along the new
    leg; meanwhile the heading is the leg's exactly, with no line correction.
    """
    ahead = p + v * LATENCY_S
    speed = float(np.hypot(*v))
    while True:
        a, b = legs[leg], legs[leg + 1]
        d = (b - a) / np.linalg.norm(b - a)
        to_go = float((b - ahead) @ d)                  # m left to this leg's end
        last = leg + 2 >= len(legs)
        if last:
            stopping = to_go <= args.stop_lead / 100.0 * np.clip(speed / LEAD_SPEED, 0.5, 1.5)
            break
        scale = np.clip(speed / LEAD_FULL_FROM, 0.4, 1.0) if speed < LEAD_SPEED else min(speed / LEAD_SPEED, 1.5)
        if to_go <= args.turn_lead / 100.0 * scale:
            leg += 1                                     # the heading steps to the next leg
            turning = True
            continue
        stopping = False
        break
    leg_heading = np.degrees(heading_of(d))
    if turning and speed > 0.05:
        course = np.degrees(heading_of(v))
        turning = abs((course - leg_heading + 180.0) % 360.0 - 180.0) > ALIGNED_DEG
    if turning:
        return leg_heading, (0 if stopping else args.raw), leg, stopping, True
    across = float(d[0] * (ahead[1] - a[1]) - d[1] * (ahead[0] - a[0]))   # + = left of the line
    nudge = float(np.clip(STEER_GAIN_DEG_PER_CM * across * 100.0, -STEER_LIMIT_DEG, STEER_LIMIT_DEG))
    return leg_heading + nudge, (0 if stopping else args.raw), leg, stopping, False


class Sensors:
    """Latest locator (cm) and velocity (cm/s) from the robot's stream, with a frame count."""

    def __init__(self):
        self.lock = threading.Lock()
        self.count = 0
        self.xy = None
        self.vxy = None
        self.yaw = np.nan

    def __call__(self, data):
        loc, vel, att = data.get("locator"), data.get("velocity"), data.get("attitude")
        with self.lock:
            self.count += 1
            if isinstance(loc, dict):
                self.xy = np.array([loc["x"], loc["y"]], dtype=float)
            if isinstance(vel, dict):
                self.vxy = np.array([vel["x"], vel["y"]], dtype=float)
            if isinstance(att, dict) and "yaw" in att:
                self.yaw = float(att["yaw"])

    def read(self):
        with self.lock:
            return (self.count, (None if self.xy is None else self.xy.copy()),
                    (None if self.vxy is None else self.vxy.copy()), self.yaw)


def run_fast(mission, api, toy, args, log):
    from sphero_unsw.utils import ToyUtil

    theta0 = mission.start_heading
    rot = sdk_to_maze_rotation(theta0)
    cells = plan_cells(mission.start_cell, mission.start_heading_deg, mission.goal_cell)
    legs = legs_of(cells)
    legs[0] = np.asarray(mission.start_xy, dtype=float)

    toy.sensor_control.set_interval(SENSOR_INTERVAL_MS)
    sensors = Sensors()
    counter = FrameCounter()
    toy.sensor_control.add_sensor_data_listener(sensors)
    toy.sensor_control.add_sensor_data_listener(counter)

    def send(maze_heading_deg, raw):
        sdk = int(round(maze_heading_deg - np.degrees(theta0))) % 360
        ToyUtil.roll_start(toy, sdk, int(raw))
        return sdk

    try:
        # Ready: aim and locator reset where the robot stands, on fresh frames.
        ToyUtil.set_locator_flags(toy, False)
        api.reset_aim()
        api.stop_roll(heading=0)
        ToyUtil.reset_locator(toy)
        if not wait_for_frames(counter, 4, time.perf_counter() + 4.0):
            raise RuntimeError("no sensor frames after the reset; reconnect and try again")
        _, origin, _, _ = sensors.read()
        if origin is None or np.abs(origin).max() > 1.0:
            raise RuntimeError(f"locator did not reset ({origin} cm); keep the robot still and try again")

        leg, stop_since, stopped_since, outcome = 0, None, None, "time-limit"
        turning, turn_since = True, 0.0             # setting off is a turn too
        recent = deque(maxlen=SPEED_MEAN_STEPS)
        start = time.perf_counter()
        next_tick = start
        while True:
            now = time.perf_counter()
            elapsed = now - start
            if elapsed > TIME_LIMIT_S:
                break
            count, xy, vxy, yaw = sensors.read()
            p = np.asarray(mission.start_xy) + rot @ ((xy - origin) / 100.0)
            v = rot @ (vxy / 100.0) if vxy is not None else np.zeros(2)
            speed = float(np.hypot(*vxy)) if vxy is not None else np.nan
            if stop_since is None:
                before = leg
                recent.append(v)
                v_mean = np.mean(recent, axis=0)
                heading, raw, leg, stopping, turning = command_for(p, v_mean, legs, leg, turning, args)
                if leg != before:
                    turn_since = elapsed
                if turning and elapsed - turn_since > ALIGN_TIMEOUT_S:
                    turning = False                      # do not hold a stale heading for ever
                if stopping:
                    stop_since = elapsed
                d = legs[leg + 1] - legs[leg]
                facing_error = (-yaw - (heading - np.degrees(theta0)) + 180.0) % 360.0 - 180.0
                raw = boosted(raw, float(v_mean @ d) / float(np.linalg.norm(d)), facing_error, args)
            else:
                raw = 0
            sdk = send(heading, raw)
            log.append({"t": round(elapsed, 4), "frames": count, "x": p[0], "y": p[1], "vx": v[0], "vy": v[1],
                        "speed_cm_s": speed, "leg": leg, "turning": int(turning), "cmd_heading_maze": heading,
                        "cmd_heading_sdk": sdk, "raw": raw, "yaw_sdk": yaw})
            if stop_since is not None:
                if speed <= STOPPED_CM_S:
                    stopped_since = elapsed if stopped_since is None else stopped_since
                    if elapsed - stopped_since >= STOPPED_FOR_S:
                        outcome = "arrived" if whole_ball_on_goal(p, mission.goal_xy) else "stopped-off-goal"
                        break
                else:
                    stopped_since = None
                if elapsed - stop_since > STOP_TIMEOUT_S:
                    outcome = "unsettled"
                    break
            next_tick += CONTROL_PERIOD_S
            time.sleep(max(0.0, next_tick - time.perf_counter()))
    finally:
        ToyUtil.roll_start(toy, int(round(-np.degrees(theta0))) % 360, 0)
        toy.sensor_control.remove_sensor_data_listener(sensors)
        toy.sensor_control.remove_sensor_data_listener(counter)
    end = log[-1] if log else None
    if end:
        miss = np.hypot(end["x"] - mission.goal_xy[0], end["y"] - mission.goal_xy[1]) * 100
        moving = next((r["t"] for r in log if r["speed_cm_s"] > 2.0), None)
        print(f"{outcome}: {end['t']:.2f} s from the first command"
              + (f" ({end['t'] - moving:.2f} s from first motion)" if moving is not None else "")
              + f", ended {miss:.1f} cm from the goal centre")
    return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description="Race a route fast: straight legs, stepped corners.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--heading", type=int, required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--raw", type=int, default=40, help="raw speed on the legs")
    parser.add_argument("--turn-lead", type=float, default=12.0, help="cm before a turning point to step the heading")
    parser.add_argument("--boost", type=int, default=40,
                        help="raw speed added when the ball is stalled or rolling back (0 = off)")
    parser.add_argument("--stop-lead", type=float, default=10.0, help="cm before the goal to tell the robot to stop")
    args = parser.parse_args(argv)

    mission = make_mission(args.start, args.heading, args.goal)
    cells = plan_cells(mission.start_cell, mission.start_heading_deg, mission.goal_cell)
    print(mission.describe())
    print(f"route {' '.join(plate_name(c) for c in cells)}: {len(cells) - 1} plates, "
          f"{count_turns(cells, mission.start_heading_deg)} turns; raw {args.raw} (+{args.boost} when stalled), "
          f"turn {args.turn_lead:.0f} cm early, stop {args.stop_lead:.0f} cm early")

    from lab2 import connect_with_retry
    from sphero_env.robot.connect import scan_and_connect

    # One run per launch: the robot goes as soon as it is connected, so it
    # must already stand on the start plate, still, facing the start heading.
    print(f"The robot must already be on {plate_name(mission.start_cell)} facing {mission.start_heading_deg} deg, still.")
    log = []
    with ExitStack() as stack:
        toy, _ = scan_and_connect()
        print(f"Selected: {toy.name}")
        api = connect_with_retry(stack, toy)
        try:
            run_fast(mission, api, toy, args, log)
        except KeyboardInterrupt:
            print("Stopped by the operator.")
        finally:
            if log:
                (sc, sr), (gc, gr) = mission.start_cell, mission.goal_cell
                path = PROJECT.parent / "logs" / (
                    f"fast_{time.strftime('%m%d_%H%M%S')}_{sc}{sr}-h{mission.start_heading_deg}-{gc}{gr}"
                    f"_raw{args.raw}_b{args.boost}_tl{args.turn_lead:g}_sl{args.stop_lead:g}.csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(log[0]))
                    writer.writeheader()
                    writer.writerows(log)
                print(f"Wrote {len(log)} steps to {path.name}")
    print("Disconnected.")


if __name__ == "__main__":
    main()
