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

What the maze added on 2026-10-09 (two routes, 15 runs, two arrivals):
- near a corner a ball faster than --corner-speed is told raw 0 first;
- a leg with a wall all along one side can be driven --wall-bias deg toward
  that wall, the locator being blind to the sideways roll the slopes give
  (off by default: on 10-16 the robot did better without it);
- with --recover, a ball that stalls is backed off to a plate centre behind
  it and set off on a new route from there, and a plate side it stalls at
  twice is avoided while there is another way (off by default: on 10-16 it
  never got the ball out, 22 recoveries in the morning, none in the demo);
  a ball stalled with the whole of it on the goal plate is stopped either way.

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
from mission import (  # noqa: E402
    count_turns, make_mission, neighbours, plan_cells, plate_name, step_plate, walled, world_to_cell,
)
from race import FrameCounter, wait_for_frames, sdk_to_maze_rotation  # noqa: E402
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
# Corners.  The run-on after the heading steps grows with speed, and the
# second corner of (0,4)->(4,0), reached at 0.37-0.61 m/s (the fastest of the
# route), is where the ball went on east into (2,2).  Within CORNER_BRAKE_M
# of where the heading steps, and through the turn itself, a ball faster than
# --corner-speed m/s is told raw 0 until it is not.
CORNER_BRAKE_M = 0.25
# Wall-side legs.  On the slope at (1,0) the ball rolled sideways, which the
# locator does not see (it counts wheel turns along the robot's heading): it
# put the ball on the top row's centre line, the line correction steered 3-5
# deg south, and the ball ended against the wall south of (3,0) (123121,
# 125825) or at the post east of (0,0), 11 cm short (125938).  A leg with a
# wall all along one side is driven --wall-bias deg toward that wall instead,
# with no locator correction: too much bias only rubs the wall.  Off by
# default: on the robot on 10-16 the runs without it went better, and leaning
# on the outer wall the locator read 4-8 cm past it (111655).
# Stalls.  On 10-09 a ball that sat still never got going again in time (12
# of 13 failed runs; two freed themselves after 10-15 s).  Moving less than
# STALL_M in STALL_S counts as stalled, except in the first QUIET_S of a run
# or of a recovery, the ball needing ~1 s to start or to turn round.  Over the
# 10-09 logs this fired in neither good run, on every stall, and twice on the
# slope where the ball moved on by itself within 0.6 s.
STALL_S = 1.0
STALL_M = 0.04
QUIET_S = 1.5
# Recovery: back to the centre of the plate the ball is on, of the one it
# came from or of the open one straight behind it, the first BACKOFF_MIN_M or
# more behind it (a run-up), then a new route from there.  The back-off leg
# is driven until BACKOFF_LEAD_M from its end, not the run-on lead: with the
# ordinary lead the ball turned back after 2-4 cm, and a back-off shorter
# than the lead was skipped (Codex review, 10-15).  Stalling twice on one
# plate in one direction closes that side to the planner while there is
# another way to the goal.
BACKOFF_MIN_M = 0.06
BACKOFF_LEAD_M = 0.02
# The plate the ball is on changes only through an open side, once the
# locator is PLATE_ENTER_M past the boundary.  But a locator a few cm off
# sideways can cut a wall's end: on 10-16 (112149) it read the ball into
# (1,3) and then west across the (0,3)|(1,3) wall, the plate stayed (1,3)
# for good, and the stall recovery planned from there into a wall.  So a
# locator that has stayed on another plate for LOST_STEPS control steps is
# believed.
PLATE_ENTER_M = 0.02
LOST_STEPS = 5
# The whole ball is on a plate when its centre is within ON_PLATE_M of the
# plate's centre on both axes (as race.whole_ball_on_goal), or up to
# WALL_SLACK_M past that towards a wall, which holds it in: leaning on a wall
# the ball sits at the limit, and the locator can read a little past it.
# Further than that it is somewhere else (stalled on (0,2), the ball is 0.59 m
# south of the goal (0,0), whose south side is a wall).
ON_PLATE_M = 0.125 - 0.0365
WALL_SLACK_M = 0.04
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


def leg_direction(a, b):
    """Nearest axis to leg a -> b (0 north, 1 east, 2 south, 3 west) and how far off it is, in degrees."""
    h = np.degrees(heading_of(np.asarray(b, dtype=float) - np.asarray(a, dtype=float)))
    k = int(round(h / 90.0)) % 4
    return k, abs((h - 90.0 * k + 180.0) % 360.0 - 180.0)


def wall_side(a, b):
    """-1 if a wall runs along the whole left of leg a -> b, +1 along the whole right, else 0."""
    k, off = leg_direction(a, b)
    first, last = world_to_cell(a), world_to_cell(b)
    if off > 10.0 or first is None or last is None:
        return 0
    plates = [first]
    while plates[-1] != last and len(plates) <= 5:
        plates.append(step_plate(plates[-1], k))
    if plates[-1] != last:
        return 0
    left = all(walled(c, (k + 3) % 4) for c in plates)
    right = all(walled(c, (k + 1) % 4) for c in plates)
    return -1 if left and not right else (1 if right and not left else 0)


def track_plate(plate, p):
    """(plate the ball is on, the open plate it came in through), given the plate it was on.

    The plate changes only through an open side, once ``p`` is PLATE_ENTER_M
    past the boundary, or two steps on through an open plate between, which
    is then the one it came through: a locator reading across a wall is not
    believed, and a back-off to the plate it came through stays inside two
    open plates.
    """
    here = world_to_cell(p)
    if here is None or here == plate:
        return plate, None
    centre = np.asarray(cell_to_world(plate), dtype=float)
    for _, other in neighbours(plate):
        if other == here:
            step = np.asarray(cell_to_world(other), dtype=float) - centre
            length = float(np.linalg.norm(step))
            past = float((np.asarray(p, dtype=float) - centre) @ step) / length - length / 2
            return (other, plate) if past >= PLATE_ENTER_M else (plate, None)
    for _, other in neighbours(plate):
        if any(beyond == here for _, beyond in neighbours(other)):
            return here, other
    return plate, None


def follow_plate(plate, came_from, lost, p):
    """(plate, came_from, lost) after the locator reads ``p``; ``lost`` counts steps it has read another plate."""
    now_on, through = track_plate(plate, p)
    if now_on != plate:
        return now_on, through, 0
    here = world_to_cell(p)
    if here is None or here == plate:
        return plate, came_from, 0
    if lost + 1 < LOST_STEPS:
        return plate, came_from, lost + 1
    # Believed now.  The plate left is where it came from only if the two are open to each other.
    return here, (plate if any(other == here for _, other in neighbours(plate)) else None), 0


def recovery(p, legs, leg, plate, came_from, goal_cell, stalls, avoid):
    """(back-off plate, new route, new legs) after a stall on ``legs[leg]``.

    Counts the stall in ``stalls`` and, at the second on the same plate and
    direction, adds that side to ``avoid`` if the leg was leaving the plate
    that way and the goal can still be reached without it.
    """
    a, b = legs[leg], legs[leg + 1]
    d = (b - a) / np.linalg.norm(b - a)
    k, _ = leg_direction(a, b)
    stalls[(plate, k)] = stalls.get((plate, k), 0) + 1
    if stalls[(plate, k)] >= 2 and world_to_cell(b) != plate:
        side = (plate, step_plate(plate, k))
        try:
            plan_cells(plate, 0, goal_cell, avoid | {side})
            avoid.add(side)
        except ValueError:
            pass                                         # the only way on: keep trying it
    p = np.asarray(p, dtype=float)
    behind = step_plate(plate, (k + 2) % 4) if not walled(plate, (k + 2) % 4) else None
    back = next((c for c in (plate, came_from, behind) if c is not None
                 and float((p - np.asarray(cell_to_world(c), dtype=float)) @ d) >= BACKOFF_MIN_M), plate)
    away = np.degrees(heading_of(-d))
    try:
        route = plan_cells(back, away, goal_cell, avoid)
    except ValueError:
        route = plan_cells(back, away, goal_cell)
    new = [p]
    for q in legs_of(route):
        if np.linalg.norm(q - new[-1]) >= 0.02:
            new.append(q)
    return back, route, new


def held_on_plate(p, cell):
    """True if the whole ball is on ``cell``, a wall side counting as holding it in."""
    off = np.asarray(p, dtype=float) - np.asarray(cell_to_world(cell), dtype=float)
    for axis, (low_side, high_side) in ((0, (3, 1)), (1, (2, 0))):     # x: west/east, y: south/north
        if abs(off[axis]) <= ON_PLATE_M:
            continue
        side = high_side if off[axis] > 0 else low_side
        if not walled(cell, side) or abs(off[axis]) > ON_PLATE_M + WALL_SLACK_M:
            return False
    return True


def boosted(raw, forward, facing_error_deg, args):
    """Raw speed with the slope boost for a ball moving ``forward`` m/s along its leg.

    No boost unless the robot faces within FACING_DEG of the command (an
    unknown yaw, NaN, counts as not facing it).
    """
    if raw <= 0 or not abs(facing_error_deg) <= FACING_DEG:
        return raw
    return raw + int(round(args.boost * np.clip((BOOST_BELOW - forward) / BOOST_BELOW, 0.0, 1.0)))


def command_for(p, v, legs, leg, turning, args, sides=None, backoff_leg=None):
    """(maze heading deg, raw speed, leg index, stopping?, turning?, braking?) for the projected position.

    ``turning`` is True from a leg change until the ball moves along the new
    leg; meanwhile the heading is the leg's exactly, with no line correction.
    ``sides[i]`` is ``wall_side`` of leg i; leg ``backoff_leg`` is a stall
    back-off, driven to BACKOFF_LEAD_M from its end.
    """
    ahead = p + v * LATENCY_S
    speed = float(np.hypot(*v))
    braking = False
    while True:
        a, b = legs[leg], legs[leg + 1]
        d = (b - a) / np.linalg.norm(b - a)
        to_go = float((b - ahead) @ d)                  # m left to this leg's end
        last = leg + 2 >= len(legs)
        if last:
            stopping = to_go <= args.stop_lead / 100.0 * np.clip(speed / LEAD_SPEED, 0.5, 1.5)
            break
        scale = np.clip(speed / LEAD_FULL_FROM, 0.4, 1.0) if speed < LEAD_SPEED else min(speed / LEAD_SPEED, 1.5)
        lead = BACKOFF_LEAD_M if leg == backoff_leg else args.turn_lead / 100.0 * scale
        if to_go <= lead:
            leg += 1                                     # the heading steps to the next leg
            turning = True
            continue
        stopping = False
        braking = to_go <= lead + CORNER_BRAKE_M
        break
    braking = 0 < args.corner_speed < speed and (braking or turning)
    raw = 0 if stopping or braking else args.raw
    leg_heading = np.degrees(heading_of(d))
    if turning and speed > 0.05:
        course = np.degrees(heading_of(v))
        turning = abs((course - leg_heading + 180.0) % 360.0 - 180.0) > ALIGNED_DEG
    if turning:
        return leg_heading, raw, leg, stopping, True, braking
    side = sides[leg] if sides is not None else 0
    if side and args.wall_bias > 0:                  # lean on the wall, not on the locator
        return leg_heading + side * args.wall_bias, raw, leg, stopping, False, braking
    across = float(d[0] * (ahead[1] - a[1]) - d[1] * (ahead[0] - a[0]))   # + = left of the line
    nudge = float(np.clip(STEER_GAIN_DEG_PER_CM * across * 100.0, -STEER_LIMIT_DEG, STEER_LIMIT_DEG))
    return leg_heading + nudge, raw, leg, stopping, False, braking


class Sensors:
    """Latest locator (cm) and velocity (cm/s) from the robot's stream, with a frame count."""

    def __init__(self):
        self.lock = threading.Lock()
        self.count = 0
        self.xy = None
        self.vxy = None
        self.attitude = {"yaw": np.nan, "pitch": np.nan, "roll": np.nan}

    def __call__(self, data):
        loc, vel, att = data.get("locator"), data.get("velocity"), data.get("attitude")
        with self.lock:
            self.count += 1
            if isinstance(loc, dict):
                self.xy = np.array([loc["x"], loc["y"]], dtype=float)
            if isinstance(vel, dict):
                self.vxy = np.array([vel["x"], vel["y"]], dtype=float)
            if isinstance(att, dict):
                for key in self.attitude:
                    if key in att:
                        self.attitude[key] = float(att[key])

    def read(self):
        """(frame count, locator cm, velocity cm/s, attitude {yaw, pitch, roll} deg)."""
        with self.lock:
            return (self.count, (None if self.xy is None else self.xy.copy()),
                    (None if self.vxy is None else self.vxy.copy()), dict(self.attitude))


def run_fast(mission, api, toy, args, log):
    from sphero_unsw.utils import ToyUtil

    theta0 = mission.start_heading
    rot = sdk_to_maze_rotation(theta0)
    cells = plan_cells(mission.start_cell, mission.start_heading_deg, mission.goal_cell)
    legs = legs_of(cells)
    legs[0] = np.asarray(mission.start_xy, dtype=float)
    sides = [wall_side(a, b) for a, b in zip(legs, legs[1:])]

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
        braking = False
        plate, came_from, lost = mission.start_cell, None, 0
        stalls, avoid, recoveries = {}, set(), 0
        anchor, anchor_t, quiet_until = None, 0.0, QUIET_S
        backoff_leg = None                          # set to 0 by a recovery: its first leg backs off
        start = time.perf_counter()
        next_tick = start
        while True:
            now = time.perf_counter()
            elapsed = now - start
            if elapsed > TIME_LIMIT_S:
                break
            count, xy, vxy, attitude = sensors.read()
            yaw = attitude["yaw"]
            p = np.asarray(mission.start_xy) + rot @ ((xy - origin) / 100.0)
            v = rot @ (vxy / 100.0) if vxy is not None else np.zeros(2)
            speed = float(np.hypot(*vxy)) if vxy is not None else np.nan
            plate, came_from, lost = follow_plate(plate, came_from, lost, p)
            if stop_since is None:
                if anchor is None or elapsed < quiet_until or np.linalg.norm(p - anchor) > STALL_M:
                    anchor, anchor_t = p, elapsed
                elif elapsed - anchor_t >= STALL_S:
                    anchor, anchor_t, quiet_until = p, elapsed, elapsed + QUIET_S
                    if held_on_plate(p, mission.goal_cell):
                        stop_since = elapsed             # stalled, but on the goal: that will do
                    elif args.recover:
                        back, route, legs = recovery(p, legs, leg, plate, came_from, mission.goal_cell,
                                                     stalls, avoid)
                        sides = [wall_side(a, b) for a, b in zip(legs, legs[1:])]
                        leg, turning, turn_since, backoff_leg = 0, True, elapsed, 0
                        recoveries += 1
                        print(f"stalled at ({p[0]:+.2f}, {p[1]:+.2f}) on {plate_name(plate)} at {elapsed:.1f} s: "
                              f"back to {plate_name(back)}, then " + " ".join(plate_name(c) for c in route[1:]))
            if stop_since is None:
                before = leg
                recent.append(v)
                v_mean = np.mean(recent, axis=0)
                heading, raw, leg, stopping, turning, braking = command_for(p, v_mean, legs, leg, turning, args,
                                                                            sides, backoff_leg)
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
                raw, braking = 0, False
            sdk = send(heading, raw)
            log.append({"t": round(elapsed, 4), "frames": count, "x": p[0], "y": p[1], "vx": v[0], "vy": v[1],
                        "speed_cm_s": speed, "leg": leg, "turning": int(turning), "cmd_heading_maze": heading,
                        "cmd_heading_sdk": sdk, "raw": raw, "yaw_sdk": yaw,
                        "pitch": attitude["pitch"], "roll": attitude["roll"], "plate": f"{plate[0]},{plate[1]}",
                        "brake": int(braking), "stopping": int(stop_since is not None), "recoveries": recoveries})
            if stop_since is not None:
                if speed <= STOPPED_CM_S:
                    stopped_since = elapsed if stopped_since is None else stopped_since
                    if elapsed - stopped_since >= STOPPED_FOR_S:
                        outcome = "arrived" if held_on_plate(p, mission.goal_cell) else "stopped-off-goal"
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
        print(f"{outcome}: {end['t']:.2f} s from the first command, {end['recoveries']} stall recoveries"
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
    parser.add_argument("--boost", type=int, default=80,
                        help="raw speed added when the ball is stalled or rolling back (0 = off)")
    parser.add_argument("--stop-lead", type=float, default=10.0, help="cm before the goal to tell the robot to stop")
    parser.add_argument("--corner-speed", type=float, default=0.35,
                        help="m/s above which the ball is braked near a corner (0 = off)")
    parser.add_argument("--wall-bias", type=float, default=0.0,
                        help="deg toward a wall that runs along a whole leg (0 = off: follow the centre line)")
    parser.add_argument("--recover", action="store_true",
                        help="after a stall, back off and re-plan (off by default)")
    args = parser.parse_args(argv)

    mission = make_mission(args.start, args.heading, args.goal)
    cells = plan_cells(mission.start_cell, mission.start_heading_deg, mission.goal_cell)
    print(mission.describe())
    print(f"route {' '.join(plate_name(c) for c in cells)}: {len(cells) - 1} plates, "
          f"{count_turns(cells, mission.start_heading_deg)} turns; raw {args.raw} (+{args.boost} when stalled), "
          f"turn {args.turn_lead:.0f} cm early, stop {args.stop_lead:.0f} cm early, "
          f"corner speed {args.corner_speed:g} m/s, wall bias {args.wall_bias:g} deg, "
          f"stall recovery {'on' if args.recover else 'off'}")

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
                    f"_raw{args.raw}_b{args.boost}_tl{args.turn_lead:g}_sl{args.stop_lead:g}"
                    f"_cs{args.corner_speed:g}_wb{args.wall_bias:g}_rec{int(args.recover)}.csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(log[0]))
                    writer.writeheader()
                    writer.writerows(log)
                print(f"Wrote {len(log)} steps to {path.name}")
    print("Disconnected.")


if __name__ == "__main__":
    main()
