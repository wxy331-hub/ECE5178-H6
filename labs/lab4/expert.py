"""The teacher: Lab 3's planner and controller, asked about one state at a time.

Lab 3's WaypointFollower keeps history -- which waypoint it is on, which it
has already reached.  A teacher for behaviour cloning has to answer for any
state it is shown, including states the learned policy reached by a route of
its own, so this one keeps none.  It plans from the state, drives at the plan's
first or second waypoint, and takes Lab 3's controller's heading.  Unlike Lab
3 it never stops to turn: a ball stopped on the slope after the second corner
rolled back (2026-10-02), so it always drives and the safety layer takes a
large turn at TURN_SPEED instead.
"""

import functools

import numpy as np

from runtime import (
    ARRIVAL_TOLERANCE,
    CRUISE_SPEED,
    DT,
    GOAL,
    GRID_RESOLUTION,
    MAX_TURN_RATE,
    PLANNER,
    WAYPOINT_TOLERANCE,
    keeps_clear,
    lab3,
    wrap_angle,
)

CONTROLLER = lab3.Controller(dt=DT)
_REACHABLE = PLANNER._connected(PLANNER._nearest_free(PLANNER.world_to_grid(GOAL)))
_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EDGE = 0.625


@functools.lru_cache(maxsize=None)
def _plan_from(cell):
    """The plan from anywhere in a grid cell: Planner.plan only uses the cell."""
    centre = PLANNER.grid_to_world(cell)
    return tuple(
        np.asarray(w, dtype=np.float64)
        for w in PLANNER.plan(np.array([centre[0], centre[1], 0.0, 0.0]), GOAL)
    )


def why_unanswerable(position):
    """None if the teacher can answer for this position, else the reason."""
    if not (abs(position[0]) < _EDGE and abs(position[1]) < _EDGE):
        return "outside"
    cell = PLANNER.world_to_grid(position)
    if cell in _REACHABLE:
        return None
    if PLANNER._is_free(cell):
        return "unreachable"  # plate (0, 1), walled in on every side
    # The grid draws walls as bands one cell wide, and the locator puts the
    # estimate a few centimetres into them on most runs; answer there, but not
    # for positions deeper than the band next to the floor.
    if any((cell[0] + dr, cell[1] + dc) in _REACHABLE for dr, dc in _NEIGHBOURS):
        return None
    return "inside-wall"


def target_for(position, heading):
    """The waypoint to drive at from here.

    The plan starts on the cell the position snapped to.  That start is
    skipped -- as Lab 3's _resume_target does after a replan -- when the
    robot is level with it or past it along the first leg and inside that
    leg's corridor, and only when the straight run to the second waypoint
    stays off the walls, so the teacher never cuts a corner into one.

    At a corner that alone would turn the robot as soon as it enters the
    corner's grid cell, 6 cm short of the centre, and a robot slower than
    the model is further back still: it turns into the wall.  Lab 3's
    follower remembered that it had not reached the corner yet; without
    memory, the heading says the same thing.  A robot still facing the way
    it came drives on to within WAYPOINT_TOLERANCE of the corner first; one
    already facing along the next leg -- drifting sideways as it drove, as on
    plate (3, 0) on 2026-09-18 -- keeps going.
    """
    waypoints = _plan_from(PLANNER.world_to_grid(position))
    start = waypoints[0]
    # A wall-band position can snap to a cell on the far side of the thin
    # wall; a target it cannot drive to straight is no label at all.
    if len(waypoints) == 1:
        return start if keeps_clear(position, start) else None
    following = waypoints[1]
    leg = following - start
    direction = leg / np.linalg.norm(leg)
    offset = position - start
    along = float(offset @ direction)
    across = abs(float(offset[0] * direction[1] - offset[1] * direction[0]))
    facing_leg = abs(wrap_angle(float(heading) - np.arctan2(direction[0], direction[1]))) < np.pi / 4
    reached = float(np.linalg.norm(offset)) <= WAYPOINT_TOLERANCE
    if (along > -WAYPOINT_TOLERANCE and across < GRID_RESOLUTION
            and (facing_leg or reached) and keeps_clear(position, following)):
        return following
    return start if keeps_clear(position, start) else None


def label_state(state):
    """The teacher's [speed, turn rate] for a state, and why if there is none.

    The turn rate is what gets the robot from its heading to the controller's
    heading in one step, clipped to what the calibrated model can turn.
    """
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (4,) or not np.all(np.isfinite(state)):
        return None, "invalid"
    position = state[:2]
    reason = why_unanswerable(position)
    if reason is not None:
        return None, reason
    if float(np.linalg.norm(position - GOAL)) <= ARRIVAL_TOLERANCE:
        return np.zeros(2), "at-goal"
    target = target_for(position, state[2])
    if target is None:
        return None, "across-wall"
    _, heading = CONTROLLER.compute_action(state, target)
    rate = float(np.clip(wrap_angle(heading - state[2]) / DT, -MAX_TURN_RATE, MAX_TURN_RATE))
    return np.array([CRUISE_SPEED, rate]), "ok"


def expert_policy(state):
    """The teacher as a controller, for collecting runs: stops where it has no answer."""
    action, _ = label_state(state)
    return np.zeros(2) if action is None else action
